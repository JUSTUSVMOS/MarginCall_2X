import os
import sys
import sqlite3
import pandas as pd
import torch
import requests
import json
import concurrent.futures
from datetime import datetime, timedelta
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import cloudscraper  # ⚠️ 新增：用於打穿 Cloudflare 防護

# --- 0. Ubuntu 路徑配置與全局 SEC 偽裝 ---
PROJECT_ROOT = "/home/margincaller/MarginCall_2X"
FINNLP_PATH = os.path.join(PROJECT_ROOT, "FinNLP-main", "FinNLP-main")
if os.path.exists(FINNLP_PATH):
    sys.path.append(FINNLP_PATH)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

from finnlp.data_sources.news.finnhub_date_range import Finnhub_Date_Range
from finnlp.data_sources.social_media.stocktwits_streaming import Stocktwits_Streaming
from finnlp.data_sources.company_announcement.sec import SEC_Announcement
from finnlp.data_sources.sec_filings.sec_filings import SECExtractor
from bs4 import BeautifulSoup
import re

# SEC 官方要求必須提供 User-Agent 與聯繫 Email
SEC_HEADERS = {
    "User-Agent": "MarginCall Bot (research@margincall.ai)",
    "Accept-Encoding": "gzip, deflate"
}

# --- 1. 資料庫配置 ---
DB_FILE = os.path.join(PROJECT_ROOT, "portfolio.db")

def get_cik(symbol):
    """
    從 SEC 官方標籤對應表動態獲取 CIK (補足 10 位)。
    """
    symbol = symbol.upper()
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        res = requests.get(url, headers={"User-Agent": "MarginCall Bot (research@margincall.ai)"}, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for key, val in data.items():
                if val['ticker'] == symbol:
                    return str(val['cik_str']).zfill(10)
    except Exception as e:
        print(f"⚠️ CIK 查詢異常: {e}")
    
    # 回退方案
    CIK_MAP = {"ARM": "0001045810", "TSLA": "0001318605", "AAPL": "0000320193", "AVGO": "0001730168"}
    return CIK_MAP.get(symbol, "0001045810")

def parse_form4_insider(content):
    if not content: return "無內容"
    try:
        soup = BeautifulSoup(content, 'xml') 
        transactions = soup.find_all(['nonDerivativeTransaction', 'derivativeTransaction'])
        if not transactions: return "【內部人變動】未發現實質交易標籤。"
        net_value = 0
        total_shares = 0
        for txn in transactions:
            try:
                code_tag = txn.find('transactionCode')
                shares_tag = txn.find('transactionShares')
                price_tag = txn.find('transactionPricePerShare')
                if not (code_tag and shares_tag): continue
                code = code_tag.find('value').get_text(strip=True) if code_tag.find('value') else code_tag.get_text(strip=True)
                shares_val = float(shares_tag.find('value').get_text(strip=True)) if shares_tag.find('value') else 0
                price_val = float(price_tag.find('value').get_text(strip=True)) if price_tag and price_tag.find('value') else 0
                if code == 'P':
                    net_value += shares_val * price_val
                    total_shares += shares_val
                elif code == 'S':
                    net_value -= shares_val * price_val
                    total_shares -= shares_val
            except: continue
        if net_value > 0: return f"【內部人增持】淨買入約 ${net_value:,.0f}"
        elif net_value < 0: return f"【內部人減持】淨賣出約 ${abs(net_value):,.0f}"
        return "【內部人變動】無顯著方向。"
    except: return "【解析異常】XML 解析失敗"

def init_nlp_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(nlp_insights)")
    columns = [column[1] for column in cursor.fetchall()]
    if columns and "nlp_alpha" not in columns:
        cursor.execute("DROP TABLE nlp_insights")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nlp_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp TEXT,
            nlp_alpha REAL,
            alpha_retail REAL,
            alpha_macro REAL,
            alpha_official REAL,
            total_items INTEGER,
            summary_text TEXT,
            insight_type TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(symbol, nlp_alpha, a_ret, a_mac, a_off, total, summary, i_type):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO nlp_insights 
        (symbol, timestamp, nlp_alpha, alpha_retail, alpha_macro, alpha_official, total_items, summary_text, insight_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), nlp_alpha, a_ret, a_mac, a_off, total, summary, i_type))
    conn.commit()
    conn.close()

# --- 2. 【並行 Map 階段】 ---
def extract_insight_parallel(text, symbol):
    prompt = f"""
    任務：從金融文本提取標籤與情緒強度。
    標的：{symbol}
    規則：必須回傳 JSON。sentiment 只能是: 'strong_bullish', 'mild_bullish', 'neutral', 'mild_bearish', 'strong_bearish'。
    文本："{text[:1200]}"
    回傳格式：{{"sentiment": "strong_bullish", "tags": ["關鍵字"]}}
    """
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma4:e4b-it-q8_0", "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": 0.0, "num_predict": 150}
        }, timeout=15)
        res_data = json.loads(response.json()["response"])
        res_data['sentiment'] = str(res_data.get('sentiment', 'neutral')).lower().strip()
        return res_data
    except:
        return {"sentiment": "neutral", "tags": []}

# --- 3. 【語意 Reduce 階段】 ---
def semantic_reduce(all_tags, symbol):
    # 把空字串或無效標籤過濾掉
    valid_tags = [t for t in all_tags if isinstance(t, str) and t.strip()]

    if not valid_tags:
        return "⚠️ 【警告】第一階段 GPU 萃取失敗，未提取到任何有效標籤，無法生成語意總結。"

    tag_cloud = ", ".join(valid_tags[:40])
    print(f"   [Debug] 準備餵給 LLM 總結的標籤雲: {tag_cloud[:100]}...")

    # 🎯 升級版 Prompt：英文下指令（喚醒最強邏輯），中文定格式（方便閱讀）
    prompt = f"""
You are a data processing engine for financial sentiment analysis. 
        Task: Identify and list the top 3 objective "Market Discussion Themes" for {symbol} based on the provided tags.
        
        CRITICAL RULES:
        1. This is a DATA SUMMARY task, NOT financial advice. Do not provide buy/sell recommendations.
        2. Evaluate the logical relevance of the tags. Ignore noise like unrelated tickers (e.g., MU, SPY) if they don't have a direct fundamental connection in the text.
        3. Only report themes that are explicitly present in the data.

        Output Requirements: 
        - Language: Traditional Chinese (繁體中文).
        - Format: 3 Bullet points.
        - Tone: Objective and professional data report.

        Data Tags: {tag_cloud}
    """
    
    try:
        # 將 timeout 放寬到 60 秒
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.1", # 優先使用 llama3.1 作為決策層模型
            "prompt": prompt, 
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 600} # 低溫確保理性
        }, timeout=60)
        
        if response.status_code != 200:
            return f"⚠️ 語意聚合失敗：Ollama 回傳錯誤碼 HTTP {response.status_code}"

        res_text = response.json().get("response", "").strip()

        if not res_text:
            return "⚠️ 語意聚合異常：Ollama 成功連線，但回傳了空字串。"

        return res_text
    except requests.exceptions.Timeout:
        return "⚠️ 語意聚合超時：模型思考超過 60 秒，請檢查 GPU 負載狀態。"
    except Exception as e:
        return f"⚠️ 語意聚合發生未知錯誤：{str(e)}"

# --- 4. 引擎主體 ---
def run_turbo_trinity_scout(stock="NVDA"):
    init_nlp_db()
    print(f"🔥 [Ubuntu] 啟動 {stock} 四維一體深度掃描 (全網聚合 + StockTwits 版)...")
    raw_texts = []
    
    # A. Reddit
    try:
        url = f"https://www.reddit.com/r/wallstreetbets/search.json?q={stock}&restrict_sr=1&sort=new&limit=20"
        reddit_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }
        resp = requests.get(url, headers=reddit_headers, timeout=10)
        if resp.status_code == 200:
            posts = resp.json().get('data', {}).get('children', [])
            valid_count = 0
            for p in posts:
                title = p['data']['title']
                body = p['data'].get('selftext', '')
                full_text = f"{title}\n{body}"
                
                # 強制檢查：文章內必須要有大寫的代碼或加上錢字號的代碼
                if re.search(rf'\b{stock}\b|\${stock}', full_text):
                    raw_texts.append(f"Reddit: {title} | {body[:200]}")
                    valid_count += 1
            print(f"   ✅ Reddit: {valid_count} 筆 (過濾後)")
    except Exception as e: print(f"   ⚠️ Reddit 異常: {e}")

    # B. StockTwits 即時多空情緒 (Cloudscraper 穿甲版)
    try:
        # 使用 cloudscraper 建立具備真實 TLS 指紋的會話
        scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
        st_url = f"https://api.stocktwits.com/api/2/streams/symbol/{stock}.json"
        st_resp = scraper.get(st_url, timeout=15)
        
        if st_resp.status_code == 200:
            st_data = st_resp.json()
            messages = st_data.get('messages', [])
            for m in messages:
                body = m.get('body', '')
                raw_texts.append(f"StockTwits: {body[:300]}")
            print(f"   ✅ StockTwits: {len(messages)} 筆 (穿甲成功)")
        else:
            print(f"   ⚠️ StockTwits 穿甲失敗 (HTTP {st_resp.status_code})")
    except Exception as e:
        print(f"   ⚠️ StockTwits 穿甲異常: {e}")

    # C. Finnhub 全網媒體聚合
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        fh_downloader = Finnhub_Date_Range({"token": FINNHUB_KEY})
        fh_downloader.download_date_range_stock(stock=stock, start_date=start_date, end_date=end_date)
        if fh_downloader.dataframe is not None and not fh_downloader.dataframe.empty:
            df_fh = fh_downloader.dataframe.head(15) 
            for _, row in df_fh.iterrows():
                source = row.get('source', 'News')
                headline = row.get('headline', '')
                summary = row.get('summary', '')[:300]
                raw_texts.append(f"Macro({source}): {headline} | {summary}")
            print(f"   ✅ Macro(Finnhub): {len(df_fh)} 筆")
    except Exception as e: 
        print(f"   ⚠️ Finnhub 抓取失敗: {e}")

    # D. SEC
    try:
        cik = get_cik(stock)
        print(f"   🔍 CIK: {cik}")
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        res = requests.get(url, headers=SEC_HEADERS, timeout=15)
        
        if res.status_code == 200:
            sec_data = res.json()
            recent_filings = sec_data.get("filings", {}).get("recent", {})
            forms = recent_filings.get("form", [])
            accessions = recent_filings.get("accessionNumber", [])
            docs = recent_filings.get("primaryDocument", [])
            dates = recent_filings.get("filingDate", [])

            # ==========================================
            # 軌道一：深度解剖「最新一份年報」(10-K / 20-F)
            # ==========================================
            print(f"   🕵️ 軌道 1：搜尋 {stock} 最新年報...")
            for i, form in enumerate(forms):
                if form in ['10-K', '20-F']:
                    acc_num = str(accessions[i]).replace('-', '')
                    doc_name = docs[i]
                    annual_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num}/{doc_name}"
                    print(f"   🎯 找到最新年報 ({form}, 發布於 {dates[i]})，準備解析...")

                    doc_res = requests.get(annual_url, headers=SEC_HEADERS, timeout=15)
                    if doc_res.status_code == 200:
                        # 分流 1：美國本土公司 (10-K)，使用 FinNLP 精準打擊
                        if form == '10-K':
                            try:
                                extractor = SECExtractor(tickers=[stock], amount=1, filing_type=form)
                                # 🎯 修正：使用 FinNLP 官方定義的 Enum 名稱 (RISK_FACTORS=1A, MANAGEMENT_DISCUSSION=7)
                                narratives, _ = extractor.pipeline_api(doc_res.text, m_section=["RISK_FACTORS", "MANAGEMENT_DISCUSSION"])
                                
                                # 取得資料時對應官方 key 與可能的變體
                                item_1a_content = narratives.get("RISK_FACTORS") or narratives.get("_ITEM_1A") or narratives.get("ITEM_1A")
                                item_7_content = narratives.get("MANAGEMENT_DISCUSSION") or narratives.get("_ITEM_7") or narratives.get("ITEM_7")

                                if item_1a_content:
                                    text_1a = " ".join([item["text"] for item in item_1a_content if "text" in item])
                                    raw_texts.append(f"SEC 10-K [風險因素]: {text_1a[:3000]}")
                                    print("   ✅ FinNLP 成功切出 RISK_FACTORS (ITEM_1A)")

                                if item_7_content:
                                    text_7 = " ".join([item["text"] for item in item_7_content if "text" in item])
                                    raw_texts.append(f"SEC 10-K [營運分析]: {text_7[:3000]}")
                                    print("   ✅ FinNLP 成功切出 MANAGEMENT_DISCUSSION (ITEM_7)")
                                    
                            except Exception as parse_e:
                                print(f"   ⚠️ FinNLP 解析 10-K 失敗 ({parse_e})，自動降級啟用智能段落解析...")
                                # 啟動我們的降級神技：智能段落抓取 (Bypass iXBRL)
                                soup = BeautifulSoup(doc_res.text, 'html.parser')
                                valid_paragraphs = []
                                for p in soup.find_all(['p', 'span']):
                                    text = p.get_text(separator=' ', strip=True)
                                    # 過濾掉太短的數字表格和包含 gaap 會計標籤的亂碼
                                    if len(text) > 120 and 'us-gaap:' not in text.lower():
                                        valid_paragraphs.append(text)
                                        
                                clean_text = " ".join(valid_paragraphs)
                                clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
                                
                                # 10-K 前面廢話很多，我們略過前 2000 字，取隨後的 4000 字
                                raw_texts.append(f"SEC 10-K [年報摘要]: {clean_text[2000:6000]}")
                                
                        # 分流 2：外國發行人 (20-F)，使用智能段落抓取避開 iXBRL 亂碼
                        elif form == '20-F':
                            print("   ℹ️ 20-F 外國年報啟用智能段落解析 (過濾 iXBRL)...")
                            soup = BeautifulSoup(doc_res.text, 'html.parser')
                            
                            valid_paragraphs = []
                            # 專門抓取段落 <p> 或文字區塊 <span>
                            for p in soup.find_all(['p', 'span']):
                                text = p.get_text(separator=' ', strip=True)
                                
                                # 🛡️ 核心濾網：長度太短不要(通常是表格數字)，包含 gaap 標籤的不要
                                if len(text) > 120 and 'us-gaap:' not in text.lower():
                                    valid_paragraphs.append(text)
                                    
                            clean_text = " ".join(valid_paragraphs)
                            clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
                            
                            upper_text = clean_text.upper() # 轉大寫方便搜尋

                            # 🎯 第一刀：尋找「風險因素 (Risk Factors)」錨點 (通常是 Item 3.D)
                            risk_idx = upper_text.find("RISK FACTOR")
                            if risk_idx != -1:
                                raw_texts.append(f"SEC 20-F [風險因素]: {clean_text[risk_idx : risk_idx + 3000]}")
                            else:
                                # 找不到標題盲切：跳過前面 1萬字 的封面廢話，抓取中間段落
                                raw_texts.append(f"SEC 20-F [年報摘要A]: {clean_text[10000 : 13000]}")

                            # 🎯 第二刀：尋找「營運分析 (Operating and Financial Review)」錨點 (通常是 Item 5)
                            op_idx = upper_text.find("OPERATING AND FINANCIAL REVIEW")
                            if op_idx == -1:
                                op_idx = upper_text.find("ITEM 5") # 備用標題
                                
                            if op_idx != -1:
                                raw_texts.append(f"SEC 20-F [營運分析]: {clean_text[op_idx : op_idx + 3000]}")
                            else:
                                # 找不到標題盲切：再往後抓一段
                                raw_texts.append(f"SEC 20-F [年報摘要B]: {clean_text[13000 : 16000]}")
                            
                    break # 找到最新的一份就跳出迴圈

            # ==========================================
            # 軌道二：近期動態監控 (45天內)
            # ==========================================
            limit_date = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
            print(f"   🕵️ 軌道 2：監控 45 天內動態 (自 {limit_date} 起)...")
            
            dynamic_count = 0
            for i, form in enumerate(forms):
                if dates[i] < limit_date or dynamic_count >= 10:
                    break
                
                if form in ['4', '8-K', '6-K', '10-Q']:
                    acc_num = str(accessions[i]).replace('-', '')
                    doc_name = docs[i]
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num}/{doc_name}"
                    doc_res = requests.get(doc_url, headers=SEC_HEADERS, timeout=10)
                    
                    if doc_res.status_code == 200:
                        if form == '4':
                            raw_texts.append(f"SEC Form 4: {parse_form4_insider(doc_res.text)}")
                        else:
                            soup = BeautifulSoup(doc_res.text, 'html.parser')
                            clean_text = soup.get_text(separator=' ', strip=True)
                            raw_texts.append(f"SEC {form} ({dates[i]}): {clean_text[:1200]}")
                        dynamic_count += 1
            print(f"   ✅ SEC 雙軌解析完成")
        else:
            print(f"   ⚠️ SEC 請求被拒絕 (HTTP {res.status_code})")
    except Exception as e: 
        print(f"   ⚠️ SEC 抓取失敗: {e}")

    total = len(raw_texts)
    if total == 0: return print("📭 無情報")

    # --- 🔍 偵錯列印：檢視抓取內容 ---
    print("\n--- 🕵️ 原始情報抽樣檢視 ---")
    sources = ["Reddit", "StockTwits", "Macro", "SEC"]
    for src in sources:
        samples = [t for t in raw_texts if t.startswith(src)]
        print(f"\n【{src} 來源抽樣 ({len(samples)} 筆)】")
        for i, s in enumerate(samples[:3]):
            print(f"  {i+1}. {s[:150]}...")
    print("\n--- 🧠 啟動 GPU 語意分析 ---")

    # 並行分析
    all_tags = []
    platform_scores = {"Reddit": 0.0, "StockTwits": 0.0, "Macro": 0.0, "SEC": 0.0}
    platform_counts = {"Reddit": 0, "StockTwits": 0, "Macro": 0, "SEC": 0}
    SENT_MAP = {"strong_bullish": 1.0, "mild_bullish": 0.0, "neutral": 0.0, "mild_bearish": 0.0, "strong_bearish": -1.0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(extract_insight_parallel, t, stock): t for t in raw_texts}
        for f in concurrent.futures.as_completed(futures):
            src = futures[f]
            res = f.result()
            score = SENT_MAP.get(res['sentiment'], 0.0)
            if src.startswith("Reddit"): platform_scores["Reddit"] += score; platform_counts["Reddit"] += 1
            elif src.startswith("StockTwits"): platform_scores["StockTwits"] += score; platform_counts["StockTwits"] += 1
            elif src.startswith("Macro"): platform_scores["Macro"] += score; platform_counts["Macro"] += 1
            elif src.startswith("SEC"): platform_scores["SEC"] += score; platform_counts["SEC"] += 1
            all_tags.extend(res.get('tags', []))

    # 計算各平台的平均情緒強度
    a_red = (platform_scores["Reddit"] / platform_counts["Reddit"]) if platform_counts["Reddit"] > 0 else 0.0
    a_stw = (platform_scores["StockTwits"] / platform_counts["StockTwits"]) if platform_counts["StockTwits"] > 0 else 0.0
    a_mac = (platform_scores["Macro"] / platform_counts["Macro"]) if platform_counts["Macro"] > 0 else 0.0
    a_sec = (platform_scores["SEC"] / platform_counts["SEC"]) if platform_counts["SEC"] > 0 else 0.0
    
    # 組合 Retail (散戶) 分數：Reddit 佔一半，StockTwits 佔一半
    a_retail = (a_red + a_stw) / 2.0 if (platform_counts["Reddit"] > 0 or platform_counts["StockTwits"] > 0) else 0.0

    # 嚴格的專家權重公式：散戶 20% (已融合兩種來源), 媒體 30%, 官方 50%
    nlp_alpha = (a_retail * 0.2) + (a_mac * 0.3) + (a_sec * 0.5)

    report = f"📊 {stock} 戰報\n綜合 Alpha: {nlp_alpha:+.2f}\n" + semantic_reduce(all_tags, stock)
    save_to_db(stock, nlp_alpha, a_retail, a_mac, a_sec, total, report, "TRINITY")
    print(f"\n{report}")

if __name__ == "__main__":
    target = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    run_turbo_trinity_scout(target)
