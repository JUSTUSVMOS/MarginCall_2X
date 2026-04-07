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
import yfinance as yf
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
    symbol = symbol.upper().replace(".", "-") # 把 BRK.B 轉成 BRK-B
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        res = requests.get(url, headers=SEC_HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for key, val in data.items():
                if val['ticker'] == symbol:
                    return str(val['cik_str']).zfill(10)
    except Exception as e:
        print(f"⚠️ CIK 查詢異常: {e}")
    
    # 🚨🚨🚨 刪掉原本的 CIK_MAP！找不到就回傳 None，寧願沒資料也不要拿 NVDA 來瞎掰！
    return None

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
    # 🎯 拔除偷懶藉口，強制模型提取商業事實
    prompt = f"""
        Task: Extract market insights and sentiment from the financial text for {symbol}.
        Rules: 
        1. Return valid JSON. 
        2. Sentiment MUST be: 'strong_bullish', 'mild_bullish', 'neutral', 'mild_bearish', 'strong_bearish'.
        3. Insights: Max 3 short sentences. 
        4. 🚨 BOILERPLATE FILTER: You MUST IGNORE standard SEC cover page text (e.g., "Indicate by check mark", "correction of an error to previously issued financial statements", "forward-looking statements"). These are NOT insights. If the text only contains this boilerplate, return an empty array [].
        5. 🚨 STRICT GROUNDING: ONLY extract facts explicitly written in the text. DO NOT invent prices or events.
        
        Text: "{text[:2500]}"
        Format: {{"sentiment": "neutral", "insights": ["Insight 1", "Insight 2"]}}
        """
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma4:e4b-it-q8_0", 
            "prompt": prompt, 
            "stream": False, 
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 250}
        }, timeout=60)
        
        raw_res = response.json().get("response", "")
        
        # 🛡️ 終極淨化：剝除 LLM 常犯的 Markdown JSON 外衣
        clean_res = re.sub(r'^```json\s*|\s*```$', '', raw_res.strip(), flags=re.IGNORECASE|re.MULTILINE)
        
        res_data = json.loads(clean_res)
        res_data['sentiment'] = str(res_data.get('sentiment', 'neutral')).lower().strip()
        
        # 確保向下相容性，把 insights 轉進 tags
        res_data['tags'] = res_data.get('insights', res_data.get('tags', []))
        return res_data
        
    except json.JSONDecodeError as je:
        # 讓錯誤浮出水面，不再默默吞掉
        print(f"   [Debug] JSON 解析失敗: 原始回傳 -> {raw_res[:100]}...")
        return {"sentiment": "neutral", "tags": []}
    except Exception as e:
        print(f"   [Debug] Gemma 萃取異常: {e}")
        return {"sentiment": "neutral", "tags": []}
    
import textwrap # 如果最上面沒有 import，記得加上去

# --- 3. 【語意 Reduce 階段】 ---
def semantic_reduce(categorized_tags, symbol, company_name, sector):
    
    # 確保標籤是乾淨的字串，避免傳入 None 或怪異型別
    sec_insights = list(set([t for t in categorized_tags["SEC"] if isinstance(t, str) and t.strip()]))[:8]
    macro_insights = list(set([t for t in categorized_tags["Macro"] if isinstance(t, str) and t.strip()]))[:8]
    retail_insights = list(set([t for t in categorized_tags["Retail"] if isinstance(t, str) and t.strip()]))[:8]
    
    sec_tags = "\n".join([f"- {t[:80]}" for t in sec_insights])
    macro_tags = "\n".join([f"- {t[:80]}" for t in macro_insights])
    retail_tags = "\n".join([f"- {t[:80]}" for t in retail_insights])

    # 🚨 全英文 Prompt：降低模型語意轉換的負擔
    prompt = f"""
        You are a Senior Wall Street Analyst at a Top-Tier Hedge Fund.
        Summarize the top 3 core investment focuses for {symbol} ({company_name}, Sector: {sector}).
        
        DATA SOURCES:
        ---
        [OFFICIAL SEC FILINGS]:
        {sec_tags if sec_tags else "None available."}
        
        [MACRO/NEWS MEDIA]:
        {macro_tags if macro_tags else "None available."}
        
        [RETAIL SOCIAL MEDIA]:
        {retail_tags if retail_tags else "None available."}
        ---
        
        CRITICAL INSTRUCTIONS:
        1. FAT-TAIL RISK OVERRIDE: If (and ONLY if) ANY insight mentions "DOJ", "Indictment", "Fraud", "Subpoena", "Investigation", or "Delist", make this the #1 bullet point and state the severe legal danger.
        2. STRICT TEMPLATE: Unless overridden by Rule 1, you MUST strictly use the exact format provided in the OUTPUT FORMAT section below. Do NOT add any extra introductory sentences (e.g., "Here is the summary...").
        3. COMPETITOR FILTER: Ignore news strictly about competitors.
        4. ZERO HALLUCINATION (CRUCIAL): You are a strict synthesizer. You MUST ONLY use the facts provided in the DATA SOURCES above. Do NOT include any historical price ranges (e.g., "$50-$55"), geopolitical events (e.g., "China bans"), or competitor actions UNLESS they are explicitly written in the provided tags.
        5.IGNORE BOILERPLATE: If the [SEC] tags only contain generic phrases like "correction of an error", "forward-looking statements", or "check mark", you MUST treat it as NO DATA and output "該維度目前無重大資訊。"
        6. NO HALLUCINATION: If a category lacks concrete data, write "該維度目前無重大資訊".
        7. NO DISCLAIMERS (STRICT): Stop generating text immediately after the 3rd bullet point.

        OUTPUT FORMAT:
                {{
                    "sec_summary": "[Insert official SEC summary here]",
                    "macro_summary": "[Insert macro/news summary here]",
                    "retail_summary": "[Insert retail sentiment summary here]"
                }}
        """
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma4:e4b-it-q8_0", 
            "prompt": prompt, 
            "stream": False,
            "format": "json", 
            "options": {
                "temperature": 0.0, # 降到 0.0 追求最大穩定性
                "num_predict": 300
            }
        }, timeout=90)

        if response.status_code != 200:
            return f"⚠️ Ollama HTTP Error: {response.status_code} - {response.text[:100]}"
            
        raw_res = response.json().get("response", "")
        clean_res = re.sub(r'^```json\s*|\s*```$', '', raw_res.strip(), flags=re.IGNORECASE|re.MULTILINE)
        
        # 嘗試解析 JSON
        data = json.loads(clean_res)
        
        sec_text = data.get("sec_summary", "No significant information available.")
        macro_text = data.get("macro_summary", "No significant information available.")
        retail_text = data.get("retail_summary", "No significant information available.")

        final_report = textwrap.dedent(f"""
                • **官方基本面**：{sec_text}
                • **總經與新聞**：{macro_text}
                • **散戶情緒**：{retail_text}
                """).strip()

        return final_report
                
    except json.JSONDecodeError:
        print(f"   [Debug] JSON 崩潰，原始回傳: {raw_res[:150]}")
        return textwrap.dedent(f"""
        • **官方基本面**：{sec_tags[:80] if sec_tags else "無"}
        • **總經與新聞**：{macro_tags[:80] if macro_tags else "無"}
        • **散戶情緒**：{retail_tags[:80] if retail_tags else "無"}
        """).strip()
    except Exception as e:
        return f"⚠️ Semantic Reduce Error: {str(e)}"

# --- 4. 引擎主體 ---
def run_turbo_trinity_scout(stock="NVDA"):
    # --- 0. 動態獲取公司背景 ---
    try:
        ticker_info = yf.Ticker(stock).info
        company_name = ticker_info.get('longName', stock)
        sector = ticker_info.get('sector', 'Unknown Sector')
        industry = ticker_info.get('industry', 'Unknown Industry')
        print(f"🧬 偵測到標的：{company_name} | 產業：{sector} / {industry}")
    except:
        company_name, sector, industry = stock, "Financial", "Technology" # 失敗時的備援

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
                    acc_num_raw = str(accessions[i])
                    acc_num = acc_num_raw.replace('-', '')
                    annual_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num}/{acc_num_raw}.txt"
                    print(f"   🎯 找到最新年報 ({form}, 發布於 {dates[i]})，準備解析...")

                    doc_res = requests.get(annual_url, headers=SEC_HEADERS, timeout=15)
                    if doc_res.status_code == 200:
                        # 分流 1：美國本土公司 (10-K)，使用 FinNLP 精準打擊
                        if form == '10-K':
                            try:
                                from finnlp.data_sources.sec_filings.prepline_sec_filings.sec_document import SECDocument
                                # 🎯 匯入 FinNLP 官方的 Enum 字典
                                from finnlp.data_sources.sec_filings.prepline_sec_filings.sections import section_string_to_enum

                                sec_doc = SECDocument.from_string(doc_res.text)

                                # 💡 修正：傳入正確的 Enum 物件，而不是數字
                                item_1a_content = sec_doc.get_section_narrative(section_string_to_enum["RISK_FACTORS"])
                                item_7_content = sec_doc.get_section_narrative(section_string_to_enum["MANAGEMENT_DISCUSSION"])

                                sec_extracted_texts = []
                                if item_1a_content:
                                    text_1a = " ".join([item["text"] for item in item_1a_content if isinstance(item, dict) and "text" in item]) if isinstance(item_1a_content, list) else str(item_1a_content)
                                    if len(text_1a) > 200:
                                        safe_start = 2500 if len(text_1a) > 5000 else 0
                                        sec_extracted_texts.append(f"SEC 10-K [風險因素]: {text_1a[safe_start : safe_start+2500]}")
                                        print(f"   ✅ FinNLP 成功抓取 RISK_FACTORS (長度: {len(text_1a)})")

                                if item_7_content:
                                    text_7 = " ".join([item["text"] for item in item_7_content if isinstance(item, dict) and "text" in item]) if isinstance(item_7_content, list) else str(item_7_content)
                                    if len(text_7) > 200:
                                        safe_start = 2500 if len(text_7) > 5000 else 0
                                        sec_extracted_texts.append(f"SEC 10-K [營運分析]: {text_7[safe_start : safe_start+2500]}")
                                        print(f"   ✅ FinNLP 成功抓取 MD&A (長度: {len(text_7)})")

                                if not sec_extracted_texts:
                                    raise ValueError("FinNLP 提取內容過短或為空")
                                
                                raw_texts.extend(sec_extracted_texts)

                            except Exception as parse_e:
                                print(f"   ⚠️ FinNLP 解析 10-K 失敗或內容為空 ({parse_e})，自動降級啟用智能段落解析...")
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
                                upper_text = clean_text.upper()
                                
                                risk_idx = upper_text.find("RISK FACTOR")
                                if risk_idx != -1:
                                    raw_texts.append(f"SEC 10-K [風險因素]: {clean_text[risk_idx : risk_idx + 3000]}")
                                    print(f"   ✅ 錨點命中 Risk Factors (位置: {risk_idx})")

                                mda_idx = upper_text.find("MANAGEMENT'S DISCUSSION AND ANALYSIS")
                                if mda_idx == -1:
                                    mda_idx = upper_text.find("MANAGEMENT\u2019S DISCUSSION") # 有些用 fancy 撇號
                                if mda_idx != -1:
                                    second_hit = upper_text.find("MANAGEMENT", mda_idx + 200)
                                    if second_hit != -1 and (second_hit - mda_idx) < 50000:
                                        mda_idx = second_hit
                                    raw_texts.append(f"SEC 10-K [營運分析]: {clean_text[mda_idx : mda_idx + 3000]}")
                                    print(f"   ✅ 錨點命中 MD&A (位置: {mda_idx})")

                                # 都找不到才用盲切，但跳過前 15000 字（封面+目錄+法律聲明）
                                if risk_idx == -1 and mda_idx == -1:
                                    raw_texts.append(f"SEC 10-K [年報摘要]: {clean_text[15000:19000]}")
                                    print(f"   ⚠️ 錨點全部未命中，盲切 [15000:19000]")
                                
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
                        # 情況 A：如果是內部人交易 (Form 4)
                        if form == '4':
                            insider_result = parse_form4_insider(doc_res.text)
                            # 🛡️ 內部濾網：如果沒有真正的買賣金額，直接丟棄這份文件！
                            if "未發現實質交易" not in insider_result and "無顯著方向" not in insider_result and "解析異常" not in insider_result:
                                raw_texts.append(f"SEC Form 4: {insider_result}")
                        
                        # 情況 B：如果是其他重大公告 (8-K, 6-K, 10-Q)
                        else:
                            soup = BeautifulSoup(doc_res.text, 'html.parser')
                            clean_text = soup.get_text(separator=' ', strip=True)
                            clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
                            upper_text = clean_text.upper()

                            extracted = ""

                            if form == '8-K':
                                # 8-K 的真正內容在 "Item X.XX" 之後
                                # 常見：Item 2.02 (財報), Item 1.01 (重大合約), Item 5.02 (人事), Item 8.01 (其他)
                                item_match = re.search(r'ITEM\s+\d+\.\d+', upper_text)
                                if item_match:
                                    start = item_match.start()
                                    extracted = clean_text[start : start + 1500]
                                    
                            elif form == '10-Q':
                                # 10-Q 的 MD&A 通常在 Item 2
                                mda_idx = upper_text.find("MANAGEMENT'S DISCUSSION")
                                if mda_idx == -1:
                                    mda_idx = upper_text.find("MANAGEMENT\u2019S DISCUSSION")
                                if mda_idx != -1:
                                    extracted = clean_text[mda_idx : mda_idx + 2000]

                            elif form == '6-K':
                                # 6-K (外國公司) 找正文開頭：通常在 "SIGNATURE" 之前的最後大段文字
                                sig_idx = upper_text.find("SIGNATURE")
                                if sig_idx > 2000:
                                    # 從中間段開始抓，避開封面
                                    mid = sig_idx // 2
                                    extracted = clean_text[mid : mid + 1500]

                            # Fallback：如果關鍵字都沒命中，跳過前 800 字（封面）再抓
                            if not extracted:
                                extracted = clean_text[800:2000]

                            # 最終過濾：如果內容超過 60% 是法律套話，直接丟棄
                            boilerplate_signals = ['check mark', 'indicate by', 'forward-looking', 
                                                   'securities registered', 'commission file']
                            boilerplate_count = sum(1 for sig in boilerplate_signals if sig in extracted.lower())
                            
                            if boilerplate_count < 3:
                                raw_texts.append(f"SEC {form} ({dates[i]}): {extracted}")
                        
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
    categorized_tags = {"SEC": [], "Macro": [], "Retail": []}
    platform_scores = {"Reddit": 0.0, "StockTwits": 0.0, "Macro": 0.0, "SEC": 0.0}
    platform_counts = {"Reddit": 0, "StockTwits": 0, "Macro": 0, "SEC": 0}
    SENT_MAP = {"strong_bullish": 1.0, "mild_bullish": 0.5, "neutral": 0.0, "mild_bearish": 0.0, "strong_bearish": -1.0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(extract_insight_parallel, t, stock): t for t in raw_texts}
        for f in concurrent.futures.as_completed(futures):
            src = futures[f]
            res = f.result()
            score = SENT_MAP.get(res['sentiment'], 0.0)
            tags = res.get('tags', [])
            
            # 👇 依照來源，把標籤放入不同的籃子
            if src.startswith("Reddit") or src.startswith("StockTwits"): 
                if src.startswith("Reddit"):
                    platform_scores["Reddit"] += score
                    platform_counts["Reddit"] += 1
                else:
                    platform_scores["StockTwits"] += score
                    platform_counts["StockTwits"] += 1
                categorized_tags["Retail"].extend(tags)
            elif src.startswith("Macro"): 
                platform_scores["Macro"] += score
                platform_counts["Macro"] += 1
                categorized_tags["Macro"].extend(tags)
            elif src.startswith("SEC"): 
                platform_scores["SEC"] += score
                platform_counts["SEC"] += 1
                categorized_tags["SEC"].extend(tags)

    # 計算各平台的平均情緒強度
    a_red = (platform_scores["Reddit"] / platform_counts["Reddit"]) if platform_counts["Reddit"] > 0 else 0.0
    a_stw = (platform_scores["StockTwits"] / platform_counts["StockTwits"]) if platform_counts["StockTwits"] > 0 else 0.0
    a_mac = (platform_scores["Macro"] / platform_counts["Macro"]) if platform_counts["Macro"] > 0 else 0.0
    a_sec = (platform_scores["SEC"] / platform_counts["SEC"]) if platform_counts["SEC"] > 0 else 0.0
    
    # 組合 Retail (散戶) 分數：Reddit 佔一半，StockTwits 佔一半
    a_retail = (a_red + a_stw) / 2.0 if (platform_counts["Reddit"] > 0 or platform_counts["StockTwits"] > 0) else 0.0

    # 嚴格的專家權重公式：散戶 20% (已融合兩種來源), 媒體 30%, 官方 50%
    nlp_alpha = (a_retail * 0.2) + (a_mac * 0.3) + (a_sec * 0.5)

    # 🚨 核彈級利空熔斷機制 (Tail-Risk Override)
    # 把所有新聞跟 SEC 的原始字串轉小寫，檢查是否有毀滅性字眼
    macro_sec_text = " ".join([t.lower() for t in raw_texts if t.startswith("Macro") or t.startswith("SEC")])
    nuclear_keywords = ['doj', 'indictment', 'subpoena', 'delist', 'fraud', 'accounting irregularity', 'investigation']
    
    triggered_nukes = [k for k in nuclear_keywords if k in macro_sec_text]
    if triggered_nukes:
        print(f"   ☢️ 警告！偵測到核彈級利空字眼: {triggered_nukes}，觸發 Alpha 熔斷機制！")
        nlp_alpha = -0.95 # 強制給予極度悲觀的量化分數

    report = f"📊 {stock} 戰報\n綜合 Alpha: {nlp_alpha:+.2f}\n" + semantic_reduce(categorized_tags, stock, company_name, sector)
    save_to_db(stock, nlp_alpha, a_retail, a_mac, a_sec, total, report, "TRINITY")
    print(f"\n{report}")

if __name__ == "__main__":
    target = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    run_turbo_trinity_scout(target)
