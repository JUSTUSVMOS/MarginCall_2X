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
            "model": "llama3.1", "prompt": prompt, "stream": False, "format": "json",
            "options": {"temperature": 0.0, "num_predict": 150}
        }, timeout=15)
        res_data = json.loads(response.json()["response"])
        res_data['sentiment'] = str(res_data.get('sentiment', 'neutral')).lower().strip()
        return res_data
    except:
        return {"sentiment": "neutral", "tags": []}

# --- 3. 【語意 Reduce 階段】 ---
def semantic_reduce(all_tags, symbol):
    if not all_tags: return "無明顯集中事件。"
    tag_cloud = ", ".join(all_tags[:40])
    prompt = f"""
    你是華爾街頂級量化研究員。請根據以下標籤，總結市場對 {symbol} 的 3 個核心關注點。
    要求：繁體中文，條列式，深度分析。標籤集：{tag_cloud}
    """
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.1", "prompt": prompt, "stream": False,
            "options": {"temperature": 0.2, "num_predict": 600}
        }, timeout=30)
        return response.json().get("response", "語意聚合失敗").strip()
    except:
        return "語意聚合失敗。"

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
            for p in posts:
                raw_texts.append(f"Reddit: {p['data']['title']} | {p['data'].get('selftext', '')[:200]}")
            print(f"   ✅ Reddit: {len(posts)} 筆")
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
            df_sec = pd.DataFrame(res.json()['filings']['recent'])
            df_sec['filingDate'] = pd.to_datetime(df_sec['filingDate'])
            df_recent = df_sec[df_sec['filingDate'] >= (datetime.now() - timedelta(days=45))]
            
            # --- 🕵️ SEC 偵察兵：檢視 API 是否存活 ---
            print(f"   🕵️ {stock} 近 45 天實際發布的表單類型有: {df_recent['form'].unique().tolist()}")
            
            high_signal = df_recent[df_recent['form'].isin(['4', '8-K', '10-K', '10-Q'])].head(5)
            for _, row in high_signal.iterrows():
                acc = str(row['accessionNumber']).replace('-', '')
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{row['primaryDocument']}"
                doc_res = requests.get(doc_url, headers=SEC_HEADERS, timeout=15)
                if doc_res.status_code == 200:
                    if row['form'] == '4': 
                        raw_texts.append(f"SEC Form 4: {parse_form4_insider(doc_res.text)}")
                    else: 
                        soup = BeautifulSoup(doc_res.text, 'html.parser')
                        clean_text = soup.get_text(separator=' ', strip=True)
                        raw_texts.append(f"SEC {row['form']}: {clean_text[:1200]}")
            print(f"   ✅ SEC 解析完成")
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
