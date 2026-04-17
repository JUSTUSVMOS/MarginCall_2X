import os
import sys
import logging
import pandas as pd
import requests
import json
import textwrap
import concurrent.futures
from datetime import datetime, timedelta
from collections import Counter
import yfinance as yf
from yf_session import get_ticker, get_download
import cloudscraper  # ⚠️ 新增：用於打穿 Cloudflare 防護
from config import PROJECT_ROOT
from src.database import db_lock, get_connection

# --- 0. 引用自建輕量版 NLP 引擎 ---
from engine_nlp import FinnhubNewsDownloader as Finnhub_Date_Range

from dotenv import load_dotenv
load_dotenv(os.path.join(str(PROJECT_ROOT), ".env"))
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

# SEC 官方要求必須提供 User-Agent 與聯繫 Email
SEC_HEADERS = {
    "User-Agent": "MarginCall Bot (research@margincall.ai)",
    "Accept-Encoding": "gzip, deflate"
}

def check_ollama():
    """快速檢查 Ollama 是否存活，避免浪費 5 分鐘等超時"""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.debug(f"Ollama check failed: {e}")
        return False

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
        logger.warning(f"⚠️ CIK 查詢異常: {e}")
    
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
            except Exception as e:
                logger.debug(f"Transaction parsing error: {e}")
                continue
        if net_value > 0: return f"【內部人增持】淨買入約 ${net_value:,.0f}"
        elif net_value < 0: return f"【內部人減持】淨賣出約 ${abs(net_value):,.0f}"
        return "【內部人變動】無顯著方向。"
    except Exception as e:
        logger.warning(f"Form 4 parsing exception: {e}")
        return "【解析異常】XML 解析失敗"

def init_nlp_db():
    with db_lock:
        conn = get_connection()
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
    with db_lock:
        conn = get_connection()
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
    # 🎯 角色識別引擎：將機構、KOL 與純散戶情緒剝離
    prompt = f"""
        Task: Analyze the text for {symbol} and categorize insights by ACTOR.
        1. "institutional": Professional analysts (JPMorgan, GS), Funds (ARK), or CEOs.
        2. "retail": Anonymous retail sentiment, forum chatter, hype.
        
        For each actor type, provide:
        - sentiment: "strong_bullish", "mild_bullish", "neutral", "mild_bearish", "strong_bearish".
        - insights: 1-2 core facts.
        
        Text: "{text[:2500]}"
        Format: {{
            "institutional": {{"sentiment": "neutral", "insights": []}},
            "retail": {{"sentiment": "neutral", "insights": []}}
        }}
        """
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma4:e4b-it-q8_0", 
            "prompt": prompt, 
            "stream": False, 
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 300}
        }, timeout=60)
        
        raw_res = response.json().get("response", "")
        clean_res = re.sub(r'^```json\s*|\s*```$', '', raw_res.strip(), flags=re.IGNORECASE|re.MULTILINE)
        
        try:
            res_data = json.loads(clean_res)
        except Exception as e:
            logger.debug(f"JSON decode error in extract_insight_parallel: {e}")
            res_data = {}

        # 🎯 格式標準化與強制填充
        final_data = {}
        for actor in ['institutional', 'retail']:
            found_key = next((k for k in res_data if k.lower() == actor), None)
            if found_key and isinstance(res_data[found_key], dict):
                final_data[actor] = res_data[found_key]
            else:
                final_data[actor] = {"sentiment": "neutral", "insights": []}
            
            final_data[actor]['sentiment'] = str(final_data[actor].get('sentiment', 'neutral')).lower().strip()
            final_data[actor]['insights'] = final_data[actor].get('insights', [])
            
        return final_data
        
    except Exception as e:
        logger.warning(f"   [Debug] Gemma 萃取異常: {e}")
        return {"institutional": {"sentiment": "neutral", "insights": []}, "retail": {"sentiment": "neutral", "insights": []}}
    

# --- 3. 【語意 Reduce 階段】 ---
def semantic_reduce(categorized_tags, symbol, company_name, sector):
    # 確保標籤是乾淨的字串，避免傳入 None 或怪異型別
    sec_insights = list(set([t for t in categorized_tags["SEC"] if isinstance(t, str) and t.strip()]))[:10]
    macro_insights = list(set([t for t in categorized_tags["Macro"] if isinstance(t, str) and t.strip()]))[:15]
    retail_insights = list(set([t for t in categorized_tags["Retail"] if isinstance(t, str) and t.strip()]))[:15]
    
    sec_tags = "\n".join([f"- {t[:150]}" for t in sec_insights])
    macro_tags = "\n".join([f"- {t[:150]}" for t in macro_insights])
    retail_tags = "\n".join([f"- {t[:150]}" for t in retail_insights])

    # 🚨 採用更強硬、更具引導性的 Prompt
    prompt = f"""
        [ROLE]
        You are a Top-Tier Quant Researcher. Your job is to extract actionable intelligence for {symbol}.
        
        [DATA SOURCES]
        Official SEC: {sec_tags if sec_tags else "EMPTY"}
        Macro News: {macro_tags if macro_tags else "EMPTY"}
        Retail Buzz: {retail_tags if retail_tags else "EMPTY"}
        
        [TASK]
        Extract exactly 1 key insight for EACH category. 
        - DO NOT say "No data provided" if there is text under the category.
        - Even if news is generic, summarize the market's current focus or bias.
        - Identify concrete numbers, project names, or legal wins/losses.
        - If a category is literally marked "EMPTY", only then say "該維度目前無重大資訊".
        
        [STRICT OUTPUT FORMAT - JSON ONLY]
        {{
            "sec": "Summary in Traditional Chinese (Taiwan)",
            "macro": "Summary in Traditional Chinese (Taiwan)",
            "retail": "Summary in Traditional Chinese (Taiwan)"
        }}
        """
    
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "gemma4:e4b-it-q8_0", 
            "prompt": prompt, 
            "stream": False,
            "format": "json", 
            "options": {"temperature": 0.1, "num_predict": 500}
        }, timeout=90)

        raw_res = response.json().get("response", "")
        clean_res = re.sub(r'^```json\s*|\s*```$', '', raw_res.strip(), flags=re.IGNORECASE|re.MULTILINE)
        data = json.loads(clean_res)
        
        # 轉換回原本的報表格式
        final_report = textwrap.dedent(f"""
                • **官方基本面**：{data.get('sec', '該維度目前無重大資訊')}
                • **總經與新聞**：{data.get('macro', '該維度目前無重大資訊')}
                • **散戶情緒**：{data.get('retail', '該維度目前無重大資訊')}
                """).strip()
        return final_report
                
    except Exception as e:
        logger.warning(f"   [Debug] Reduce 異常: {e}")
        return f"• **官方基本面**：分析中...\n• **總經與新聞**：分析中...\n• **散戶情緒**：分析中..."

def extract_section(text, start_keyword, stop_keywords, max_len=5000):
    """從 start_keyword 開始，到 stop_keywords 任一出現就停"""
    upper = text.upper()
    
    # 找起點（跳過目錄，取第二次出現）
    first = upper.find(start_keyword)
    if first == -1:
        return ""
    start = upper.find(start_keyword, first + 500)
    if start == -1:
        start = first
        
    # 找終點：掃描所有可能的「下一章標題」，取最近的
    end = start + max_len  # 預設上限
    for kw in stop_keywords:
        hit = upper.find(kw, start + 200)
        if hit != -1 and hit < end:
            end = hit
            
    section = text[start:end].strip()
    
    # 太長的話取頭尾（給 LLM 看重點就好）
    if len(section) > 5000:
        return section[:2500] + " [...] " + section[-2000:]
    return section

def fetch_earning_call_from_fool(symbol):
    """從 Motley Fool 抓取最新的財報逐字稿"""
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
    
    # 策略 1：總覽頁 (抓取全市場最新的 20-30 篇)
    list_url = "https://www.fool.com/earnings-call-transcripts/"
    transcript_url = None
    try:
        resp = scraper.get(list_url, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.find_all('a', href=True)
            for link in links:
                text = link.get_text().upper()
                href = link['href'].lower()
                if (symbol.upper() in text or f"-{symbol.lower()}-" in href) and "/earnings-call-transcripts/" in href:
                    transcript_url = "https://www.fool.com" + href if href.startswith('/') else href
                    break
        
        # 策略 2：搜尋接口備援 (利用 Motley Fool 內部搜尋)
        if not transcript_url:
            search_api = f"https://www.fool.com/search/solr-proxy/?q={symbol}+earnings+call+transcript&sort=publish_date&order=desc"
            s_resp = scraper.get(search_api, timeout=10)
            if s_resp.status_code == 200:
                s_data = s_resp.json()
                results = s_data.get('results', [])
                for res in results:
                    path = res.get('url', '')
                    title = res.get('title', '').upper()
                    if "EARNINGS CALL TRANSCRIPT" in title and symbol.upper() in title:
                        transcript_url = "https://www.fool.com" + path if path.startswith('/') else path
                        break
        
        # 策略 3：標的專屬列表頁 (備援)
        if not transcript_url:
            for exchange in ['nasdaq', 'nyse']:
                quote_url = f"https://www.fool.com/quote/{exchange}/{symbol.lower()}/transcripts/"
                q_resp = scraper.get(quote_url, timeout=10)
                if q_resp.status_code == 200:
                    q_soup = BeautifulSoup(q_resp.text, 'html.parser')
                    # 尋找列表中的第一篇
                    q_links = q_soup.find_all('a', href=True)
                    for q_link in q_links:
                        if "/earnings-call-transcripts/" in q_link['href'].lower():
                            href = q_link['href'].lower()
                            transcript_url = "https://www.fool.com" + href if href.startswith('/') else href
                            break
                if transcript_url: break
        
        if not transcript_url: return None
        
        article_resp = scraper.get(transcript_url, timeout=15)
        if article_resp.status_code != 200: return None
        
        article_soup = BeautifulSoup(article_resp.text, 'html.parser')
        content_div = article_soup.find('div', class_='tailwind-article-body') or article_soup.find('div', class_='article-body')
        
        if content_div:
            for unwanted in content_div.find_all(['div', 'section'], class_=re.compile(r'pitch|promo|ads|sidebar')):
                unwanted.decompose()
            return content_div.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.warning(f"Error fetching earning call for {symbol}: {e}")
    return None

def adjust_retail_score(raw_score, source_count):
    """
    🎯 解決缺陷 6：散戶情緒校正 (反向指標)
    參考 FinNLP 哲學：極端值反著看，中間區域當雜訊
    """
    if source_count < 5:
        return 0.0  # 樣本太少，不可信
    
    if raw_score >= 0.8:
        # 散戶極度狂熱 -> 反向指標，大概率是頂部
        return -0.3
    elif raw_score <= -0.7:
        # 散戶極度恐慌 -> 反向指標，大概率是底部
        return 0.3
    elif -0.3 <= raw_score <= 0.3:
        # 中間區域 -> 沒有訊號意義 (Neutral)
        return 0.0
    else:
        # 輕微偏多/偏空 -> 保留但權重降低，避免過度影響
        return raw_score * 0.5


TRINITY_CATEGORY_ROUTING = {
    "SEC": ("sec", "institutional"),
    "Macro": ("macro", "institutional"),
    "Retail": ("retail", "retail"),
}

TRINITY_BASE_WEIGHTS = {
    "sec": 0.45,
    "macro": 0.30,
    "retail": 0.25,
}

# 一份 SEC filing 通常已足夠有份量；新聞與散戶訊號則需要更多來源才給滿信心。
TRINITY_CONFIDENCE_TARGETS = {
    "sec": 1,
    "macro": 3,
    "retail": 8,
}

TRINITY_SENT_MAP = {
    "strong_bullish": 1.0,
    "mild_bullish": 0.3,
    "neutral": 0.0,
    "mild_bearish": -0.3,
    "strong_bearish": -1.0,
}


def score_sentiment_groups(groups, stock):
    """
    將 SEC / Macro / Retail 三類資料拆成真正獨立的維度分數。
    SEC、Macro 只看 institutional；Retail 只看 retail。
    """
    categorized_tags = {"SEC": [], "Macro": [], "Retail": []}
    dimension_scores = {"sec": 0.0, "macro": 0.0, "retail": 0.0}

    for category, texts in groups.items():
        if not texts:
            continue

        combined = "\n---\n".join([t[:500] for t in texts])[:4000]
        logger.info(f"    🧠 分析 {category} ( {len(texts)} 篇合併 )...")

        res_data = extract_insight_parallel(combined, stock)
        target_dim, actor_key = TRINITY_CATEGORY_ROUTING[category]
        actor_data = res_data.get(actor_key, {})
        categorized_tags[category].extend(actor_data.get("insights", []))

        sentiment = actor_data.get("sentiment", "neutral")
        raw_score = TRINITY_SENT_MAP.get(sentiment, 0.0)

        if target_dim == "retail":
            dimension_scores[target_dim] += adjust_retail_score(raw_score, len(texts))
        else:
            dimension_scores[target_dim] += raw_score

    return dimension_scores, categorized_tags


def compose_alpha_signal(a_sec, n_sec, a_mac, n_mac, a_ret, n_ret):
    """
    將三個獨立維度做來源感知的加權合成。
    與其硬套 Bayesian 先驗把單一 SEC 重大訊號過度稀釋，這裡採用：
    1. 固定基礎權重（SEC > Macro > Retail）
    2. 缺資料時自動重分配權重
    3. 各維度依來源數量逐步拉滿信心，但不改變既有分數方向
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for key, score, count in (
        ("sec", a_sec, n_sec),
        ("macro", a_mac, n_mac),
        ("retail", a_ret, n_ret),
    ):
        count = max(0, int(count or 0))
        if count == 0:
            continue

        target = TRINITY_CONFIDENCE_TARGETS[key]
        confidence = min(1.0, count / target)
        effective_weight = TRINITY_BASE_WEIGHTS[key] * confidence
        weighted_sum += score * effective_weight
        total_weight += effective_weight

    if total_weight <= 0:
        return 0.0

    return max(-1.0, min(1.0, weighted_sum / total_weight))


def _direction_for_score(score):
    if score > 0.2:
        return "bullish"
    if score < -0.2:
        return "bearish"
    return "neutral"


def _verify_nuclear_threat(stock, macro_sec_text, triggered_nukes):
    if not triggered_nukes:
        return False, ""

    context_snippets = []
    for kw in triggered_nukes:
        idx = macro_sec_text.find(kw)
        start = max(0, idx - 100)
        end = min(len(macro_sec_text), idx + 200)
        snippet = macro_sec_text[start:end].replace("\n", " ")
        context_snippets.append(f"[{kw.upper()}]: ...{snippet}...")

    verify_prompt = f"""
        Identify if the following context indicates a SEVERE LEGAL or FINANCIAL THREAT (e.g., fraud, DOJ/SEC investigation INTO {stock}, delisting)
        to {stock} itself, or if the keyword is used in a benign/common way (e.g., "investigating new markets", "won a lawsuit", "legal victory", "routine investigation").

        Keywords detected: {triggered_nukes}
        Context: {" | ".join(context_snippets[:3])}

        Question: Is this a REAL catastrophic threat or just benign/positive news?
        Answer ONLY with "REAL_THREAT" or "BENIGN".
        """

    try:
        v_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma4:e4b-it-q8_0",
                "prompt": verify_prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10},
            },
            timeout=15,
        )

        if v_response.status_code == 200:
            answer = v_response.json().get("response", "").strip().upper()
            if "REAL_THREAT" in answer:
                logger.warning(f"   ☢️ LLM 確認核彈級利空！字眼: {triggered_nukes}")
                return True, f"☢️ 偵測到重大法律或會計風險 (經 LLM 核實: {triggered_nukes})"

            logger.info(f"   ✅ LLM 判定為良性用法 ({triggered_nukes})，解除警報。")
            return False, ""

        return False, f"⚠️ 偵測到敏感關鍵字 {triggered_nukes}，但 LLM 核實服務超時，請人工確認。"
    except Exception as ve:
        logger.warning(f"   ⚠️ 核彈核實異常: {ve}")
        return False, f"⚠️ 偵測到敏感關鍵字 {triggered_nukes}，但 LLM 核實過程異常，請人工確認。"

def _fetch_sec_data(stock, raw_texts):
    """SEC 資料抓取，回傳 None 表示跳過"""
    cik = get_cik(stock)
    logger.info(f"    🔍 CIK: {cik}")
    if not cik:
        logger.warning(f"    ⚠️ {stock} 無法查到 CIK，跳過 SEC 分析")
        return
        
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
        logger.info(f"   🕵️ 軌道 1：搜尋 {stock} 最新年報...")
        for i, form in enumerate(forms):
            if form in ['10-K', '20-F']:
                acc_num_raw = str(accessions[i])
                acc_num = acc_num_raw.replace('-', '')
                doc_name = docs[i]
                annual_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num}/{doc_name}"
                logger.info(f"  🎯 找到最新年報 ({form}, 發布於 {dates[i]}), 下載 primaryDocument...")

                doc_res = requests.get(annual_url, headers=SEC_HEADERS, timeout=30)
                if doc_res.status_code == 200:
                    if form == '10-K':
                        soup = BeautifulSoup(doc_res.text, 'html.parser')
                        valid_paragraphs = []
                        for p in soup.find_all(['p', 'span']):
                            text = p.get_text(separator=' ', strip=True)
                            if len(text) > 120 and 'us-gaap:' not in text.lower():
                                valid_paragraphs.append(text)

                        clean_text = " ".join(valid_paragraphs)
                        clean_text = re.sub(r'http[s]?://\S+', '', clean_text)

                        risk_text = extract_section(clean_text,
                            "RISK FACTOR",
                            ["UNRESOLVED STAFF", "ITEM 1B", "ITEM 2", "PROPERTIES"]
                        )
                        if risk_text:
                            raw_texts.append(f"SEC 10-K [風險因素]: {risk_text}")
                            logger.info("      ✅ 錨點命中 Risk Factors")

                        mda_text = extract_section(clean_text,
                            "MANAGEMENT'S DISCUSSION",
                            ["ITEM 7A", "ITEM 8", "QUANTITATIVE AND QUALITATIVE", "FINANCIAL STATEMENTS"]
                        )
                        if mda_text:
                            raw_texts.append(f"SEC 10-K [營運分析]: {mda_text}")
                            logger.info("      ✅ 錨點命中 MD&A")

                        if not risk_text and not mda_text:
                            raw_texts.append(f"SEC 10-K [年報摘要]: {clean_text[15000:19000]}")
                            logger.warning("      ⚠️ 錨點全部未命中，盲切 [15000:19000]")

                    # 分流 2：外國發行人 (20-F)，使用智能段落抓取避開 iXBRL 亂碼
                    elif form == '20-F':
                        logger.info("   ℹ️ 20-F 外國年報啟用智能段落解析 (過濾 iXBRL)...")
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
                        risk_text = extract_section(clean_text, 
                            "RISK FACTOR", 
                            ["ITEM 4", "ITEM 5", "INFORMATION ON THE COMPANY"]
                        )
                        if risk_text:
                            raw_texts.append(f"SEC 20-F [風險因素]: {risk_text}")
                        else:
                            # 找不到標題盲切：跳過前面 1萬字 的封面廢話，抓取中間段落
                            raw_texts.append(f"SEC 20-F [年報摘要A]: {clean_text[10000 : 13000]}")

                        # 🎯 第二刀：尋找「營運分析 (Operating and Financial Review)」錨點 (通常是 Item 5)
                        mda_text = extract_section(clean_text, 
                            "OPERATING AND FINANCIAL REVIEW", 
                            ["ITEM 6", "ITEM 7", "DIRECTORS", "MAJOR SHAREHOLDERS"]
                        )
                        if mda_text:
                            raw_texts.append(f"SEC 20-F [營運分析]: {mda_text}")
                        else:
                            # 找不到標題盲切：再往後抓一段
                            raw_texts.append(f"SEC 20-F [年報摘要B]: {clean_text[13000 : 16000]}")
                        
                break # 找到最新的一份就跳出迴圈

        # ==========================================
        # 軌道二：近期動態監控 (45天內)
        # ==========================================
        limit_date = (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d')
        logger.info(f"   🕵️ 軌道 2：監控 45 天內動態 (自 {limit_date} 起)...")
        
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
        logger.info("   ✅ SEC 雙軌解析完成")

        # ==========================================
        # 軌道三：最新一季財報電話會議 (Earning Call)
        # ==========================================
        logger.info(f"   🚀 軌道 3：搜尋 {stock} 最新 Earning Call (via Motley Fool)...")
        try:
            content = fetch_earning_call_from_fool(stock)
            if content and len(content) > 500:
                upper_content = content.upper()
                # 1. 尋找 Q&A 段落
                qa_keywords = ["QUESTIONS AND ANSWERS", "QUESTION AND ANSWER", "Q&A SESSION", "Q & A"]
                qa_start = -1
                for kw in qa_keywords:
                    idx = upper_content.find(kw, len(content) // 4) 
                    if idx != -1:
                        qa_start = idx
                        break
                if qa_start != -1:
                    raw_texts.append(f"SEC EarningCall [Q&A]: {content[qa_start : qa_start + 3000]}")
                    logger.info("   ✅ Earning Call Q&A 命中")
                else:
                    raw_texts.append(f"SEC EarningCall [後半段]: {content[len(content)//2 : len(content)//2 + 3000]}")
                    logger.info("   ✅ Earning Call 取後半段")

                # 2. 尋找前瞻指引
                guidance_keywords = ["GUIDANCE", "OUTLOOK", "EXPECT", "FORECAST"]
                for kw in guidance_keywords:
                    g_idx = upper_content.find(kw)
                    if g_idx != -1 and g_idx < len(content) // 2: 
                        raw_texts.append(f"SEC EarningCall [指引]: {content[g_idx : g_idx + 1500]}")
                        logger.info("   ✅ Earning Call 前瞻指引命中")
                        break
            else:
                logger.warning("   ⚠️ Earning Call 未在 Motley Fool 總覽頁找到")
        except Exception as e:
            logger.warning(f"   ⚠️ Earning Call 抓取失敗: {e}")
    else:
        logger.warning(f"    ⚠️ SEC 請求被拒絕 (HTTP {res.status_code})")

# --- 4. 引擎主體 ---
def run_turbo_trinity_scout(stock="NVDA"):
    """
    Performs a deep NLP scan by aggregating sentiment and insights from Reddit, StockTwits, 
    Finnhub news, and SEC filings. Uses LLM-based semantic reduction to generate 
    a strategic "Trinity" report with quantified Alpha scores.
    """
    # --- 0. Ollama 預檢 ---
    if not check_ollama():
        logger.warning("❌ Ollama 未啟動或無法連線，中止分析。")
        sys.exit(1)

    # --- 0. 動態獲取公司背景 ---
    try:
        ticker_info = get_ticker(stock, cache_level="daily").info
        company_name = ticker_info.get('longName', stock)
        sector = ticker_info.get('sector', 'Unknown Sector')
        industry = ticker_info.get('industry', 'Unknown Industry')
        logger.info(f"🧬 偵測到標的：{company_name} | 產業：{sector} / {industry}")
    except Exception as e:
        logger.warning(f"Error getting ticker info for {stock}: {e}")
        company_name, sector, industry = stock, "Financial", "Technology" # 失敗時的備援

    init_nlp_db()
    logger.info(f"🔥 [Ubuntu] 啟動 {stock} 四維一體深度掃描 (全網聚合 + StockTwits 版)...")
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
            logger.info(f"   ✅ Reddit: {valid_count} 筆 (過濾後)")
    except Exception as e:
        logger.warning(f"   ⚠️ Reddit 異常: {e}")

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
            logger.info(f"   ✅ StockTwits: {len(messages)} 筆 (穿甲成功)")
        else:
            logger.warning(f"   ⚠️ StockTwits 穿甲失敗 (HTTP {st_resp.status_code})")
    except Exception as e:
        logger.warning(f"   ⚠️ StockTwits 穿甲異常: {e}")

    # C. Finnhub 全網媒體聚合
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
        
        # 確保抓到 KEY
        fh_key = os.getenv("FINNHUB_API_KEY")
        if not fh_key:
            logger.warning("   ⚠️ 未設定 FINNHUB_API_KEY，跳過新聞抓取。")
        else:
            fh_downloader = Finnhub_Date_Range(fh_key)
            fh_downloader.download_date_range_stock(stock=stock, start_date=start_date, end_date=end_date)
            
            # 只有在 downloader 存在且有資料時才處理
            if hasattr(fh_downloader, 'dataframe') and fh_downloader.dataframe is not None and not fh_downloader.dataframe.empty:
                df_fh = fh_downloader.dataframe.head(15) 
                for _, row in df_fh.iterrows():
                    source = row.get('source', 'News')
                    headline = row.get('headline', '')
                    summary = row.get('summary', '')[:300]
                    raw_texts.append(f"Macro({source}): {headline} | {summary}")
                logger.info(f"   ✅ Macro(Finnhub): {len(df_fh)} 筆")
    except Exception as e: 
        logger.warning(f"   ⚠️ Finnhub 流程異常: {e}")

    # D. SEC
    try:
        _fetch_sec_data(stock, raw_texts)
    except Exception as e:
        logger.warning(f"    ⚠️ SEC 抓取失敗: {e}")

    total = len(raw_texts)

    if total == 0:
        logger.info("📭 無情報")
        return

    # --- 🔍 偵錯列印：檢視抓取內容 ---
    logger.info("\n--- 🕵️ 原始情報抽樣檢視 ---")
    sources = ["Reddit", "StockTwits", "Macro", "SEC"]
    for src in sources:
        samples = [t for t in raw_texts if t.startswith(src)]
        logger.info(f"\n【{src} 來源抽樣 ({len(samples)} 筆)】")
        for i, s in enumerate(samples[:3]):
            logger.info(f"  {i+1}. {s[:150]}...")
    logger.info("\n--- 🧠 啟動 GPU 語意分析 ---")

    # --- 分組合併，只跑 3 次 LLM ( Map 階段 ) ---
    groups = {
        "SEC":    [t for t in raw_texts if t.startswith("SEC")],
        "Macro":  [t for t in raw_texts if t.startswith("Macro")],
        "Retail": [t for t in raw_texts if t.startswith("Reddit") or t.startswith("StockTwits")],
    }

    dimension_scores, categorized_tags = score_sentiment_groups(groups, stock)
    a_sec = dimension_scores["sec"]
    a_mac = dimension_scores["macro"]
    a_retail = dimension_scores["retail"]

    # 🎯 矛盾偵測 (Divergence Detection)
    divergence_alert = ""
    sec_dir = _direction_for_score(a_sec)
    mac_dir = _direction_for_score(a_mac)
    ret_dir = _direction_for_score(a_retail)

    if ret_dir == "bullish" and sec_dir == "bearish":
        divergence_alert = "⚠️ 散戶情緒看多 vs SEC 官方偏空 -> 散戶陷阱風險"
    elif ret_dir == "bearish" and sec_dir == "bullish":
        divergence_alert = "🔍 散戶恐慌 vs SEC 官方偏多 -> 潛在反轉機會"
    elif ret_dir == "bullish" and mac_dir == "bearish":
        divergence_alert = "⚠️ 散戶情緒看多 vs 宏觀新聞偏空 -> 注意逆風"

    # 🚨 核彈級利空熔斷機制 (Tail-Risk Override)
    macro_sec_text = " ".join(groups["Macro"] + groups["SEC"]).lower()
    nuclear_keywords = ['doj', 'indictment', 'subpoena', 'delist', 'fraud', 'accounting irregularity', 'investigation']

    triggered_nukes = [k for k in nuclear_keywords if k in macro_sec_text]
    nuclear_confirmed, nuclear_alert = _verify_nuclear_threat(stock, macro_sec_text, triggered_nukes)
    if nuclear_alert:
        divergence_alert = nuclear_alert

    nlp_alpha = compose_alpha_signal(
        a_sec,
        len(groups["SEC"]),
        a_mac,
        len(groups["Macro"]),
        a_retail,
        len(groups["Retail"]),
    )
    if nuclear_confirmed:
        nlp_alpha = -0.95

    signal_pack = {
        "sec_stance": sec_dir,
        "sec_score": round(a_sec, 3),
        "sec_detail": categorized_tags["SEC"][:3],
        "macro_stance": mac_dir,
        "macro_score": round(a_mac, 3),
        "macro_detail": categorized_tags["Macro"][:3],
        "retail_stance": ret_dir,
        "retail_score": round(a_retail, 3),
        "retail_detail": categorized_tags["Retail"][:3],
        "divergence": divergence_alert or "無",
        "nuclear_alert": nuclear_confirmed,
        "source_counts": {
            "sec": len(groups["SEC"]),
            "macro": len(groups["Macro"]),
            "retail": len(groups["Retail"]),
        },
        "composite_alpha": round(nlp_alpha, 4),
    }

    semantic_summary = semantic_reduce(categorized_tags, stock, company_name, sector)
    report_header = f"📊 {stock} 戰報\n綜合 Alpha: {nlp_alpha:+.2f} | SEC:{sec_dir} Macro:{mac_dir} Retail:{ret_dir}\n"
    if divergence_alert:
        report_header += f"{divergence_alert}\n"

    report = report_header + semantic_summary
    storage_payload = json.dumps(
        {"signal_pack": signal_pack, "semantic_summary": semantic_summary},
        ensure_ascii=False,
    )
    save_to_db(stock, nlp_alpha, a_retail, a_mac, a_sec, total, storage_payload, "TRINITY_V2")
    logger.info("\n%s", report)

if __name__ == "__main__":
    target = sys.argv[1].upper() if len(sys.argv) > 1 else "NVDA"
    run_turbo_trinity_scout(target)
