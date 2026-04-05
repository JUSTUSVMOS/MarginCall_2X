import os
import telebot
import logging
import datetime
import yfinance as yf
import json
import sqlite3
import re  # 補回此行
from google import genai
import engine_market as market
import engine_risk as risk

# 設定日誌
logger = logging.getLogger(__name__)

# --- 0. 資料庫路徑與初始化 ---
DB_FILE = os.path.join(os.path.dirname(__file__), "portfolio.db")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

def get_my_user_id():
    val = os.getenv("TELEGRAM_USER_ID")
    return int(val) if val else 0

bot = telebot.TeleBot(BOT_TOKEN) if BOT_TOKEN else None
genai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

def detect_symbols(text: str) -> list:
    """
    從用戶輸入中提取股票代號。優先使用 LLM，失敗則使用 Regex 備援。
    """
    symbols = []
    
    # --- 1. 嘗試使用 LLM 偵測 (精準度高) ---
    if genai_client:
        try:
            prompt = f"請從以下文字中提取提到的股票代號或公司名稱，並轉換成 yfinance 格式的代號 (例如: TSLA, 2330.TW, BRK-B)。只需回傳代號並以逗號分隔，若無則回傳 'None'。\n文字：{text}"
            response = genai_client.models.generate_content(
                model="gemini-2.0-flash-lite", 
                contents=prompt
            )
            res_text = response.text.strip()
            if res_text != "None" and res_text:
                symbols = [s.strip().upper() for s in res_text.split(',') if s.strip()]
        except Exception as e:
            logger.warning(f"LLM Symbol detection failed (Quota?): {e}")

    # --- 2. Regex 備援 (確保系統不掛掉) ---
    if not symbols:
        # 尋找 2-5 個大寫字母 (美股) 或 數字.TW (台股)
        regex_patterns = [r'\b[A-Z]{2,5}\b', r'\b\d{4}\.TW\b']
        for p in regex_patterns:
            matches = re.findall(p, text)
            symbols.extend(matches)
    
    return list(set(symbols)) # 去重

def fetch_nlp_alpha(symbol: str) -> dict:
    """
    從資料庫讀取最新的 NLP Alpha 因子與語意報告。
    增加時間檢查：若資料超過 10 分鐘則視為過期。
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # 抓取該標的最新的紀錄，包含時間戳
        cursor.execute("""
            SELECT nlp_alpha, alpha_retail, alpha_macro, alpha_official, summary_text, timestamp 
            FROM nlp_insights 
            WHERE symbol = ? 
            ORDER BY timestamp DESC LIMIT 1
        """, (symbol,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # 檢查時間新鮮度 (10 分鐘內)
            try:
                data_time = datetime.datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S')
                if (datetime.datetime.now() - data_time).total_seconds() > 600:
                    return {"error": "NLP data expired (over 10 mins). Needs refresh."}
            except: pass # 若格式不對則跳過時間檢查

            return {
                "nlp_alpha": round(row[0], 4),
                "alpha_retail": round(row[1], 4),
                "alpha_macro": round(row[2], 4),
                "alpha_official": round(row[3], 4),
                "semantic_summary": row[4],
                "timestamp": row[5]
            }
        return {"error": "No NLP data found for this symbol. Please run nlp_worker."}
    except Exception as e:
        logger.error(f"Failed to fetch NLP Alpha for {symbol}: {e}")
        return {"error": str(e)}

def fetch_strat_data(symbol: str) -> dict:
    """
    根據資產類型分流抓取數據，並實作 CVD & NLP 雙重熔斷中斷。
    """
    symbol = symbol.upper()
    profile = market.get_asset_profile(symbol)
    asset_type = profile.get('asset_type', 'Unknown')
    
    # --- 核心升級：抓取 NLP Alpha 因子 ---
    nlp_data = fetch_nlp_alpha(symbol)
    
    data = {
        "symbol": symbol,
        "asset_type": asset_type,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "nlp_insights": nlp_data, # 注入語意情緒
        "metrics": {},
        "raw_profile": profile
    }

    try:
        # 【重要】情緒熔斷：如果 SEC 官方訊號低於 -0.7 (強烈利空/風險)
        alpha_off = nlp_data.get("alpha_official", 0)
        if isinstance(alpha_off, (int, float)) and alpha_off < -0.7 and bot:
            alert_msg = f"🛑 【SEC 深度預警】{symbol} 偵測到官方重大風險！\n官方 Alpha: {alpha_off:.2f}\n語意摘要: {nlp_data.get('semantic_summary', '無')[:150]}..."
            bot.send_message(get_my_user_id(), alert_msg)
            logger.warning(f"SEC Sentiment Alert for {symbol}: {alpha_off}")

        if asset_type == 'Tech_Momentum':
            # 抓取 5分K CVD
            ticker = yf.Ticker(symbol)
            df_5m = ticker.history(period="1d", interval="5m")
            cvd = risk.calculate_buying_pressure(df_5m)
            
            # 【重要】硬體中斷：CVD < -0.9 立即警報
            if cvd < -0.9 and bot:
                alert_msg = f"🚨 【硬體中斷】偵測到 {symbol} 恐慌性拋售！\n當前 CVD: {cvd:.4f}\n請立即檢查盤勢！"
                bot.send_message(get_my_user_id(), alert_msg)
                logger.warning(f"CVD Alert triggered for {symbol}: {cvd}")

            # 抓取技術面 (RSI, MACD 等)
            tech_report = market.get_technical_analysis(symbol)
            data["metrics"] = {
                "cvd": round(cvd, 4),
                "technical_analysis": tech_report,
                "live_insight": market.get_us_realtime_insight(symbol)
            }

        elif asset_type == 'Value_Holding':
            # 抓取基本面 (P/B, EPS 等)
            fundamental_report = market.get_fundamental_data(symbol)
            data["metrics"] = {
                "fundamental_analysis": fundamental_report,
                "news": market.get_stock_news(symbol)
            }

        elif asset_type == 'Macro_Hedge':
            # 抓取價格趨勢 + 總經指標
            market_sentiment = market.get_market_sentiment()
            data["metrics"] = {
                "market_sentiment": market_sentiment,
                "news": market.get_stock_news(symbol),
                "price": market.get_live_price(symbol)
            }
        
        else:
            # 預設回傳基礎數據
            data["metrics"] = {
                "price": market.get_live_price(symbol),
                "news": market.get_stock_news(symbol)
            }

    except Exception as e:
        logger.error(f"Error in fetch_strat_data for {symbol}: {e}")
        data["error"] = str(e)

    return data

def get_strat_context(user_text: str) -> str:
    """
    整合偵測與抓取，產生供 LLM 使用的 Context。
    """
    symbols = detect_symbols(user_text)
    if not symbols: return ""
    
    context = "\n【🛡️ 策略路由系統已啟動】\n"
    for sym in symbols:
        data = fetch_strat_data(sym)
        context += f"\n--- 標的: {sym} ({data.get('asset_type')}) ---\n"
        # 整合技術指標與語意情緒
        combined_metrics = {
            "market_data": data.get('metrics', {}),
            "nlp_sentiment_alpha": data.get('nlp_insights', {})
        }
        context += json.dumps(combined_metrics, indent=2, ensure_ascii=False)
    
    return context
