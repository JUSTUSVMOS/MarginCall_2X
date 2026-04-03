import os
import telebot
import logging
import datetime
import yfinance as yf
import json
from google import genai
import engine_market as market
import engine_risk as risk

# 設定日誌
logger = logging.getLogger(__name__)

# 初始化 Bot 用於緊急警報
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

def get_my_user_id():
    val = os.getenv("TELEGRAM_USER_ID")
    return int(val) if val else 0

bot = telebot.TeleBot(BOT_TOKEN) if BOT_TOKEN else None
genai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

def detect_symbols(text: str) -> list:
    """
    使用 LLM 從用戶輸入中提取股票代號。
    """
    if not genai_client: return []
    try:
        prompt = f"請從以下文字中提取提到的股票代號或公司名稱，並轉換成 yfinance 格式的代號 (例如: TSLA, 2330.TW, BRK-B)。只需回傳代號並以逗號分隔，若無則回傳 'None'。\n文字：{text}"
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash-lite", # 使用輕量模型加速
            contents=prompt
        )
        res_text = response.text.strip()
        if res_text == "None" or not res_text: return []
        return [s.strip().upper() for s in res_text.split(',') if s.strip()]
    except Exception as e:
        logger.error(f"Symbol detection failed: {e}")
        return []

def fetch_strat_data(symbol: str) -> dict:
    """
    根據資產類型分流抓取數據，並實作 CVD 緊急中斷。
    """
    symbol = symbol.upper()
    profile = market.get_asset_profile(symbol)
    asset_type = profile.get('asset_type', 'Unknown')
    
    data = {
        "symbol": symbol,
        "asset_type": asset_type,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "metrics": {},
        "raw_profile": profile
    }

    try:
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
        context += json.dumps(data.get('metrics', {}), indent=2, ensure_ascii=False)
    
    return context
