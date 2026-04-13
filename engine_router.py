import os
import logging
import datetime
import yfinance as yf
from yf_session import get_ticker, get_download
import json
import re  # 補回此行
import engine_market as market
import engine_risk as risk
import engine_fundamentals as fundamentals
from src.database import db_lock, get_connection

# 設定日誌
logger = logging.getLogger(__name__)

def get_my_user_id():
    val = os.getenv("TELEGRAM_USER_ID")
    return int(val) if val else 0

bot = None # 改為延後初始化或從外部注入

def set_bot(external_bot):
    global bot
    bot = external_bot

# --- 0. 全域 Regex 配置 (預編譯提高效能) ---

# 俗稱與中文對照表 (Lookup Table)：將口語直接對應 yfinance 標準代號
TICKER_ALIASES = {
    # 科技巨頭 & 常見標的
    'APPLE': 'AAPL', '蘋果': 'AAPL',
    'TESLA': 'TSLA', '特斯拉': 'TSLA',
    'NVIDIA': 'NVDA', '輝達': 'NVDA', 'NV': 'NVDA',
    'BROADCOM': 'AVGO', '博通': 'AVGO',
    'PALANTIR': 'PLTR',
    'TELEDYNE': 'TDY',
    'BERKSHIRE': 'BRK-B', '波克夏': 'BRK-B',
    'COREWEAVE': 'COREWEAVE',
    
    # 台股口語 (直接轉好 .TW)
    '台積電': '2330.TW', '神山': '2330.TW',
    '聯發科': '2454.TW', '發哥': '2454.TW'
}

# 提取對照表的 keys，組成 Regex 條件 (例如：APPLE|蘋果|TESLA...)
alias_pattern = '|'.join(map(re.escape, TICKER_ALIASES.keys()))

# 支援：1.自訂中英俗稱 2.純英文字母(1-6碼，可帶點) 3.台股數字/ETF(4-6碼數字，可帶字母)
# 使用 Lookaround (?<!...) (?!) 確保不會被中文字黏住而抓不到
regex_str = rf'(?<![a-zA-Z0-9])({alias_pattern}|[a-zA-Z]{{1,6}}(?:\.[a-zA-Z])?|\d{{4,6}}[A-Za-z]?|\d{{4}}\.(?:tw|two|TW|TWO))(?![a-zA-Z0-9])'
SYMBOL_PATTERN = re.compile(regex_str, re.IGNORECASE)

# 黑名單：防止 Regex 抓到常用金融術語
STOP_WORDS = {
    'BUY', 'SELL', 'CALL', 'PUT', 'INFO', 'NEWS', 'CHAT', 'THE', 'AND', 
    'FOR', 'STOCK', 'PRICE', 'GOOD', 'BAD', 'RISK', 'TECH', 'USER', 'LIST',
    'LONG', 'SHORT', 'OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOL', 'BULL', 'BEAR'
}

def detect_symbols(text: str) -> list:
    """
    Regex-First: 
    1. 使用不分大小寫的 Regex 抓取候選字。
    2. 若 Regex 沒抓到，才將原始文字丟給 LLM 判斷語意。
    """
    symbols = _regex_fallback(text)
    if symbols:
        return symbols

    try:
        from src.llm import quick_call

        prompt = f"請從以下文字中提取提到的股票代號或公司名稱，並轉換成 yfinance 格式的代號 (例如：TSLA, 2330.TW, BRK-B)。\n只需回傳代號並以逗號分隔，若無則回傳 'None'。\n文字: {text}"
        res_text = quick_call(prompt)
        if res_text:
            res_text = res_text.strip()
            if res_text != "None":
                symbols = [s.strip().upper() for s in res_text.split(',') if s.strip()]
                if symbols:
                    logger.info(f"LLM detected symbols: {symbols}")
                    return list(set(symbols))
    except Exception as e:
        logger.warning(f"LLM symbol detection failed: {e}")

    return []

def _regex_fallback(text: str) -> list:
    """
    優化後的 Regex 邏輯：
    1. 查表轉換：若是俗稱，直接轉為標準代號。
    2. 自動補綴：若是台股代號 (4-6碼數字/ETF) 且未帶後綴，自動補上 .TW。
    3. 排除雜訊：過濾掉 STOP_WORDS 黑名單。
    """
    matches = SYMBOL_PATTERN.findall(text)
    
    results = []
    for m in matches:
        upper_m = m.upper()
        
        # 1. 優先檢查是否在俗稱對照表內 (完全不花 AI 額度)
        if upper_m in TICKER_ALIASES:
            results.append(TICKER_ALIASES[upper_m])
            continue
            
        # 2. 如果不是俗稱，檢查是否為誤抓的英文黑名單
        if upper_m not in STOP_WORDS:
            # 3. 如果是 4~6 碼純數字 (或數字加一字母如 00981A)，且沒有小數點
            # 代表是台股口語，自動幫 yfinance 補齊格式
            if re.match(r'^\d{4,6}[A-Z]?$', upper_m) and '.' not in upper_m:
                upper_m = f"{upper_m}.TW" 
                
            results.append(upper_m)
            
    return list(set(results))

def fetch_nlp_alpha(symbol: str) -> dict:
    """
    從資料庫讀取最新的 NLP Alpha 因子與語意報告。
    增加時間檢查：若資料超過 10 分鐘則視為過期。
    """
    try:
        with db_lock:
            conn = get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT nlp_alpha, alpha_retail, alpha_macro, alpha_official, summary_text, timestamp 
                    FROM nlp_insights 
                    WHERE symbol = ? 
                    ORDER BY timestamp DESC LIMIT 1
                """, (symbol,))
                row = cursor.fetchone()
            finally:
                conn.close()
        
        if row:
            # 檢查時間新鮮度 (10 分鐘內)
            try:
                data_time = datetime.datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S')
                if (datetime.datetime.now() - data_time).total_seconds() > 1800:
                    return {"error": "NLP data expired (over 30 mins). Needs refresh."}
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

def get_relative_move(symbol):
    """比較個股 vs 大盤，區分系統性和個股風險"""
    try:
        # 抓取 2 天數據以計算最新一日漲跌幅 (今天 vs 昨天收盤)
        stock = get_ticker(symbol).history(period="2d")
        spy = get_ticker("SPY").history(period="2d")
        
        if len(stock) < 2 or len(spy) < 2:
            return "UNKNOWN", 0.0

        stock_ret = (stock['Close'].iloc[-1] / stock['Close'].iloc[-2]) - 1
        spy_ret = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2]) - 1
        
        excess_return = stock_ret - spy_ret # 超額報酬
        
        if abs(excess_return) < 0.01:
            return "SYSTEMATIC", excess_return # 跟大盤同步 -> 系統性風險
        elif excess_return < -0.02:
            return "IDIOSYNCRATIC_BAD", excess_return # 獨自大跌 -> 個股利空
        elif excess_return > 0.02:
            return "IDIOSYNCRATIC_GOOD", excess_return # 獨自大漲 -> 個股利多
        return "NORMAL", excess_return
    except Exception as e:
        logger.warning(f"Relative move calculation failed for {symbol}: {e}")
        return "UNKNOWN", 0.0

def parse_pc_ratio(insight_text: str) -> float:
    """從 get_us_realtime_insight 的文字報告中提取 P/C Ratio"""
    try:
        match = re.search(r'P/C Ratio:\s*([\d\.]+)', insight_text)
        if match:
            return float(match.group(1))
    except:
        pass
    return None

def fetch_strat_data(symbol: str) -> dict:
    """
    根據資產類型分流抓取數據，並實作 CVD & NLP 雙重熔斷中斷。
    """
    symbol = market.normalize_ticker(symbol)
    profile = market.get_asset_profile(symbol)
    asset_type = profile.get('asset_type', 'Unknown')
    
    # --- 核心升級：抓取 NLP Alpha 因子 ---
    nlp_data = fetch_nlp_alpha(symbol)
    
    # --- 核心升級：區分系統性 vs 個股風險 ---
    risk_type, excess = get_relative_move(symbol)
    
    data = {
        "symbol": symbol,
        "asset_type": asset_type,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "nlp_insights": nlp_data, # 注入語意情緒
        "relative_move": {
            "risk_type": risk_type,
            "excess_return": round(excess, 4)
        },
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
            ticker = get_ticker(symbol, cache_level="live")
            df_5m = ticker.history(period="1d", interval="5m")
            cvd = risk.calculate_buying_pressure(df_5m)
            
            # 【重要】硬體中斷：CVD < -0.9 立即警報
            if cvd < -0.9 and bot:
                alert_msg = f"🚨 【硬體中斷】偵測到 {symbol} 恐慌性拋售！\n當前 CVD: {cvd:.4f}\n請立即檢查盤勢！"
                bot.send_message(get_my_user_id(), alert_msg)
                logger.warning(f"CVD Alert triggered for {symbol}: {cvd}")

            # 抓取技術面 (RSI, MACD 等)
            tech_report = market.get_technical_analysis(symbol)
            live_insight = market.get_us_realtime_insight(symbol)
            
            data["metrics"] = {
                "cvd": round(cvd, 4),
                "technical_analysis": tech_report,
                "live_insight": live_insight
            }

            # 🎯 解決缺陷 7：整合領先指標修正 NLP Alpha
            if "nlp_alpha" in nlp_data:
                # 修正因子 1：CVD (資金流向)
                if cvd < -0.5:
                    nlp_data["nlp_alpha"] -= 0.15
                    logger.info(f"Leading Indicator Correction: {symbol} CVD {cvd} -> Alpha -0.15")
                elif cvd > 0.5:
                    nlp_data["nlp_alpha"] += 0.1
                    logger.info(f"Leading Indicator Correction: {symbol} CVD {cvd} -> Alpha +0.1")
                
                # 修正因子 2：P/C Ratio (市場避險情緒)
                pc_ratio = parse_pc_ratio(live_insight)
                if pc_ratio:
                    if pc_ratio > 1.5:
                        nlp_data["nlp_alpha"] -= 0.1
                        logger.info(f"Leading Indicator Correction: {symbol} P/C {pc_ratio} -> Alpha -0.1")
                    elif pc_ratio < 0.5:
                        nlp_data["nlp_alpha"] += 0.1
                        logger.info(f"Leading Indicator Correction: {symbol} P/C {pc_ratio} -> Alpha +0.1")

        elif asset_type == 'Value_Holding':
            # 抓取深度基本面 (趨勢、ROE、債務)
            fundamental_report = fundamentals.get_deep_fundamentals(symbol)
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
