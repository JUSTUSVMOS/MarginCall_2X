import os
import datetime
import requests
import time
import pandas as pd
import numpy as np
import yfinance as yf
import fubon  # 引用現有的 fubon.py
import sqlite3
from google import genai
from google.genai import types

import logging

# 引入共用鎖與連線
from engine_risk import db_lock, get_db_connection

# 設定基礎日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

FMP_KEY = os.getenv("FMP_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# 初始化 Gemini 客戶端 (用於 Stage 2)
genai_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None

def normalize_ticker(symbol: str) -> str:
    """
    將使用者輸入的代號正規化。
    特別處理美股中帶點的代號 (如 BRK.B -> BRK-B)
    """
    symbol = symbol.upper().strip()
    # 如果不是台股 (不含數字且長度不符合台股規則)
    is_taiwan = any(char.isdigit() for char in symbol) and (len(symbol.replace('.TW','').replace('.TWO','')) <= 6)
    
    if not is_taiwan:
        # 排除常見的交易所後綴，其餘的點 (如 BRK.B) 轉換為橫槓 (BRK-B)
        suffixes = (".TW", ".TWO", ".HK", ".SS", ".SZ", ".L", ".DE", ".AS", ".AX", ".T", ".PA", ".MI", ".TO", ".V")
        if "." in symbol and not symbol.endswith(suffixes):
            return symbol.replace(".", "-")
    return symbol

def get_asset_profile(symbol: str) -> dict:
    """
    【核心】資產分類器：Stage 1 (規則) + Stage 2 (LLM Fallback)
    """
    symbol = normalize_ticker(symbol)
    
    # 1. 檢查 SQLite 快取
    with db_lock:
        conn = get_db_connection()
        try:
            df = pd.read_sql("SELECT * FROM asset_profile_cache WHERE symbol = ?", conn, params=(symbol,))
            if not df.empty:
                logger.info(f"Cache Hit: {symbol}")
                return df.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Cache check failed: {e}")
        finally: conn.close()

    logger.info(f"Cache Miss: {symbol}, starting classifier...")
    
    # Hard-coded Overrides
    overrides = {
        'BRK-B': 'Value_Holding',
        'IAUM': 'Macro_Hedge',
        'MLPS.L': 'Macro_Hedge'
    }
    
    asset_type = "Unknown"
    sector = "Unknown"
    industry = "Unknown"
    risk_score = 1.0 # 預設

    # Stage 1: Rule-based (YF Info)
    if symbol in overrides:
        asset_type = overrides[symbol]
        try:
            info = yf.Ticker(symbol).info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
        except: pass
    else:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            if sector in ['Technology', 'Communication Services']:
                asset_type = 'Tech_Momentum'
            elif sector in ['Energy', 'Utilities'] or 'Oil' in industry or 'Gas' in industry:
                asset_type = 'Macro_Hedge'
            elif sector == 'Financial Services':
                market_cap = info.get('marketCap', 0)
                if market_cap > 100_000_000_000: # 100B
                    asset_type = 'Value_Holding'
            elif any(kw in (sector + industry) for kw in ['Gold', 'Metal', 'Commodity']):
                asset_type = 'Macro_Hedge'
        except Exception as e:
            logger.warning(f"Stage 1 fetching failed for {symbol}: {e}")

    # Stage 2: LLM Fallback (支援多模型降級)
    if asset_type == "Unknown" and genai_client:
        logger.info(f"Starting Stage 2 LLM Classifier for {symbol}")
        # 這裡借用 main.py 的邏輯，但為了不循環引用，我們簡單列出
        fallback_models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]
        prompt = f"請將標的 {symbol} (Sector: {sector}, Industry: {industry}) 分類為以下三類之一：Tech_Momentum, Value_Holding, Macro_Hedge。僅回傳分類名稱。"
        
        for model_name in fallback_models:
            try:
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response.text:
                    llm_type = response.text.strip()
                    if llm_type in ['Tech_Momentum', 'Value_Holding', 'Macro_Hedge']:
                        asset_type = llm_type
                        break
            except Exception as e:
                logger.warning(f"Stage 2 LLM classification failed with {model_name}: {e}")
                continue

    # 3. 持久化到 SQLite
    with db_lock:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO asset_profile_cache (symbol, asset_type, sector, industry, risk_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (symbol, asset_type, sector, industry, risk_score, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            logger.info(f"Cached {symbol} as {asset_type}")
        except Exception as e:
            logger.error(f"Failed to cache {symbol}: {e}")
        finally: conn.close()

    return {
        "symbol": symbol,
        "asset_type": asset_type,
        "sector": sector,
        "industry": industry,
        "risk_score": risk_score
    }

import pytz

def is_tw_market_open() -> bool:
    tw_tz = pytz.timezone('Asia/Taipei')
    now = datetime.datetime.now(tw_tz)
    # 台股交易時間: 周一至周五 09:00 - 13:30
    if now.weekday() >= 5: return False
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end = now.replace(hour=13, minute=30, second=0, microsecond=0)
    return start <= now <= end

def is_us_market_open() -> bool:
    us_tz = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(us_tz)
    # 美股交易時間: 周一至周五 09:30 - 16:00 (美東時間)
    if now.weekday() >= 5: return False
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now <= end

def resolve_symbol_identity(symbol: str) -> str:
    """
    【核心修復】專門解決機器人不認識新標的(如 009816)的問題。
    在任何分析前，先呼叫此工具確認標的名與真實性。
    """
    symbol = normalize_ticker(symbol).replace('.TW', '').replace('.TWO', '')
    is_taiwan = any(char.isdigit() for char in symbol) and (len(symbol) <= 6)
    
    if is_taiwan and fubon.fubon_ready:
        try:
            # 利用富邦歷史統計功能來抓取官方名稱
            stats = fubon.get_historical_stats(symbol)
            if "未知" not in stats and "異常" not in stats:
                return stats
        except Exception as e:
            logger.debug(f"Fubon historical stats error for {symbol}: {e}")
        
    try:
        s = f"{symbol}.TW" if is_taiwan and not symbol.endswith('.TW') else symbol
        ticker = yf.Ticker(s)
        info = ticker.info
        name = info.get('longName') or info.get('shortName') or "未知標的"
        return f"🔍 識別結果: {symbol} ({name}) | 類型: {info.get('quoteType', '未知')}"
    except Exception as e:
        logger.error(f"Failed to resolve symbol identity for {symbol}: {e}")
        return f"❌ 無法識別標的: {symbol}，請確認代號是否正確。"

def get_live_price(symbol: str) -> str:
    symbol = normalize_ticker(symbol)
    clean_symbol = symbol.replace('.TW', '').replace('.TWO', '')
    if clean_symbol == "2454_ESOP": clean_symbol = "2454"
    # 支援新形態 ETF (009816 等 6 碼)
    is_taiwan_stock = any(char.isdigit() for char in clean_symbol) and (len(clean_symbol) <= 6)
    
    price = None
    if is_taiwan_stock and fubon.fubon_ready:
        try:
            reststock = fubon.fubon_sdk.marketdata.rest_client.stock
            quote_data = reststock.intraday.quote(symbol=clean_symbol)
            is_dict = isinstance(quote_data, dict)
            price = quote_data.get('closePrice') or quote_data.get('lastPrice') if is_dict else getattr(quote_data, 'closePrice', getattr(quote_data, 'lastPrice', None))
            if price and price > 0: 
                # 順便抓一下名字，讓回報更有感
                name = getattr(quote_data, 'name', '台股')
                return f"{name} {clean_symbol} 現價: {round(float(price), 2)} (來源: Fubon)"
        except Exception as e:
            logger.warning(f"Fubon real-time price fetch failed for {clean_symbol}: {e}")

    if not is_taiwan_stock and FMP_KEY and is_us_market_open():
        try:
            url = f"https://financialmodelingprep.com/stable/quote?symbol={symbol}&apikey={FMP_KEY}"
            res = requests.get(url, timeout=5).json()
            if isinstance(res, list) and len(res) > 0:
                return f"{round(float(res[0]['price']), 2)} (來源: FMP)"
        except Exception as e:
            logger.warning(f"FMP real-time price fetch failed for {symbol}: {e}")

    search_list = [symbol, f"{symbol}.TW", f"{symbol}.TWO"] if is_taiwan_stock else [symbol]
    for s in search_list:
        try:
            ticker = yf.Ticker(s)
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not price:
                hist = ticker.history(period="1d")
                if not hist.empty: price = hist['Close'].iloc[-1]
            if price and price > 0: return f"{round(float(price), 2)} (來源: YF)"
        except Exception as e:
            logger.debug(f"YFinance fetch failed for {s}: {e}")
            continue
    return "無法取得報價"

def get_us_realtime_insight(symbol: str) -> str:
    symbol = normalize_ticker(symbol)
    try:
        ticker = yf.Ticker(symbol)
        full_df = ticker.history(period="1d", interval="5m")
        if full_df.empty: return f"❌ {symbol} 目前無盤中數據。"
        df = full_df.tail(10)
        info = ticker.info
        bid, ask = info.get('bid', 0), info.get('ask', 0)
        ba_ratio = (info.get('bidSize', 1) / info.get('askSize', 1)) if info.get('askSize', 0) > 0 else 1
        
        # 🎭 Put/Call Ratio 計算 (優化：跳過超短期周選)
        pc_report = "N/A"
        try:
            expirations = ticker.options
            if expirations:
                total_calls, total_puts, valid_fetched = 0, 0, 0
                target_count = 4
                min_days = 7
                today = datetime.datetime.now()
                for date_str in expirations:
                    if valid_fetched >= target_count: break
                    try:
                        expiry_date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
                        if (expiry_date - today).days < min_days: continue
                        chain = ticker.option_chain(date_str)
                        c_sum = chain.calls['volume'].sum() if not chain.calls.empty else 0
                        p_sum = chain.puts['volume'].sum() if not chain.puts.empty else 0
                        total_calls += (c_sum if not np.isnan(c_sum) else 0)
                        total_puts += (p_sum if not np.isnan(p_sum) else 0)
                        valid_fetched += 1
                    except: continue
                if total_calls > 0:
                    pc_report = f"{total_puts / total_calls:.2f}"
        except Exception as e:
            logger.warning(f"Put/Call ratio calculation failed for {symbol}: {e}")

        # 成交量密集區 (POC)
        day_min, day_max = full_df['Low'].min(), full_df['High'].max()
        bins = np.linspace(day_min, day_max, 11)
        full_df['bin'] = pd.cut(full_df['Close'], bins=bins)
        vp = full_df.groupby('bin', observed=True)['Volume'].sum()
        poc_bin = vp.idxmax()
        poc_price = (poc_bin.left + poc_bin.right) / 2
        vp_status = "🛡️ 支撐" if df['Close'].iloc[-1] > poc_price else "🧱 壓力"

        # 📊 成交量爆發力 (Volume Ratio) - 修正時間加權 Bug
        vol_ratio_report = "N/A"
        try:
            avg_vol = info.get('averageVolume')
            curr_vol = info.get('regularMarketVolume')
            if avg_vol and curr_vol and avg_vol > 0:
                import pytz
                est = pytz.timezone('US/Eastern')
                now_est = datetime.datetime.now(est)
                open_time = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
                
                # 計算已開盤分鐘數 (最多 390 分鐘)
                if now_est < open_time:
                    vol_ratio_report = "N/A (未開盤)"
                else:
                    elapsed_mins = min(390, (now_est - open_time).total_seconds() / 60)
                    if elapsed_mins <= 0:
                        vol_ratio_report = "N/A"
                    else:
                        expected_vol_at_now = (avg_vol / 390) * elapsed_mins
                        vol_ratio = curr_vol / expected_vol_at_now
                        vol_ratio_report = f"{vol_ratio:.2f}x"
        except: pass

        report = f"🚀 === {symbol} 美股即時戰情 ===\n"
        report += f"● 現價: {df['Close'].iloc[-1]:.2f} | 買賣比: {ba_ratio:.2f} | P/C Ratio: {pc_report}\n"
        report += f"● 成交量能比: {vol_ratio_report} | POC 密集區: {poc_price:.2f} ({vp_status})\n"
        report += "【📊 最近 5 根 K 線】\n"
        for _, row in df.tail(5).iterrows():
            report += f"  [{row.name.strftime('%H:%M')}] {'🟢' if row['Close']>row['Open'] else '🔴'} C:{row['Close']:.2f} | 量:{int(row['Volume'])}\n"
        return report
    except Exception as e: return f"❌ 美股掃描失敗: {e}"

def get_market_sentiment() -> str:
    indicators = {
        "^TWII": "台股(加權)", "TSM": "台積ADR", "EWT": "台灣ETF",
        "^GSPC": "標普500(大盤)", "^IXIC": "那指(科技)", "^SOX": "費半(基石)", "^RUT": "羅素2000(水溫)",
        "^TNX": "美債10Y(重力)", "TLT": "20Y美債(避風港)",
        "DX-Y.NYB": "美元(水龍頭)", "TWD=X": "台幣(外資)", "JPY=X": "日圓(套利)",
        "^VIX": "恐慌(絞肉機)", "HYG": "高收債(風險)", "XLU": "公用事業(防禦)",
        "GC=F": "黃金(避險)", "CL=F": "原油(通膨)", "BZ=F": "布蘭特(地緣)", "HG=F": "銅(景氣)",
        "BTC-USD": "BTC"
    }
    report = "【🌐 全球宏觀資金流向雷達】\n"
    for symbol, name in indicators.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="10d")
            if not hist.empty:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((curr - prev) / prev) * 100
                
                # 計算量能比 (今日成交量 / 前 5 日平均成交量) - 加入時間加權修正
                vol_ratio_str = ""
                if 'Volume' in hist.columns:
                    today_vol = hist['Volume'].iloc[-1]
                    avg_vol = hist['Volume'].iloc[-6:-1].mean()
                    if avg_vol > 0:
                        v_ratio = today_vol / avg_vol
                        
                        # 時間加權修正 (如果是美股/全球市場)
                        is_global = any(kw in symbol for kw in ['^', 'BTC', '=F', 'DX-Y', 'X'])
                        if is_global:
                            import pytz
                            est = pytz.timezone('US/Eastern')
                            now_est = datetime.datetime.now(est)
                            # 粗略估計美股開盤進度 (09:30 - 16:00)
                            open_time = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
                            if now_est >= open_time:
                                elapsed_mins = min(390, (now_est - open_time).total_seconds() / 60)
                                if 10 < elapsed_mins < 390:
                                    v_ratio = v_ratio / (elapsed_mins / 390)

                        # 過濾掉異常過大的期貨換月雜訊
                        if 0.1 < v_ratio < 10:
                            vol_ratio_str = f" [量:{v_ratio:.1f}x]"

                # 判定狀態圖示
                if change > 1.5: emoji = '🚀'
                elif change > 0: emoji = '📈'
                elif change < -1.5: emoji = '💀'
                else: emoji = '📉'
                
                report += f"{emoji} {name}: {curr:.2f} ({change:+.2f}%){vol_ratio_str}\n"
        except Exception as e:
            logger.debug(f"Market sentiment fetch failed for {symbol}: {e}")
    return report

def get_stock_news(symbol: str) -> str:
    try:
        symbol = normalize_ticker(symbol)
        search_symbol = symbol.upper()
        if search_symbol.isdigit(): search_symbol += ".TW"
        ticker = yf.Ticker(search_symbol)
        news_list = ticker.news[:10]
        if not news_list: return "無新聞數據。"
        report = f"【📰 {symbol} 最新情報】\n"
        for i, item in enumerate(news_list):
            title = item.get('title') or item.get('content', {}).get('title')
            publisher = item.get('publisher') or item.get('content', {}).get('provider', {}).get('displayName', '媒體')
            report += f"{i+1}. [{publisher}] {title}\n"
        return report
    except Exception as e: return f"新聞異常: {e}"

def get_fundamental_data(symbol: str) -> str:
    try:
        symbol = normalize_ticker(symbol)
        s = symbol.upper()
        if s.isdigit(): s += ".TW"
        ticker = yf.Ticker(s)
        info = ticker.info
        
        # 提取更多關鍵指標
        eps = info.get('trailingEps', 'N/A')
        pe = info.get('trailingPE', 'N/A')
        pb = info.get('priceToBook', 'N/A')
        short_ratio = info.get('shortRatio', 'N/A')
        inst_own = info.get('heldPercentInstitutions')
        inst_own_str = f"{inst_own*100:.1f}%" if inst_own is not None else "N/A"
        
        report = f"【📊 {symbol} 深度基本面】\n"
        report += f"● EPS: {eps} | P/E: {pe} | P/B: {pb}\n"
        report += f"● 空頭回補天數 (Days to Cover): {short_ratio}\n"
        report += f"● 機構持倉比: {inst_own_str}"
        
        return report
    except Exception as e:
        return f"基本面數據獲取失敗: {e}"

def get_technical_analysis(symbol: str) -> str:
    try:
        symbol = normalize_ticker(symbol)
        s = symbol.upper()
        clean_symbol = s.replace('.TW', '').replace('.TWO', '')
        is_taiwan = any(char.isdigit() for char in clean_symbol) and (len(clean_symbol) <= 6)
        
        # --- 台股使用 Fubon SDK 官方數據 ---
        if is_taiwan and fubon.fubon_ready:
            return fubon.get_fubon_technical(clean_symbol)
            
        # --- 美股使用 yfinance + pandas 自行計算 ---
        ticker = yf.Ticker(s)
        df = ticker.history(period="6mo")
        if df.empty: return f"❌ {s} 無法取得歷史數據。"
        
        close = df['Close']
        # 1. 計算 RSI (14) - 修正為標準 Wilder's Smoothing (EWM)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 2. 計算 MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        dif = exp12 - exp26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = dif - dea
        
        # 3. 計算布林通道 (20)
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper = ma20 + (std20 * 2)
        lower = ma20 - (std20 * 2)
        
        # 4. 取得 52 週高低 (yf info)
        info = ticker.info
        h52 = info.get('fiftyTwoWeekHigh', df['High'].max())
        l52 = info.get('fiftyTwoWeekLow', df['Low'].min())
        curr = close.iloc[-1]
        
        report = f"🇺🇸 === {s} 美股全武裝分析 ===\n"
        report += f"● 現價: {curr:.2f} | 52週高: {h52:.2f} | 52週低: {l52:.2f}\n"
        report += f"● RSI(14): {rsi.iloc[-1]:.2f} ({'🔥超買' if rsi.iloc[-1]>70 else '❄️超跌' if rsi.iloc[-1]<30 else '⚖️中性'})\n"
        report += f"● MACD: DIF:{dif.iloc[-1]:.2f} | 柱狀體:{macd_hist.iloc[-1]:.2f} ({'📈多頭增強' if macd_hist.iloc[-1]>0 else '📉空頭衰退'})\n"
        report += f"● 布林通道: 上軌:{upper.iloc[-1]:.2f} | 下軌:{lower.iloc[-1]:.2f}\n"
        
        # 戰術建議 (優化：結合 RSI 濾網)
        curr_rsi = rsi.iloc[-1]
        if curr >= upper.iloc[-1]:
            if curr_rsi > 75:
                report += f"⚠️ 戰略：觸及布林上軌且 RSI 極度過熱 ({curr_rsi:.2f})，短線噴發過頭，不建議追高。\n"
            elif 55 < curr_rsi <= 75:
                report += f"🔥 戰略：強勢沿上軌攀升中 (RSI: {curr_rsi:.2f})，留意跌破均線停利。\n"
            else:
                report += "⚠️ 戰略：觸及布林上軌，留意拉回風險。\n"
        elif curr <= lower.iloc[-1]:
            if curr_rsi < 25:
                report += f"🎯 戰略：觸及布林下軌且極度超跌 ({curr_rsi:.2f})，具備技術性反彈潛力！\n"
            elif 25 <= curr_rsi < 45:
                report += f"⚠️ 戰略：沿下軌弱勢下跌中 ({curr_rsi:.2f})，切勿盲目抄底。\n"
            else:
                report += "🎯 戰略：觸及布林下軌，具備反彈潛力。\n"
        elif curr_rsi < 30:
            report += f"🔥 戰略：RSI 極度超跌 ({curr_rsi:.2f})，隨時可能暴力反彈。\n"
        else:
            report += "🧘 戰略：目前位階中性，建議分批佈局或等待關鍵突破。\n"
        
        return report
    except Exception as e: return f"❌ 技術分析失敗: {e}"

def get_market_history(symbol: str, days: int) -> str:
    try:
        symbol = normalize_ticker(symbol)
        s = symbol.upper()
        if s.isdigit() and not s.endswith('.TW'): s += '.TW'
        hist = yf.Ticker(s).history(period="1mo").tail(days)
        report = f"【📅 {symbol} 歷史走勢】\n"
        for date, row in hist.iterrows():
            report += f"[{date.strftime('%m/%d')}] 收:{row['Close']:.2f} | 量:{int(row['Volume'])}\n"
        return report
    except Exception as e:
        logger.error(f"Market history fetch failed for {symbol}: {e}")
        return "歷史數據獲取失敗。"
