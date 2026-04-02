import os
import datetime
import requests
import time
import pandas as pd
import numpy as np
import yfinance as yf
import fubon  # 引用現有的 fubon.py

FMP_KEY = os.getenv("FMP_API_KEY")

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
    symbol = symbol.upper().replace('.TW', '').replace('.TWO', '')
    is_taiwan = any(char.isdigit() for char in symbol) and (len(symbol) <= 6)
    
    if is_taiwan and fubon.fubon_ready:
        try:
            # 利用富邦歷史統計功能來抓取官方名稱
            stats = fubon.get_historical_stats(symbol)
            if "未知" not in stats and "異常" not in stats:
                return stats
        except: pass
        
    try:
        s = f"{symbol}.TW" if is_taiwan and not symbol.endswith('.TW') else symbol
        ticker = yf.Ticker(s)
        info = ticker.info
        name = info.get('longName') or info.get('shortName') or "未知標的"
        return f"🔍 識別結果: {symbol} ({name}) | 類型: {info.get('quoteType', '未知')}"
    except:
        return f"❌ 無法識別標的: {symbol}，請確認代號是否正確。"

def get_live_price(symbol: str) -> str:
    symbol = symbol.upper()
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
        except: pass

    if not is_taiwan_stock and FMP_KEY and is_us_market_open():
        try:
            url = f"https://financialmodelingprep.com/stable/quote?symbol={symbol}&apikey={FMP_KEY}"
            res = requests.get(url, timeout=5).json()
            if isinstance(res, list) and len(res) > 0:
                return f"{round(float(res[0]['price']), 2)} (來源: FMP)"
        except: pass

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
        except: continue
    return "無法取得報價"

def get_us_realtime_insight(symbol: str) -> str:
    symbol = symbol.upper()
    try:
        ticker = yf.Ticker(symbol)
        full_df = ticker.history(period="1d", interval="5m")
        if full_df.empty: return f"❌ {symbol} 目前無盤中數據。"
        df = full_df.tail(10)
        info = ticker.info
        bid, ask = info.get('bid', 0), info.get('ask', 0)
        ba_ratio = (info.get('bidSize', 1) / info.get('askSize', 1)) if info.get('askSize', 0) > 0 else 1
        
        # 🎭 Put/Call Ratio 計算 (整合自 test_yf.py)
        pc_report = "N/A"
        try:
            if ticker.options:
                opt = ticker.option_chain(ticker.options[0])
                calls_vol = opt.calls['volume'].sum()
                puts_vol = opt.puts['volume'].sum()
                pc_ratio = puts_vol / calls_vol if calls_vol > 0 else 0
                pc_report = f"{pc_ratio:.2f}"
        except: pass

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
                # 計算美股開盤至今經過的比例 (美東 09:30 ~ 16:00, 共 6.5 小時)
                import pytz
                est = pytz.timezone('US/Eastern')
                now_est = datetime.datetime.now(est)
                open_time = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
                
                # 計算已開盤分鐘數 (最多 390 分鐘)
                elapsed_mins = max(1, min(390, (now_est - open_time).total_seconds() / 60))
                expected_vol_at_now = (avg_vol / 390) * elapsed_mins
                vol_ratio = curr_vol / expected_vol_at_now
                vol_ratio_report = f"{vol_ratio:.2f}x"
        except: pass

        report = f"🚀 === {symbol} 美股即時戰情 ===\n"
        report += f"● 現價: {df['Close'].iloc[-1]:.2f} | 買賣比: {ba_ratio:.2f} | P/C Ratio: {pc_report}\n"
        report += f"● 成交量能比: {vol_ratio_report} | POC 密集區: {poc_price:.2f} ({vp_status})\n"
        report += "【📊 最近 5 根 K 線】\n"
        for _, row in df.tail(5).iterrows():
            report += f"  [{row.name.strftime('%H:%M')}] {'🔴' if row['Close']>row['Open'] else '🟢'} C:{row['Close']:.2f} | 量:{int(row['Volume'])}\n"
        return report
    except Exception as e: return f"❌ 美股掃描失敗: {e}"

def get_market_sentiment() -> str:
    indicators = {
        "ES=F": "標普期", "NQ=F": "那指期", "YM=F": "道瓊期",
        "WTX=F": "台指期", "TSM": "台積ADR", # 補上台股核心領先指標
        "^TNX": "美債10Y", "DX-Y.NYB": "美元", "GC=F": "黃金",
        "^VIX": "恐慌", "BTC-USD": "BTC"
    }
    report = "【🌐 全球資金流向與台美連動】\n"
    for symbol, name in indicators.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((curr - prev) / prev) * 100
                
                # 針對 TSM 與 台指期 增加戰術標記
                emoji = '🚀' if change > 1.5 else '📈' if change > 0 else '📉' if change > -1.5 else '💀'
                report += f"{emoji} {name}: {curr:.2f} ({change:+.2f}%)\n"
        except: pass
    return report

def get_stock_news(symbol: str) -> str:
    try:
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
        report += f"● Short Ratio (軋空比): {short_ratio}\n"
        report += f"● 機構持倉比: {inst_own_str}"
        
        return report
    except Exception as e:
        return f"基本面數據獲取失敗: {e}"

def get_technical_analysis(symbol: str) -> str:
    try:
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
        # 1. 計算 RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 2. 計算 MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        dif = exp12 - exp26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd_hist = (dif - dea) * 2
        
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
        
        # 戰術建議
        if curr >= upper.iloc[-1]: report += "⚠️ 戰略：股價觸及布林上軌，短線噴發過頭，不建議追高。\n"
        elif curr <= lower.iloc[-1]: report += "🎯 戰略：股價觸及布林下軌，且 RSI 偏低，具備反彈潛力！\n"
        elif rsi.iloc[-1] < 30: report += "🔥 戰略：RSI 極度超跌，隨時可能暴力反彈。\n"
        else: report += "🧘 戰略：目前位階中性，建議分批佈局或等待關鍵突破。\n"
        
        return report
    except Exception as e: return f"❌ 技術分析失敗: {e}"

def get_market_history(symbol: str, days: int) -> str:
    try:
        s = symbol.upper()
        if s.isdigit() and not s.endswith('.TW'): s += '.TW'
        hist = yf.Ticker(s).history(period="1mo").tail(days)
        report = f"【📅 {symbol} 歷史走勢】\n"
        for date, row in hist.iterrows():
            report += f"[{date.strftime('%m/%d')}] 收:{row['Close']:.2f} | 量:{int(row['Volume'])}\n"
        return report
    except: return "歷史數據獲取失敗。"
