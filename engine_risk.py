import io
import time
import sqlite3
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta

DB_FILE = "portfolio.db"

def init_market_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_history (
            date TEXT PRIMARY KEY,
            SPX REAL, VIX REAL, DXY REAL, TNX REAL, GOLD REAL, SKEW REAL,
            dix REAL, gex REAL
        )
    """)
    conn.commit()
    conn.close()

def update_market_db():
    init_market_db()
    conn = sqlite3.connect(DB_FILE)
    last_date_df = pd.read_sql("SELECT MAX(date) as last_date FROM market_history", conn)
    last_date_str = last_date_df['last_date'].iloc[0]
    period = "1y" if not last_date_str else "7d"
    
    tickers = {'SPX': '^GSPC', 'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'GOLD': 'GC=F', 'SKEW': '^SKEW'}
    yf_dfs = []
    for name, ticker in tickers.items():
        hist = yf.Ticker(ticker).history(period=period)
        if not hist.empty:
            s = hist['Close'].rename(name)
            s.index = s.index.tz_localize(None).strftime('%Y-%m-%d')
            yf_dfs.append(s)
    
    if not yf_dfs: return
    new_yf_df = pd.concat(yf_dfs, axis=1, sort=True)

    url = 'https://squeezemetrics.com/monitor/static/DIX.csv'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        req = requests.get(url, headers=headers, timeout=5)
        sm_df = pd.read_csv(io.StringIO(req.text))
        sm_df['date'] = pd.to_datetime(sm_df['date']).dt.strftime('%Y-%m-%d')
        sm_df.set_index('date', inplace=True)
        sm_data = sm_df[['dix', 'gex']]
    except:
        sm_data = pd.DataFrame(columns=['dix', 'gex'])

    final_new_df = pd.merge(new_yf_df, sm_data, left_index=True, right_index=True, how='left').ffill()
    
    conn_cursor = conn.cursor()
    for date, row in final_new_df.iterrows():
        conn_cursor.execute("""
            INSERT OR REPLACE INTO market_history (date, SPX, VIX, DXY, TNX, GOLD, SKEW, dix, gex)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, row['SPX'], row['VIX'], row['DXY'], row['TNX'], row['GOLD'], row['SKEW'], row['dix'], row['gex']))
    conn.commit()
    conn.close()

def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def get_realtime_spy_gex():
    """計算 SPY GEX (單位: Billions)"""
    try:
        spy = yf.Ticker("SPY")
        spot = spy.history(period="1d")["Close"].iloc[-1]
        expirations = spy.options[:3]
        total_gex = 0
        for exp in expirations:
            opt = spy.option_chain(exp)
            T = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365.0
            if T <= 0: T = 0.001
            
            # SPY 合約單位為 100 股
            calls = opt.calls.dropna()
            puts = opt.puts.dropna()
            for _, row in calls.iterrows():
                g = calculate_gamma(spot, row['strike'], T, 0.04, row['impliedVolatility'])
                total_gex += row['openInterest'] * 100 * g * (spot**2) * 0.01
            for _, row in puts.iterrows():
                g = calculate_gamma(spot, row['strike'], T, 0.04, row['impliedVolatility'])
                total_gex -= row['openInterest'] * 100 * g * (spot**2) * 0.01
        return total_gex / 10**9 
    except: return None

def get_market_sentiment_score():
    """整合新聞情緒分析 (取代冗長新聞清單)"""
    try:
        news = yf.Ticker("SPY").news[:10]
        if not news: return 0.0, "無數據"
        bear_keywords = ['drop', 'fall', 'recession', 'lower', 'fear', 'warn', 'weak', 'risk', 'inflation', 'sell']
        bull_keywords = ['rise', 'rally', 'growth', 'strong', 'gain', 'support', 'buy', 'optimism', 'beat']
        score = 0
        for item in news:
            title = (item.get('title') or "").lower()
            for w in bear_keywords:
                if w in title: score -= 1
            for w in bull_keywords:
                if w in title: score += 1
        normalized_score = max(-1.0, min(1.0, score / 10.0))
        summary = "偏多" if normalized_score > 0.2 else "偏空" if normalized_score < -0.2 else "中性"
        return normalized_score, summary
    except:
        return 0.0, "分析失敗"

def add_dynamic_metrics(df, column_name, window=120):
    if column_name not in df.columns: return df
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    rolling_mean = df[column_name].rolling(window=window).mean()
    rolling_std = df[column_name].rolling(window=window).std()
    df[f'{column_name}_Z'] = np.where(rolling_std == 0, 0, (df[column_name] - rolling_mean) / rolling_std)
    df[f'{column_name}_PR'] = df[column_name].rolling(window=window).apply(
        lambda x: stats.percentileofscore(x, x[-1], kind='weak') / 100.0, raw=True
    )
    df[f'{column_name}_10MA'] = df[column_name].rolling(window=10).mean()
    df[f'{column_name}_20MA'] = df[column_name].rolling(window=20).mean()
    return df

def fetch_all_market_data():
    try:
        update_market_db()
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql("SELECT * FROM market_history ORDER BY date ASC", conn)
        conn.close()
        if df.empty: return df
        df.set_index('date', inplace=True)
        for col in ['SPX', 'VIX', 'DXY', 'TNX', 'GOLD', 'SKEW', 'dix', 'gex']:
            if col in df.columns:
                df = add_dynamic_metrics(df, col, window=120)
        return df.dropna(subset=['SPX', 'SPX_20MA'])
    except: return pd.DataFrame()

_risk_cache = {"report": "", "timestamp": 0, "expiry": 1200}

def get_global_risk_radar() -> str:
    global _risk_cache
    current_time = time.time()
    if _risk_cache["report"] and (current_time - _risk_cache["timestamp"] < _risk_cache["expiry"]):
        return _risk_cache["report"] + "\n(⚡ DB-Cached)"

    try:
        df = fetch_all_market_data()
        if df.empty: return "❌ 雷達掃描失敗。"
        latest = df.iloc[-1]
        
        rt_gex = get_realtime_spy_gex()
        final_gex = rt_gex if rt_gex is not None else (latest.get('gex', 0) / 10**9)
        sent_score, sent_label = get_market_sentiment_score()
        
        # --- 移植自 test.py 的核心計分邏輯 ---
        base_risk = 0
        reasons = []

        # 1. 環境與籌碼底分 (Armed 狀態判斷)
        if latest.get('DXY_Z', 0) > 1.5 or latest.get('TNX_Z', 0) > 1.5:
            base_risk += 25
            reasons.append("🔴 [Armed] 資金緊縮 (美元/美債突波)")
        
        if latest.get('VIX_Z', 0) > 2.0 or final_gex < 0:
            base_risk += 25
            reasons.append(f"🔴 [Armed] 波動率失控 / 負 Gamma ({final_gex:.2f}B)")
        
        if latest.get('SKEW_PR', 0) > 0.90 or latest.get('GOLD_PR', 0) > 0.85:
            base_risk += 15
            reasons.append("🔴 [Armed] 尾部風險升溫 (黑天鵝/黃金擁擠)")
            
        # 滅火器：大戶吸籌
        if latest.get('dix_PR', 0) > 0.85:
            base_risk = max(0, base_risk - 20)
            reasons.append("🟢 [Safe] 暗池吸籌，大戶提供下檔支撐")

        # 整合新聞情緒 (額外加權)
        if sent_score < -0.4:
            base_risk += 10
            reasons.append(f"📰 新聞極度偏空 ({sent_label})")

        is_armed = base_risk >= 40
        final_score = base_risk
        
        # 2. 技術扳機 (動態權重)
        spx = latest.get('SPX', 0)
        ma10 = latest.get('SPX_10MA', 0)
        ma20 = latest.get('SPX_20MA', 0)

        if spx < ma20:
            if is_armed:
                final_score += 40
                reasons.append("🚨 [Trigger] 趨勢破滅：跌破月線且環境惡化！")
            else:
                final_score += 15
                reasons.append("🟠 [中期轉弱] 跌破月線，部位應收斂。")
        elif spx < ma10:
            if is_armed:
                final_score += 25
                reasons.append("🚨 [Trigger] 致命破線：環境不佳且跌破 10MA。")
            else:
                final_score += 5
                reasons.append("🟡 [技術回檔] 跌破 10MA，籌碼尚可。")

        score = min(100, final_score)
        state = "🟢 多頭" if score < 30 else "🟡 整理" if score < 45 else "🔴 警戒" if score < 75 else "💀 系統風險"
        
        msg = f"📊 *【MarginCall_2X 全局雷達】*\n🔥 風險分數：{score} ({state})\n"
        msg += "\n".join(reasons) if reasons else "🟢 指標目前健康"
        
        msg += f"\n\n- DIX_PR: {latest.get('dix_PR', 0):.2f}\n- GEX: {final_gex:.2f}B\n- Sentiment: {sent_label}({sent_score:.2f})"
        msg += f"\n- SPX: {spx:.1f} (10MA:{ma10:.1f}, 20MA:{ma20:.1f})"
        
        _risk_cache["report"] = msg
        _risk_cache["timestamp"] = current_time
        return msg
    except Exception as e: return f"❌ 雷達異常: {e}"
