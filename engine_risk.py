import io
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats

def add_dynamic_metrics(df, column_name, window=120):
    if column_name not in df.columns: return df
    rolling_mean = df[column_name].rolling(window=window).mean()
    rolling_std = df[column_name].rolling(window=window).std()
    df[f'{column_name}_Z'] = np.where(rolling_std == 0, 0, (df[column_name] - rolling_mean) / rolling_std)
    df[f'{column_name}_PR'] = df[column_name].rolling(window=window).apply(
        lambda x: stats.percentileofscore(x, x[-1], kind='weak') / 100.0, raw=True
    )
    df[f'{column_name}_10MA'] = df[column_name].rolling(window=10).mean()
    df[f'{column_name}_20MA'] = df[column_name].rolling(window=20).mean()
    return df

def fetch_squeezemetrics_data():
    url = 'https://squeezemetrics.com/monitor/static/DIX.csv'
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        req = requests.get(url, headers=headers, timeout=10)
        sm_df = pd.read_csv(io.StringIO(req.text))
        sm_df['date'] = pd.to_datetime(sm_df['date'])
        sm_df.set_index('date', inplace=True)
        return sm_df[['dix', 'gex']]
    except: return pd.DataFrame()

def fetch_all_market_data():
    sm_df = fetch_squeezemetrics_data()
    tickers = {'SPX': '^GSPC', 'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'TNX': '^TNX', 'GOLD': 'GC=F', 'SKEW': '^SKEW'}
    df_list = []
    for name, ticker in tickers.items():
        hist = yf.Ticker(ticker).history(period='1y')
        if not hist.empty:
            s = hist['Close'].rename(name)
            s.index = s.index.tz_localize(None)
            df_list.append(s)
            
    yf_df = pd.concat(df_list, axis=1, sort=True)
    market_df = pd.merge(yf_df, sm_df, left_index=True, right_index=True, how='left').ffill()
    
    for col in ['SPX', 'VIX', 'DXY', 'TNX', 'GOLD', 'SKEW', 'dix', 'gex']:
        if col in market_df.columns:
            market_df = add_dynamic_metrics(market_df, col, window=120)
    return market_df.dropna(subset=['SPX', 'SPX_20MA'])

def get_global_risk_radar() -> str:
    try:
        df = fetch_all_market_data()
        if df.empty: return "❌ 雷達掃描失敗。"
        latest = df.iloc[-1]
        
        score = 0
        reasons = []
        if latest.get('DXY_Z', 0) > 1.5: score += 15; reasons.append("🔴 美元強勢")
        if latest.get('VIX_Z', 0) > 2.0: score += 20; reasons.append("🔴 恐慌噴發")
        if latest.get('gex', 0) < 0: score += 20; reasons.append("🔴 負 Gamma 環境")
        if latest.get('SPX', 0) < latest.get('SPX_20MA', 0): score += 25; reasons.append("🚨 跌破月線")

        state = "🟢 多頭" if score < 30 else "🟡 整理" if score < 50 else "🔴 警戒" if score < 80 else "💀 系統風險"
        msg = f"📊 *【MarginCall_2X 全局雷達】*\n🔥 風險分數：{score} ({state})\n"
        msg += "\n".join(reasons) if reasons else "🟢 指標健康"
        msg += f"\n- DIX_PR: {latest.get('dix_PR', 0):.2f}\n- GEX: {latest.get('gex', 0):.0f}"
        return msg
    except Exception as e: return f"❌ 雷達異常: {e}"
