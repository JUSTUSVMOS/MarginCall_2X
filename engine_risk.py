import io
import time
import sqlite3
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import threading
import logging
from scipy import stats
from datetime import datetime, timedelta

# 設定日誌
logger = logging.getLogger(__name__)

from config import DB_FILE

db_lock = threading.Lock() # 【V4 加固】資料庫互斥鎖

def init_market_db():
    with db_lock:
        conn = sqlite3.connect(str(DB_FILE))
        cursor = conn.cursor()
        # 【V5 終極加固】WAL 模式啟動指令
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_history (
                date TEXT PRIMARY KEY,
                SPX REAL, VIX REAL, DXY REAL, TNX REAL, GOLD REAL, SKEW REAL,
                dix REAL, gex REAL
            )
        """)
        # 新增 V 轉狀態追蹤表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS v_turn_state (
                id INTEGER PRIMARY KEY,
                is_confirmed INTEGER,
                day1_date TEXT,
                day1_price REAL,
                ftd_date TEXT,
                last_check_date TEXT
            )
        """)
        # 【重構】資產類型快取表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS asset_profile_cache (
                symbol TEXT PRIMARY KEY,
                asset_type TEXT,
                sector TEXT,
                industry TEXT,
                risk_score REAL,
                last_updated DATETIME
            )
        """)
        conn.commit()
        conn.close()

def get_db_connection():
    """取得開啟 WAL 模式的資料庫連線"""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def get_v_turn_state():
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        try:
            df = pd.read_sql("SELECT * FROM v_turn_state WHERE id = 1", conn)
            return df.iloc[0] if not df.empty else None
        except Exception as e:
            logger.error(f"Failed to get v_turn_state: {e}")
            return None
        finally: conn.close()

def save_v_turn_state(is_confirmed, day1_date, day1_price, ftd_date):
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO v_turn_state (id, is_confirmed, day1_date, day1_price, ftd_date, last_check_date)
                VALUES (1, ?, ?, ?, ?, ?)
            """, (is_confirmed, day1_date, day1_price, ftd_date, datetime.now().strftime('%Y-%m-%d')))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save v_turn_state: {e}")
        finally: conn.close()

def calculate_buying_pressure(df, window=5):
    """
    【V5 強化】計算 K 線的淨買盤壓力 (CVD 邏輯替代 $TICK)
    """
    if df is None or df.empty or len(df) < window:
        return 0.0
    epsilon = 1e-8 
    # 計算買盤強度權重 (留長下影線權重高，留長上影線權重低)
    weight = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + epsilon)
    buy_vol = df['Volume'] * weight
    sell_vol = df['Volume'] - buy_vol
    net_vol = buy_vol - sell_vol
    recent_net_vol = net_vol.tail(window).sum()
    recent_total_vol = df['Volume'].tail(window).sum()
    if recent_total_vol == 0: return 0.0
    return recent_net_vol / recent_total_vol

def update_market_db():
    init_market_db()
    conn = sqlite3.connect(DB_FILE)
    try:
        last_date_df = pd.read_sql("SELECT MAX(date) as last_date FROM market_history", conn)
        last_date_str = last_date_df['last_date'].iloc[0]
        period = "1y" if not last_date_str else "7d"
        
        tickers = {
            'SPX': '^GSPC', 'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 
            'TNX': '^TNX', 'GOLD': 'GC=F', 'SKEW': '^SKEW',
            'SOX': '^SOX', 'HYG': 'HYG', 'OIL': 'CL=F'
        }
        yf_dfs = []
        for name, ticker in tickers.items():
            try:
                hist = yf.Ticker(ticker).history(period=period)
                if not hist.empty:
                    s = hist['Close'].rename(name)
                    s.index = s.index.tz_localize(None).strftime('%Y-%m-%d')
                    yf_dfs.append(s)
            except Exception as e:
                logger.warning(f"Failed to fetch market ticker {ticker}: {e}")
        
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
        except Exception as e:
            logger.warning(f"Failed to fetch DIX data: {e}")
            sm_data = pd.DataFrame(columns=['dix', 'gex'])

        final_new_df = pd.merge(new_yf_df, sm_data, left_index=True, right_index=True, how='left').ffill()
        
        conn_cursor = conn.cursor()
        for date, row in final_new_df.iterrows():
            conn_cursor.execute("""
                INSERT OR REPLACE INTO market_history (date, SPX, VIX, DXY, TNX, GOLD, SKEW, SOX, HYG, OIL, dix, gex)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (date, row.get('SPX'), row.get('VIX'), row.get('DXY'), row.get('TNX'), 
                  row.get('GOLD'), row.get('SKEW'), row.get('SOX'), row.get('HYG'), 
                  row.get('OIL'), row.get('dix'), row.get('gex')))
        conn.commit()
    except Exception as e:
        logger.error(f"Market DB update failed: {e}")
    finally:
        conn.close()

def calculate_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    except Exception as e:
        logger.debug(f"Gamma calculation error: {e}")
        return 0

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
    except Exception as e:
        logger.error(f"Real-time GEX calculation failed: {e}")
        return None

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
    except Exception as e:
        logger.error(f"Market sentiment score failed: {e}")
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
    df[f'{column_name}_200MA'] = df[column_name].rolling(window=200).mean()
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
    except Exception as e:
        logger.error(f"Fetch all market data failed: {e}")
        return pd.DataFrame()

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
        risk_multiplier = 1.0
        reasons = []

        # 1. 環境與籌碼底分 (Multiplicative Factors)
        if latest.get('DXY_Z', 0) > 1.5 or latest.get('TNX_Z', 0) > 1.5:
            risk_multiplier *= 1.5    # 資金緊縮
            reasons.append("🔴 資金緊縮 (美元/美債突波)")
        
        if latest.get('VIX_Z', 0) > 2.0 or final_gex < 0:
            risk_multiplier *= 1.6    # 波動率失控
            reasons.append(f"🔴 波動率失控 / 負 Gamma ({final_gex:.2f}B)")
        
        if latest.get('SKEW_PR', 0) > 0.90:
            risk_multiplier *= 1.3    # 尾部風險
            reasons.append("🟠 尾部風險升溫")
            
        # 滅火器：大戶吸籌 (打折但不清零)
        if latest.get('dix_PR', 0) > 0.85:
            risk_multiplier *= 0.7
            reasons.append("🟢 暗池吸籌，大戶提供下檔支撐")

        # 整合新聞情緒 (乘法因子)
        if sent_score < -0.4:
            risk_multiplier *= 1.2
            reasons.append(f"📰 新聞極度偏空 ({sent_label})")

        # 2. 技術扳機 (動態權重，改為乘法)
        spx = latest.get('SPX', 0)
        ma10 = latest.get('SPX_10MA', 0)
        ma20 = latest.get('SPX_20MA', 0)
        ma200 = latest.get('SPX_200MA', 0) 

        if ma200 > 0 and spx < ma200:
            risk_multiplier *= 1.4
            reasons.append("🚨 [Trigger] 熊市區間：跌破 200MA 均線！")
        elif spx < ma20:
            risk_multiplier *= 1.25
            reasons.append("🚨 [Trigger] 趨勢破滅：跌破月線！")
        elif spx < ma10:
            risk_multiplier *= 1.15
            reasons.append("🚨 [Trigger] 短期轉弱：跌破 10MA。")

        # 轉換成 0-100 分
        # 1.0 = 0分 (無風險)，3.0+ = 100分 (極端風險)
        score = max(0, min(100, int((risk_multiplier - 1.0) / 2.0 * 100)))
        state = "🟢 多頭" if score < 30 else "🟡 整理" if score < 45 else "🔴 警戒" if score < 75 else "💀 系統風險"
        
        msg = f"📊 *【MarginCall_2X 全局雷達】*\n🔥 風險分數：{score} ({state})\n"
        msg += "\n".join(reasons) if reasons else "🟢 指標目前健康"
        
        msg += f"\n\n- DIX_PR: {latest.get('dix_PR', 0):.2f}\n- GEX: {final_gex:.2f}B\n- Sentiment: {sent_label}({sent_score:.2f})"
        msg += f"\n- SPX: {spx:.1f} (10MA:{ma10:.1f}, 20MA:{ma20:.1f}, 200MA:{ma200:.1f})"
        
        _risk_cache["report"] = msg
        _risk_cache["timestamp"] = current_time
        return msg
    except Exception as e:
        logger.error(f"Risk radar analysis failed: {e}")
        return f"❌ 雷達異常: {e}"

def get_v_turn_confirmation() -> str:
    """
    【終極 V 轉確認模組 V5 - 事務安全與 CVD 強化版】
    """
    # 【V5 核心】Transaction 安全鎖，包圍整個 Read-Modify-Write 流程
    with db_lock:
        try:
            init_market_db()
            # 1. 【極速優化】使用 yf.download 批量下載歷史數據，減少連線開銷
            symbols = ["SPLG", "RSP", "HYG", "LQD", "CL=F"]
            hist_data = yf.download(symbols, period="30d", group_by='ticker', progress=False)
            
            # 分配數據
            splg = hist_data['SPLG'].dropna()
            rsp = hist_data['RSP'].dropna()
            hyg = hist_data['HYG'].dropna()
            lqd = hist_data['LQD'].dropna()
            oil = hist_data['CL=F'].dropna()
            
            # 2. 抓取盤中即時指標 (15m/5m)
            vix_df = yf.Ticker("^VIX").history(period="2d", interval="15m")
            vix3m_df = yf.Ticker("^VIX3M").history(period="2d", interval="15m")
            vvix_df = yf.Ticker("^VVIX").history(period="2d", interval="15m")
            spy_5m = yf.Ticker("SPY").history(period="1d", interval="5m")

            if splg.empty or rsp.empty:
                return "❌ yfinance 數據下載失敗，請檢查網路連線。"

            # --- 模組一：狀態恢復與 FTD 判定 (Transaction 保護中) ---
            conn = sqlite3.connect(DB_FILE)
            # 在同一連線內處理事務
            cursor = conn.cursor()
            state_df = pd.read_sql("SELECT * FROM v_turn_state WHERE id = 1", conn)
            state = state_df.iloc[0] if not state_df.empty else None
            
            window = splg.tail(25)
            current_low_idx = window['Close'].idxmin()
            current_low_price = float(window.loc[current_low_idx, 'Close'])
            current_low_date = current_low_idx.strftime('%Y-%m-%d')
            
            # 判斷重置或繼承
            if state is None or current_low_price < float(state['day1_price']):
                is_confirmed, day1_date, day1_price, ftd_date = 0, current_low_date, current_low_price, ""
            else:
                is_confirmed, day1_date, day1_price, ftd_date = int(state['is_confirmed']), state['day1_date'], float(state['day1_price']), state['ftd_date']

            rally_period = splg.loc[day1_date:]
            day_count = len(rally_period)
            today_ftd = False
            if 4 <= day_count <= 20 and is_confirmed == 0:
                today_price, prev_price = rally_period['Close'].iloc[-1], rally_period['Close'].iloc[-2]
                today_vol, prev_vol = rally_period['Volume'].iloc[-1], rally_period['Volume'].iloc[-2]
                if (today_price - prev_price)/prev_price >= 0.015 and today_vol > prev_vol:
                    today_ftd, is_confirmed, ftd_date = True, 1, datetime.now().strftime('%Y-%m-%d')
            
            # 立即更新狀態 (尚未關閉連線)
            cursor.execute("""
                INSERT OR REPLACE INTO v_turn_state (id, is_confirmed, day1_date, day1_price, ftd_date, last_check_date)
                VALUES (1, ?, ?, ?, ?, ?)
            """, (is_confirmed, day1_date, day1_price, ftd_date, datetime.now().strftime('%Y-%m-%d')))
            conn.commit()
            conn.close()

            # --- 模組二：護法判定 (CVD 升級) ---
            # 🎯 解決缺陷 8：市場寬度確認 (5D RSP vs SPLG)
            rsp_5d = (rsp['Close'].iloc[-1] / rsp['Close'].iloc[-5]) - 1 if len(rsp) >= 5 else 0
            splg_5d = (splg['Close'].iloc[-1] / splg['Close'].iloc[-5]) - 1 if len(splg) >= 5 else 0
            breadth_val = rsp_5d - splg_5d
            breadth_safe = (breadth_val > -0.005) # 至少不能輸大盤太多 (健康反彈門檻)
            
            vix_p = vix_df['Close'].iloc[-1] if not vix_df.empty else yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
            vix3m_p = vix3m_df['Close'].iloc[-1] if not vix3m_df.empty else yf.Ticker("^VIX3M").history(period="5d")['Close'].iloc[-1]
            vix_term = vix_p / vix3m_p
            vix_term_safe = (vix_term < 1.0)
            
            # 【V5 CVD 淨買盤壓力判定】
            bp_ratio = calculate_buying_pressure(spy_5m, window=5)
            tick_safe = (bp_ratio > 0.15) 
            tick_emoji = '🔥' if bp_ratio > 0.3 else '🟢' if tick_safe else '⚪'
            tick_msg = f"{bp_ratio:+.1%}"

            vvix_val = vvix_df['Close'].iloc[-1] if not vvix_df.empty else yf.Ticker("^VVIX").history(period="5d")['Close'].iloc[-1]
            vvix_safe = (vvix_val < 110)
            credit_ratio = (hyg['Close'] / lqd['Close']).iloc[-1]
            credit_ma = (hyg['Close'] / lqd['Close']).rolling(20).mean().iloc[-1]
            credit_safe = (credit_ratio > credit_ma)
            ma20 = splg['Close'].rolling(20).mean().iloc[-1]
            ma20_safe = (splg['Close'].iloc[-1] > ma20)

            all_macro_safe = (vix_term_safe and vvix_safe and credit_safe and breadth_safe)
            
            # --- 報告輸出 ---
            status_txt = "🛡️ 偵測底盤中" if is_confirmed == 0 else "🚀 強勢反彈中"
            report = f"📊 *【MarginCall_2X V 轉戰報 V5】*\n當前狀態：{status_txt}\n"
            report += f"- Day 1 低點：{day1_price:.2f} ({day1_date})\n"
            report += f"- 目前進度：Day {day_count}\n"
            if is_confirmed: report += f"- ✅ FTD 點火日：{ftd_date}\n"
            
            report += f"\n🌡️ *核心護法狀態 (CVD 強化)：*\n"
            report += f"- VIX 期限結構: {vix_term:.2f} {'🟢' if vix_term_safe else '🔴'}\n"
            report += f"- VVIX 恐慌速率: {vvix_val:.1f} {'🟢' if vvix_safe else '🔴'}\n"
            report += f"- 信用市場(HYG/LQD): {'🟢' if credit_safe else '🔴'}\n"
            report += f"- 市場寬度(RSP/SPLG): {breadth_val:+.2%} {'🟢' if breadth_safe else '🔴'}\n"
            report += f"- 買盤推力(CVD): {tick_msg} {tick_emoji}\n"
            report += f"- MA20 技術位階: {'🟢' if ma20_safe else '🔴'}\n"
            
            if is_confirmed and all_macro_safe and ma20_safe:
                report += "\n🏁 *【最終判定：發射訊號！】*\n👉 機構確認進場，CVD 買盤力道強勁。建議分批建倉。"
            else:
                report += "\n🏁 *【最終判定：維持現狀】*\n👉 市場尚未出現轉強信號或條件未齊。"
            return report
        except Exception as e:
            logger.error(f"V-turn confirmation failed: {e}")
            return f"❌ V 轉監測失敗: {e}"

def get_capital_flow_matrix() -> str:
    """
    【宏觀資金流向矩陣】
    專門用於計算不同板塊、資產間的「比值 (Ratios)」與「量能」，來判斷資金的流向與避險情緒。
    回傳一段結構化的文字戰報，提供給 AI 進行深度解讀。
    """
    try:
        symbols = ['^SOX', 'XLU', 'HG=F', 'GC=F', '^TNX', 'TLT', 'DX-Y.NYB', 'TWD=X', 'JPY=X', '^VIX']
        hist_data = yf.download(symbols, period="1mo", group_by='ticker', progress=False)
        
        def get_recent(ticker_data):
            if ticker_data is None or ticker_data.empty: return None, None
            df = ticker_data.dropna()
            if df.empty or len(df) < 2: return None, None
            return float(df['Close'].iloc[-1]), float(df['Close'].iloc[-2])
            
        def get_vol_ratio(ticker_data):
            if ticker_data is None or ticker_data.empty or 'Volume' not in ticker_data: return 1.0
            df = ticker_data.dropna()
            if len(df) < 6: return 1.0
            today_vol = float(df['Volume'].iloc[-1])
            avg_vol = float(df['Volume'].iloc[-6:-1].mean())
            if avg_vol > 0:
                v_ratio = today_vol / avg_vol
                return v_ratio if 0.1 < v_ratio < 10 else 1.0
            return 1.0

        sox, sox_prev = get_recent(hist_data.get('^SOX'))
        xlu, xlu_prev = get_recent(hist_data.get('XLU'))
        hg, hg_prev = get_recent(hist_data.get('HG=F'))
        gc, gc_prev = get_recent(hist_data.get('GC=F'))
        tnx, tnx_prev = get_recent(hist_data.get('^TNX'))
        tlt, tlt_prev = get_recent(hist_data.get('TLT'))
        jpy, jpy_prev = get_recent(hist_data.get('JPY=X'))
        
        tlt_vol = get_vol_ratio(hist_data.get('TLT'))
        xlu_vol = get_vol_ratio(hist_data.get('XLU'))
        
        report = "🧠 *【Capital Flow Matrix 資金流向矩陣】*\n"
        
        # 1. 景氣與板塊輪動
        if sox is not None and xlu is not None:
            sox_chg = (sox - sox_prev) / sox_prev * 100
            xlu_chg = (xlu - xlu_prev) / xlu_prev * 100
            tech_def_spread = sox_chg - xlu_chg
            if tech_def_spread < -1.5 and xlu_vol > 1.2:
                report += f"🔄 **板塊輪動 (Risk-Off):** 資金從科技股(SOX {sox_chg:+.2f}%) 撤退，防禦性公用事業(XLU {xlu_chg:+.2f}%) 放量({xlu_vol:.1f}x)承接。\n"
            elif tech_def_spread > 1.5:
                report += f"🔥 **風險偏好 (Risk-On):** 資金集中攻擊科技股 (SOX {sox_chg:+.2f}%)，公用事業跑輸大盤。\n"
            else:
                report += f"⚖️ **板塊狀態中性:** SOX({sox_chg:+.2f}%) vs XLU({xlu_chg:+.2f}%) 輪動不明顯。\n"
                
        # 2. 實體景氣 (銅金比)
        if hg is not None and gc is not None:
            hg_chg = (hg - hg_prev) / hg_prev * 100
            gc_chg = (gc - gc_prev) / gc_prev * 100
            cg_spread = hg_chg - gc_chg
            if cg_spread < -1.0:
                report += f"📉 **衰退疑慮 (銅金比轉弱):** 銅博士({hg_chg:+.2f}%)走弱，黃金({gc_chg:+.2f}%)避險升溫，實體經濟預期放緩。\n"
            elif cg_spread > 1.0:
                report += f"🏭 **復甦預期 (銅金比轉強):** 銅價({hg_chg:+.2f}%)跑贏黃金，工業/實體需求強勁。\n"
                
        # 3. 匯率與套利平倉 (Carry Trade)
        if jpy is not None:
            jpy_chg = (jpy - jpy_prev) / jpy_prev * 100 # JPY=X 是 USD/JPY，數字變小代表日圓升值
            if jpy_chg < -0.8: # 日圓單日急升超過 0.8%
                report += f"🚨 **套利平倉警戒 (Carry Trade Unwind):** 日圓急升({jpy_chg:+.2f}%)，高度留意全球流動性收緊與跨國資產拋售。\n"
            elif jpy_chg > 0.8:
                report += f"💸 **套利資金寬鬆:** 日圓貶值({jpy_chg:+.2f}%)，有利於全球風險資產的槓桿資金池。\n"

        # 4. 長債避風港
        if tnx is not None and tlt is not None:
            tlt_chg = (tlt - tlt_prev) / tlt_prev * 100
            tnx_chg = (tnx - tnx_prev) / tnx_prev * 100
            if tnx_chg > 2.0:
                report += f"🎈 **估值重力壓迫:** 10年期美債殖利率飆升({tnx_chg:+.2f}%)，將對科技股估值造成壓力。\n"
            elif tlt_chg > 1.0 and tlt_vol > 1.3:
                report += f"🛡️ **終極避風港進駐:** 20年期美債(TLT) 放量上漲({tlt_chg:+.2f}%, 量:{tlt_vol:.1f}x)，大資金正在尋求絕對避險。\n"
                
        if report == "🧠 *【Capital Flow Matrix 資金流向矩陣】*\n":
            return report + "目前無明顯異常資金流向信號。\n"
            
        return report
    except Exception as e:
        logger.error(f"Capital Flow Matrix calculation failed: {e}")
        return f"❌ 資金流向矩陣計算失敗: {e}\n"
