import io
import math
import time
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from yf_session import get_ticker, get_download
import logging
import os
from typing import Any, Dict
# ... (保留原本的 import)
from scipy import stats
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import engine_market as market
from engine_technical import IndicatorCalculator

# 新增 FRED 相關 import
import json
from src.database import db_lock, get_connection
from src.tools import format_tool_error, tool

# 設定日誌
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY") # 建議使用者在 .env 加入，若無則使用備援邏輯

# --- [新增] FRED 宏觀引擎 (The Macro Sentinel) ---
class MacroEngine:
    """
    專門對接 FRED (聯準會) 的宏觀數據引擎。
    涵蓋：利率、通膨、就業、貨幣供給、殖利率曲線。
    """
    def __init__(self, api_key=None):
        self.api_key = api_key or FRED_API_KEY
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def _fetch_fred(self, series_id):
        """核心請求邏輯 (帶有備援機制)"""
        if not self.api_key:
            # 如果沒有 API Key，嘗試透過 yfinance 模擬部分宏觀指標 (如利率/殖利率)
            return self._fetch_fallback(series_id)

        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 5
        }
        try:
            r = requests.get(self.base_url, params=params, timeout=10)
            data = r.json()
            if 'observations' in data:
                return float(data['observations'][0]['value'])
        except Exception as e:
            logger.error(f"FRED Fetch Error ({series_id}): {e}")
        return self._fetch_fallback(series_id)

    def _fetch_fallback(self, series_id):
        """當 FRED API 失效或無 Key 時的備援 (利用 yfinance 趨勢模擬)"""
        mapping = {
            'T10Y2Y': '^TNX', # 粗略代替，實際上需要 10Y - 2Y
            'FEDFUNDS': '^IRX', # 13周國庫券作為利率基準
        }
        if series_id in mapping:
            try:
                val = get_ticker(mapping[series_id]).history(period="1d")['Close'].iloc[-1]
                return val if series_id != 'T10Y2Y' else val - 4.0 # 假定 2Y 在 4%
            except Exception as e:
                logger.debug(f"GEX calculation error: {e}")
                pass
        return None

    def get_macro_dashboard(self):
        """獲取所有核心宏觀指標"""
        indicators = {
            "Yield_Curve_10Y2Y": "T10Y2Y",
            "Fed_Funds_Rate": "FEDFUNDS",
            "CPI_Inflation": "CPIAUCSL",
            "Non_Farm_Payrolls": "PAYEMS",
            "M2_Money_Supply": "M2SL",
            "Recession_Prob": "RECPROUSM156N"
        }
        results = {}
        for name, sid in indicators.items():
            val = self._fetch_fred(sid)
            results[name] = val
        return results

# ... (保留原本的 init_market_db 等函數)

def init_market_db():
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        # 【V5 終極加固】WAL 模式啟動指令
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_history (
                date TEXT PRIMARY KEY,
                SPX REAL, VIX REAL, DXY REAL, TNX REAL, GOLD REAL, SKEW REAL,
                SOX REAL, HYG REAL, OIL REAL,
                dix REAL, gex REAL
            )
        """)
        # 遷移：若舊表缺少 SOX/HYG/OIL 欄位，手動補上
        for col in ['SOX', 'HYG', 'OIL']:
            try:
                cursor.execute(f"ALTER TABLE market_history ADD COLUMN {col} REAL")
            except Exception as e:
                logger.debug(f"Market history migration skipped for {col}: {e}")
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

def get_v_turn_state():
    with db_lock:
        conn = get_connection()
        try:
            df = pd.read_sql("SELECT * FROM v_turn_state WHERE id = 1", conn)
            return df.iloc[0] if not df.empty else None
        except Exception as e:
            logger.error(f"Failed to get v_turn_state: {e}")
            return None
        finally: conn.close()

def save_v_turn_state(is_confirmed, day1_date, day1_price, ftd_date):
    with db_lock:
        conn = get_connection()
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
    【V5 強化】計算 K 線的淨買盤壓力 (CLV 邏輯替代 $TICK)
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

    # 階段 1：短 DB 讀 (持鎖)
    with db_lock:
        conn = get_connection()
        try:
            last_date_df = pd.read_sql(
                "SELECT MAX(date) as last_date FROM market_history", conn)
            last_date_str = last_date_df['last_date'].iloc[0]
        finally:
            conn.close()

    period = "1y" if not last_date_str else "7d"

    # 階段 2：網路 I/O (不持鎖、不開 conn)
    tickers = {
        'SPX': '^GSPC', 'VIX': '^VIX', 'DXY': 'DX-Y.NYB',
        'TNX': '^TNX', 'GOLD': 'GC=F', 'SKEW': '^SKEW',
        'SOX': '^SOX', 'HYG': 'HYG', 'OIL': 'CL=F'
    }
    yf_dfs = []
    for name, ticker in tickers.items():
        try:
            hist = get_ticker(ticker).history(period=period)
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

    final_new_df = pd.merge(
        new_yf_df, sm_data, left_index=True, right_index=True, how='left'
    ).ffill()

    # 階段 3：短 DB 寫 (持鎖)
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            for date, row in final_new_df.iterrows():
                cursor.execute("""
                    INSERT OR REPLACE INTO market_history
                    (date, SPX, VIX, DXY, TNX, GOLD, SKEW, SOX, HYG, OIL, dix, gex)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (date, row.get('SPX'), row.get('VIX'), row.get('DXY'),
                      row.get('TNX'), row.get('GOLD'), row.get('SKEW'),
                      row.get('SOX'), row.get('HYG'), row.get('OIL'),
                      row.get('dix'), row.get('gex')))
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


def _get_risk_free_rate() -> float:
    try:
        return float(get_ticker("^TNX").history(period="1d")['Close'].iloc[-1]) / 100.0
    except Exception as e:
        logger.debug(f"TNX fallback rate used: {e}")
        return 0.04


def _select_future_expirations(expirations, *, min_days: int = 3, max_count: int = 4) -> list[str]:
    selected = []
    today = datetime.now().date()
    for date_str in expirations or []:
        if len(selected) >= max_count:
            break
        try:
            expiry_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if (expiry_date - today).days < min_days:
            continue
        selected.append(date_str)
    return selected


def _interpolate_zero_cross(x1: float, y1: float, x2: float, y2: float) -> float:
    if y1 == y2:
        return (x1 + x2) / 2
    return x1 + ((0 - y1) * (x2 - x1) / (y2 - y1))


def get_spy_gex_levels(symbol: str = "SPY") -> Dict[str, Any]:
    """計算 SPY / ETF 的 GEX 總量、Gamma Flip proxy 與 Max Pain。"""
    try:
        ticker = get_ticker(symbol)
        spot_hist = ticker.history(period="1d")
        if spot_hist.empty:
            return {
                "total_gex_billions": None,
                "gamma_flip_level": None,
                "max_pain": None,
                "spot": None,
                "above_flip": None,
                "below_flip": None,
            }

        spot = float(spot_hist["Close"].iloc[-1])
        rate = _get_risk_free_rate()
        strike_gex = {}
        call_oi_by_strike = {}
        put_oi_by_strike = {}

        expirations = _select_future_expirations(getattr(ticker, "options", None), min_days=3, max_count=4)
        for exp in expirations:
            try:
                opt = ticker.option_chain(exp)
            except Exception as chain_exc:
                logger.debug(f"GEX chain fetch failed for {symbol} @ {exp}: {chain_exc}")
                continue

            time_to_expiry = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365.0
            if time_to_expiry <= 0:
                time_to_expiry = 0.001

            calls = getattr(opt, "calls", None)
            puts = getattr(opt, "puts", None)
            calls = calls if isinstance(calls, pd.DataFrame) else pd.DataFrame()
            puts = puts if isinstance(puts, pd.DataFrame) else pd.DataFrame()

            for frame, direction, oi_store in ((calls, 1, call_oi_by_strike), (puts, -1, put_oi_by_strike)):
                if frame.empty:
                    continue
                subset = frame[["strike", "impliedVolatility", "openInterest"]].dropna()
                subset = subset[(subset["strike"] > 0) & (subset["impliedVolatility"] > 0) & (subset["openInterest"] > 0)]
                for _, row in subset.iterrows():
                    strike = float(row["strike"])
                    open_interest = float(row["openInterest"])
                    gamma = calculate_gamma(spot, strike, time_to_expiry, rate, float(row["impliedVolatility"]))
                    exposure = open_interest * 100 * gamma * (spot ** 2) * 0.01
                    strike_gex[strike] = strike_gex.get(strike, 0.0) + (direction * exposure)
                    oi_store[strike] = oi_store.get(strike, 0.0) + open_interest

        total_gex = sum(strike_gex.values()) / 10 ** 9 if strike_gex else None
        flip_level = None
        if strike_gex:
            sorted_pairs = sorted(strike_gex.items())
            strikes = [pair[0] for pair in sorted_pairs]
            cumulative_gex = np.cumsum([pair[1] for pair in sorted_pairs])
            for idx in range(len(strikes) - 1):
                left_gex = cumulative_gex[idx]
                right_gex = cumulative_gex[idx + 1]
                if left_gex == 0:
                    flip_level = strikes[idx]
                    break
                if (left_gex < 0 < right_gex) or (left_gex > 0 > right_gex):
                    flip_level = _interpolate_zero_cross(strikes[idx], left_gex, strikes[idx + 1], right_gex)
                    break
            if flip_level is None and len(strikes) > 0 and np.any(np.abs(cumulative_gex) > 0):
                flip_level = strikes[int(np.argmin(np.abs(cumulative_gex)))]

        max_pain = None
        candidate_strikes = sorted(set(call_oi_by_strike) | set(put_oi_by_strike))
        if candidate_strikes:
            def _pain_at(settlement: float) -> float:
                call_pain = sum(max(0.0, settlement - strike) * oi * 100 for strike, oi in call_oi_by_strike.items())
                put_pain = sum(max(0.0, strike - settlement) * oi * 100 for strike, oi in put_oi_by_strike.items())
                return call_pain + put_pain

            max_pain = min(candidate_strikes, key=_pain_at)

        return {
            "total_gex_billions": round(float(total_gex), 3) if total_gex is not None else None,
            "gamma_flip_level": round(float(flip_level), 2) if flip_level is not None else None,
            "max_pain": round(float(max_pain), 2) if max_pain is not None else None,
            "spot": round(float(spot), 2),
            "above_flip": bool(flip_level is not None and spot > flip_level) if flip_level is not None else None,
            "below_flip": bool(flip_level is not None and spot < flip_level) if flip_level is not None else None,
        }
    except Exception as e:
        logger.error(f"Real-time GEX profile calculation failed for {symbol}: {e}")
        return {
            "total_gex_billions": None,
            "gamma_flip_level": None,
            "max_pain": None,
            "spot": None,
            "above_flip": None,
            "below_flip": None,
        }


def get_realtime_spy_gex():
    """相容舊介面：回傳 SPY GEX 總量 (Billions)。"""
    return get_spy_gex_levels().get("total_gex_billions")

def get_market_sentiment_score():
    """整合新聞情緒分析 (LLM 優先，關鍵字備援)"""
    try:
        news = get_ticker("SPY").news[:10]
        if not news: return 0.0, "無數據"
        
        titles = [(item.get('title') or "") for item in news]
        all_titles = "\n".join([f"- {t}" for t in titles])

        # 優先嘗試 LLM (透過統一管理器)
        try:
            from src.llm import quick_call, LIGHT_MODELS

            prompt = f"""請分析以下美股新聞標題的綜合市場情緒：
{all_titles}
請僅回傳一個浮點數，範圍從 -1.0 (極度悲觀/利空) 到 1.0 (極度樂觀/利多)。不要回傳任何其他文字。"""
            result = quick_call(prompt, models=LIGHT_MODELS)
            if result:
                normalized_score = float(result.strip())
                normalized_score = max(-1.0, min(1.0, normalized_score))
                summary = "偏多" if normalized_score > 0.2 else "偏空" if normalized_score < -0.2 else "中性"
                return normalized_score, summary
        except Exception as e:
            logger.warning(f"LLM sentiment analysis failed, falling back to keywords: {e}")

        # 備援：關鍵字計分
        bear_keywords = ['drop', 'fall', 'recession', 'lower', 'fear', 'warn', 'weak', 'risk', 'inflation', 'sell', 'plunge', 'crisis']
        bull_keywords = ['rise', 'rally', 'growth', 'strong', 'gain', 'support', 'buy', 'optimism', 'beat', 'surge', 'rebound']
        score = 0
        for title in titles:
            t_lower = title.lower()
            for w in bear_keywords:
                if w in t_lower: score -= 1
            for w in bull_keywords:
                if w in t_lower: score += 1
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
        with db_lock:
            conn = get_connection()
            try:
                df = pd.read_sql("SELECT * FROM market_history ORDER BY date ASC", conn)
            finally:
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

_risk_cache = {"report": "", "snapshot": None, "timestamp": 0, "expiry": 1200}
_risk_cache_lock = threading.Lock()

DIX_SUPPORT_THRESHOLD = 0.85
DIX_SUPPORT_OFFSET_POINTS = 12
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLC', 'XLB', 'XLRE']
BREADTH_RISK_THRESHOLD = 30.0
V_TURN_BREADTH_SAFE_THRESHOLD = 40.0

def _safe_float(value, digits: int = 2):
    try:
        if value is None:
            return None
        val = float(value)
        if not math.isfinite(val):
            return None
        return round(val, digits)
    except (TypeError, ValueError):
        return None


def _extract_close_series_from_download(payload, symbol: str) -> pd.Series | None:
    if isinstance(payload, dict):
        frame = payload.get(symbol)
        if isinstance(frame, pd.DataFrame) and "Close" in frame.columns:
            series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            return series if not series.empty else None
        return None

    if not isinstance(payload, pd.DataFrame) or payload.empty:
        return None

    if isinstance(payload.columns, pd.MultiIndex):
        if symbol not in payload.columns.levels[0]:
            return None
        frame = payload[symbol]
        if "Close" not in frame.columns:
            return None
        series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        return series if not series.empty else None

    if "Close" in payload.columns:
        series = pd.to_numeric(payload["Close"], errors="coerce").dropna()
        return series if not series.empty else None
    return None


def get_market_breadth() -> Dict[str, Any]:
    """以 11 個 sector ETF 衡量市場廣度。"""
    try:
        data = get_download(SECTOR_ETFS, period="1y", group_by='ticker', progress=False)
    except Exception as e:
        logger.warning(f"Market breadth fetch failed: {e}")
        return {"pct_above_200ma": None, "pct_above_50ma": None, "total_sectors": 0, "breadth_signal": "unknown"}

    above_200ma = 0
    above_50ma = 0
    total = 0

    for etf in SECTOR_ETFS:
        close = _extract_close_series_from_download(data, etf)
        if close is None or len(close) < 200:
            continue
        ma200 = close.rolling(200).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        if pd.notna(ma200) and current > ma200:
            above_200ma += 1
        if pd.notna(ma50) and current > ma50:
            above_50ma += 1
        total += 1

    if total == 0:
        return {"pct_above_200ma": None, "pct_above_50ma": None, "total_sectors": 0, "breadth_signal": "unknown"}

    pct_above_200ma = above_200ma / total * 100
    pct_above_50ma = above_50ma / total * 100
    breadth_signal = "healthy" if pct_above_200ma > 70 else "deteriorating" if pct_above_200ma > 40 else "weak"
    return {
        "pct_above_200ma": round(pct_above_200ma, 1),
        "pct_above_50ma": round(pct_above_50ma, 1),
        "total_sectors": total,
        "breadth_signal": breadth_signal,
    }


def get_rolling_correlations(window: int = 60) -> Dict[str, Any]:
    """追蹤美股與債券 / 黃金 / 美元的 60 日滾動相關性。"""
    symbols = ("SPY", "TLT", "GLD", "DX-Y.NYB")
    try:
        data = get_download(list(symbols), period="6mo", group_by='ticker', progress=False)
    except Exception as e:
        logger.warning(f"Rolling correlation fetch failed: {e}")
        return {}

    returns = {}
    for sym in symbols:
        close = _extract_close_series_from_download(data, sym)
        if close is None:
            continue
        pct = close.pct_change().dropna()
        if not pct.empty:
            returns[sym] = pct

    spy_returns = returns.get("SPY")
    if spy_returns is None:
        return {}

    correlations = {}
    mapping = {
        "TLT": "spyTltCorr60d",
        "GLD": "spyGldCorr60d",
        "DX-Y.NYB": "spyDxyCorr60d",
    }
    for sym, key in mapping.items():
        asset_returns = returns.get(sym)
        if asset_returns is None:
            continue
        aligned = pd.concat([spy_returns, asset_returns], axis=1).dropna()
        if len(aligned) < window:
            continue
        corr_value = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1]).iloc[-1]
        if pd.notna(corr_value):
            correlations[key] = round(float(corr_value), 3)
    return correlations

def _build_global_risk_summary(score: int, state: str, reasons) -> str:
    if score >= 75:
        lead = "系統風險進入高警戒，先以防守和流動性管理為優先。"
    elif score >= 45:
        lead = "市場進入警戒帶，偏向控槓桿、降追價、等確認。"
    elif score >= 30:
        lead = "市場處於整理盤，適合等待更清楚的方向再擴大部位。"
    else:
        lead = "市場仍維持偏多結構，但要留意短線波動升溫。"
    top_reasons = "；".join(reasons[:3]) if reasons else "目前主要風險指標穩定。"
    return f"{lead} 當前 regime：{state}，風險分數 {score}。核心觀察：{top_reasons}"


def _score_risk_multiplier(risk_multiplier: float, *, dix_support_active: bool = False):
    if risk_multiplier <= 1.0:
        gross_score = 0
    else:
        raw_score = (math.log(risk_multiplier) / math.log(3.0)) * 100
        gross_score = int(raw_score)

    dix_offset = -DIX_SUPPORT_OFFSET_POINTS if dix_support_active and gross_score > 0 else 0
    score = max(0, min(100, gross_score + dix_offset))
    return gross_score, score, dix_offset


def _get_spx_trend_snapshot():
    try:
        spx_df = get_ticker("^GSPC", cache_level="daily").history(period="6mo")
        if spx_df is None or spx_df.empty:
            return None, "unknown"

        calc = IndicatorCalculator()
        adx_payload = calc.ADX(
            spx_df["High"].astype(float).values,
            spx_df["Low"].astype(float).values,
            spx_df["Close"].astype(float).values,
        )
        adx_series = pd.Series(adx_payload["adx"]).dropna()
        adx_value = float(adx_series.iloc[-1]) if not adx_series.empty else None
        return adx_value, adx_payload.get("trend_regime", "unknown")
    except Exception as e:
        logger.debug(f"SPX ADX snapshot failed: {e}")
        return None, "unknown"


def _select_ma_break_weight(base_weight: float, adx_value: float | None) -> float:
    if adx_value is None or adx_value > 25:
        return base_weight
    if base_weight >= 1.35:
        return 1.1
    if base_weight >= 1.2:
        return 1.08
    return 1.05

def _build_global_risk_snapshot() -> Dict[str, Any]:
    df = fetch_all_market_data()
    if df.empty:
        raise RuntimeError("雷達掃描失敗。")

    latest = df.iloc[-1]
    macro = MacroEngine().get_macro_dashboard()
    breadth = get_market_breadth()
    gex_profile = get_spy_gex_levels()
    final_gex = gex_profile.get("total_gex_billions")
    if final_gex is None:
        final_gex = latest.get('gex', 0) / 10**9
    sent_score, sent_label = get_market_sentiment_score()
    spx_adx, spx_trend_regime = _get_spx_trend_snapshot()
    try:
        spy_vol_context = market.build_option_volatility_context("SPY")
    except Exception as e:
        logger.debug(f"SPY volatility context failed: {e}")
        spy_vol_context = {}
    rolling_corrs = get_rolling_correlations()

    risk_multiplier = 1.0
    reasons = []

    yc = macro.get("Yield_Curve_10Y2Y")
    if yc is not None and yc < 0:
        risk_multiplier *= 1.2
        reasons.append(f"⚠️ 殖利率曲線倒掛 ({yc:.2f}) - 衰退隱憂")

    ffr = macro.get("Fed_Funds_Rate")
    if ffr is not None and ffr > 5.0:
        risk_multiplier *= 1.1
        reasons.append(f"🏦 高利率環境 ({ffr:.2f}%) - 估值壓力")

    if latest.get('DXY_Z', 0) > 1.5 or latest.get('TNX_Z', 0) > 1.5:
        risk_multiplier *= 1.5
        reasons.append("🔴 資金緊縮 (美元/美債突波)")

    if latest.get('VIX_Z', 0) > 2.0 or final_gex < 0:
        risk_multiplier *= 1.6
        reasons.append(f"🔴 波動率失控 / 負 Gamma ({final_gex:.2f}B)")

    if latest.get('SKEW_PR', 0) > 0.90:
        risk_multiplier *= 1.3
        reasons.append("🟠 尾部風險升溫")

    breadth_200 = breadth.get("pct_above_200ma")
    breadth_50 = breadth.get("pct_above_50ma")
    if breadth_200 is not None and breadth_200 < BREADTH_RISK_THRESHOLD:
        risk_multiplier *= 1.15
        reasons.append(f"🔻 Sector breadth 偏弱 (200MA 上方僅 {breadth_200:.1f}%)")

    dix_support_active = latest.get('dix_PR', 0) > DIX_SUPPORT_THRESHOLD
    if dix_support_active:
        reasons.append("🟢 暗池吸籌，大戶提供下檔支撐")

    if sent_score < -0.4:
        risk_multiplier *= 1.2
        reasons.append(f"📰 新聞極度偏空 ({sent_label})")

    gamma_flip_level = gex_profile.get("gamma_flip_level")
    below_gamma_flip = gex_profile.get("below_flip")
    if below_gamma_flip and final_gex is not None and final_gex >= 0:
        risk_multiplier *= 1.1
        reasons.append(f"⚠️ SPY 位於 Gamma Flip 下方 ({gamma_flip_level:.2f})")

    spx = latest.get('SPX', 0)
    ma10 = latest.get('SPX_10MA', 0)
    ma20 = latest.get('SPX_20MA', 0)
    ma200 = latest.get('SPX_200MA', 0)

    if ma200 > 0 and spx < ma200:
        ma_weight = _select_ma_break_weight(1.4, spx_adx)
        risk_multiplier *= ma_weight
        suffix = f" (ADX {spx_adx:.1f}，震盪盤降權)" if spx_adx is not None and ma_weight < 1.4 else ""
        reasons.append(f"🚨 [Trigger] 熊市區間：跌破 200MA 均線！{suffix}")
    elif spx < ma20:
        ma_weight = _select_ma_break_weight(1.25, spx_adx)
        risk_multiplier *= ma_weight
        suffix = f" (ADX {spx_adx:.1f}，震盪盤降權)" if spx_adx is not None and ma_weight < 1.25 else ""
        reasons.append(f"🚨 [Trigger] 趨勢破滅：跌破月線！{suffix}")
    elif spx < ma10:
        ma_weight = _select_ma_break_weight(1.15, spx_adx)
        risk_multiplier *= ma_weight
        suffix = f" (ADX {spx_adx:.1f}，震盪盤降權)" if spx_adx is not None and ma_weight < 1.15 else ""
        reasons.append(f"🚨 [Trigger] 短期轉弱：跌破 10MA。{suffix}")

    gross_score, score, dix_offset = _score_risk_multiplier(
        risk_multiplier,
        dix_support_active=dix_support_active,
    )

    state = "🟢 多頭" if score < 30 else "🟡 整理" if score < 45 else "🔴 警戒" if score < 75 else "💀 系統風險"
    reasons = reasons or ["🟢 指標目前健康"]

    return {
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "grossRiskScore": gross_score,
        "riskScore": score,
        "state": state,
        "riskMultiplier": round(risk_multiplier, 4),
        "scoreAdjustments": {"dixSupport": dix_offset},
        "summary": _build_global_risk_summary(score, state, reasons),
        "reasons": reasons,
        "signals": {
            "yieldCurve10Y2Y": _safe_float(yc, 3),
            "fedFundsRate": _safe_float(ffr, 2),
            "dixPr": _safe_float(latest.get('dix_PR', 0), 2),
            "dixSupportActive": dix_support_active,
            "dixSupportOffset": dix_offset,
            "gexBillions": _safe_float(final_gex, 2),
            "spySpot": _safe_float(gex_profile.get("spot"), 2),
            "spyGammaFlipLevel": _safe_float(gamma_flip_level, 2),
            "spyMaxPain": _safe_float(gex_profile.get("max_pain"), 2),
            "spyBelowGammaFlip": below_gamma_flip,
            "sentimentScore": _safe_float(sent_score, 2),
            "sentimentLabel": sent_label,
            "spx": _safe_float(spx, 1),
            "spx10Ma": _safe_float(ma10, 1),
            "spx20Ma": _safe_float(ma20, 1),
            "spx200Ma": _safe_float(ma200, 1),
            "spxAdx": _safe_float(spx_adx, 2),
            "spxTrendRegime": spx_trend_regime,
            "sectorBreadth50": _safe_float(breadth_50, 1),
            "sectorBreadth200": _safe_float(breadth_200, 1),
            "sectorBreadthState": breadth.get("breadth_signal"),
            "spyCurrentIv": _safe_float(spy_vol_context.get("current_iv"), 1),
            "spyRealizedVol30d": _safe_float(spy_vol_context.get("realized_vol_30d"), 1),
            "spyVrp": _safe_float(spy_vol_context.get("vrp"), 1),
            "spyIvVsRvPercentile": _safe_float(spy_vol_context.get("iv_vs_rv_percentile"), 1),
            "spyVolSignal": spy_vol_context.get("signal"),
            "spyTltCorr60d": rolling_corrs.get("spyTltCorr60d"),
            "spyGldCorr60d": rolling_corrs.get("spyGldCorr60d"),
            "spyDxyCorr60d": rolling_corrs.get("spyDxyCorr60d"),
            "dxyZ": _safe_float(latest.get('DXY_Z', 0), 2),
            "tnxZ": _safe_float(latest.get('TNX_Z', 0), 2),
            "vixZ": _safe_float(latest.get('VIX_Z', 0), 2),
            "skewPr": _safe_float(latest.get('SKEW_PR', 0), 2)
        }
    }

def format_global_risk_snapshot(snapshot: Dict[str, Any]) -> str:
    signals = snapshot.get("signals", {})
    msg = f"📊 *【MarginCall_2X 全局雷達 (含宏觀)】*\n🔥 風險分數：{snapshot.get('riskScore', 'N/A')} ({snapshot.get('state', '未初始化')})\n"
    msg += "\n".join(snapshot.get("reasons", [])) if snapshot.get("reasons") else "🟢 指標目前健康"
    msg += f"\n\n- Yield Curve: {signals.get('yieldCurve10Y2Y', 'N/A') if signals.get('yieldCurve10Y2Y') is not None else 'N/A'}"
    msg += f"\n- Fed Funds: {signals.get('fedFundsRate', 'N/A') if signals.get('fedFundsRate') is not None else 'N/A'}%"
    msg += f"\n- DIX_PR: {signals.get('dixPr', 'N/A') if signals.get('dixPr') is not None else 'N/A'}"
    if signals.get("dixSupportOffset"):
        msg += f" | DIX 抵扣: {signals['dixSupportOffset']}"
    msg += f" | GEX: {signals.get('gexBillions', 'N/A') if signals.get('gexBillions') is not None else 'N/A'}B"
    msg += f"\n- Sentiment: {signals.get('sentimentLabel', 'N/A')}({signals.get('sentimentScore', 'N/A') if signals.get('sentimentScore') is not None else 'N/A'})"
    msg += f"\n- SPX: {signals.get('spx', 'N/A') if signals.get('spx') is not None else 'N/A'} (MA20:{signals.get('spx20Ma', 'N/A') if signals.get('spx20Ma') is not None else 'N/A'}, MA200:{signals.get('spx200Ma', 'N/A') if signals.get('spx200Ma') is not None else 'N/A'})"
    if signals.get("sectorBreadth200") is not None:
        msg += (
            f"\n- Breadth: 50MA {signals.get('sectorBreadth50', 'N/A')}% | "
            f"200MA {signals['sectorBreadth200']}% ({signals.get('sectorBreadthState', 'unknown')})"
        )
    if signals.get("spyGammaFlipLevel") is not None or signals.get("spyMaxPain") is not None:
        msg += (
            f"\n- Gamma Levels: Spot {signals.get('spySpot', 'N/A')} | "
            f"Flip {signals.get('spyGammaFlipLevel', 'N/A')} | Max Pain {signals.get('spyMaxPain', 'N/A')}"
        )
        if signals.get("spyBelowGammaFlip") is not None:
            msg += " | 低於 Flip" if signals["spyBelowGammaFlip"] else " | 高於 Flip"
    if signals.get("spyVrp") is not None:
        msg += (
            f"\n- SPY Vol Context: IV {signals.get('spyCurrentIv', 'N/A')}% | "
            f"RV30 {signals.get('spyRealizedVol30d', 'N/A')}% | VRP {signals['spyVrp']}pt"
        )
        if signals.get("spyVolSignal"):
            msg += f" ({signals['spyVolSignal']})"
    corr_parts = []
    if signals.get("spyTltCorr60d") is not None:
        corr_parts.append(f"SPY/TLT {signals['spyTltCorr60d']}")
    if signals.get("spyGldCorr60d") is not None:
        corr_parts.append(f"SPY/GLD {signals['spyGldCorr60d']}")
    if signals.get("spyDxyCorr60d") is not None:
        corr_parts.append(f"SPY/DXY {signals['spyDxyCorr60d']}")
    if corr_parts:
        msg += "\n- Corr60: " + " | ".join(corr_parts)
    return msg

def get_global_risk_snapshot(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Returns the structured risk radar snapshot so other modules can persist and
    reason about the current macro regime without parsing markdown output.
    """
    global _risk_cache
    current_time = time.time()
    if not force_refresh:
        with _risk_cache_lock:
            if (
                _risk_cache["snapshot"] is not None
                and (current_time - _risk_cache["timestamp"] < _risk_cache["expiry"])
            ):
                return {**_risk_cache["snapshot"], "cached": True}

    try:
        snapshot = _build_global_risk_snapshot()
        report = format_global_risk_snapshot(snapshot)
        with _risk_cache_lock:
            _risk_cache["snapshot"] = snapshot
            _risk_cache["report"] = report
            _risk_cache["timestamp"] = time.time()
        return {**snapshot, "cached": False}
    except Exception as e:
        logger.error(f"Risk snapshot analysis failed: {e}")
        raise

@tool()
def get_global_risk_radar(force_refresh: bool = False) -> str:
    """
    Analyzes systemic risk by aggregating macro indicators (Yield Curve, Fed Funds),
    market technicals (SPX MA20/MA200), and volatility metrics (VIX, GEX, DIX).
    Returns a risk score (0-100) and strategic summary.
    """
    try:
        snapshot = get_global_risk_snapshot(force_refresh=force_refresh)
        with _risk_cache_lock:
            report = _risk_cache["report"]
        report = report or format_global_risk_snapshot(snapshot)
        return report + ("\n(⚡ DB-Cached)" if snapshot.get("cached") else "")
    except Exception as e:
        logger.error(f"Risk radar analysis failed: {e}")
        return format_tool_error(f"❌ 雷達異常: {e}", transient=True)

def build_v_turn_report() -> str:
    """Pure V-turn logic for direct callers and tests."""
    try:
        init_market_db()

        # === 階段一：網路 I/O ( 不持鎖 ) ===
        symbols = ["SPLG", "RSP", "HYG", "LQD", "CL=F"]
        hist_data = get_download(symbols, period="60d", group_by='ticker', progress=False)

        splg = hist_data['SPLG'].dropna()
        rsp = hist_data['RSP'].dropna()
        hyg = hist_data['HYG'].dropna()
        lqd = hist_data['LQD'].dropna()
        oil = hist_data['CL=F'].dropna()

        vix_df = get_ticker("^VIX").history(period="2d", interval="15m")
        vix3m_df = get_ticker("^VIX3M").history(period="2d", interval="15m")
        vvix_df = get_ticker("^VVIX").history(period="2d", interval="15m")
        spy_5m = get_ticker("SPY").history(period="2d", interval="5m")
        breadth_snapshot = get_market_breadth()

        if splg.empty or rsp.empty:
            return "❌ yfinance 數據下載失敗，請檢查網路連線。"

        # === 階段二：DB 讀取 + 計算 + DB 寫入 ( 持鎖，極短 ) ===
        window = splg.tail(25)
        current_low_idx = window['Close'].idxmin()
        current_low_price = float(window.loc[current_low_idx, 'Close'])
        current_low_date = current_low_idx.strftime('%Y-%m-%d')

        with db_lock:
            conn = get_connection()
            try:
                cursor = conn.cursor()
                state_df = pd.read_sql("SELECT * FROM v_turn_state WHERE id = 1", conn)
                state = state_df.iloc[0] if not state_df.empty else None

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

                cursor.execute("""
                    INSERT OR REPLACE INTO v_turn_state (id, is_confirmed, day1_date, day1_price, ftd_date, last_check_date)
                    VALUES (1, ?, ?, ?, ?, ?)
                """, (is_confirmed, day1_date, day1_price, ftd_date, datetime.now().strftime('%Y-%m-%d')))
                conn.commit()
            finally:
                conn.close()

        # === 階段三：計算 ( 不持鎖；若階段一數據為空則 fallback 會做少量網路 I/O) ===
        rsp_5d = (rsp['Close'].iloc[-1] / rsp['Close'].iloc[-5]) - 1 if len(rsp) >= 5 else 0
        splg_5d = (splg['Close'].iloc[-1] / splg['Close'].iloc[-5]) - 1 if len(splg) >= 5 else 0
        breadth_val = rsp_5d - splg_5d
        proxy_breadth_safe = (breadth_val > -0.005)
        sector_breadth_50 = breadth_snapshot.get("pct_above_50ma")
        sector_breadth_safe = sector_breadth_50 is not None and sector_breadth_50 > V_TURN_BREADTH_SAFE_THRESHOLD
        breadth_safe = proxy_breadth_safe and (sector_breadth_safe if sector_breadth_50 is not None else True)

        vix_p = vix_df['Close'].iloc[-1] if not vix_df.empty else get_ticker("^VIX").history(period="5d")['Close'].iloc[-1]
        vix3m_p = vix3m_df['Close'].iloc[-1] if not vix3m_df.empty else get_ticker("^VIX3M").history(period="5d")['Close'].iloc[-1]
        vix_term = vix_p / vix3m_p
        vix_term_safe = (vix_term < 0.93)

        bp_ratio = calculate_buying_pressure(spy_5m, window=5)
        tick_safe = (bp_ratio > 0.15)
        tick_emoji = '🔥' if bp_ratio > 0.3 else '🟢' if tick_safe else '⚪'
        tick_msg = f"{bp_ratio:+.1%}"

        vvix_val = vvix_df['Close'].iloc[-1] if not vvix_df.empty else get_ticker("^VVIX").history(period="5d")['Close'].iloc[-1]
        vvix_safe = (vvix_val < 110)
        credit_ratio = (hyg['Close'] / lqd['Close']).iloc[-1]
        credit_ma = (hyg['Close'] / lqd['Close']).rolling(20).mean().iloc[-1]
        credit_safe = (credit_ratio > credit_ma)
        ma20 = splg['Close'].rolling(20).mean().iloc[-1]
        ma20_safe = (splg['Close'].iloc[-1] > ma20)

        all_macro_safe = (vix_term_safe and vvix_safe and credit_safe and breadth_safe)

        status_txt = "🔵 偵測底盤中" if is_confirmed == 0 else "🚀 強勢反彈中"
        report = f"📊 *【MarginCall_2X V 轉戰報 V5】*\n當前狀態: {status_txt}\n"
        report += f"- Day 1 低點: {day1_price:.2f} ({day1_date})\n"
        report += f"- 目前進度: Day {day_count}\n"
        if is_confirmed: report += f"- ✅ FTD 點火日: {ftd_date}\n"

        report += f"\n🪬 *核心護法狀態 (CLV 強化):*\n"
        report += f"- VIX 期限結構: {vix_term:.2f} {'🟢' if vix_term_safe else '🔴'}\n"
        report += f"- VVIX 恐慌速率: {vvix_val:.1f} {'🟢' if vvix_safe else '🔴'}\n"
        report += f"- 信用市場(HYG/LQD): {'🟢' if credit_safe else '🔴'}\n"
        report += f"- 市場寬度(RSP/SPLG): {breadth_val:+.2%} {'🟢' if breadth_safe else '🔴'}\n"
        if sector_breadth_50 is not None:
            report += f"- Sector Breadth(50MA): {sector_breadth_50:.1f}% {'🟢' if sector_breadth_safe else '🔴'}\n"
        report += f"- K線推力(CLV): {tick_msg} {tick_emoji}\n"
        report += f"- MA20 技術位階: {'🟢' if ma20_safe else '🔴'}\n"
        
        if is_confirmed and all_macro_safe and ma20_safe:
            report += "\n🏁 *【最終判定：發射訊號！】*\n👉 機構確認進場，CLV 推力強勁。建議分批建倉。"
        else:
            report += "\n🏁 *【最終判定：維持現狀】*\n👉 市場尚未出現轉強信號或條件未齊。"
        return report
    except Exception as e:
        logger.error(f"V-turn confirmation failed: {e}")
        return format_tool_error(f"❌ V 轉監測失敗: {e}", transient=True)

@tool()
def get_v_turn_confirmation() -> str:
    """
    Monitors market bottoming signals and "Follow-Through Day" (FTD) events.
    Combines price action with internal breadth, VIX term structure, and credit spreads.
    """
    return build_v_turn_report()

def build_capital_flow_report() -> str:
    """Pure capital-flow logic for direct callers and tests."""
    try:
        symbols = ['^SOX', 'XLU', 'HG=F', 'GC=F', '^TNX', 'TLT', 'DX-Y.NYB', 'TWD=X', 'JPY=X', '^VIX']
        hist_data = get_download(symbols, period="1mo", group_by='ticker', progress=False)
        
        def get_chg_5d(ticker_data):
            if ticker_data is None or ticker_data.empty: return None
            df = ticker_data.dropna()
            if len(df) < 5: 
                # Fallback to max available
                return (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100 if not df.empty else 0
            return (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
            
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

        sox_chg = get_chg_5d(hist_data.get('^SOX'))
        xlu_chg = get_chg_5d(hist_data.get('XLU'))
        hg_chg = get_chg_5d(hist_data.get('HG=F'))
        gc_chg = get_chg_5d(hist_data.get('GC=F'))
        tnx_chg = get_chg_5d(hist_data.get('^TNX'))
        tlt_chg = get_chg_5d(hist_data.get('TLT'))
        jpy_chg = get_chg_5d(hist_data.get('JPY=X')) # JPY=X 是 USD/JPY
        
        tlt_vol = get_vol_ratio(hist_data.get('TLT'))
        xlu_vol = get_vol_ratio(hist_data.get('XLU'))
        
        report = "🧠 *【Capital Flow Matrix 資金流向矩陣】*\n"
        
        # 1. 景氣與板塊輪動
        if sox_chg is not None and xlu_chg is not None:
            tech_def_spread = sox_chg - xlu_chg
            if tech_def_spread < -2.5 and xlu_vol > 1.2:
                report += f"🔄 **板塊輪動 (Risk-Off):** 資金從科技股(SOX 5D {sox_chg:+.2f}%) 撤退，防禦性公用事業(XLU 5D {xlu_chg:+.2f}%) 放量({xlu_vol:.1f}x)承接。\n"
            elif tech_def_spread > 2.5:
                report += f"🔥 **風險偏好 (Risk-On):** 資金集中攻擊科技股 (SOX 5D {sox_chg:+.2f}%)，公用事業跑輸大盤。\n"
            else:
                report += f"⚖️ **板塊狀態中性:** SOX({sox_chg:+.2f}%) vs XLU({xlu_chg:+.2f}%) 5日輪動不明顯。\n"
                
        # 2. 實體景氣 (銅金比)
        if hg_chg is not None and gc_chg is not None:
            cg_spread = hg_chg - gc_chg
            if cg_spread < -2.0:
                report += f"📉 **衰退疑慮 (銅金比轉弱):** 銅博士(5D {hg_chg:+.2f}%)走弱，黃金(5D {gc_chg:+.2f}%)避險升溫，實體經濟預期放緩。\n"
            elif cg_spread > 2.0:
                report += f"🏭 **復甦預期 (銅金比轉強):** 銅價(5D {hg_chg:+.2f}%)跑贏黃金，工業/實體需求強勁。\n"
                
        # 3. 匯率與套利平倉 (Carry Trade)
        if jpy_chg is not None:
            if jpy_chg < -1.5: # 日圓 5 日急升
                report += f"🚨 **套利平倉警戒 (Carry Trade Unwind):** 日圓週漲幅({-jpy_chg:+.2f}%)顯著，高度留意全球流動性收緊。\n"
            elif jpy_chg > 1.5:
                report += f"💸 **套利資金寬鬆:** 日圓週貶值({jpy_chg:+.2f}%)，有利於全球風險資產的槓桿資金池。\n"

        # 4. 長債避風港
        if tnx_chg is not None and tlt_chg is not None:
            if tnx_chg > 5.0:
                report += f"🎈 **估值重力壓迫:** 10年期美債殖利率週飆升({tnx_chg:+.2f}%)，將對科技股估值造成壓力。\n"
            elif tlt_chg > 2.0 and tlt_vol > 1.3:
                report += f"🛡️ **終極避風港進駐:** 20年期美債(TLT) 週放量上漲({tlt_chg:+.2f}%, 量:{tlt_vol:.1f}x)，大資金正在尋求絕對避險。\n"
                
        if report == "🧠 *【Capital Flow Matrix 資金流向矩陣】*\n":
            return report + "目前無明顯異常資金流向信號。\n"
            
        return report
    except Exception as e:
        logger.error(f"Capital Flow Matrix calculation failed: {e}")
        return format_tool_error(f"❌ 資金流向矩陣計算失敗: {e}", transient=True) + "\n"

@tool()
def get_capital_flow_matrix() -> str:
    """
    Calculates ratios and volume dynamics between different sectors and asset classes.
    Identifies capital migration between tech, utilities, bonds, and currencies.
    """
    return build_capital_flow_report()

if __name__ == "__main__":
    print("🚀 === MarginCall_2X 引擎自檢測試 ===")
    
    # 1. 測試宏觀引擎
    print("\n[1] 正在抓取 FRED 宏觀數據...")
    macro_eng = MacroEngine()
    dashboard = macro_eng.get_macro_dashboard()
    for k, v in dashboard.items():
        print(f"  - {k}: {v}")

    # 2. 測試資金流向
    print("\n[2] 正在計算資金流向矩陣...")
    print(get_capital_flow_matrix())

    # 3. 測試全局雷達 (含整合分數)
    print("\n[3] 正在生成全局風險雷達...")
    print(get_global_risk_radar())

    # 4. 測試 V 轉監測
    print("\n[4] 正在執行 V 轉確認偵測...")
    print(get_v_turn_confirmation())
