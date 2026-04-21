import os
import re
import datetime
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import time
import pandas as pd
import numpy as np
import yfinance as yf
from yf_session import get_ticker, get_download

import logging
from engine_technical import IndicatorCalculator, analyze_obv_signal, summarize_divergence
from config import WATCH_LIST
from src.database import db_lock, get_connection
from src.symbols import normalize_ticker
from src.tools import format_tool_error, tool

logger = logging.getLogger(__name__)

FMP_KEY = os.getenv("FMP_API_KEY")

_fubon_provider = None
_CANDIDATE_PANEL_MAX_WORKERS = 5
_SENTIMENT_BATCH_CACHE_TTL_SECONDS = 600
_SENTIMENT_BATCH_CACHE = {"entries": {}}
_SENTIMENT_BATCH_CACHE_LOCK = threading.Lock()
_SENTIMENT_INDICATORS = {
    "^TWII": "台股(加權)",
    "TSM": "台積ADR",
    "EWT": "台灣ETF",
    "^GSPC": "標普500(大盤)",
    "^IXIC": "那指(科技)",
    "^SOX": "費半(基石)",
    "^RUT": "羅素2000(水溫)",
    "^TNX": "美債10Y(重力)",
    "TLT": "20Y美債(避風港)",
    "DX-Y.NYB": "美元(水龍頭)",
    "TWD=X": "台幣(外資)",
    "JPY=X": "日圓(套利)",
    "^VIX": "恐慌(絞肉機)",
    "HYG": "高收債(風險)",
    "XLU": "公用事業(防禦)",
    "GC=F": "黃金(避險)",
    "CL=F": "原油(通膨)",
    "BZ=F": "布蘭特(地緣)",
    "HG=F": "銅(景氣)",
    "BTC-USD": "BTC",
}

_ASSET_PROFILE_CACHE_COLUMNS = {
    "name": "TEXT",
    "currency": "TEXT",
    "quote_type": "TEXT",
    "is_etf": "INTEGER DEFAULT 0",
    "category": "TEXT",
    "fund_family": "TEXT",
    "geo_focus": "TEXT",
    "strategy_type": "TEXT",
    "tracking_index": "TEXT",
    "concentration_bucket": "TEXT",
}

_TRACKING_INDEX_PATTERNS = (
    (re.compile(r"S&P\s*500\s+([A-Za-z& ]+?)\s+Sector", re.IGNORECASE), lambda m: f"S&P 500 {m.group(1).strip()} Sector"),
    (re.compile(r"NASDAQ-?100", re.IGNORECASE), lambda m: "NASDAQ-100"),
    (re.compile(r"S&P\s*500", re.IGNORECASE), lambda m: "S&P 500"),
    (re.compile(r"MSCI\s+[A-Za-z0-9& \-]+", re.IGNORECASE), lambda m: m.group(0).strip()),
    (re.compile(r"FTSE\s+[A-Za-z0-9& \-]+", re.IGNORECASE), lambda m: m.group(0).strip()),
    (re.compile(r"Russell\s+\d{3,4}", re.IGNORECASE), lambda m: m.group(0).strip()),
    (re.compile(r"Dow Jones\s+[A-Za-z0-9& \-]+", re.IGNORECASE), lambda m: m.group(0).strip()),
    (re.compile(r"Morningstar\s+[A-Za-z0-9& \-]+", re.IGNORECASE), lambda m: m.group(0).strip()),
    (re.compile(r"Technology Select Sector", re.IGNORECASE), lambda m: "Technology Select Sector"),
)

_LOCAL_ETF_PROFILE_REGISTRY = {
    "00997A": {
        "name": "群益美國增長主動式ETF",
        "quote_type": "ETF",
        "is_etf": True,
        "currency": "TWD",
        "fund_family": "群益投信",
        "geo_focus": "US",
        "strategy_type": "Active US Growth ETF",
        "tracking_index": "Active ETF Strategy",
        "category": "US Large-Cap Growth",
        "sector": "Unknown",
        "industry": "Active ETF",
        "asset_type": "Tech_Momentum",
        "concentration_bucket": "Technology",
    },
    "00987A": {
        "name": "主動台新台灣優勢成長ETF",
        "quote_type": "ETF",
        "is_etf": True,
        "currency": "TWD",
        "fund_family": "台新投信",
        "geo_focus": "Taiwan",
        "strategy_type": "Active Taiwan Advantage Growth ETF",
        "tracking_index": "Active ETF Strategy",
        "category": "Taiwan Technology Growth",
        "sector": "Unknown",
        "industry": "Active ETF",
        "asset_type": "Tech_Momentum",
        "concentration_bucket": "Technology",
    },
    "00981A": {
        "name": "主動統一台股增長ETF",
        "quote_type": "ETF",
        "is_etf": True,
        "currency": "TWD",
        "fund_family": "統一投信",
        "geo_focus": "Taiwan",
        "strategy_type": "Active Taiwan Growth ETF",
        "tracking_index": "Active ETF Strategy",
        "category": "Taiwan Technology Growth",
        "sector": "Unknown",
        "industry": "Active ETF",
        "asset_type": "Tech_Momentum",
        "concentration_bucket": "Technology",
    },
    "FBCG": {
        "name": "Fidelity Blue Chip Growth ETF",
        "quote_type": "ETF",
        "is_etf": True,
        "currency": "USD",
        "fund_family": "Fidelity Investments",
        "geo_focus": "US",
        "strategy_type": "Blue Chip Growth Catalyst Strategy",
        "tracking_index": "Active ETF Strategy",
        "category": "US Blue Chip Growth",
        "sector": "Unknown",
        "industry": "Active ETF",
        "asset_type": "Tech_Momentum",
        "concentration_bucket": "Technology",
    },
    "TCHP": {
        "name": "T. Rowe Price Blue Chip Growth ETF",
        "quote_type": "ETF",
        "is_etf": True,
        "currency": "USD",
        "fund_family": "T. Rowe Price",
        "geo_focus": "US",
        "strategy_type": "Quality Blue Chip Growth Strategy",
        "tracking_index": "Active ETF Strategy",
        "category": "US Quality Growth",
        "sector": "Unknown",
        "industry": "Active ETF",
        "asset_type": "Value_Holding",
        "concentration_bucket": "Diversified Equity",
    },
    "CGGR": {
        "name": "Capital Group Growth ETF",
        "quote_type": "ETF",
        "is_etf": True,
        "currency": "USD",
        "fund_family": "Capital Group",
        "geo_focus": "Global",
        "strategy_type": "Global Multi-Manager Growth Strategy",
        "tracking_index": "Active ETF Strategy",
        "category": "Global Growth",
        "sector": "Unknown",
        "industry": "Active ETF",
        "asset_type": "Value_Holding",
        "concentration_bucket": "Diversified Equity",
    },
}

def set_fubon_provider(provider):
    global _fubon_provider
    _fubon_provider = provider


def _has_fubon_provider() -> bool:
    return _fubon_provider is not None and getattr(_fubon_provider, "fubon_ready", False)


def _resolve_technical_interval(interval: str | None) -> tuple[str, str]:
    requested_interval = (interval or "1d").lower()
    tech_period_by_interval = {
        "1d": "6mo",
        "1wk": "3y",
        "1mo": "10y",
    }
    history_interval = requested_interval if requested_interval in tech_period_by_interval else "1d"
    return history_interval, tech_period_by_interval[history_interval]


def _latest_numeric(value, digits: int = 2):
    series = pd.Series(value).dropna() if isinstance(value, (np.ndarray, pd.Series, list, tuple)) else None
    if series is not None:
        if series.empty:
            return None
        value = series.iloc[-1]
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _normalize_lookup_symbol(symbol: str) -> str:
    symbol = normalize_ticker(symbol)
    s = symbol.upper()
    # 支援台股代碼：純數字或包含數字且長度 <= 6 (涵蓋權證如 00981A)
    is_taiwan_format = any(char.isdigit() for char in s) and (
        len(s.replace(".TW", "").replace(".TWO", "")) <= 6
    )
    if is_taiwan_format and not (s.endswith(".TW") or s.endswith(".TWO")):
        try:
            # 優先嘗試 .TW (上市)
            ticker = get_ticker(s + ".TW", cache_level="daily")
            info = ticker.fast_info
            if getattr(info, "last_price", None) is not None or getattr(info, "previous_close", None) is not None:
                s += ".TW"
            else:
                # 否則 fallback 到 .TWO (上櫃/權證)
                s += ".TWO"
        except Exception:
            s += ".TWO"
    return s


def _safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        result = float(value)
        if not np.isfinite(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _fetch_daily_close_series(symbol: str, period: str = "2y") -> pd.Series:
    ticker = get_ticker(symbol, cache_level="daily")
    history = ticker.history(period=period, interval="1d")
    if history.empty or "Close" not in history.columns:
        return pd.Series(dtype=float)
    close = (
        pd.to_numeric(history["Close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    close.index = pd.to_datetime(close.index)
    return close.astype(float)


def _estimate_half_life(series: pd.Series) -> float | None:
    clean = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 10:
        return None
    lagged = clean.shift(1).dropna()
    current = clean.loc[lagged.index]
    if lagged.empty:
        return None
    x = lagged.to_numpy(dtype=float)
    y = current.to_numpy(dtype=float)
    x_var = float(np.var(x))
    if x_var <= 0:
        return None
    beta = float(np.mean((x - x.mean()) * (y - y.mean())) / x_var)
    if 0 < beta < 1:
        return float(-np.log(2) / np.log(beta))
    return None


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_volume = df['Volume'].cumsum().replace(0, np.nan)
    return (typical * df['Volume']).cumsum() / cumulative_volume


def _classify_dual_anchor_state(last_price: float, vwap: float | None, poc_price: float) -> str:
    if vwap is None or pd.isna(vwap) or vwap <= 0:
        return "N/A"
    above_vwap = last_price >= vwap
    above_poc = last_price >= poc_price
    if above_vwap and above_poc:
        return "🟢 多方完全控盤"
    if (not above_vwap) and above_poc:
        return "🟡 短線回調但量價仍有支撐"
    if above_vwap and (not above_poc):
        return "⚠️ 站上 VWAP 但仍受 POC 壓制"
    return "🔴 空方控盤"


def _select_option_expirations(expirations, *, min_days: int = 7, max_count: int = 4) -> list[str]:
    selected = []
    today = datetime.datetime.now().date()
    for date_str in expirations or []:
        if len(selected) >= max_count:
            break
        try:
            expiry_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            continue
        if (expiry_date - today).days < min_days:
            continue
        selected.append(date_str)
    return selected


def _extract_atm_iv_samples(option_df: pd.DataFrame | None, spot: float, contracts_per_side: int = 2) -> list[float]:
    if option_df is None or option_df.empty or spot <= 0:
        return []
    if "strike" not in option_df.columns or "impliedVolatility" not in option_df.columns:
        return []
    subset = option_df[["strike", "impliedVolatility"]].dropna()
    subset = subset[(subset["strike"] > 0) & (subset["impliedVolatility"] > 0)]
    if subset.empty:
        return []
    nearest = subset.iloc[(subset["strike"] - spot).abs().argsort()[:contracts_per_side]]
    return [float(iv) * 100 for iv in nearest["impliedVolatility"].tolist()]


def _scan_option_derivatives(ticker, symbol: str, spot: float) -> dict:
    derivatives = {
        "pc_ratio": None,
        "pc_ratio_report": "N/A",
        "current_iv": None,
        "current_iv_expiry": None,
    }
    total_calls = 0.0
    total_puts = 0.0

    expirations = _select_option_expirations(getattr(ticker, "options", None), min_days=7, max_count=4)
    for date_str in expirations:
        try:
            chain = ticker.option_chain(date_str)
        except Exception as chain_exc:
            logger.debug(f"Option chain fetch failed for {symbol} @ {date_str}: {chain_exc}")
            continue

        calls = getattr(chain, "calls", None)
        puts = getattr(chain, "puts", None)
        calls = calls if isinstance(calls, pd.DataFrame) else pd.DataFrame()
        puts = puts if isinstance(puts, pd.DataFrame) else pd.DataFrame()

        if not calls.empty and "volume" in calls.columns:
            total_calls += float(pd.to_numeric(calls["volume"], errors="coerce").fillna(0).sum())
        if not puts.empty and "volume" in puts.columns:
            total_puts += float(pd.to_numeric(puts["volume"], errors="coerce").fillna(0).sum())

        if derivatives["current_iv"] is None:
            iv_samples = _extract_atm_iv_samples(calls, spot) + _extract_atm_iv_samples(puts, spot)
            if iv_samples:
                derivatives["current_iv"] = round(float(np.mean(iv_samples)), 1)
                derivatives["current_iv_expiry"] = date_str

    if total_calls > 0:
        pc_ratio = total_puts / total_calls
        derivatives["pc_ratio"] = round(float(pc_ratio), 2)
        derivatives["pc_ratio_report"] = f"{pc_ratio:.2f}"
    return derivatives


def _build_option_volatility_context_from_history(
    history_df: pd.DataFrame,
    symbol: str,
    *,
    current_iv: float | None = None,
    expiry_used: str | None = None,
) -> dict:
    context = {
        "symbol": symbol,
        "current_iv": round(float(current_iv), 1) if isinstance(current_iv, (int, float)) else None,
        "realized_vol_30d": None,
        "vrp": None,
        "iv_vs_rv_percentile": None,
        "vol_premium_pct": None,
        "signal": "⚪ 無期權波動資料",
        "summary": "N/A",
        "expiry_used": expiry_used,
    }

    if history_df is None or history_df.empty or "Close" not in history_df.columns:
        return context

    close = pd.to_numeric(history_df["Close"], errors="coerce").dropna()
    if len(close) < 35:
        return context

    log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    realized_vol_series = (log_returns.rolling(30).std() * np.sqrt(252) * 100).replace([np.inf, -np.inf], np.nan).dropna()
    if realized_vol_series.empty:
        return context

    latest_rv = float(realized_vol_series.iloc[-1])
    context["realized_vol_30d"] = round(latest_rv, 1)

    if context["current_iv"] is None or context["current_iv"] <= 0:
        context["summary"] = f"RV30 {latest_rv:.1f}%"
        return context

    current_iv = float(context["current_iv"])
    iv_vs_rv_percentile = float((realized_vol_series < current_iv).sum() / len(realized_vol_series) * 100)
    vrp = current_iv - latest_rv
    vol_premium_pct = ((current_iv / latest_rv) - 1) * 100 if latest_rv > 0 else None

    if vrp >= 10 or (iv_vs_rv_percentile >= 80 and vrp > 5):
        signal = "🔥 恐慌定價"
    elif vrp <= -5 or (iv_vs_rv_percentile <= 30 and vrp < 0):
        signal = "⚠️ 波動低估"
    elif vrp > 0:
        signal = "🟡 避險偏貴"
    else:
        signal = "⚪ 中性"

    context.update(
        {
            "vrp": round(vrp, 1),
            "iv_vs_rv_percentile": round(iv_vs_rv_percentile, 1),
            "vol_premium_pct": round(vol_premium_pct, 1) if vol_premium_pct is not None else None,
            "signal": signal,
            "summary": f"ATM IV {current_iv:.1f}% | RV30 {latest_rv:.1f}% | VRP {vrp:+.1f}pt ({signal})",
        }
    )
    return context


def build_option_volatility_context(symbol: str) -> dict:
    symbol = normalize_ticker(symbol)
    s = symbol.upper()
    if s.isdigit():
        s += ".TW"

    ticker = get_ticker(s)
    history_df = ticker.history(period="1y", interval="1d")
    if history_df.empty:
        return _build_option_volatility_context_from_history(pd.DataFrame(), s)

    spot = float(pd.to_numeric(history_df["Close"], errors="coerce").dropna().iloc[-1])
    derivatives = _scan_option_derivatives(ticker, s, spot)
    return _build_option_volatility_context_from_history(
        history_df,
        s,
        current_iv=derivatives.get("current_iv"),
        expiry_used=derivatives.get("current_iv_expiry"),
    )


def get_mtf_confluence(symbol: str) -> dict:
    symbol = normalize_ticker(symbol)
    s = symbol.upper()
    if s.isdigit():
        s += ".TW"

    calc = IndicatorCalculator()
    scores = {}
    for interval, label in (("1wk", "weekly"), ("1d", "daily"), ("1h", "intraday_1h")):
        try:
            rsi_values = calc.RSI(calc.CLOSE(s, interval))
            rsi_series = pd.Series(rsi_values).replace([np.inf, -np.inf], np.nan).dropna()
            scores[label] = round(float(rsi_series.iloc[-1]), 2) if not rsi_series.empty else None
        except Exception as exc:
            logger.debug(f"MTF RSI fetch failed for {s} @ {interval}: {exc}")
            scores[label] = None

    valid = {key: value for key, value in scores.items() if value is not None}
    oversold_count = sum(1 for value in valid.values() if value < 30)
    overbought_count = sum(1 for value in valid.values() if value > 70)

    if oversold_count >= 2:
        signal = "strong_oversold"
        signal_label = "🟢 強超賣共振"
    elif overbought_count >= 2:
        signal = "strong_overbought"
        signal_label = "🔴 強過熱共振"
    elif oversold_count == 1:
        signal = "mild_oversold"
        signal_label = "🟡 輕度超賣"
    elif overbought_count == 1:
        signal = "mild_overbought"
        signal_label = "🟡 輕度過熱"
    else:
        signal = "neutral"
        signal_label = "⚪ 中性"

    strength = max(oversold_count, overbought_count)
    reliability = "HIGH" if strength >= 2 else "MEDIUM" if strength == 1 else "NORMAL"
    return {
        "rsi_by_timeframe": scores,
        "confluence_signal": signal,
        "signal_label": signal_label,
        "confluence_strength": strength,
        "signal_reliability": reliability,
    }


def build_technical_snapshot(symbol: str, interval: str = "1d", mean_reversion_lookback: int = 60) -> dict:
    symbol = normalize_ticker(symbol)
    s = symbol.upper()
    if s.isdigit():
        s += ".TW"

    history_interval, history_period = _resolve_technical_interval(interval)
    if history_interval != (interval or "1d").lower():
        logger.debug(f"Unsupported technical interval '{interval}' for {s}, falling back to 1d")

    ticker = get_ticker(s)
    df = ticker.history(period=history_period, interval=history_interval)
    if df.empty:
        raise ValueError(f"{s} 無法取得歷史數據")

    calc = IndicatorCalculator()
    close = df['Close'].astype(float)
    high = df['High'].astype(float)
    low = df['Low'].astype(float)
    volume = df['Volume'].astype(float)

    rsi_values = calc.RSI(close.values)
    macd_payload = calc.MACD(close.values)
    dif = pd.Series(macd_payload['macd'], index=df.index)
    macd_hist = pd.Series(macd_payload['histogram'], index=df.index)
    adx_payload = calc.ADX(high.values, low.values, close.values)
    divergence = calc.DIVERGENCE(close.values, rsi_values)
    divergence_label, divergence_details = summarize_divergence(divergence)
    obv_values = calc.OBV(close.values, volume.values)
    obv_signal = analyze_obv_signal(close.values, obv_values)
    mtf_rsi = get_mtf_confluence(s)
    mean_reversion = calc.MEAN_REVERSION(close.values, lookback=mean_reversion_lookback)

    low_9 = low.rolling(window=9).min()
    high_9 = high.rolling(window=9).max()
    rsv = (close - low_9) / (high_9 - low_9) * 100
    vk = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    vd = vk.ewm(alpha=1 / 3, adjust=False).mean()
    vj = 3 * vk - 2 * vd

    ma20 = close.rolling(window=20).mean()
    ma60 = close.rolling(window=60).mean()
    std20 = close.rolling(window=20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)

    try:
        info = ticker.info or {}
    except Exception as e:
        logger.debug(f"Failed to fetch info for technical snapshot {s}: {e}")
        info = {}

    current_price = float(close.iloc[-1])
    rsi_latest = _latest_numeric(rsi_values)
    macd_hist_latest = _latest_numeric(macd_hist)
    high_52w = info.get('fiftyTwoWeekHigh')
    low_52w = info.get('fiftyTwoWeekLow')
    return {
        "symbol": s,
        "history_interval": history_interval,
        "current_price": current_price,
        "high_52w": float(high_52w) if high_52w is not None and not pd.isna(high_52w) else float(high.max()),
        "low_52w": float(low_52w) if low_52w is not None and not pd.isna(low_52w) else float(low.min()),
        "ma20": _latest_numeric(ma20),
        "ma60": _latest_numeric(ma60),
        "rsi": {
            "value": rsi_latest,
            "state": "🔥超買" if rsi_latest is not None and rsi_latest > 70 else "❄️超跌" if rsi_latest is not None and rsi_latest < 30 else "⚖️中性",
        },
        "macd": {
            "dif": _latest_numeric(dif),
            "histogram": macd_hist_latest,
            "state": "📈多頭增強" if (macd_hist_latest or 0) > 0 else "📉空頭衰退",
        },
        "kdj": {
            "k": _latest_numeric(vk, 1),
            "d": _latest_numeric(vd, 1),
            "j": _latest_numeric(vj, 1),
        },
        "bbands": {
            "upper": _latest_numeric(upper),
            "lower": _latest_numeric(lower),
        },
        "adx": {
            "value": _latest_numeric(adx_payload["adx"]),
            "plus_di": _latest_numeric(adx_payload["plus_di"]),
            "minus_di": _latest_numeric(adx_payload["minus_di"]),
            "trend_regime": adx_payload["trend_regime"],
        },
        "divergence": {
            **divergence,
            "label": divergence_label,
            "details": divergence_details,
        },
        "obv": {
            "value": _latest_numeric(obv_values, 0),
            **obv_signal,
        },
        "mtf_rsi": mtf_rsi,
        "mean_reversion": mean_reversion,
    }


def _ensure_asset_profile_cache_schema():
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_profile_cache (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    asset_type TEXT,
                    sector TEXT,
                    industry TEXT,
                    currency TEXT,
                    risk_score REAL,
                    quote_type TEXT,
                    is_etf INTEGER DEFAULT 0,
                    category TEXT,
                    fund_family TEXT,
                    geo_focus TEXT,
                    strategy_type TEXT,
                    tracking_index TEXT,
                    concentration_bucket TEXT,
                    last_updated DATETIME
                )
                """
            )
            existing_columns = {
                row[1]
                for row in cursor.execute("PRAGMA table_info(asset_profile_cache)").fetchall()
            }
            for column, ddl in _ASSET_PROFILE_CACHE_COLUMNS.items():
                if column not in existing_columns:
                    cursor.execute(f"ALTER TABLE asset_profile_cache ADD COLUMN {column} {ddl}")
            conn.commit()
        finally:
            conn.close()


def _clean_profile_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "nan"} else text


def _normalize_cached_asset_profile(row: dict) -> dict:
    normalized = dict(row)
    normalized["quote_type"] = _clean_profile_text(normalized.get("quote_type")) or "Unknown"
    normalized["name"] = _clean_profile_text(normalized.get("name")) or "Unknown"
    normalized["category"] = _clean_profile_text(normalized.get("category"))
    normalized["fund_family"] = _clean_profile_text(normalized.get("fund_family"))
    normalized["geo_focus"] = _clean_profile_text(normalized.get("geo_focus"))
    normalized["strategy_type"] = _clean_profile_text(normalized.get("strategy_type"))
    normalized["tracking_index"] = _clean_profile_text(normalized.get("tracking_index")) or None
    normalized["concentration_bucket"] = _clean_profile_text(normalized.get("concentration_bucket")) or "Unknown"
    normalized["sector"] = _clean_profile_text(normalized.get("sector")) or "Unknown"
    normalized["industry"] = _clean_profile_text(normalized.get("industry")) or "Unknown"
    normalized["currency"] = _clean_profile_text(normalized.get("currency")) or "USD"
    raw_is_etf = normalized.get("is_etf")
    if isinstance(raw_is_etf, str):
        normalized["is_etf"] = raw_is_etf.strip().lower() in {"1", "true", "yes"}
    else:
        normalized["is_etf"] = bool(raw_is_etf)
    return normalized


def _get_local_profile_override(symbol: str, lookup_symbol: str | None = None) -> dict:
    candidates = [normalize_ticker(symbol)]
    if lookup_symbol:
        normalized_lookup = normalize_ticker(lookup_symbol)
        candidates.append(normalized_lookup)
        candidates.append(normalized_lookup.replace(".TW", "").replace(".TWO", ""))
    for candidate in dict.fromkeys(candidates):
        override = _LOCAL_ETF_PROFILE_REGISTRY.get(candidate)
        if override:
            return dict(override)
    return {}


def _merge_profile_override(base: dict, override: dict) -> dict:
    if not override:
        return base
    merged = dict(base)
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged


def _infer_tracking_index(info: dict, symbol: str, is_etf: bool) -> str | None:
    if not is_etf:
        return None

    text_parts = [
        info.get("longName"),
        info.get("shortName"),
        info.get("category"),
        info.get("longBusinessSummary"),
    ]
    combined = " | ".join(part for part in (_clean_profile_text(v) for v in text_parts) if part)

    for pattern, formatter in _TRACKING_INDEX_PATTERNS:
        match = pattern.search(combined)
        if match:
            return formatter(match)

    lowered = combined.lower()
    if "gold" in lowered or "bullion" in lowered:
        return "Physical Gold"
    if "innovation" in lowered and "active" in lowered:
        return "Active Innovation Basket"
    if "active" in lowered or "actively-managed" in lowered:
        return "Active ETF Strategy"
    if "technology" in lowered and "sector" in lowered:
        return "Technology Sector Basket"
    if "energy" in lowered and "sector" in lowered:
        return "Energy Sector Basket"
    if "mlp" in lowered or "pipeline" in lowered:
        return "Energy Infrastructure / MLP Basket"
    if "total market" in lowered:
        return "Total Market"
    if "world" in lowered or "all-country" in lowered:
        return "Global Equity"
    return None


def _infer_concentration_bucket(
    *,
    asset_type: str,
    sector: str,
    industry: str,
    is_etf: bool,
    tracking_index: str | None,
    category: str,
    name: str,
) -> str:
    clean_sector = _clean_profile_text(sector)
    if clean_sector and clean_sector not in {"Unknown", "ETF"}:
        return clean_sector

    text = " ".join(
        part.lower()
        for part in (
            tracking_index,
            category,
            name,
            industry,
            asset_type,
        )
        if _clean_profile_text(part)
    )

    keyword_buckets = (
        (("technology", "tech", "software", "semiconductor", "nasdaq-100", "innovation", "artificial intelligence", "cloud"), "Technology"),
        (("communication", "internet", "media", "telecom", "social"), "Communication Services"),
        (("energy", "oil", "gas", "mlp", "pipeline"), "Energy"),
        (("gold", "silver", "bullion", "metal", "commodity"), "Macro Hedge"),
        (("utility", "utilities"), "Utilities"),
        (("financial", "bank", "insurance"), "Financial Services"),
        (("healthcare", "biotech", "pharma"), "Healthcare"),
        (("real estate", "reit"), "Real Estate"),
        (("industrial", "aerospace", "defense", "manufacturing"), "Industrials"),
        (("consumer staples",), "Consumer Staples"),
        (("consumer discretionary", "retail"), "Consumer Discretionary"),
        (("materials", "mining", "chemicals"), "Materials"),
        (("s&p 500", "msci world", "ftse", "russell 1000", "russell 2000", "total market", "large blend", "large value"), "Diversified Equity"),
    )
    for keywords, bucket in keyword_buckets:
        if any(keyword in text for keyword in keywords):
            return bucket

    if is_etf:
        return "Diversified Equity"
    if asset_type == "Tech_Momentum":
        return "Technology"
    if asset_type == "Macro_Hedge":
        return "Macro Hedge"
    if asset_type == "Value_Holding":
        return "Diversified Equity"
    return "Unknown"


def get_asset_profile(symbol: str) -> dict:
    """
    【核心】資產分類器：Stage 1 (規則) + Stage 2 (LLM Fallback)
    """
    symbol = normalize_ticker(symbol)
    lookup_symbol = _normalize_lookup_symbol(symbol)
    _ensure_asset_profile_cache_schema()
    local_override = _get_local_profile_override(symbol, lookup_symbol)

    cache_candidates = list(dict.fromkeys([symbol, lookup_symbol]))
    with db_lock:
        conn = get_connection()
        try:
            for cache_symbol in cache_candidates:
                df = pd.read_sql("SELECT * FROM asset_profile_cache WHERE symbol = ?", conn, params=(cache_symbol,))
                if df.empty:
                    continue
                cached = _normalize_cached_asset_profile(df.iloc[0].to_dict())
                merged_cached = _merge_profile_override(cached, local_override)
                merged_cached.setdefault("symbol", symbol)
                merged_cached.setdefault("lookup_symbol", lookup_symbol)
                if merged_cached.get("quote_type") not in {"", "Unknown"} or local_override:
                    logger.info(f"Cache Hit: {cache_symbol}")
                    return merged_cached
        except Exception as e:
            logger.error(f"Cache check failed: {e}")
        finally:
            conn.close()

    logger.info(f"Cache Miss: {symbol}, starting classifier...")

    overrides = {
        'BRK-B': 'Value_Holding',
        'IAUM': 'Macro_Hedge',
        'MLPS.L': 'Macro_Hedge'
    }

    info = {}
    asset_type = _clean_profile_text(local_override.get("asset_type")) or "Unknown"
    sector = _clean_profile_text(local_override.get("sector")) or "Unknown"
    industry = _clean_profile_text(local_override.get("industry")) or "Unknown"
    currency = _clean_profile_text(local_override.get("currency")) or "USD"
    risk_score = 1.0
    quote_type = _clean_profile_text(local_override.get("quote_type")) or "Unknown"
    category = _clean_profile_text(local_override.get("category"))
    fund_family = _clean_profile_text(local_override.get("fund_family"))
    geo_focus = _clean_profile_text(local_override.get("geo_focus"))
    strategy_type = _clean_profile_text(local_override.get("strategy_type"))
    tracking_index = _clean_profile_text(local_override.get("tracking_index")) or None
    concentration_bucket = _clean_profile_text(local_override.get("concentration_bucket")) or "Unknown"
    is_etf = bool(local_override.get("is_etf"))
    name_hint = _clean_profile_text(local_override.get("name")) or symbol

    try:
        ticker = get_ticker(lookup_symbol)
        info = ticker.info or {}
        sector = _clean_profile_text(info.get('sector')) or "Unknown"
        industry = _clean_profile_text(info.get('industry')) or "Unknown"
        currency = _clean_profile_text(info.get('currency')) or "USD"
        quote_type = _clean_profile_text(info.get('quoteType')) or "Unknown"
        category = _clean_profile_text(info.get('category'))
        fund_family = _clean_profile_text(info.get('fundFamily'))
        legal_type = _clean_profile_text(info.get("legalType")).lower()
        is_etf = quote_type.upper() == "ETF" or "exchange traded fund" in legal_type
        tracking_index = _infer_tracking_index(info, symbol, is_etf)
        concentration_bucket = _infer_concentration_bucket(
            asset_type=asset_type,
            sector=sector,
            industry=industry,
            is_etf=is_etf,
            tracking_index=tracking_index,
            category=category,
            name=_clean_profile_text(info.get("longName")) or _clean_profile_text(info.get("shortName")) or name_hint,
        )
    except Exception as e:
        logger.warning(f"Stage 1 fetching failed for {symbol}: {e}")

    if symbol in overrides:
        asset_type = overrides[symbol]
    elif sector in ['Technology', 'Communication Services']:
        asset_type = 'Tech_Momentum'
    elif sector in ['Energy', 'Utilities'] or 'Oil' in industry or 'Gas' in industry:
        asset_type = 'Macro_Hedge'
    elif sector == 'Financial Services':
        market_cap = info.get('marketCap', 0)
        if market_cap > 100_000_000_000:
            asset_type = 'Value_Holding'
    elif any(kw in (sector + industry) for kw in ['Gold', 'Metal', 'Commodity']):
        asset_type = 'Macro_Hedge'
    elif is_etf:
        if concentration_bucket in {"Technology", "Communication Services"}:
            asset_type = "Tech_Momentum"
        elif concentration_bucket in {"Energy", "Utilities", "Macro Hedge"}:
            asset_type = "Macro_Hedge"
        elif concentration_bucket == "Diversified Equity":
            asset_type = "Value_Holding"

    is_taiwan_lookup = lookup_symbol.endswith(".TW") or lookup_symbol.endswith(".TWO")
    should_use_llm_fallback = asset_type == "Unknown" and not local_override and not is_taiwan_lookup
    if should_use_llm_fallback:
        logger.info(f"Starting Stage 2 LLM Classifier for {symbol}")
        try:
            from src.llm import quick_call, LIGHT_MODELS

            prompt = (
                f"請將標的 {symbol} (Sector: {sector}, Industry: {industry}, "
                f"QuoteType: {quote_type}, TrackingIndex: {tracking_index or 'Unknown'}) "
                "分類為以下三類之一：Tech_Momentum, Value_Holding, Macro_Hedge。\n僅回傳分類名稱。"
            )
            result = quick_call(prompt, models=LIGHT_MODELS)
            if result:
                llm_type = result.strip()
                if llm_type in ['Tech_Momentum', 'Value_Holding', 'Macro_Hedge']:
                    asset_type = llm_type
        except Exception as e:
            logger.warning(f"Stage 2 LLM classification failed: {e}")

    concentration_bucket = _infer_concentration_bucket(
        asset_type=asset_type,
        sector=sector,
        industry=industry,
        is_etf=is_etf,
        tracking_index=tracking_index,
        category=category,
        name=_clean_profile_text(info.get("longName")) or _clean_profile_text(info.get("shortName")) or name_hint,
    )

    profile_payload = {
        "symbol": symbol,
        "lookup_symbol": lookup_symbol,
        "name": _clean_profile_text(info.get('longName')) or _clean_profile_text(info.get('shortName')) or name_hint,
        "asset_type": asset_type,
        "sector": sector,
        "industry": industry,
        "currency": currency,
        "risk_score": risk_score,
        "quote_type": quote_type,
        "is_etf": is_etf,
        "category": category,
        "fund_family": fund_family,
        "geo_focus": geo_focus,
        "strategy_type": strategy_type,
        "tracking_index": tracking_index,
        "concentration_bucket": concentration_bucket,
    }
    profile_payload = _merge_profile_override(profile_payload, local_override)

    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            for cache_symbol in cache_candidates:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO asset_profile_cache (
                        symbol, name, asset_type, sector, industry, currency, risk_score, quote_type, is_etf,
                        category, fund_family, geo_focus, strategy_type, tracking_index, concentration_bucket, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_symbol,
                        profile_payload["name"],
                        profile_payload["asset_type"],
                        profile_payload["sector"],
                        profile_payload["industry"],
                        profile_payload["currency"],
                        risk_score,
                        profile_payload["quote_type"],
                        int(bool(profile_payload["is_etf"])),
                        profile_payload["category"] or None,
                        profile_payload["fund_family"] or None,
                        profile_payload["geo_focus"] or None,
                        profile_payload["strategy_type"] or None,
                        profile_payload["tracking_index"],
                        profile_payload["concentration_bucket"],
                        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    ),
                )
            conn.commit()
            logger.info(f"Cached {symbol} as {asset_type}")
        except Exception as e:
            logger.error(f"Failed to cache {symbol}: {e}")
        finally:
            conn.close()

    return profile_payload

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

def build_symbol_identity_report(symbol: str) -> str:
    """Pure symbol-identity logic for direct callers and tests."""
    symbol = normalize_ticker(symbol).replace('.TW', '').replace('.TWO', '')
    is_taiwan = any(char.isdigit() for char in symbol) and (len(symbol) <= 6)
    
    if is_taiwan and _has_fubon_provider():
        try:
            # 利用富邦歷史統計功能來抓取官方名稱
            stats = _fubon_provider.build_historical_stats_report(symbol)
            if "未知" not in stats and "異常" not in stats:
                return stats
        except Exception as e:
            logger.debug(f"Fubon historical stats error for {symbol}: {e}")
        
    try:
        s = f"{symbol}.TW" if is_taiwan and not symbol.endswith('.TW') else symbol
        ticker = get_ticker(s)
        info = ticker.info
        name = info.get('longName') or info.get('shortName') or "未知標的"
        return f"🔍 識別結果: {symbol} ({name}) | 類型: {info.get('quoteType', '未知')}"
    except Exception as e:
        logger.error(f"Failed to resolve symbol identity for {symbol}: {e}")
        return format_tool_error(f"❌ 無法識別標的: {symbol}，請確認代號是否正確。", data_unavailable=True)


@tool()
def resolve_symbol_identity(symbol: str) -> str:
    """
    Identifies and validates a ticker symbol. Returns the official name and asset type.
    Use this to resolve unknown or new symbols before further analysis.
    """
    return build_symbol_identity_report(symbol)

def fetch_live_price(symbol: str) -> str:
    """Pure price-fetching logic for direct callers and tests."""
    symbol = normalize_ticker(symbol)
    clean_symbol = symbol.replace('.TW', '').replace('.TWO', '')
    if clean_symbol == "2454_ESOP": clean_symbol = "2454"
    # 支援新形態 ETF (009816 等 6 碼)
    is_taiwan_stock = any(char.isdigit() for char in clean_symbol) and (len(clean_symbol) <= 6)
    
    price = None
    if is_taiwan_stock and _has_fubon_provider():
        try:
            reststock = _fubon_provider.fubon_sdk.marketdata.rest_client.stock
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
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            res = response.json()
            if isinstance(res, list) and len(res) > 0:
                return f"{round(float(res[0]['price']), 2)} (來源: FMP)"
            logger.warning(f"FMP returned empty quote payload for {symbol}")
        except requests.RequestException as e:
            logger.warning(f"FMP real-time price fetch failed for {symbol}: {e}")
        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(f"FMP real-time price parsing failed for {symbol}: {e}")
        except Exception:
            logger.exception(f"Unexpected FMP real-time price failure for {symbol}")

    search_list = [symbol, f"{symbol}.TW", f"{symbol}.TWO"] if is_taiwan_stock else [symbol]
    for s in search_list:
        try:
            ticker = get_ticker(s, cache_level="daily")
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            if not price:
                hist = ticker.history(period="1d")
                if not hist.empty: price = hist['Close'].iloc[-1]
            if price and price > 0: return f"{round(float(price), 2)} (來源: YF)"
        except Exception as e:
            logger.debug(f"YFinance fetch failed for {s}: {e}")
            continue
    return format_tool_error("無法取得報價", data_unavailable=True)

@tool()
def get_live_price(symbol: str) -> str:
    """
    Fetches the real-time or most recent price for a given ticker symbol.
    Supports US and Taiwan markets.
    """
    return fetch_live_price(symbol)

def build_realtime_insight(symbol: str) -> str:
    """Pure intraday insight logic for direct callers and tests."""
    symbol = normalize_ticker(symbol)
    try:
        ticker = get_ticker(symbol)
        full_df = ticker.history(period="2d", interval="5m")
        if full_df.empty: return f"❌ {symbol} 目前無盤中數據。"
        df = full_df.tail(10)
        try:
            daily_history = ticker.history(period="1y", interval="1d")
        except Exception as daily_exc:
            logger.debug(f"Daily history fetch failed for {symbol}: {daily_exc}")
            daily_history = pd.DataFrame()
        info = ticker.info
        bid, ask = info.get('bid', 0), info.get('ask', 0)
        ba_ratio = (info.get('bidSize', 1) / info.get('askSize', 1)) if info.get('askSize', 0) > 0 else 1
        
        spot_for_derivatives = float(df["Close"].iloc[-1])
        if not daily_history.empty and "Close" in daily_history.columns:
            daily_close = pd.to_numeric(daily_history["Close"], errors="coerce").dropna()
            if not daily_close.empty:
                spot_for_derivatives = float(daily_close.iloc[-1])
        derivatives = _scan_option_derivatives(ticker, symbol, spot_for_derivatives)
        pc_report = derivatives["pc_ratio_report"]
        volatility_context = _build_option_volatility_context_from_history(
            daily_history,
            symbol,
            current_iv=derivatives.get("current_iv"),
            expiry_used=derivatives.get("current_iv_expiry"),
        )
        vol_context_report = volatility_context.get("summary", "N/A")

        # 成交量密集區 (POC)
        day_min, day_max = full_df['Low'].min(), full_df['High'].max()
        bins = np.linspace(day_min, day_max, 11)
        full_df['bin'] = pd.cut(full_df['Close'], bins=bins)
        vp = full_df.groupby('bin', observed=True)['Volume'].sum()
        poc_bin = vp.idxmax()
        poc_price = (poc_bin.left + poc_bin.right) / 2
        vp_status = "🛡️ 支撐" if df['Close'].iloc[-1] > poc_price else "🧱 壓力"
        vwap_series = _compute_vwap(full_df)
        current_vwap = _latest_numeric(vwap_series)
        vwap_report = "N/A"
        dual_anchor_state = "N/A"
        if current_vwap is not None and current_vwap > 0:
            vwap_delta_pct = ((df['Close'].iloc[-1] / current_vwap) - 1) * 100
            vwap_status = "上方" if df['Close'].iloc[-1] >= current_vwap else "下方"
            vwap_report = f"{current_vwap:.2f} ({vwap_status} {vwap_delta_pct:+.2f}%)"
            dual_anchor_state = _classify_dual_anchor_state(df['Close'].iloc[-1], current_vwap, poc_price)

        # 📊 成交量爆發力與換手率 (Volume & Turnover)
        vol_ratio_report = "N/A"
        turnover_report = "N/A"
        try:
            avg_vol = info.get('averageVolume')
            curr_vol = info.get('regularMarketVolume')
            base_shares = next(
                (
                    value
                    for value in (info.get('floatShares'), info.get('sharesOutstanding'))
                    if value is not None and not pd.isna(value) and value > 0
                ),
                None,
            )

            # 1. Volume Ratio (時間加權成交量能比) - 保持您原本的邏輯
            if (
                avg_vol is not None and not pd.isna(avg_vol) and avg_vol > 0
                and curr_vol is not None and not pd.isna(curr_vol) and curr_vol >= 0
            ):
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
                        
            # [新增] 2. 換手率 (Turnover Rate) 與 AI 訊號
            if curr_vol is not None and not pd.isna(curr_vol) and curr_vol >= 0:
                if base_shares:
                    turnover_rate = (curr_vol / base_shares) * 100
                    turnover_report = f"{turnover_rate:.2f}%"
                    if turnover_rate > 5.0:  # 訊號生成: 日換手 >5% 視為籌碼極度活躍
                        turnover_report += " 🔥籌碼活躍"
        except Exception as e:
            logger.debug(f"Volume/Turnover metrics fetching failed for {symbol}: {e}")
            pass

        report = f"🚀 === {symbol} 美股即時戰情 ===\n"
        report += f"● 現價: {df['Close'].iloc[-1]:.2f} | 買賣比: {ba_ratio:.2f} | P/C Ratio: {pc_report}\n"
        report += f"● 成交量能比: {vol_ratio_report} | 換手率: {turnover_report} | VWAP: {vwap_report}\n"
        report += f"● POC 密集區: {poc_price:.2f} ({vp_status}) | 雙錨點: {dual_anchor_state}\n"
        report += f"● 波動定價: {vol_context_report}\n"
        report += "【📊 最近 5 根 K 線】\n"
        for _, row in df.tail(5).iterrows():
            report += f"  [{row.name.strftime('%H:%M')}] {'🟢' if row['Close']>row['Open'] else '🔴'} C:{row['Close']:.2f} | 量:{int(row['Volume'])}\n"
        return report
    except Exception as e:
        logger.error(f"Realtime insight failed for {symbol}: {e}")
        return format_tool_error(f"❌ 美股掃描失敗: {e}", transient=True)

@tool()
def get_us_realtime_insight(symbol: str) -> str:
    """Tool schema for LLM."""
    return build_realtime_insight(symbol)


def _extract_download_history_frame(histories: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(histories, pd.DataFrame) or histories.empty:
        return pd.DataFrame()
    try:
        if isinstance(histories.columns, pd.MultiIndex):
            if symbol not in histories.columns.get_level_values(0):
                return pd.DataFrame()
            frame = histories[symbol].copy()
        else:
            frame = histories.copy()
    except Exception as exc:
        logger.debug(f"Batch history extract failed for {symbol}: {exc}")
        return pd.DataFrame()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return frame.dropna(how="all")


def _get_cached_sentiment_histories(symbols: list[str], period: str = "10d") -> pd.DataFrame:
    normalized_symbols = tuple(dict.fromkeys(symbols))
    cache_key = (normalized_symbols, period, "1d")
    now = time.time()
    with _SENTIMENT_BATCH_CACHE_LOCK:
        cached = _SENTIMENT_BATCH_CACHE["entries"].get(cache_key)
        if cached and (now - cached["timestamp"] < _SENTIMENT_BATCH_CACHE_TTL_SECONDS):
            return cached["data"].copy()

    try:
        histories = get_download(
            list(normalized_symbols),
            period=period,
            interval="1d",
            group_by="ticker",
            progress=False,
        )
    except Exception as exc:
        logger.debug(f"Sentiment batch download failed: {exc}")
        return pd.DataFrame()

    if not isinstance(histories, pd.DataFrame) or histories.empty:
        return pd.DataFrame()

    with _SENTIMENT_BATCH_CACHE_LOCK:
        _SENTIMENT_BATCH_CACHE["entries"][cache_key] = {
            "timestamp": now,
            "data": histories.copy(),
        }
    return histories.copy()


def _load_sentiment_history(symbol: str, batch_histories: pd.DataFrame, period: str = "10d") -> pd.DataFrame:
    history = _extract_download_history_frame(batch_histories, symbol)
    if not history.empty:
        return history
    try:
        return get_ticker(symbol).history(period=period)
    except Exception as exc:
        logger.debug(f"Market sentiment fallback fetch failed for {symbol}: {exc}")
        return pd.DataFrame()


def build_sentiment_report() -> str:
    """Pure market sentiment logic for direct callers and tests."""
    indicators = _SENTIMENT_INDICATORS
    batch_histories = _get_cached_sentiment_histories(list(indicators.keys()), period="10d")
    report = "【🌐 全球宏觀資金流向雷達】\n"
    for symbol, name in indicators.items():
        try:
            hist = _load_sentiment_history(symbol, batch_histories, period="10d")
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

@tool()
def get_market_sentiment() -> str:
    """
    Analyzes global market sentiment by monitoring key indices, bonds, and commodities.
    Provides a macro-level overview of capital flows and risk appetite.
    """
    return build_sentiment_report()

def build_stock_news_report(symbol: str) -> str:
    """Pure stock-news logic for direct callers and tests."""
    try:
        symbol = normalize_ticker(symbol)
        search_symbol = symbol.upper()
        if search_symbol.isdigit(): search_symbol += ".TW"
        ticker = get_ticker(search_symbol)
        news_list = ticker.news[:10]
        if not news_list:
            return format_tool_error("無新聞數據。", data_unavailable=True)
        report = f"【📰 {symbol} 最新情報】\n"
        for i, item in enumerate(news_list):
            title = item.get('title') or item.get('content', {}).get('title')
            publisher = item.get('publisher') or item.get('content', {}).get('provider', {}).get('displayName', '媒體')
            report += f"{i+1}. [{publisher}] {title}\n"
        return report
    except Exception as e:
        logger.error(f"Stock news fetch failed for {symbol}: {e}")
        return format_tool_error(f"新聞異常: {e}", transient=True)


@tool()
def get_stock_news(symbol: str) -> str:
    """
    Retrieves the latest news headlines for a specific stock symbol.
    """
    return build_stock_news_report(symbol)


def build_fundamental_report(symbol: str) -> str:
    """Pure fundamental snapshot logic for direct callers and tests."""
    try:
        symbol = normalize_ticker(symbol)
        s = symbol.upper()
        if s.isdigit(): s += ".TW"
        ticker = get_ticker(s)
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
        logger.error(f"Fundamental data fetch failed for {symbol}: {e}")
        return format_tool_error(f"基本面數據獲取失敗: {e}", data_unavailable=True)


@tool()
def get_fundamental_data(symbol: str) -> str:
    """
    Retrieves key fundamental metrics (EPS, P/E, P/B, Institutional Ownership) for a stock.
    """
    return build_fundamental_report(symbol)

def build_technical_report(symbol: str, interval: str = "1d") -> str:
    """Pure technical-analysis logic for direct callers and tests."""
    try:
        symbol = normalize_ticker(symbol)
        s = symbol.upper()
        clean_symbol = s.replace('.TW', '').replace('.TWO', '')
        is_taiwan = any(char.isdigit() for char in clean_symbol) and (len(clean_symbol) <= 6)
        
        # --- 台股使用 Fubon SDK 官方數據 ---
        if is_taiwan and _has_fubon_provider():
            return _fubon_provider.get_fubon_technical(clean_symbol)
            
        snapshot = build_technical_snapshot(symbol, interval)
        curr = snapshot["current_price"]
        curr_ma20 = snapshot["ma20"] if snapshot["ma20"] is not None else float("nan")
        curr_ma60 = snapshot["ma60"] if snapshot["ma60"] is not None else float("nan")
        curr_rsi = snapshot["rsi"]["value"] if snapshot["rsi"]["value"] is not None else float("nan")
        curr_k = snapshot["kdj"]["k"] if snapshot["kdj"]["k"] is not None else float("nan")
        curr_d = snapshot["kdj"]["d"] if snapshot["kdj"]["d"] is not None else float("nan")
        curr_j = snapshot["kdj"]["j"] if snapshot["kdj"]["j"] is not None else float("nan")
        upper = snapshot["bbands"]["upper"] if snapshot["bbands"]["upper"] is not None else float("nan")
        lower = snapshot["bbands"]["lower"] if snapshot["bbands"]["lower"] is not None else float("nan")
        divergence = snapshot["divergence"]
        adx = snapshot["adx"]
        obv = snapshot["obv"]
        mtf_rsi = snapshot.get("mtf_rsi", {})
        mean_reversion = snapshot.get("mean_reversion", {})
        trend_regime = adx.get("trend_regime", "unknown")
        trend_label = "📈趨勢盤" if trend_regime == "trending" else "🌀震盪盤"
        mtf_scores = mtf_rsi.get("rsi_by_timeframe", {})
        mtf_fragments = []
        for key, label in (("weekly", "W"), ("daily", "D"), ("intraday_1h", "H1")):
            value = mtf_scores.get(key)
            if value is not None:
                mtf_fragments.append(f"{label}:{value:.1f}")
        mtf_score_line = " | ".join(mtf_fragments) if mtf_fragments else "N/A"
        
        report = f"🇺🇸 === {s} 美股全武裝分析 ===\n"
        report += f"● 現價: {curr:.2f} | 52週高: {snapshot['high_52w']:.2f} | 52週低: {snapshot['low_52w']:.2f}\n"
        report += f"● 均線位階: MA20:{curr_ma20:.2f} | MA60:{curr_ma60:.2f}\n"
        report += f"● KDJ(9,3,3): K:{curr_k:.1f} | D:{curr_d:.1f} | J:{curr_j:.1f}\n"
        report += f"● RSI(14): {curr_rsi:.2f} ({snapshot['rsi']['state']})\n"
        report += (
            f"● 多時間框 RSI: {mtf_score_line} -> "
            f"{mtf_rsi.get('signal_label', '⚪ 中性')} ({mtf_rsi.get('signal_reliability', 'NORMAL')})\n"
        )
        report += f"● MACD: DIF:{snapshot['macd']['dif']:.2f} | 柱狀體:{snapshot['macd']['histogram']:.2f} ({snapshot['macd']['state']})\n"
        report += f"● ADX(14): {adx['value']:.2f} | +DI:{adx['plus_di']:.2f} | -DI:{adx['minus_di']:.2f} ({trend_label})\n"
        report += f"● RSI 背離: {divergence['label']}"
        if divergence.get("details") and divergence["details"] not in {"無明顯背離", "資料不足"}:
            report += f" | {divergence['details']}"
        report += "\n"
        report += f"● OBV 趨勢: {obv['label']} | {obv['signal']}"
        if obv.get("obv_ma20") is not None:
            report += f" | OBV20MA:{obv['obv_ma20']:.2f}"
        report += "\n"
        zscore = mean_reversion.get("zscore")
        zscore_text = f"{zscore:+.2f}" if isinstance(zscore, (int, float)) else "N/A"
        half_life = mean_reversion.get("half_life_days")
        half_life_text = f"{half_life:.1f} 期" if isinstance(half_life, (int, float)) else "N/A"
        report += (
            f"● 均值回歸: Z:{zscore_text} | 半衰期:{half_life_text} | "
            f"{mean_reversion.get('signal_label', '⚪ 中性')}\n"
        )
        report += f"● 布林通道: 上軌:{upper:.2f} | 下軌:{lower:.2f}\n"
        
        # 戰術建議 (優化：結合 RSI, KDJ 與 MA 濾網)
        if curr >= upper:
            if curr_rsi > 75:
                report += f"⚠️ 戰略：觸及布林上軌且 RSI 極度過熱 ({curr_rsi:.2f})，短線噴發過頭，不建議追高。\n"
            elif 55 < curr_rsi <= 75:
                report += f"🔥 戰略：強勢沿上軌攀升中 (RSI: {curr_rsi:.2f})，留意跌破均線停利。\n"
            else:
                report += "⚠️ 戰略：觸及布林上軌，留意拉回風險。\n"
        elif curr <= lower:
            if curr_rsi < 25:
                report += f"🎯 戰略：觸及布林下軌且極度超跌 ({curr_rsi:.2f})，具備技術性反彈潛力！\n"
            elif 25 <= curr_rsi < 45:
                report += f"⚠️ 戰略：沿下軌弱勢下跌中 ({curr_rsi:.2f})，切勿盲目抄底。\n"
            else:
                report += "🎯 戰略：觸及布林下軌，具備反彈潛力。\n"
        elif curr_k > curr_d and curr_k < 30:
            report += f"🚀 戰略：KDJ 低檔金叉 (K:{curr_k:.1f})，轉折噴發信號！\n"
        elif curr_k < curr_d and curr_k > 70:
            report += f"🥀 戰略：KDJ 高檔死叉 (K:{curr_k:.1f})，波段見頂信號。\n"
        elif curr_j > 100:
            report += "🔥 戰略：J 線噴發過度，留意隨時拉回。\n"
        elif curr_j < 0:
            report += "❄️ 戰略：J 線極度耗竭，反彈將至。\n"
        elif curr > curr_ma20 and curr > curr_ma60:
            report += "📈 戰略：股價站上月線與季線，多頭排列建立，回檔即買點。\n"
        elif curr_rsi < 30:
            report += f"🔥 戰略：RSI 極度超跌 ({curr_rsi:.2f})，隨時可能暴力反彈。\n"
        else:
            report += "🧘 戰略：目前位階中性，建議分批佈局過等待關鍵突破。\n"

        if divergence.get("bullish_divergence"):
            report += "🟢 補充：RSI 底背離成立，賣壓動能正在衰竭。\n"
        elif divergence.get("bearish_divergence"):
            report += "🔴 補充：RSI 頂背離成立，上攻動能開始鈍化。\n"

        if trend_regime == "ranging" and adx.get("value") is not None:
            report += f"🌀 補充：ADX 僅 {adx['value']:.2f}，當前偏震盪盤，均線突破需二次確認。\n"

        if mean_reversion.get("reversion_candidate"):
            if mean_reversion.get("signal") in {"strong_buy", "buy"}:
                report += "🧲 補充：價格偏離均值過深且半衰期收斂，若出現止跌確認可留意回歸反彈。\n"
            elif mean_reversion.get("signal") in {"strong_sell", "sell"}:
                report += "🧲 補充：價格高於均值過多且半衰期收斂，追價風險正在上升。\n"
        elif mean_reversion.get("signal") != "neutral" and mean_reversion.get("zscore") is not None:
            report += "⚠️ 補充：雖有偏離均值，但半衰期不收斂，較像趨勢延伸而非穩定回歸 edge。\n"
        
        return report
    except Exception as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}")
        return format_tool_error(f"❌ 技術分析失敗: {e}", data_unavailable=True)

@tool()
def get_technical_analysis(symbol: str, interval: str = "1d") -> str:
    """
    Performs multi-indicator technical analysis (RSI, MACD, KDJ, Bollinger Bands, mean reversion).
    Provides a strategic outlook based on indicator alignment.
    """
    return build_technical_report(symbol, interval)


def build_mean_reversion_report(symbol: str, interval: str = "1d", lookback: int = 60) -> str:
    try:
        snapshot = build_technical_snapshot(symbol, interval, mean_reversion_lookback=lookback)
        signal = snapshot.get("mean_reversion", {})
        zscore = signal.get("zscore")
        zscore_text = f"{zscore:+.2f}" if isinstance(zscore, (int, float)) else "N/A"
        half_life = signal.get("half_life_days")
        half_life_text = f"{half_life:.1f} 期" if isinstance(half_life, (int, float)) else "N/A"
        lookback_used = signal.get("lookback_used") or lookback

        report = f"🧲 === {snapshot['symbol']} 均值回歸信號 ===\n"
        report += (
            f"● Z-Score({lookback_used}): {zscore_text} | 半衰期: {half_life_text} | "
            f"{signal.get('signal_label', '⚪ 中性')}\n"
        )
        report += f"● 解讀: {signal.get('details', 'N/A')}\n"

        if signal.get("reversion_candidate"):
            report += "● 結論: 偏離與回歸速度都達標，屬可交易的均值回歸候選。\n"
        elif signal.get("zscore") is not None and abs(float(signal["zscore"])) >= 1:
            report += "● 結論: 偏離存在，但半衰期不收斂，暫時更像趨勢延伸而非回歸 edge。\n"
        else:
            report += "● 結論: 目前偏離有限，沒有明顯均值回歸優勢。\n"

        return report
    except Exception as e:
        logger.error(f"Mean reversion report failed for {symbol}: {e}")
        return format_tool_error(f"❌ 均值回歸信號失敗: {e}", data_unavailable=True)


@tool()
def get_mean_reversion_signal(symbol: str, interval: str = "1d", lookback: int = 60) -> str:
    """Evaluates price-vs-mean deviation and half-life for mean-reversion setups."""
    return build_mean_reversion_report(symbol, interval, lookback)


def compute_pair_trade_signal(symbol_a: str, symbol_b: str, lookback: int = 120, period: str = "2y") -> dict:
    pair_a = _normalize_lookup_symbol(symbol_a)
    pair_b = _normalize_lookup_symbol(symbol_b)
    if pair_a == pair_b:
        return {"error": "配對交易需要兩個不同標的。"}

    effective_lookback = max(int(lookback), 60)
    series_a = _fetch_daily_close_series(pair_a, period=period)
    series_b = _fetch_daily_close_series(pair_b, period=period)
    aligned = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1, join="inner").dropna()
    if len(aligned) < effective_lookback:
        return {"error": f"{pair_a}/{pair_b} 重疊歷史資料不足 ({len(aligned)})。"}

    window = aligned.tail(effective_lookback).copy()
    x = window["b"].to_numpy(dtype=float)
    y = window["a"].to_numpy(dtype=float)
    design = np.column_stack([x, np.ones(len(x))])
    hedge_ratio, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = (hedge_ratio * window["b"]) + intercept
    spread = (window["a"] - fitted).astype(float)
    spread_std = float(spread.std(ddof=0)) if len(spread) > 1 else 0.0
    if spread_std <= 0:
        return {"error": f"{pair_a}/{pair_b} 價差波動過低，無法判讀配對偏離。"}

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"error": "缺少 statsmodels，無法執行 ADF 協整檢驗。"}

    adf_pvalue = None
    try:
        adf_pvalue = float(adfuller(spread.to_numpy(dtype=float), autolag="AIC")[1])
    except Exception as exc:
        logger.debug(f"ADF test failed for {pair_a}/{pair_b}: {exc}")

    zscore = float((spread.iloc[-1] - spread.mean()) / spread_std)
    corr = float(np.corrcoef(y, x)[0, 1]) if len(window) > 1 else float("nan")
    r_squared = (corr ** 2) if np.isfinite(corr) else None
    half_life = _estimate_half_life(spread)
    cointegrated = adf_pvalue is not None and adf_pvalue < 0.05
    slow_reversion = half_life is not None and half_life > 60

    if not cointegrated:
        signal = "不構成配對 (spread 未達協整)"
    elif zscore <= -2 and not slow_reversion:
        signal = f"做多 {pair_a} / 做空 {pair_b}"
    elif zscore >= 2 and not slow_reversion:
        signal = f"做空 {pair_a} / 做多 {pair_b}"
    elif abs(zscore) >= 2 and slow_reversion:
        signal = "價差偏離達標，但半衰期過長，先列監控"
    else:
        signal = "無交易訊號"

    return {
        "pair": f"{pair_a}/{pair_b}",
        "symbol_a": pair_a,
        "symbol_b": pair_b,
        "lookback": effective_lookback,
        "hedge_ratio": round(float(hedge_ratio), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_squared), 4) if r_squared is not None else None,
        "adf_pvalue": round(adf_pvalue, 4) if adf_pvalue is not None else None,
        "cointegrated": cointegrated,
        "spread_zscore": round(zscore, 2),
        "spread_half_life": round(half_life, 1) if half_life is not None else None,
        "signal": signal,
        "methodology": "以 OLS hedge ratio 建 spread，並用 ADF 檢驗殘差平穩性；屬統計配對訊號，不等於無風險套利。",
    }


def build_pairs_trade_report(symbol_a: str, symbol_b: str, lookback: int = 120) -> str:
    payload = compute_pair_trade_signal(symbol_a, symbol_b, lookback=lookback)
    if payload.get("error"):
        return format_tool_error(f"❌ {payload['error']}", data_unavailable=True)

    r2_text = f"{payload['r_squared']:.3f}" if payload.get("r_squared") is not None else "N/A"
    adf_text = f"{payload['adf_pvalue']:.4f}" if payload.get("adf_pvalue") is not None else "N/A"
    half_life = payload.get("spread_half_life")
    half_life_text = f"{half_life:.1f} 期" if isinstance(half_life, (int, float)) else "N/A"
    coint_text = "是" if payload.get("cointegrated") else "否"

    report = f"🔗 === {payload['pair']} 配對/協整監控 ===\n"
    report += (
        f"● Hedge Ratio: {payload['hedge_ratio']:.3f} | R²: {r2_text} | "
        f"ADF p-value: {adf_text} | 協整: {coint_text}\n"
    )
    report += (
        f"● Spread Z-Score: {payload['spread_zscore']:+.2f} | "
        f"半衰期: {half_life_text}\n"
    )
    report += f"● 結論: {payload['signal']}\n"
    report += f"● 註記: {payload['methodology']}"
    return report


@tool()
def get_pairs_trade_signal(symbol_a: str, symbol_b: str, lookback: int = 120) -> str:
    """Evaluates a pair spread with OLS hedge ratio, ADF cointegration, and spread z-score."""
    return build_pairs_trade_report(symbol_a, symbol_b, lookback)


def compute_factor_snapshot(symbol: str) -> dict:
    s = _normalize_lookup_symbol(symbol)
    ticker = get_ticker(s, cache_level="daily")
    history = ticker.history(period="2y", interval="1d")
    if history.empty or "Close" not in history.columns:
        return {"error": f"{s} 無法取得歷史價格。"}

    close = (
        pd.to_numeric(history["Close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(close) < 22:
        return {"error": f"{s} 歷史價格不足，無法計算單股因子。"}

    try:
        info = ticker.info or {}
    except Exception as exc:
        logger.debug(f"Factor snapshot info fetch failed for {s}: {exc}")
        info = {}

    trailing_pe = _safe_float(info.get("trailingPE"))
    price_to_book = _safe_float(info.get("priceToBook"))
    market_cap = _safe_float(info.get("marketCap"))
    roe = _safe_float(info.get("returnOnEquity"))
    gross_margin = _safe_float(info.get("grossMargins"))
    debt_to_equity = _safe_float(info.get("debtToEquity"))

    momentum_12_1 = float((close.iloc[-22] / close.iloc[-252]) - 1) if len(close) >= 252 else None
    reversal_1m = float(-((close.iloc[-1] / close.iloc[-22]) - 1)) if len(close) >= 22 else None
    earnings_yield = float(1 / trailing_pe) if trailing_pe is not None and trailing_pe > 0 else None
    book_price = float(1 / price_to_book) if price_to_book is not None and price_to_book > 0 else None

    leverage_penalty = 1.0
    if debt_to_equity is not None:
        leverage_penalty = 1 / (1 + max(debt_to_equity, 0) / 100)
    quality_raw = (
        float(roe * gross_margin * leverage_penalty)
        if roe is not None and gross_margin is not None
        else None
    )

    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    realized_vol_60d = float(returns.tail(60).std(ddof=0) * np.sqrt(252)) if len(returns) >= 60 else None
    low_vol_score = -realized_vol_60d if realized_vol_60d is not None else None
    size_log_mcap = float(math.log(market_cap)) if market_cap is not None and market_cap > 0 else None

    return {
        "symbol": s,
        "momentum_12_1": round(momentum_12_1, 4) if momentum_12_1 is not None else None,
        "reversal_1m": round(reversal_1m, 4) if reversal_1m is not None else None,
        "earnings_yield": round(earnings_yield, 4) if earnings_yield is not None else None,
        "book_price": round(book_price, 4) if book_price is not None else None,
        "quality_raw": round(quality_raw, 4) if quality_raw is not None else None,
        "roe": round(roe, 4) if roe is not None else None,
        "gross_margin": round(gross_margin, 4) if gross_margin is not None else None,
        "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity is not None else None,
        "low_vol_score": round(low_vol_score, 4) if low_vol_score is not None else None,
        "realized_vol_60d": round(realized_vol_60d, 4) if realized_vol_60d is not None else None,
        "size_log_mcap": round(size_log_mcap, 4) if size_log_mcap is not None else None,
        "methodology": "單股 raw 因子快照；未做截面標準化、產業中性化或組合排序，因此不能直接視為正式多因子 alpha rank。",
    }


def build_factor_snapshot_report(symbol: str) -> str:
    payload = compute_factor_snapshot(symbol)
    if payload.get("error"):
        return format_tool_error(f"❌ {payload['error']}", data_unavailable=True)

    def _pct(value):
        return f"{value * 100:+.1f}%" if isinstance(value, (int, float)) else "N/A"

    def _raw(value, digits: int = 2):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "N/A"

    report = f"🧬 === {payload['symbol']} 單股因子快照 ===\n"
    report += (
        f"● Momentum(12-1): {_pct(payload.get('momentum_12_1'))} | "
        f"Reversal(1M): {_pct(payload.get('reversal_1m'))} | "
        f"RV60: {_pct(payload.get('realized_vol_60d'))}\n"
    )
    report += (
        f"● Value: E/P {_pct(payload.get('earnings_yield'))} | "
        f"B/P {_raw(payload.get('book_price'), 3)} | "
        f"Size ln(MCap): {_raw(payload.get('size_log_mcap'), 2)}\n"
    )
    report += (
        f"● Quality(raw): {_raw(payload.get('quality_raw'), 4)} | "
        f"ROE {_pct(payload.get('roe'))} | "
        f"毛利率 {_pct(payload.get('gross_margin'))} | "
        f"D/E {_raw(payload.get('debt_to_equity'), 1)}\n"
    )
    report += f"● 註記: {payload['methodology']}"
    return report


@tool()
def get_factor_snapshot(symbol: str) -> str:
    """Builds a single-name raw factor snapshot (momentum, reversal, value, quality, low-vol, size)."""
    return build_factor_snapshot_report(symbol)


def _signal_symbol_variants(symbol: str) -> list[str]:
    base = _normalize_lookup_symbol(symbol)
    variants = [base]
    clean = base.replace(".TW", "").replace(".TWO", "")
    if clean not in variants:
        variants.append(clean)
    if base.endswith(".TW"):
        two_variant = base.replace(".TW", ".TWO")
        if two_variant not in variants:
            variants.append(two_variant)
    return variants


def _parse_signal_timestamp(value):
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def compute_nlp_signal_ic(
    symbol: str,
    horizon_days: int = 5,
    lookback_signals: int = 120,
    rolling_window: int = 20,
    min_samples: int = 20,
) -> dict:
    s = _normalize_lookup_symbol(symbol)
    variants = _signal_symbol_variants(symbol)
    placeholders = ", ".join(["?"] * len(variants))
    query = (
        f"SELECT timestamp, nlp_alpha FROM nlp_insights "
        f"WHERE symbol IN ({placeholders}) AND nlp_alpha IS NOT NULL "
        f"ORDER BY timestamp DESC LIMIT ?"
    )
    try:
        with db_lock:
            conn = get_connection()
            try:
                signals = pd.read_sql(query, conn, params=[*variants, max(lookback_signals * 3, rolling_window * 3)])
            finally:
                conn.close()
    except Exception as exc:
        logger.debug(f"NLP IC history load failed for {s}: {exc}")
        return {"error": f"{s} 尚未建立可用的 nlp_insights 歷史。"}

    if signals.empty:
        return {"error": f"{s} 缺少 nlp_insights 歷史資料。"}

    horizon = max(int(horizon_days), 1)
    rolling_window = max(int(rolling_window), 5)
    min_samples = max(int(min_samples), rolling_window)

    signals["timestamp"] = signals["timestamp"].apply(_parse_signal_timestamp)
    signals["nlp_alpha"] = pd.to_numeric(signals["nlp_alpha"], errors="coerce")
    signals = signals.dropna(subset=["timestamp", "nlp_alpha"]).sort_values("timestamp")
    if signals.empty:
        return {"error": f"{s} 的 nlp_insights 時間戳或 alpha 無法解析。"}

    signals["signal_day"] = signals["timestamp"].dt.normalize()
    signals = signals.drop_duplicates(subset=["signal_day"], keep="last").tail(lookback_signals)

    close = _fetch_daily_close_series(s, period="5y")
    if close.empty:
        return {"error": f"{s} 無法取得日線價格，無法計算 IC。"}

    trade_index = pd.DatetimeIndex(pd.to_datetime(close.index))
    if trade_index.tz is not None:
        trade_index = trade_index.tz_convert(None)
    trade_days = trade_index.normalize()

    observations = []
    for row in signals.itertuples(index=False):
        anchor_idx = int(trade_days.searchsorted(row.signal_day, side="right"))
        exit_idx = anchor_idx + horizon
        if anchor_idx >= len(close) or exit_idx >= len(close):
            continue
        entry_price = float(close.iloc[anchor_idx])
        exit_price = float(close.iloc[exit_idx])
        if entry_price <= 0:
            continue
        observations.append(
            {
                "signal": float(row.nlp_alpha),
                "forward_return": (exit_price / entry_price) - 1.0,
                "anchor_day": trade_days[anchor_idx],
            }
        )

    sample = pd.DataFrame(observations).sort_values("anchor_day")
    if len(sample) < min_samples:
        return {"error": f"{s} 可對齊的訊號樣本僅 {len(sample)} 筆，至少需要 {min_samples} 筆。"}

    ic_full = sample["signal"].corr(sample["forward_return"], method="spearman")
    ic_full = float(ic_full) if pd.notna(ic_full) else None

    rolling_ics = []
    if len(sample) >= rolling_window:
        for end in range(rolling_window, len(sample) + 1):
            window = sample.iloc[end - rolling_window:end]
            ic_value = window["signal"].corr(window["forward_return"], method="spearman")
            if pd.notna(ic_value):
                rolling_ics.append(float(ic_value))

    rolling_mean = float(np.mean(rolling_ics)) if rolling_ics else None
    latest_rolling = rolling_ics[-1] if rolling_ics else None
    quality_ref = rolling_mean if rolling_mean is not None else ic_full

    if quality_ref is None:
        quality = "noise"
        directionality = "undetermined"
        interpretation = "樣本雖足夠，但 IC 無法穩定估計。"
    else:
        directionality = "positive" if quality_ref > 0 else "negative" if quality_ref < 0 else "flat"
        abs_ic = abs(quality_ref)
        if abs_ic >= 0.05:
            quality = "strong"
            interpretation = "訊號具可觀統計預測力。"
        elif abs_ic >= 0.02:
            quality = "weak"
            interpretation = "訊號微弱但仍可追蹤。"
        else:
            quality = "noise"
            interpretation = "訊號目前接近雜訊。"
        if quality_ref < 0:
            interpretation += " 當前更像反向/contrarian edge。"
        elif quality_ref > 0:
            interpretation += " 當前屬順向 edge。"

    return {
        "symbol": s,
        "horizon_days": horizon,
        "rolling_window": rolling_window,
        "sample_count": int(len(sample)),
        "ic_full_sample": round(ic_full, 4) if ic_full is not None else None,
        "ic_rolling_mean": round(rolling_mean, 4) if rolling_mean is not None else None,
        "ic_latest_rolling": round(latest_rolling, 4) if latest_rolling is not None else None,
        "signal_quality": quality,
        "directionality": directionality,
        "interpretation": interpretation,
        "methodology": (
            "使用 nlp_insights 中每日最新 nlp_alpha，對齊到訊號後下一個交易日收盤作為起點，"
            f"計算 {horizon} 日 forward return；IC 採 Spearman rank correlation。"
        ),
    }


def build_nlp_signal_ic_report(symbol: str, horizon_days: int = 5, lookback_signals: int = 120) -> str:
    payload = compute_nlp_signal_ic(symbol, horizon_days=horizon_days, lookback_signals=lookback_signals)
    if payload.get("error"):
        return format_tool_error(f"❌ {payload['error']}", data_unavailable=True)

    def _fmt_ic(value):
        return f"{value:+.3f}" if isinstance(value, (int, float)) else "N/A"

    report = f"🧪 === {payload['symbol']} NLP Alpha IC 追蹤 ===\n"
    report += (
        f"● Horizon: {payload['horizon_days']}D | Samples: {payload['sample_count']} | "
        f"Rolling Window: {payload['rolling_window']}\n"
    )
    report += (
        f"● Full IC: {_fmt_ic(payload.get('ic_full_sample'))} | "
        f"Rolling Mean: {_fmt_ic(payload.get('ic_rolling_mean'))} | "
        f"Latest Rolling: {_fmt_ic(payload.get('ic_latest_rolling'))}\n"
    )
    report += f"● 品質: {payload['signal_quality']} | 方向: {payload['directionality']}\n"
    report += f"● 解讀: {payload['interpretation']}\n"
    report += f"● 註記: {payload['methodology']}"
    return report


@tool()
def get_nlp_signal_ic(symbol: str, horizon_days: int = 5, lookback_signals: int = 120) -> str:
    """Tracks the Information Coefficient of persisted NLP alpha signals versus future returns."""
    return build_nlp_signal_ic_report(symbol, horizon_days, lookback_signals)


def _parse_candidate_universe(symbols: str | list[str] | None) -> list[str]:
    raw_items = WATCH_LIST if symbols in (None, "", []) else symbols
    if not isinstance(raw_items, list):
        raw_items = str(raw_items).replace(",", " ").split()

    universe = []
    for item in raw_items:
        normalized = _normalize_lookup_symbol(str(item or "").strip())
        if normalized and normalized not in universe:
            universe.append(normalized)
    return universe


def _apply_cross_section_score(
    rows: list[dict],
    source_key: str,
    target_key: str,
    *,
    invert: bool = False,
):
    valid_values = [
        float(row[source_key])
        for row in rows
        if isinstance(row.get(source_key), (int, float)) and np.isfinite(float(row[source_key]))
    ]
    mean_value = float(np.mean(valid_values)) if valid_values else 0.0
    std_value = float(np.std(valid_values, ddof=0)) if len(valid_values) >= 2 else 0.0

    for row in rows:
        value = row.get(source_key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)) or std_value <= 0:
            score = 0.0
        else:
            score = (float(value) - mean_value) / std_value
        row[target_key] = round((-score if invert else score), 4)


def _compute_liquidity_proxy(symbol: str, period: str = "6mo") -> tuple[float | None, float | None]:
    try:
        history = get_ticker(symbol, cache_level="daily").history(period=period, interval="1d")
    except Exception as exc:
        logger.debug(f"Liquidity proxy fetch failed for {symbol}: {exc}")
        return None, None
    if history.empty or "Close" not in history.columns or "Volume" not in history.columns:
        return None, None

    close = pd.to_numeric(history["Close"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    volume = pd.to_numeric(history["Volume"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    dollar_volume = (close * volume).dropna()
    if dollar_volume.empty:
        return None, None

    avg_dollar_volume_20d = float(dollar_volume.tail(20).mean())
    liquidity_proxy = float(math.log1p(avg_dollar_volume_20d)) if avg_dollar_volume_20d > 0 else None
    return liquidity_proxy, avg_dollar_volume_20d


def _infer_mean_reversion_edge(mean_reversion: dict) -> float | None:
    if not isinstance(mean_reversion, dict):
        return None
    zscore = _safe_float(mean_reversion.get("zscore"))
    if zscore is None:
        return None

    edge = -zscore
    if not mean_reversion.get("reversion_candidate"):
        edge *= 0.5
    half_life = _safe_float(mean_reversion.get("half_life_days"))
    if half_life is not None and half_life > 30:
        edge *= 0.7
    return round(float(edge), 4)


def _calibrate_candidate_forecast(row: dict, risk_state: str, portfolio_overlay: dict) -> dict:
    final_score = float(row.get("final_alpha_score") or 0.0)
    alpha_adjusted = float(row.get("alpha_adjusted") or 0.0)
    asset_type = str(row.get("asset_type") or "Unknown")
    ic_quality = str(row.get("alpha_ic_quality") or "unknown")
    reversion_candidate = bool(row.get("reversion_candidate"))
    half_life = _safe_float(row.get("mr_half_life_days"))

    ic_multiplier = {"strong": 1.0, "weak": 0.8, "noise": 0.55, "unknown": 0.65}.get(ic_quality, 0.65)
    positive_bias = (final_score >= 0) or (alpha_adjusted >= 0)
    if positive_bias:
        if str(risk_state).startswith("🟢"):
            regime_multiplier = 1.0
        elif str(risk_state).startswith("🟡"):
            regime_multiplier = 0.9
        elif str(risk_state).startswith("🔴"):
            regime_multiplier = 0.7
        else:
            regime_multiplier = 0.5
        drawdown_multiplier = float(portfolio_overlay.get("size_multiplier", 1.0))
    else:
        regime_multiplier = 1.0 if not str(risk_state).startswith("💀") else 1.05
        drawdown_multiplier = 1.0

    signal_strength = min(1.0, (abs(final_score) / 2.5) + (abs(alpha_adjusted) / 1.5))
    confidence = (0.35 + (0.35 * signal_strength)) * ic_multiplier * regime_multiplier * max(drawdown_multiplier, 0.5)
    if reversion_candidate:
        confidence *= 1.05
    forecast_confidence = round(float(max(0.1, min(0.95, confidence))), 4)

    base_return_bps = (final_score * 85.0) + (alpha_adjusted * 90.0)
    expected_return_bps = round(
        float(max(-250.0, min(250.0, base_return_bps * max(forecast_confidence, 0.35)))),
        1,
    )

    if reversion_candidate and half_life is not None:
        holding_horizon_days = int(max(3, min(15, round(half_life))))
    elif asset_type == "Value_Holding":
        holding_horizon_days = 20 if expected_return_bps >= 0 else 10
    elif asset_type == "Macro_Hedge":
        holding_horizon_days = 10
    else:
        holding_horizon_days = 5 if abs(expected_return_bps) >= 60 else 8

    if expected_return_bps >= 80:
        signal_strength_label = "high_conviction_long"
    elif expected_return_bps >= 20:
        signal_strength_label = "long_bias"
    elif expected_return_bps <= -80:
        signal_strength_label = "avoid_or_short_bias"
    elif expected_return_bps <= -20:
        signal_strength_label = "underweight_bias"
    else:
        signal_strength_label = "neutral"

    return {
        "expected_return_bps": expected_return_bps,
        "forecast_confidence": forecast_confidence,
        "holding_horizon_days": holding_horizon_days,
        "forecast_label": signal_strength_label,
    }


def _seed_candidate_beta_series_cache(portfolio_module, benchmark: str, period: str) -> dict[tuple[str, str], pd.Series]:
    try:
        benchmark_symbol, benchmark_returns, benchmark_error = portfolio_module._load_daily_return_series(
            benchmark,
            period=period,
            series_cache={},
        )
    except Exception as exc:
        logger.debug(f"Candidate alpha panel benchmark prefetch failed: {exc}")
        return {}

    if benchmark_error or not benchmark_symbol or benchmark_returns.empty:
        return {}
    return {(benchmark_symbol, period): benchmark_returns.copy()}


def _compute_candidate_alpha_row(
    symbol: str,
    *,
    risk_snapshot: dict,
    portfolio_overlay: dict,
    risk_state: str,
    benchmark: str,
    period: str,
    horizon_days: int,
    lookback_signals: int,
    portfolio_module,
    router_module,
    beta_series_seed: dict[tuple[str, str], pd.Series],
) -> dict:
    profile = get_asset_profile(symbol)
    nlp_data = router_module.fetch_nlp_alpha(symbol)
    ic_payload = compute_nlp_signal_ic(
        symbol,
        horizon_days=horizon_days,
        lookback_signals=lookback_signals,
    )
    alpha_overlay = router_module._build_alpha_confidence_overlay(
        symbol,
        nlp_data,
        risk_snapshot=risk_snapshot,
        portfolio_overlay=portfolio_overlay,
        ic_payload=ic_payload,
    )
    factor = compute_factor_snapshot(symbol)
    technical_snapshot = {}
    try:
        technical_snapshot = build_technical_snapshot(symbol)
    except Exception as exc:
        logger.debug(f"Candidate alpha panel technical snapshot failed for {symbol}: {exc}")
    mean_reversion = technical_snapshot.get("mean_reversion", {}) if isinstance(technical_snapshot, dict) else {}

    local_beta_cache = {key: value.copy() for key, value in beta_series_seed.items()}
    beta_payload = portfolio_module.compute_portfolio_beta_attribution(
        {symbol: 1.0},
        benchmark=benchmark,
        period=period,
        series_cache=local_beta_cache,
    )
    beta_row = next(iter(beta_payload.get("positions", {}).values()), {}) if not beta_payload.get("error") else {}
    liquidity_proxy, avg_dollar_volume_20d = _compute_liquidity_proxy(symbol, period=period)

    ic_mean = ic_payload.get("ic_rolling_mean")
    beta_value = beta_row.get("beta")
    return {
        "symbol": symbol,
        "asset_type": profile.get("asset_type", "Unknown"),
        "sector": profile.get("sector", "Unknown"),
        "industry": profile.get("industry", "Unknown"),
        "alpha_raw": _safe_float(nlp_data.get("nlp_alpha")),
        "alpha_adjusted": _safe_float(alpha_overlay.get("effective_alpha")),
        "alpha_scale": _safe_float(alpha_overlay.get("combined_multiplier")),
        "alpha_ic_quality": ic_payload.get("signal_quality", "unknown"),
        "alpha_ic_mean": _safe_float(ic_mean),
        "directionality": ic_payload.get("directionality", "undetermined"),
        "momentum_12_1": _safe_float(factor.get("momentum_12_1")),
        "reversal_1m": _safe_float(factor.get("reversal_1m")),
        "quality_composite": _safe_float(factor.get("quality_raw")),
        "earnings_yield": _safe_float(factor.get("earnings_yield")),
        "book_price": _safe_float(factor.get("book_price")),
        "mr_zscore": _safe_float(mean_reversion.get("zscore")),
        "mr_half_life_days": _safe_float(mean_reversion.get("half_life_days")),
        "reversion_candidate": bool(mean_reversion.get("reversion_candidate")),
        "mean_reversion_edge": _infer_mean_reversion_edge(mean_reversion),
        "beta": _safe_float(beta_value),
        "idio_vol": _safe_float(beta_row.get("idio_vol")),
        "liquidity_proxy": _safe_float(liquidity_proxy),
        "avg_dollar_volume_20d": round(avg_dollar_volume_20d, 2) if avg_dollar_volume_20d is not None else None,
        "beta_penalty_raw": max(0.0, float(beta_value) - 1.0) if isinstance(beta_value, (int, float)) else None,
        "risk_state": risk_state,
        "portfolio_trade_mode": portfolio_overlay.get("trade_mode_label"),
    }


def compute_candidate_alpha_panel(
    symbols: str | list[str] | None = None,
    benchmark: str = "SPY",
    period: str = "6mo",
    horizon_days: int = 5,
    lookback_signals: int = 120,
) -> dict:
    universe = _parse_candidate_universe(symbols)
    if not universe:
        return {"error": "沒有可用候選股可建立 alpha panel。"}

    import engine_portfolio as portfolio
    import engine_router as router
    import engine_risk as risk_engine

    try:
        risk_snapshot = risk_engine.get_global_risk_snapshot()
    except Exception as exc:
        logger.debug(f"Candidate alpha panel risk snapshot failed: {exc}")
        risk_snapshot = {}
    try:
        portfolio_overlay = portfolio.compute_portfolio_risk_overlay(benchmark=benchmark, period=period)
    except Exception as exc:
        logger.debug(f"Candidate alpha panel portfolio overlay failed: {exc}")
        portfolio_overlay = {}

    risk_state = str(risk_snapshot.get("state") or "🟡 整理")
    beta_series_seed = _seed_candidate_beta_series_cache(portfolio, benchmark, period)

    if len(universe) == 1:
        rows = [
            _compute_candidate_alpha_row(
                universe[0],
                risk_snapshot=risk_snapshot,
                portfolio_overlay=portfolio_overlay,
                risk_state=risk_state,
                benchmark=benchmark,
                period=period,
                horizon_days=horizon_days,
                lookback_signals=lookback_signals,
                portfolio_module=portfolio,
                router_module=router,
                beta_series_seed=beta_series_seed,
            )
        ]
    else:
        rows = []
        max_workers = min(len(universe), _CANDIDATE_PANEL_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _compute_candidate_alpha_row,
                    symbol,
                    risk_snapshot=risk_snapshot,
                    portfolio_overlay=portfolio_overlay,
                    risk_state=risk_state,
                    benchmark=benchmark,
                    period=period,
                    horizon_days=horizon_days,
                    lookback_signals=lookback_signals,
                    portfolio_module=portfolio,
                    router_module=router,
                    beta_series_seed=beta_series_seed,
                ): symbol
                for symbol in universe
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"Candidate alpha panel failed for {symbol}: {exc}") from exc

    for source_key, target_key, invert in (
        ("alpha_adjusted", "alpha_score_cs", False),
        ("momentum_12_1", "momentum_score_cs", False),
        ("quality_composite", "quality_score_cs", False),
        ("mean_reversion_edge", "mean_reversion_score_cs", False),
        ("alpha_ic_mean", "ic_score_cs", False),
        ("beta_penalty_raw", "beta_penalty_cs", True),
        ("idio_vol", "idio_vol_score_cs", True),
        ("liquidity_proxy", "liquidity_score_cs", False),
    ):
        _apply_cross_section_score(rows, source_key, target_key, invert=invert)

    for row in rows:
        final_alpha_score = (
            (row.get("alpha_score_cs", 0.0) * 0.35)
            + (row.get("momentum_score_cs", 0.0) * 0.15)
            + (row.get("quality_score_cs", 0.0) * 0.10)
            + (row.get("mean_reversion_score_cs", 0.0) * 0.10)
            + (row.get("ic_score_cs", 0.0) * 0.10)
            + (row.get("liquidity_score_cs", 0.0) * 0.05)
            + (row.get("beta_penalty_cs", 0.0) * 0.10)
            + (row.get("idio_vol_score_cs", 0.0) * 0.05)
        )
        row["final_alpha_score"] = round(float(final_alpha_score), 4)
        row.update(_calibrate_candidate_forecast(row, risk_state, portfolio_overlay))

    rows.sort(
        key=lambda row: (
            float(row.get("final_alpha_score") or 0.0),
            float(row.get("expected_return_bps") or 0.0),
            float(row.get("forecast_confidence") or 0.0),
        ),
        reverse=True,
    )
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    return {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark": _normalize_lookup_symbol(benchmark),
        "risk_state": risk_state,
        "portfolio_trade_mode": portfolio_overlay.get("trade_mode_label"),
        "portfolio_gross_scale": portfolio_overlay.get("recommended_gross_scale"),
        "universe": universe,
        "rows": rows,
        "methodology": (
            "把現有 adjusted alpha、IC、單股因子、均值回歸、beta/idio vol、流動性做截面標準化後合成 ranking score，"
            "再依 risk state 與 portfolio governor 校準 expected return / confidence / horizon。"
        ),
    }


def build_candidate_alpha_report(
    symbols: str | list[str] | None = None,
    benchmark: str = "SPY",
    period: str = "6mo",
    horizon_days: int = 5,
    lookback_signals: int = 120,
) -> str:
    payload = compute_candidate_alpha_panel(
        symbols=symbols,
        benchmark=benchmark,
        period=period,
        horizon_days=horizon_days,
        lookback_signals=lookback_signals,
    )
    if payload.get("error"):
        return format_tool_error(f"❌ {payload['error']}", data_unavailable=True)

    report = "🧭 === Candidate Alpha Panel ===\n"
    report += (
        f"● Universe: {len(payload['rows'])} | Benchmark: {payload['benchmark']} | "
        f"Risk: {payload.get('risk_state', 'N/A')} | Portfolio Mode: {payload.get('portfolio_trade_mode', 'N/A')}\n"
    )
    if isinstance(payload.get("portfolio_gross_scale"), (int, float)):
        report += f"● Gross Scale: {payload['portfolio_gross_scale']:.2f}x\n"

    for row in payload["rows"][:10]:
        beta_text = f"{row['beta']:.2f}" if isinstance(row.get("beta"), (int, float)) else "N/A"
        conf_text = f"{row['forecast_confidence'] * 100:.0f}%" if isinstance(row.get("forecast_confidence"), (int, float)) else "N/A"
        report += (
            f"{row['rank']}. {row['symbol']}: Score {row['final_alpha_score']:+.2f} | "
            f"ER {row['expected_return_bps']:+.0f}bps | Conf {conf_text} | "
            f"Horizon {row['holding_horizon_days']}D | β {beta_text} | "
            f"IC {row.get('alpha_ic_quality', 'unknown')} | Sector {row.get('sector', 'Unknown')}\n"
        )

    if len(payload["rows"]) > 10:
        report += f"● 其餘 {len(payload['rows']) - 10} 檔未展開。\n"
    report += f"● 註記: {payload['methodology']}"
    return report


@tool()
def get_candidate_alpha_panel(
    symbols: str = "",
    benchmark: str = "SPY",
    period: str = "6mo",
    horizon_days: int = 5,
    lookback_signals: int = 120,
) -> str:
    """Builds a cross-sectional candidate ranking panel with calibrated return/confidence forecasts."""
    return build_candidate_alpha_report(symbols, benchmark, period, horizon_days, lookback_signals)


def build_movers_report() -> str:
    """Pure market-movers logic for direct callers and tests."""
    report = "🚀 === 市場異動排行榜 (Movers) ===\n"
    
    if FMP_KEY:
        try:
            # 獲取漲幅榜
            gainers = requests.get(f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={FMP_KEY}", timeout=5).json()
            # 獲取跌幅榜
            losers = requests.get(f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={FMP_KEY}", timeout=5).json()
            
            report += "【📈 領漲榜】\n"
            for s in gainers[:5]:
                report += f"  - {s['symbol']}: {s['price']} ({s['changesPercentage']}%)\n"
            
            report += "\n【📉 領跌榜】\n"
            for s in losers[:5]:
                report += f"  - {s['symbol']}: {s['price']} ({s['changesPercentage']}%)\n"
            return report
        except Exception as e:
            logger.warning(f"FMP Movers failed: {e}")

    # Fallback: YF 模擬 (批量掃描 S&P500/Nasdaq100 核心權值股 + 熱門股)
    report += "(數據來源: YF 大盤權值股批量掃描)\n"
    # 擴大監控池 (包含科技巨頭、半導體、金融、傳產、加密貨幣概念等)
    watch_list = [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
        'V', 'JPM', 'UNH', 'MA', 'PG', 'JNJ', 'HD', 'CVX', 'MRK', 'ABBV', 'COST',
        'AMD', 'NFLX', 'CRM', 'PEP', 'TMO', 'WMT', 'KO', 'DIS', 'CSCO', 'INTC', 'IBM',
        'COIN', 'MARA', 'MSTR', 'PLTR', 'SMCI', 'ARM', 'UBER', 'RST', 'QCOM', 'TXN'
    ]
    
    try:
        # 使用批量下載以提升效能
        data = get_download(watch_list, period="2d", group_by="ticker", progress=False)
        results = []
        
        # yf.download 回傳的欄位結構會依據 ticker 數量變化
        if len(watch_list) > 1:
            for s in watch_list:
                try:
                    if s in data.columns.levels[0]:
                        close_series = data[s]['Close']
                        if len(close_series) >= 2:
                            prev_close = close_series.iloc[-2]
                            curr_close = close_series.iloc[-1]
                            if pd.notna(prev_close) and pd.notna(curr_close) and prev_close > 0:
                                chg = ((curr_close / prev_close) - 1) * 100
                                results.append({'s': s, 'p': curr_close, 'c': chg})
                except Exception as scan_exc:
                    logger.debug(f"Mover scan failed for {s}: {scan_exc}")
                    continue
                
        # 排序
        sorted_gainers = sorted(results, key=lambda x: x['c'], reverse=True)
        sorted_losers = sorted(results, key=lambda x: x['c'])
        
        report += "【📈 領漲榜】\n"
        for r in sorted_gainers[:5]:
            report += f"  - {r['s']}: {r['p']:.2f} ({r['c']:+.2f}%)\n"
            
        report += "\n【📉 領跌榜】\n"
        for r in sorted_losers[:5]:
            report += f"  - {r['s']}: {r['p']:.2f} ({r['c']:+.2f}%)\n"
            
    except Exception as e:
        logger.error(f"Market movers scan failed: {e}")
        report += format_tool_error(f"掃描失敗: {e}", transient=True) + "\n"
        
    return report

@tool()
def get_market_movers() -> str:
    """
    Retrieves top gainers, losers, and most active stocks.
    Uses FMP API as primary and YFinance as fallback.
    """
    return build_movers_report()

def build_market_history_report(symbol: str, days: int = 14) -> str:
    """Pure market-history logic for direct callers and tests."""
    try:
        symbol = normalize_ticker(symbol)
        s = symbol.upper()
        if s.isdigit() and not s.endswith('.TW'): s += '.TW'
        hist = get_ticker(s).history(period="1mo").tail(days)
        if hist.empty:
            return format_tool_error(f"❌ {symbol} 無法取得歷史數據。", data_unavailable=True)
        
        report = f"【📅 {symbol} 最近 {len(hist)} 日歷史走勢】\n"
        # 反轉順序，由新到舊顯示
        for date, row in hist.iloc[::-1].iterrows():
            report += f"[{date.strftime('%m/%d')}] 收:{row['Close']:.2f} | 量:{int(row['Volume'])}\n"
        return report
    except Exception as e:
        logger.error(f"Market history fetch failed for {symbol}: {e}")
        return format_tool_error(f"❌ 歷史數據獲取失敗: {e}", data_unavailable=True)


@tool()
def get_market_history(symbol: str, days: int = 14) -> str:
    """
    Fetches historical closing prices and volumes for a specific stock (up to 1 month).
    """
    return build_market_history_report(symbol, days)

def get_market_calendar() -> str:
    """
    獲取市場日曆 (財報、重要經濟事件)。
    """
    report = "📅 === 市場關鍵日曆 (未來 7 天) ===\n"
    
    # 1. 重要財報 (Earnings)
    # yfinance 沒辦法直接查全市場日曆，我們用一個聰明的方法：
    # 查詢熱門股的 earnings_dates
    hot_tickers = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOG', 'META']
    events = []
    now = datetime.datetime.now()
    end_date = now + datetime.timedelta(days=7)
    
    for s in hot_tickers:
        try:
            t = get_ticker(s)
            dates = t.earnings_dates
            if dates is not None and not dates.empty:
                dates.index = dates.index.tz_localize(None)
                upcoming = dates[(dates.index >= now) & (dates.index <= end_date)]
                for d, _ in upcoming.iterrows():
                    events.append(f"  - {d.strftime('%m/%d')} | {s} 財報發布")
        except Exception as e:
            logger.debug(f"Market calendar fetch failed for {s}: {e}")
            continue

    if events:
        report += "【📣 重點財報】\n" + "\n".join(events) + "\n"
    else:
        report += "【📣 重點財報】近期無巨頭財報。\n"

    # 2. 宏觀日曆預留 (未來可對接 FRED 或 News)
    report += "\n【💡 提示】建議關注週五非農就業數據 (NFP) 或 CPI 發布。"
    
    return report

if __name__ == "__main__":
    # 自檢測試
    print(get_market_sentiment())
    print("\n")
    print(get_market_movers())
    print("\n")
    print(get_market_calendar())
