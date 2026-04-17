import os
import datetime
import requests
import time
import pandas as pd
import numpy as np
import yfinance as yf
from yf_session import get_ticker, get_download

import logging
from engine_technical import IndicatorCalculator, analyze_obv_signal, summarize_divergence
from src.database import db_lock, get_connection
from src.symbols import normalize_ticker
from src.tools import format_tool_error, tool

logger = logging.getLogger(__name__)

FMP_KEY = os.getenv("FMP_API_KEY")

_fubon_provider = None

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


def build_technical_snapshot(symbol: str, interval: str = "1d") -> dict:
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
    }

def get_asset_profile(symbol: str) -> dict:
    """
    【核心】資產分類器：Stage 1 (規則) + Stage 2 (LLM Fallback)
    """
    symbol = normalize_ticker(symbol)
    
    # 1. 檢查 SQLite 快取
    with db_lock:
        conn = get_connection()
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
            info = get_ticker(symbol).info
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
        except Exception as e:
            logger.debug(f"Stage 1 info fetching failed for {symbol}: {e}")
            pass
    else:
        try:
            ticker = get_ticker(symbol)
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

    # Stage 2: LLM Fallback (透過統一管理器)
    if asset_type == "Unknown":
        logger.info(f"Starting Stage 2 LLM Classifier for {symbol}")
        try:
            from src.llm import quick_call, LIGHT_MODELS

            prompt = f"請將標的 {symbol} (Sector: {sector}, Industry: {industry}) 分類為以下三類之一：Tech_Momentum, Value_Holding, Macro_Hedge。\n僅回傳分類名稱。"
            result = quick_call(prompt, models=LIGHT_MODELS)
            if result:
                llm_type = result.strip()
                if llm_type in ['Tech_Momentum', 'Value_Holding', 'Macro_Hedge']:
                    asset_type = llm_type
        except Exception as e:
            logger.warning(f"Stage 2 LLM classification failed: {e}")

    # 3. 持久化到 SQLite
    with db_lock:
        conn = get_connection()
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

def build_sentiment_report() -> str:
    """Pure market sentiment logic for direct callers and tests."""
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
            ticker = get_ticker(symbol)
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
        
        return report
    except Exception as e:
        logger.error(f"Technical analysis failed for {symbol}: {e}")
        return format_tool_error(f"❌ 技術分析失敗: {e}", data_unavailable=True)

@tool()
def get_technical_analysis(symbol: str, interval: str = "1d") -> str:
    """
    Performs multi-indicator technical analysis (RSI, MACD, KDJ, Bollinger Bands).
    Provides a strategic outlook based on indicator alignment.
    """
    return build_technical_report(symbol, interval)

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
        'V', 'JPM', 'UNH', 'MA', 'PG', 'JNJ', 'HD', 'HD', 'CVX', 'MRK', 'ABBV', 'COST',
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
