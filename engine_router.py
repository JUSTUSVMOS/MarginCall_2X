import os
import logging
import datetime
import time
import threading
import yfinance as yf
from yf_session import get_ticker, get_download
import json
import re  # 補回此行
from typing import Callable, Optional
import engine_market as market
import engine_risk as risk
import engine_fundamentals as fundamentals
from src.database import db_lock, get_connection

# 設定日誌
logger = logging.getLogger(__name__)

_nlp_ic_cache = {"entries": {}, "expiry": 1800}
_nlp_ic_cache_lock = threading.Lock()


def _safe_round(value, digits=4):
    return round(value, digits) if isinstance(value, (int, float)) else value


def _get_cached_nlp_signal_ic(symbol: str, horizon_days: int = 5, lookback_signals: int = 120) -> dict:
    cache_key = (market.normalize_ticker(symbol), int(horizon_days), int(lookback_signals))
    now = time.time()
    with _nlp_ic_cache_lock:
        cached = _nlp_ic_cache["entries"].get(cache_key)
        if cached and (now - cached["timestamp"] < _nlp_ic_cache["expiry"]):
            return dict(cached["payload"])

    payload = market.compute_nlp_signal_ic(symbol, horizon_days=horizon_days, lookback_signals=lookback_signals)
    with _nlp_ic_cache_lock:
        _nlp_ic_cache["entries"][cache_key] = {"timestamp": now, "payload": dict(payload)}
    return dict(payload)


def _build_alpha_confidence_overlay(
    symbol: str,
    nlp_data: dict,
    risk_snapshot: dict | None = None,
    portfolio_overlay: dict | None = None,
    ic_payload: dict | None = None,
) -> dict:
    raw_alpha = nlp_data.get("nlp_alpha")
    overlay = {
        "raw_alpha": raw_alpha,
        "effective_alpha": raw_alpha,
        "combined_multiplier": 1.0,
        "ic_multiplier": 1.0,
        "regime_multiplier": 1.0,
        "drawdown_multiplier": 1.0,
        "ic_quality": "unknown",
        "ic_rolling_mean": None,
        "directionality": "undetermined",
        "summary": "NLP alpha 尚未進行風控縮放。",
        "reasons": [],
    }
    if not isinstance(raw_alpha, (int, float)):
        return overlay

    ic_payload_data = {}
    ic_multiplier = 0.9
    reasons = []
    try:
        ic_payload_data = dict(ic_payload) if isinstance(ic_payload, dict) else _get_cached_nlp_signal_ic(
            symbol,
            horizon_days=5,
            lookback_signals=120,
        )
    except Exception as exc:
        logger.debug(f"Alpha IC overlay failed for {symbol}: {exc}")
        ic_payload_data = {"error": str(exc)}

    ic_quality = ic_payload_data.get("signal_quality", "unknown")
    directionality = ic_payload_data.get("directionality", "undetermined")
    ic_mean = ic_payload_data.get("ic_rolling_mean")

    if ic_payload_data.get("error"):
        ic_multiplier = 0.9
        reasons.append("IC 樣本不足，先維持輕度保守縮放")
    elif directionality == "negative":
        ic_multiplier = 0.35
        reasons.append("NLP IC 轉負，模型近期更像反向訊號")
    elif ic_quality == "strong":
        ic_multiplier = 1.0
        reasons.append("NLP IC 穩定為正，可維持原始 alpha")
    elif ic_quality == "weak":
        ic_multiplier = 0.75
        reasons.append("NLP IC 偏弱，alpha 降權")
    else:
        ic_multiplier = 0.55
        reasons.append("NLP IC 接近雜訊，alpha 明顯降權")

    state = (risk_snapshot or {}).get("state", "🟡 整理")
    if raw_alpha >= 0:
        if str(state).startswith("🟢"):
            regime_multiplier = 1.0
        elif str(state).startswith("🟡"):
            regime_multiplier = 0.85
        elif str(state).startswith("🔴"):
            regime_multiplier = 0.65
        else:
            regime_multiplier = 0.35
        if regime_multiplier < 1.0:
            reasons.append(f"市場 regime {state}，正向 alpha 進一步降權")
    else:
        if str(state).startswith("💀"):
            regime_multiplier = 1.10
            reasons.append("系統風險偏高，偏空/防守 alpha 可略提高權重")
        elif str(state).startswith("🔴"):
            regime_multiplier = 1.05
            reasons.append("警戒 regime 下，防守型 alpha 保持較高權重")
        else:
            regime_multiplier = 1.0

    drawdown_multiplier = 1.0
    if raw_alpha >= 0 and isinstance(portfolio_overlay, dict) and not portfolio_overlay.get("error"):
        drawdown_multiplier = float(portfolio_overlay.get("size_multiplier", 1.0))
        if drawdown_multiplier < 1.0:
            reasons.append(f"組合回撤節流 {portfolio_overlay.get('trade_mode_label', '')}，新增多單需縮倉")

    combined = max(0.0, min(1.25, ic_multiplier * regime_multiplier * drawdown_multiplier))
    effective_alpha = round(float(raw_alpha) * combined, 4)
    overlay.update(
        {
            "effective_alpha": effective_alpha,
            "combined_multiplier": round(combined, 4),
            "ic_multiplier": round(ic_multiplier, 4),
            "regime_multiplier": round(regime_multiplier, 4),
            "drawdown_multiplier": round(drawdown_multiplier, 4),
            "ic_quality": ic_quality,
            "ic_rolling_mean": _safe_round(ic_mean, 4),
            "directionality": directionality,
            "summary": (
                f"Raw {raw_alpha:+.2f} -> Adjusted {effective_alpha:+.2f} "
                f"(IC x{ic_multiplier:.2f} / Regime x{regime_multiplier:.2f} / DD x{drawdown_multiplier:.2f})"
            ),
            "reasons": reasons,
        }
    )
    return overlay


def _decode_nlp_summary_payload(summary_text):
    signal_pack = None
    semantic_summary = summary_text

    if not summary_text:
        return signal_pack, semantic_summary

    try:
        payload = json.loads(summary_text)
    except (TypeError, json.JSONDecodeError):
        return signal_pack, semantic_summary

    if isinstance(payload, dict):
        if "signal_pack" in payload or "semantic_summary" in payload:
            signal_pack = payload.get("signal_pack")
            semantic_summary = payload.get("semantic_summary")
        else:
            signal_pack = payload
            semantic_summary = None

    return signal_pack, semantic_summary

_alert_callback: Optional[Callable[[str], None]] = None


def set_alert_callback(cb: Optional[Callable[[str], None]]):
    global _alert_callback
    _alert_callback = cb


def _emit_alert(message: str):
    if _alert_callback is None:
        return
    try:
        _alert_callback(message)
    except Exception:
        logger.exception("Router alert callback failed")

# --- 0. 全域 Regex 配置 (預編譯提高效能) ---

def load_aliases():
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'aliases.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load aliases.json: {e}")
        return {}

# 俗稱與中文對照表 (Lookup Table)：將口語直接對應 yfinance 標準代號
TICKER_ALIASES = load_aliases()

# 提取對照表的 keys，組成 Regex 條件 (例如：APPLE|蘋果|TESLA...)
alias_pattern = '|'.join(map(re.escape, TICKER_ALIASES.keys())) if TICKER_ALIASES else '(?!)'

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
        from src.llm import quick_call, LIGHT_MODELS

        prompt = f"""Extract stock symbols or company names mentioned in the text and convert them to yfinance compatible ticker formats (e.g., TSLA, 2330.TW, BRK-B).
Return ONLY a valid JSON object in this exact format, with no markdown formatting or extra text:
{{"symbols": ["AAPL", "TSLA"]}}
If no symbols are found, return {{"symbols": []}}.

Text: {text}"""
        
        res_text = quick_call(prompt, models=LIGHT_MODELS)
        if res_text:
            # Clean markdown code blocks if AI wrapped the JSON
            cleaned_json = re.sub(r'^```(?:json)?\s*(.*?)\s*```$', r'\1', res_text.strip(), flags=re.DOTALL)
            
            try:
                data = json.loads(cleaned_json)
                extracted = data.get("symbols", [])
                if extracted:
                    symbols = [str(s).strip().upper() for s in extracted if str(s).strip()]
                    if symbols:
                        logger.info(f"LLM detected symbols via JSON: {symbols}")
                        return list(set(symbols))
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse LLM JSON response: {res_text} - Error: {je}")
                
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
    增加時間檢查：若資料超過 30 分鐘則視為過期。
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
            # 檢查時間新鮮度 (30 分鐘內)
            try:
                data_time = datetime.datetime.strptime(row[5], '%Y-%m-%d %H:%M:%S')
                if (datetime.datetime.now() - data_time).total_seconds() > 1800:
                    return {"error": "NLP data expired (over 30 mins). Needs refresh."}
            except Exception as e:
                logger.debug(f"Cache time check error: {e}")
                pass # 若格式不對則跳過時間檢查

            signal_pack, semantic_summary = _decode_nlp_summary_payload(row[4])
            nlp_alpha = _safe_round(row[0], 4)
            return {
                "nlp_alpha": nlp_alpha,
                "alpha_retail": _safe_round(row[1], 4),
                "alpha_macro": _safe_round(row[2], 4),
                "alpha_official": _safe_round(row[3], 4),
                "signal_pack": signal_pack,
                "semantic_summary": semantic_summary,
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
    except Exception as e:
        logger.debug(f"Failed to parse P/C ratio: {e}")
    return None

def fetch_strat_data(
    symbol: str,
    *,
    risk_snapshot: dict | None = None,
    portfolio_overlay: dict | None = None,
    alpha_ic_payload: dict | None = None,
) -> dict:
    """
    根據資產類型分流抓取數據，並實作 CVD & NLP 雙重熔斷中斷。
    警報交由可選 callback 發送，避免資料層直接依賴 Telegram。
    V2: NLP 分數保持唯讀，領先指標以獨立欄位輸出。
    """
    symbol = market.normalize_ticker(symbol)
    profile = market.get_asset_profile(symbol)
    asset_type = profile.get('asset_type', 'Unknown')
    
    # --- 核心升級：抓取 NLP Alpha 因子 ---
    nlp_data = fetch_nlp_alpha(symbol)
    if risk_snapshot is None:
        try:
            risk_snapshot = risk.get_global_risk_snapshot()
        except Exception as risk_exc:
            logger.debug(f"Risk snapshot load failed for {symbol}: {risk_exc}")
            risk_snapshot = {"error": str(risk_exc)}
    if portfolio_overlay is None:
        try:
            import engine_portfolio as portfolio
            portfolio_overlay = portfolio.compute_portfolio_risk_overlay()
        except Exception as overlay_exc:
            logger.debug(f"Portfolio overlay load failed for {symbol}: {overlay_exc}")
            portfolio_overlay = {"error": str(overlay_exc)}
    alpha_overlay = _build_alpha_confidence_overlay(
        symbol,
        nlp_data,
        risk_snapshot=risk_snapshot,
        portfolio_overlay=portfolio_overlay,
        ic_payload=alpha_ic_payload,
    )
    nlp_data = {**nlp_data, "alpha_overlay": alpha_overlay}
    
    # --- 核心升級：區分系統性 vs 個股風險 ---
    risk_type, excess = get_relative_move(symbol)
    
    data = {
        "symbol": symbol,
        "name": profile.get('name', 'Unknown'),
        "asset_type": asset_type,
        "currency": profile.get('currency', 'USD'),
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "nlp_insights": nlp_data, # 注入語意情緒
        "relative_move": {
            "risk_type": risk_type,
            "excess_return": round(excess, 4)
        },
        "portfolio_overlay": portfolio_overlay,
        "risk_snapshot": {
            "state": risk_snapshot.get("state"),
            "riskScore": risk_snapshot.get("riskScore"),
        },
        "leading_indicators": {
            "alpha_raw": _safe_round(nlp_data.get("nlp_alpha"), 4),
            "alpha_adjusted": alpha_overlay.get("effective_alpha"),
            "alpha_scale": alpha_overlay.get("combined_multiplier"),
            "alpha_governor": alpha_overlay.get("summary"),
            "alpha_ic_quality": alpha_overlay.get("ic_quality"),
            "alpha_ic_mean": alpha_overlay.get("ic_rolling_mean"),
            "portfolio_trade_mode": portfolio_overlay.get("trade_mode_label"),
            "portfolio_gross_scale": portfolio_overlay.get("recommended_gross_scale"),
            "risk_state": risk_snapshot.get("state"),
            "risk_score": risk_snapshot.get("riskScore"),
        },
        "metrics": {},
        "raw_profile": profile
    }

    try:
        composite_alpha = nlp_data.get("nlp_alpha", 0)
        signal_pack = nlp_data.get("signal_pack")
        if not isinstance(signal_pack, dict):
            signal_pack = None

        if isinstance(composite_alpha, (int, float)) and composite_alpha < -0.7:
            sec_facts = "; ".join(signal_pack.get("sec_detail", [])[:2]) if signal_pack else ""
            macro_facts = "; ".join(signal_pack.get("macro_detail", [])[:2]) if signal_pack else ""
            fact_summary = sec_facts or macro_facts or (nlp_data.get("semantic_summary", "無") or "無")
            divergence = signal_pack.get("divergence", "無") if signal_pack else "無"
            alert_prefix = "☢️ 【NLP 核心預警】" if signal_pack and signal_pack.get("nuclear_alert") else "🔴 【NLP 深度預警】"
            alert_msg = (
                f"{alert_prefix}{symbol} 綜合 Alpha = {composite_alpha:+.2f}\n"
                f"事實摘要: {fact_summary[:180]}\n"
                f"矛盾偵測: {divergence}"
            )
            _emit_alert(alert_msg)
            logger.warning(f"NLP Composite Alert for {symbol}: {composite_alpha}")

        if asset_type == 'Tech_Momentum':
            # 抓取 5分K CVD
            ticker = get_ticker(symbol, cache_level="live")
            df_5m = ticker.history(period="1d", interval="5m")
            cvd = risk.calculate_buying_pressure(df_5m)
            technical_snapshot = market.build_technical_snapshot(symbol)
            divergence = technical_snapshot.get("divergence", {})
            divergence_label = divergence.get("label", "⚪ 無明顯背離")
            has_bearish_divergence = bool(divergence.get("bearish_divergence"))
            mtf_rsi = technical_snapshot.get("mtf_rsi", {})
              
            # 【升級】改為分級警報：CVD 極端先告警；CVD + 熊背離才升級為硬體中斷
            if cvd < -0.9 and has_bearish_divergence:
                alert_msg = (
                    f"🚨 【硬體中斷】偵測到 {symbol} 恐慌性拋售 + 熊背離確認！\n"
                    f"當前 CVD: {cvd:.4f}\n"
                    f"RSI 結構: {divergence_label}\n"
                    "請立即檢查盤勢！"
                )
                _emit_alert(alert_msg)
                logger.warning(f"CVD + divergence hard alert triggered for {symbol}: {cvd}")
            elif cvd < -0.9:
                alert_msg = (
                    f"⚠️ 【盤中拋壓警報】{symbol} 出現極端賣壓，但尚未完成熊背離確認。\n"
                    f"當前 CVD: {cvd:.4f}\n"
                    f"RSI 結構: {divergence_label}"
                )
                _emit_alert(alert_msg)
                logger.warning(f"CVD soft alert triggered for {symbol}: {cvd}")

            # 抓取技術面 (RSI, MACD 等)
            tech_report = market.build_technical_report(symbol)
            live_insight = market.build_realtime_insight(symbol)
            pc_ratio = parse_pc_ratio(live_insight)
            try:
                option_vol_context = market.build_option_volatility_context(symbol)
            except Exception as vol_exc:
                logger.debug(f"Option volatility context failed for {symbol}: {vol_exc}")
                option_vol_context = {}

            pc_context = ""
            vrp = option_vol_context.get("vrp")
            vol_signal = option_vol_context.get("signal")
            if isinstance(pc_ratio, (int, float)) and pc_ratio > 1.5:
                if vol_signal == "🔥 恐慌定價":
                    pc_context = "🟡 P/C 偏高且權利金昂貴，偏向恐慌避險定價"
                elif isinstance(vrp, (int, float)) and vrp <= -3:
                    pc_context = "🔴 P/C 偏高但 VRP 為負，避險壓力不可輕忽"
                else:
                    pc_context = "⚠️ P/C 偏高，避險需求升溫"
            elif isinstance(pc_ratio, (int, float)) and pc_ratio < 0.5:
                pc_context = "🟢 P/C 偏低，短線情緒偏樂觀"
             
            data["metrics"] = {
                "cvd": round(cvd, 4),
                "technical_snapshot": technical_snapshot,
                "technical_analysis": tech_report,
                "live_insight": live_insight,
                "option_volatility": option_vol_context,
            }

            data["leading_indicators"].update({
                "cvd": round(cvd, 4),
                "pc_ratio": _safe_round(pc_ratio, 4),
                "rsi_divergence": divergence_label,
                "adx": technical_snapshot.get("adx", {}).get("value"),
                "trend_regime": technical_snapshot.get("adx", {}).get("trend_regime"),
                "obv_signal": technical_snapshot.get("obv", {}).get("signal"),
                "volatility_context": option_vol_context.get("summary"),
                "volatility_signal": option_vol_context.get("signal"),
                "pc_context": pc_context,
                "mtf_rsi_signal": mtf_rsi.get("signal_label"),
                "mtf_rsi_strength": mtf_rsi.get("confluence_strength"),
                "signal_reliability": mtf_rsi.get("signal_reliability", "NORMAL"),
                "cvd_signal": "🔴 拋壓" if cvd < -0.5 else "🟢 買壓" if cvd > 0.5 else "⚪ 中性",
                "pc_signal": (
                    "🔴 避險"
                    if isinstance(pc_ratio, (int, float)) and pc_ratio > 1.5
                    else "🟢 貪婪"
                    if isinstance(pc_ratio, (int, float)) and pc_ratio < 0.5
                    else "⚪ 中性"
                ),
            })

        elif asset_type == 'Value_Holding':
            # 抓取深度基本面 (趨勢、ROE、債務)
            fundamental_report = fundamentals.get_deep_fundamentals(symbol)
            data["metrics"] = {
                "fundamental_analysis": fundamental_report,
                "news": market.get_stock_news(symbol)
            }

        elif asset_type == 'Macro_Hedge':
            # 抓取價格趨勢 + 總經指標
            market_sentiment = market.build_sentiment_report()
            data["metrics"] = {
                "market_sentiment": market_sentiment,
                "news": market.get_stock_news(symbol),
                "price": market.fetch_live_price(symbol)
            }
        
        else:
            # 預設回傳基礎數據
            data["metrics"] = {
                "price": market.fetch_live_price(symbol),
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

    try:
        shared_risk_snapshot = risk.get_global_risk_snapshot()
    except Exception as exc:
        logger.debug(f"Shared risk snapshot load failed: {exc}")
        shared_risk_snapshot = {"error": str(exc)}
    try:
        import engine_portfolio as portfolio
        shared_portfolio_overlay = portfolio.compute_portfolio_risk_overlay()
    except Exception as exc:
        logger.debug(f"Shared portfolio overlay load failed: {exc}")
        shared_portfolio_overlay = {"error": str(exc)}
    
    context = "\n【🛡️ 策略路由系統已啟動】\n"
    for sym in symbols:
        data = fetch_strat_data(
            sym,
            risk_snapshot=shared_risk_snapshot,
            portfolio_overlay=shared_portfolio_overlay,
        )
        context += f"\n--- 標的: {sym} ({data.get('asset_type')}) ---\n"
        # 整合技術指標與語意情緒
        combined_metrics = {
            "market_data": data.get('metrics', {}),
            "leading_indicators": data.get('leading_indicators', {}),
            "relative_move": data.get('relative_move', {}),
            "portfolio_overlay": data.get('portfolio_overlay', {}),
            "risk_snapshot": data.get('risk_snapshot', {}),
            "nlp_sentiment_alpha": data.get('nlp_insights', {})
        }
        context += json.dumps(combined_metrics, ensure_ascii=False, separators=(',', ':'))

    return context
