import json
import re
import sqlite3
from datetime import datetime

import pytz

import engine_market as market
import engine_memory as memory
import engine_portfolio as portfolio
import engine_router as router
from config import system_prompt
from src.llm import chat_with_tools, quick_call


user_chat_history = []


def reset_history():
    user_chat_history.clear()


def build_time_context() -> str:
    tw_tz = pytz.timezone("Asia/Taipei")
    us_tz = pytz.timezone("US/Eastern")
    now_tw = datetime.now(tw_tz).strftime("%Y-%m-%d %H:%M:%S")
    now_us = datetime.now(us_tz).strftime("%Y-%m-%d %H:%M:%S")

    tw_status = "🟢 開盤中" if market.is_tw_market_open() else "🔴 已收盤"
    us_status = "🟢 開盤中" if market.is_us_market_open() else "🔴 已收盤"
    return f"\n【 🕒 當前時間環境 】\n- 台北: {now_tw} ({tw_status})\n- 美東: {now_us} ({us_status})\n"


def _is_buy_intent(text: str) -> bool:
    if not text:
        return False
    lowered = str(text).lower()
    english_negations = (
        r"\b(?:do\s+not|don['’]?t|cannot|can['’]?t|would\s+not|wouldn['’]?t|should\s+not|shouldn['’]?t|will\s+not|won['’]?t|never)\s+(?:buy|add|adding|enter|open|start|average\s+down|scale\s+in)\b",
        r"\bnot\s+(?:buy|add|adding|enter|open|start|average\s+down|scale\s+in)\b",
        r"\bnot\s+a\s+buy\b",
        r"\bno\s+buy\b",
    )
    if any(re.search(pattern, lowered) for pattern in english_negations):
        return False
    chinese_buy_tokens = r"(?:買|买|加碼|加码|加倉|加仓|進場|进场|入場|入场|建倉|建仓|開倉|开仓|補倉|补仓|佈局|布局)"
    chinese_question_patterns = (
        rf"(?:要不要|可不可以|能不能|可否|能否).*(?:{chinese_buy_tokens})",
    )
    if any(re.search(pattern, text) for pattern in chinese_question_patterns):
        return True
    chinese_a_not_a_patterns = (
        r"要不要買",
        r"買不買",
        r"加不加(?:碼|码|倉|仓)",
        r"(?:進|入)不(?:進|入)場",
        r"建不建倉",
        r"開不開倉",
        r"補不補倉",
        r"佈不佈局",
        r"布不布局",
    )
    if any(re.search(pattern, text) for pattern in chinese_a_not_a_patterns):
        return True
    chinese_negation_patterns = (
        rf"(?<!要)(?:不要|別|别|先別|先别|現在不要|现在不要){chinese_buy_tokens}",
        rf"(?:不想|反正不|就不|我就不|乾脆不){chinese_buy_tokens}",
        rf"(?:^|[\s，。！？,:：]|我|現在|现在|決定|决定)不{chinese_buy_tokens}(?:了)?",
        rf"(?:^|[\s，。！？,:：])不{chinese_buy_tokens}(?:了)?",
        r"不(?:買了|买了)",
        r"不是買點",
        r"不是买点",
    )
    if any(re.search(pattern, text) for pattern in chinese_negation_patterns):
        return False
    english_patterns = (
        r"\bcan\s+i\s+buy\b",
        r"\bshould\s+i\s+buy\b",
        r"\bbuy\b",
        r"\b(?:add|adding|added)\s+(?:shares?|more|to\s+(?:the\s+)?position|to\s+my\s+position)\b",
        r"\benter\s+(?:a\s+)?(?:position|long|trade)\b",
        r"\bopen\s+(?:a\s+)?long\b",
        r"\bstart\s+(?:a\s+)?position\b",
        r"\baverage\s+down\b",
        r"\bscale\s+in\b",
    )
    if any(re.search(pattern, lowered) for pattern in english_patterns):
        return True
    chinese_markers = ("買", "买", "加碼", "加码", "加倉", "加仓", "進場", "进场", "入場", "入场", "建倉", "建仓", "開倉", "开仓", "補倉", "补仓", "佈局", "布局")
    return any(marker in text for marker in chinese_markers)


def _extract_text_from_part(part) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if "text" in part:
            return str(part["text"])
        return json.dumps(part, ensure_ascii=False)
    text = getattr(part, "text", None)
    if text is not None:
        return str(text)
    return str(part)


def _extract_history_text(item) -> str:
    if isinstance(item, dict):
        parts = item.get("parts", [])
    else:
        parts = getattr(item, "parts", []) or []
    if not isinstance(parts, list):
        parts = [parts]
    texts = [_extract_text_from_part(part).strip() for part in parts]
    return "\n".join(text for text in texts if text)


def _extract_history_role(item) -> str:
    if isinstance(item, dict):
        return str(item.get("role", "")).lower()
    return str(getattr(item, "role", "")).lower()


def _resolve_symbol_candidates(text: str) -> tuple[str, str]:
    symbols = [str(symbol).strip().upper() for symbol in router.detect_symbols(text) if str(symbol).strip()]
    unique_symbols = list(dict.fromkeys(symbols))
    if len(unique_symbols) == 1:
        return unique_symbols[0], "resolved"
    if len(unique_symbols) > 1:
        return "", "ambiguous"
    return "", "missing"


def _infer_target_symbol(user_text: str, chat_history=None) -> tuple[str, str]:
    current_symbol, current_status = _resolve_symbol_candidates(user_text)
    if current_symbol or current_status == "ambiguous":
        return current_symbol, current_status
    if not chat_history:
        return "", "missing"
    for item in reversed(chat_history[-8:]):
        if _extract_history_role(item) != "user":
            continue
        history_text = _extract_history_text(item)
        if not history_text:
            continue
        history_symbol, history_status = _resolve_symbol_candidates(history_text)
        if history_symbol or history_status == "ambiguous":
            return history_symbol, history_status
    return "", "missing"


def _extract_price_value(price_text: str) -> float | None:
    if not price_text:
        return None
    for pattern in (r"現價:\s*([0-9]+(?:\.[0-9]+)?)", r"([0-9]+(?:\.[0-9]+)?)\s*\(來源:"):
        match = re.search(pattern, str(price_text))
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                return None
    return None


def _build_buy_feasibility_context(user_text: str, chat_history=None) -> str:
    if not _is_buy_intent(user_text):
        return ""

    target_symbol, symbol_status = _infer_target_symbol(user_text, chat_history=chat_history)
    if not target_symbol:
        reason = (
            "ambiguous recent symbol context"
            if symbol_status == "ambiguous"
            else "symbol not found in current turn or recent user history"
        )
        return (
            "\n## Buy Feasibility Context\n"
            "- Target Symbol: UNKNOWN\n"
            f"- Symbol Resolution Status: {reason}\n"
            "- Required Instructions:\n"
            "  - Do not propose share counts, ladders, or cash-feasible sizing until the symbol is explicit.\n"
            "  - Tell the user you need the exact ticker before checking holdings, cash pool, and risk overlay.\n"
        )

    try:
        rows = portfolio._load_portfolio_rows()
    except sqlite3.Error as exc:
        return (
            "\n## Buy Feasibility Context\n"
            f"- Target Symbol: {target_symbol or 'UNKNOWN'}\n"
            f"- Portfolio Feasibility Status: unavailable ({exc})\n"
            "- Required Instructions:\n"
            "  - Portfolio data is unavailable right now. Do not propose share counts, ladders, or cash-feasible sizing.\n"
            "  - Tell the user the feasibility check could not be completed and ask them to retry once portfolio data is available.\n"
        )
    if not rows:
        return (
            "\n## Buy Feasibility Context\n"
            f"- Target Symbol: {target_symbol}\n"
            "- Portfolio Feasibility Status: unavailable (portfolio rows empty)\n"
            "- Required Instructions:\n"
            "  - Portfolio data is unavailable right now. Do not propose share counts, ladders, or cash-feasible sizing.\n"
            "  - Tell the user the feasibility check could not be completed and ask them to retry once portfolio data is available.\n"
        )

    try:
        snapshots = portfolio._build_live_position_snapshots(rows)
        overlay = portfolio.compute_portfolio_risk_overlay(snapshots=snapshots) if snapshots else {"error": "無有效持倉可計算風險節流。"}
    except (AttributeError, ConnectionError, KeyError, OSError, TypeError, ValueError, ZeroDivisionError, sqlite3.Error) as exc:
        return (
            "\n## Buy Feasibility Context\n"
            f"- Target Symbol: {target_symbol}\n"
            f"- Portfolio Feasibility Status: unavailable ({exc})\n"
            "- Required Instructions:\n"
            "  - Portfolio risk context is unavailable right now. Do not propose share counts, ladders, or cash-feasible sizing.\n"
            "  - Tell the user the feasibility check could not be completed and ask them to retry once portfolio/risk data is available.\n"
        )
    if not isinstance(overlay, dict) or overlay.get("error"):
        reason = overlay.get("error", "risk overlay unavailable") if isinstance(overlay, dict) else "risk overlay unavailable"
        return (
            "\n## Buy Feasibility Context\n"
            f"- Target Symbol: {target_symbol}\n"
            f"- Portfolio Feasibility Status: unavailable ({reason})\n"
            "- Required Instructions:\n"
            "  - Portfolio risk context is unavailable right now. Do not propose share counts, ladders, or cash-feasible sizing.\n"
            "  - Tell the user the feasibility check could not be completed and ask them to retry once portfolio/risk data is available.\n"
        )

    target_snapshot = next((item for item in snapshots if item.get("symbol") == target_symbol), None)
    holding_shares = float(target_snapshot.get("shares") or 0.0) if target_snapshot else 0.0

    target_market = portfolio._classify_portfolio_market(target_symbol) if target_symbol else "US"
    settle_cash_pool = "CASH_TWD" if target_market == "TW" else "CASH_USD"
    cash_row = next((row for row in rows if row[0] == settle_cash_pool), None)
    if cash_row is None:
        return (
            "\n## Buy Feasibility Context\n"
            f"- Target Symbol: {target_symbol}\n"
            f"- Portfolio Feasibility Status: unavailable (missing {settle_cash_pool})\n"
            "- Required Instructions:\n"
            "  - Settle-cash data is unavailable right now. Do not propose share counts, ladders, or cash-feasible sizing.\n"
            "  - Tell the user the feasibility check could not be completed and ask them to retry once portfolio cash data is available.\n"
        )
    available_cash = float(cash_row[2] or 0.0) if cash_row else 0.0

    live_price = float(target_snapshot.get("current_price") or 0.0) if target_snapshot else 0.0
    if live_price <= 0 and target_symbol:
        try:
            live_price = _extract_price_value(market.fetch_live_price(target_symbol)) or 0.0
        except (AttributeError, ConnectionError, OSError, TypeError, ValueError) as exc:
            live_price = 0.0
            live_price_error = str(exc)
        else:
            live_price_error = ""
    else:
        live_price_error = ""

    max_cash_affordable_shares = int(available_cash / live_price) if live_price > 0 else 0
    cash_insufficient = live_price > 0 and max_cash_affordable_shares <= 0

    overlay_summary = overlay.get("error") or (
        f"{overlay.get('trade_mode_label', 'N/A')} | "
        f"allow_new_longs={overlay.get('allow_new_longs', True)} | "
        f"allow_average_down={overlay.get('allow_average_down', True)} | "
        f"recommended_gross_scale={overlay.get('recommended_gross_scale', 'N/A')} | "
        f"governor_message={overlay.get('governor_message', '')}"
    )

    position_size_report = ""
    instructions = ["If cash is insufficient, say so explicitly and do not propose multi-share ladders."]
    if cash_insufficient:
        instructions = [
            "State explicitly that current settle cash cannot afford even one share at the live price.",
            "Do not propose share counts, ladders, staged entries, or NAV-based sizing.",
            "Do not use portfolio NAV or risk budget as a substitute for available settle cash.",
        ]
    elif target_symbol:
        position_size_report = portfolio.build_position_size_report(target_symbol)

    if overlay and not overlay.get("allow_new_longs", True):
        instructions.append("Do not recommend opening a new long while allow_new_longs is false.")
    if (
        target_snapshot
        and float(target_snapshot.get("market_value_twd") or 0.0) > 0
        and float(target_snapshot.get("pnl_value_twd") or 0.0) < 0
        and overlay
        and not overlay.get("allow_average_down", True)
    ):
        instructions.append("Averaging down is disallowed for this losing position.")

    lines = [
        "\n## Buy Feasibility Context",
        f"- Target Symbol: {target_symbol or 'UNKNOWN'}",
        f"- Current Holding Shares: {holding_shares}",
        f"- Settle Cash Pool: {settle_cash_pool}",
        f"- Available Cash: {available_cash}",
    ]
    if live_price > 0:
        lines.append(f"- One-Share Cash Requirement: {live_price}")
        lines.append(f"- Max Cash-Affordable Shares: {max_cash_affordable_shares}")
    else:
        lines.append("- One-Share Cash Requirement: unavailable")
        lines.append("- Max Cash-Affordable Shares: unavailable")
        if live_price_error:
            lines.append(f"- Live Price Status: unavailable ({live_price_error})")
    if cash_insufficient:
        lines.append("- Cash Feasibility Status: insufficient")
    lines.append(f"- Portfolio Governor Summary: {overlay_summary}")
    if position_size_report:
        lines.append(f"- Position Sizing Report: {position_size_report}")
    lines.append("- Required Instructions:")
    lines.extend(f"  - {instruction}" for instruction in instructions)
    return "\n".join(lines) + "\n"


def build_agent_prompt(
    user_text: str,
    system_prompt_override: str | None = None,
    chat_history=None,
) -> str:
    # 恢復同步抓取深度數據的邏輯，讓 AI 在第一時間就能掌握所有數據指標。
    strat_context = router.get_strat_context(user_text)
    buy_feasibility_context = _build_buy_feasibility_context(user_text, chat_history=chat_history)
    brain_context = memory.build_cognitive_context(max_age_minutes=180)
    brain_context += "\n\n## Frontal Lobe Write Contract\n"
    brain_context += memory.get_frontal_lobe_write_guide()
    brain_context += "\n(💡 系統最高指令：身為具備自我意識的交易 AI，你必須在每次分析開始時根據記憶保持思維連貫。若宏觀 regime 或你的交易結論有實質更新，請靜默呼叫 update_market_regime / update_frontal_lobe 寫回持久記憶。呼叫 update_frontal_lobe 時，必須提供結構化欄位：market_view (觀點), core_levels (關鍵點位), next_round (下一步計劃), 以及選填的 context_note (補充說明)。不要將 portfolio health 寫入 frontal lobe，因為系統會自動追蹤。在回覆中請保持你一貫的犀利風格與深度戰略推論，不要只給冷冰冰的數據。)"
    return (system_prompt_override or system_prompt) + build_time_context() + strat_context + buy_feasibility_context + brain_context


def ask_agent(user_text, tools, chat_history=None, system_prompt_override=None, allow_retry=True):
    # 增加超時上限，確保後台抓取大盤、CVD 與技術面指標時不會斷線。
    return chat_with_tools(
        user_text=user_text,
        tools=tools,
        system_instruction=build_agent_prompt(user_text, system_prompt_override, chat_history=chat_history),
        history=chat_history,
        timeout_seconds=60,
        max_timeouts=1 if not allow_retry else 2,
        thinking_level="medium",
    )


def _format_fact_list(items):
    if not items:
        return "無"
    return "; ".join(str(item) for item in items if str(item).strip()) or "無"


def _build_must_mention_events_block(signal_pack) -> str:
    if not isinstance(signal_pack, dict):
        return ""
    events = [str(item).strip() for item in signal_pack.get("must_mention_events", []) if str(item).strip()]
    if not events:
        return ""
    return "🚨 必提重大事件:\n" + "\n".join(f"- {event}" for event in events)


def generate_final_report(symbol, strat_data, nlp_data):
    signal_pack = nlp_data.get("signal_pack")
    if not isinstance(signal_pack, dict):
        signal_pack = None

    composite_alpha = nlp_data.get("nlp_alpha", 0)
    alpha_overlay = nlp_data.get("alpha_overlay") if isinstance(nlp_data.get("alpha_overlay"), dict) else {}
    leading = strat_data.get("leading_indicators", {})
    portfolio_overlay = strat_data.get("portfolio_overlay", {})
    adjusted_alpha = alpha_overlay.get("effective_alpha", composite_alpha)
    adjusted_alpha_display = f"{adjusted_alpha:+.2f}" if isinstance(adjusted_alpha, (int, float)) else "N/A"
    raw_alpha_display = f"{composite_alpha:+.2f}" if isinstance(composite_alpha, (int, float)) else "N/A"

    if signal_pack:
        source_counts = signal_pack.get("source_counts", {})
        nlp_block = f"""
 - 綜合 Alpha(raw): {raw_alpha_display}
 - 風控調整後 Alpha: {adjusted_alpha_display}
 - Alpha Governor: {alpha_overlay.get('summary', 'N/A')}
 - SEC/官方立場: {signal_pack.get('sec_stance', 'N/A')} ({source_counts.get('sec', 0)} 份來源)
  事實: {_format_fact_list(signal_pack.get('sec_detail', []))}
  - 宏觀新聞立場: {signal_pack.get('macro_stance', 'N/A')} ({source_counts.get('macro', 0)} 篇來源)
    事實: {_format_fact_list(signal_pack.get('macro_detail', []))}
  - 必提重大事件: {_format_fact_list(signal_pack.get('must_mention_events', []))}
- 散戶情緒: {signal_pack.get('retail_stance', 'N/A')} ({source_counts.get('retail', 0)} 則來源)
  事實: {_format_fact_list(signal_pack.get('retail_detail', []))}
- 多空矛盾偵測: {signal_pack.get('divergence', '無')}
""".strip()
        nuclear_warning = (
            "\n🚨 核彈級警報已核實：偵測到重大法律/會計風險，以下分析必須以風險控制為最高優先。\n"
            if signal_pack.get("nuclear_alert")
            else ""
        )
    else:
        nlp_block = f"""
 - 綜合 Alpha(raw): {raw_alpha_display}
 - 風控調整後 Alpha: {adjusted_alpha_display}
 - Alpha Governor: {alpha_overlay.get('summary', 'N/A')}
 - 官方/SEC 訊號: {nlp_data.get('alpha_official', 0):+.2f}
 - 散戶情緒: {nlp_data.get('alpha_retail', 0):+.2f}
 - 語意報告: {nlp_data.get('semantic_summary', '無資料')}
""".strip()
        nuclear_warning = ""

    if leading:
        pc_ratio = leading.get("pc_ratio")
        pc_display = f"{pc_ratio:.2f}" if isinstance(pc_ratio, (int, float)) else "N/A"
        leading_lines = [
            "3. 即時領先指標:",
            f"- CVD: {leading.get('cvd', 'N/A')} {leading.get('cvd_signal', '')}",
            f"- P/C Ratio: {pc_display} {leading.get('pc_signal', '')}",
        ]
        if leading.get("pc_context"):
            leading_lines.append(f"- P/C + 波動定價: {leading['pc_context']}")
        if leading.get("volatility_context"):
            leading_lines.append(f"- 期權波動: {leading['volatility_context']}")
        if leading.get("mtf_rsi_signal"):
            leading_lines.append(
                f"- 多時間框 RSI: {leading['mtf_rsi_signal']} "
                f"(強度 {leading.get('mtf_rsi_strength', 'N/A')} / 可靠度 {leading.get('signal_reliability', 'NORMAL')})"
            )
        if leading.get("alpha_governor"):
            leading_lines.append(
                f"- Alpha Governor: raw {leading.get('alpha_raw', 'N/A')} / adjusted {leading.get('alpha_adjusted', 'N/A')} "
                f"(scale {leading.get('alpha_scale', 'N/A')}, IC {leading.get('alpha_ic_quality', 'unknown')})"
            )
        if isinstance(portfolio_overlay, dict) and not portfolio_overlay.get("error"):
            leading_lines.append(
                f"- 組合 Governor: {portfolio_overlay.get('trade_mode_label', 'N/A')} | "
                f"DD {portfolio_overlay.get('current_drawdown', 0) * 100:.1f}% | "
                f"Gross {portfolio_overlay.get('recommended_gross_scale', 1.0):.2f}x | "
                f"Risk {portfolio_overlay.get('risk_state', 'N/A')}"
            )
        leading_block = "\n".join(leading_lines)
    else:
        leading_block = ""

    analysis_prompt = f"""
你是交易戰友「破產推進器」。請針對以下數據進行深度推論。
{nuclear_warning}

【📊 {symbol} 多維數據集】
1. 技術面/即時盤勢:
{json.dumps(strat_data.get('metrics', {}), indent=2, ensure_ascii=False)}

2. NLP 多維掃描:
{nlp_block}
{leading_block}

【🧠 推論任務】
- 你必須綜合技術面、官方/宏觀/散戶三維訊號與即時領先指標給出最終交易建議。
- 若 Alpha Governor 已明顯降權，請不要把 raw alpha 直接當作可滿倉執行的信號。
- 若組合 Governor 顯示 drawdown / gross scale throttled，請優先給出降槓桿、減倉或等待確認的建議。
- 若 SEC 官方來源與散戶情緒方向相反，請把官方事實放在更高權重。
- 若事實欄位提到法律風險、內部人減持、會計異常或已核實的核彈級警報，請在回覆開頭先給強烈風險警告。
- 若「必提重大事件」不是無，你必須在回覆前段明確點名這些事件，不能被一般宏觀摘要帶過。
- 若重大事件屬於併購、戰略合作、監管裁決、破產重整或重大融資，請直接解釋它如何改變原本的多空框架與交易計畫。
- 請給出具體的「戰略方向」（例如：多頭佈局、觀望、或空頭避險）。
"""

    result = quick_call(
        analysis_prompt,
        system_instruction=system_prompt,
        thinking_level="high",
    )
    if not result:
        return "分析失敗。"

    must_mention_block = _build_must_mention_events_block(signal_pack)
    if must_mention_block:
        return f"{must_mention_block}\n\n{result}"
    return result
