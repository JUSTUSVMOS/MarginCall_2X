import json
from datetime import datetime

import pytz

import engine_market as market
import engine_memory as memory
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


def build_agent_prompt(user_text: str, system_prompt_override: str | None = None) -> str:
    # 恢復同步抓取深度數據的邏輯，讓 AI 在第一時間就能掌握所有數據指標。
    strat_context = router.get_strat_context(user_text)
    brain_context = memory.build_cognitive_context(max_age_minutes=180)
    brain_context += "\n\n## Frontal Lobe Write Contract\n"
    brain_context += memory.get_frontal_lobe_write_guide()
    brain_context += "\n(💡 系統最高指令：身為具備自我意識的交易 AI，你必須在每次分析開始時根據記憶保持思維連貫。若宏觀 regime 或你的交易結論有實質更新，請靜默呼叫 update_market_regime / update_frontal_lobe 寫回持久記憶。呼叫 update_frontal_lobe 時，必須遵守上面的四段式專業交易筆記格式。在回覆中請保持你一貫的犀利風格與深度戰略推論，不要只給冷冰冰的數據。)"
    return (system_prompt_override or system_prompt) + build_time_context() + strat_context + brain_context


def ask_agent(user_text, tools, chat_history=None, system_prompt_override=None, allow_retry=True):
    # 增加超時上限，確保後台抓取大盤、CVD 與技術面指標時不會斷線。
    return chat_with_tools(
        user_text=user_text,
        tools=tools,
        system_instruction=build_agent_prompt(user_text, system_prompt_override),
        history=chat_history,
        timeout_seconds=60,
        max_timeouts=1 if not allow_retry else 2,
        thinking_level="medium",
    )


def _format_fact_list(items):
    if not items:
        return "無"
    return "; ".join(str(item) for item in items if str(item).strip()) or "無"


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
- 請給出具體的「戰略方向」（例如：多頭佈局、觀望、或空頭避險）。
"""

    result = quick_call(
        analysis_prompt,
        system_instruction=system_prompt,
        thinking_level="high",
    )
    return result if result else "分析失敗。"
