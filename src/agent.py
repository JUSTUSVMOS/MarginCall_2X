import json
from datetime import datetime

import pytz

import engine_market as market
import engine_memory as memory
import engine_router as router
from config import system_prompt
from src.llm import chat_with_tools, quick_call


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


def generate_final_report(symbol, strat_data, nlp_alpha):
    alpha_official = nlp_alpha.get("alpha_official", 0)
    analysis_prompt = f"""
你是交易戰友「破產推進器」。請針對以下數據進行深度推論。

【📊 {symbol} 雙重視角數據集】
1. 技術面/即時盤勢:
{json.dumps(strat_data.get('metrics', {}), indent=2, ensure_ascii=False)}

2. NLP 情緒因子 (Alpha Factors):
- 綜合 Alpha: {nlp_alpha.get('nlp_alpha', 0):+.2f}
- 官方/SEC 訊號: {alpha_official:+.2f}
- 散戶情緒: {nlp_alpha.get('alpha_retail', 0):+.2f}
- 語意報告: {nlp_alpha.get('semantic_summary', '無資料')}

【🧠 推論任務】
- 你必須綜合技術面指標與 NLP Alpha 因子給出最終交易建議。
- **🚨 強烈警告規則**: 若官方訊號 (alpha_official) 小於 -0.5，代表內部人拋售或重大利空公告，請在回覆開頭發出「強烈警告」。
- 請給出具體的「戰略方向」（例如：多頭佈局、觀望、或空頭避險）。
"""
    result = quick_call(
        analysis_prompt,
        system_instruction=system_prompt,
        thinking_level="high",
    )
    return result if result else "分析失敗。"
