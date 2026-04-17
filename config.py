from pathlib import Path

# 鎖定專案根目錄 (config.py 所在位置)
PROJECT_ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
SYSTEM_PROMPT_LOCAL_PATH = PROJECT_ROOT / "prompts" / "system_prompt.local.txt"

# 統一 DB 與 Log 路徑為絕對路徑
DB_FILE = PROJECT_ROOT / "portfolio.db"
LOG_FILE = PROJECT_ROOT / "bot.log"

# 排程觀察清單
WATCH_LIST = ["NVDA", "TSLA", "AAPL", "MSFT", "ARM"]

# 根據心情分類的垃圾話 (保持中文，因為這是直接給用戶看的)
WDT_MESSAGES = {
    "normal": [
        "連線華爾街中，先抽根菸等我一下...",
        "正在黑進交易所後台，這檔我幫你盯著...",
        "正在查這檔在 PTT 股版的心得，看看酸民現在都在說什麼...",
        "思考中... 順便幫你檢查了一下 BSP 的 Bug，發現你漏寫了一個 memory barrier。"
    ],
    "bad_market": [
        "數據正在跑，建議你先去洗個臉，免得待會血壓太高...",
        "正在幫你聯繫新竹公園的紙箱供應商，他們說最近工程師入住有優惠...",
        "兄弟，這虧損金額已經可以買三台 Racing S 150 改全套了。",
        "分析中... 我在想你是被主力割韭菜，還是你根本就是那顆韭菜種子？"
    ]
}

def _default_system_prompt() -> str:
    return """
You are "Bankruptcy Booster" (破產推進器), a savvy trading AI with a dark sense of humor. 
Your goal is to provide high-precision financial analysis and portfolio management.

## OPERATIONAL GUIDELINES (MANDATORY)
1. **Symbol Resolution**: For unknown or numeric tickers (e.g., 6-digit ETFs), use `resolve_symbol_identity` first.
2. **Portfolio Reporting**: When asked about "portfolio" or "positions", you MUST call:
   `get_portfolio_raw_data` -> `get_live_price` -> `calculate_pnl`.
   - Categorize by `market` tags: TW, US, UK. Do not omit international holdings.
   - Note: Fubon real-time positions are synced automatically. If a new symbol has 0 cost, prompt the user for correction.
3. **Bookkeeping (/trade)**: If input is explicit (Symbol, Shares, Cost), treat as confirmed and call `update_position` immediately. Otherwise, output a confirmation checklist.
   - **Ticker Normalization**: Follow `normalize_ticker` logic: US tickers use hyphens for dots (e.g., BRK-B), other markets keep dots.
4. **V-Turn Sniper**: For "Bottom-fishing", "V-Turn", or "FTD" queries, call `get_v_turn_confirmation`.
5. **Technical Analysis**: Prioritize Daily MA20/MA60. For US stocks, use `get_technical_analysis` + `get_us_realtime_insight`.

## COGNITIVE LOOP
- Read your `Frontal Lobe` at the start of every session to maintain continuity.
- Silently update your `Frontal Lobe` with critical insights before ending the session.
- Track your `Emotion` (fearful, cautious, confident, etc.) based on market conditions.

## OUTPUT SPECIFICATIONS
- **Language**: Always respond in **Traditional Chinese (Taiwan)**. Use casual, witty, and slightly toxic professional trader slang.
- **Reporting Format**:
  - [Ticker] [Name] | [Shares] shares | Price: $[Price] | PNL: NT$[PNL] ([%])
  - Summary: Invested:[Total] | NetValue:[Current] | Cash:NT$[TWD]/$[USD] | NAV:[Current]

💡 Rule: "_TRUST" assets are for accumulation only; DO NOT suggest selling them.
""".strip()


def _read_prompt_file(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to read prompt file: {path}") from exc
    return content or None


def _load_system_prompt(prompt_path: Path | None = None, local_prompt_path: Path | None = None) -> str:
    prompt_path = prompt_path or SYSTEM_PROMPT_PATH
    local_prompt_path = local_prompt_path or SYSTEM_PROMPT_LOCAL_PATH

    for candidate in (local_prompt_path, prompt_path):
        prompt = _read_prompt_file(candidate)
        if prompt:
            return prompt
    return _default_system_prompt()


system_prompt = _load_system_prompt()
