import logging
import os
import random
import re
import subprocess
import sys
import threading
import time

try:
    import telebot
    _TELEBOT_API_EXCEPTION = telebot.apihelper.ApiTelegramException
except ModuleNotFoundError as exc:
    telebot = None
    _TELEBOT_IMPORT_ERROR = exc

    class _TELEBOT_API_EXCEPTION(Exception):
        pass
else:
    _TELEBOT_IMPORT_ERROR = None

import engine_fundamentals as fundamentals
import engine_journal
import engine_market as market
import engine_portfolio as portfolio
import engine_technical as technical
import fubon
from config import PROJECT_ROOT, WDT_MESSAGES
from src.tools import format_tool_error, get_tools

try:
    import engine_risk as risk
except ModuleNotFoundError:
    risk = None


logger = logging.getLogger(__name__)


AUTHORIZED_USER_ID = None

# 先載入所有工具模組，再從共享 registry 取出可用工具。
_TOOL_MODULES = tuple(module for module in (portfolio, risk, fundamentals, technical, market, fubon, engine_journal) if module is not None)

bot = None
user_chat_history = []

MEMORY_WRITE_TOOL_NAMES = {
    "update_frontal_lobe",
    "update_emotion",
    "update_market_regime",
}
WRITE_TOOLS = get_tools("write")
FULL_AGENT_TOOLS = get_tools("all")
READ_ONLY_TOOLS = get_tools("read") + [
    tool_func for tool_func in WRITE_TOOLS if tool_func.__name__ in MEMORY_WRITE_TOOL_NAMES
]
_handlers_registered = False
_runtime_initialized = False
_v_turn_active = False
_TRADE_ACTION_PATTERNS = (
    ("sell", re.compile(r"(賣出|卖出|出場|减码|減碼|\bsell\b)", re.IGNORECASE)),
    ("set", re.compile(r"(校正|更正|調整|调整|\bset\b)", re.IGNORECASE)),
    ("buy", re.compile(r"(買入|买入|加碼|加码|補倉|补仓|進場|进场|\bbuy\b)", re.IGNORECASE)),
)
_TRADE_SHARES_PATTERN = re.compile(r"(?P<shares>\d+(?:\.\d+)?)\s*(?:股|shares?)", re.IGNORECASE)
_TRADE_PRICE_PATTERN = re.compile(r"(?:(?:^|[\s(])(?:@|at|price|價格|均價|成本)\s*\$?|\$)(?P<price>\d+(?:\.\d+)?)", re.IGNORECASE)
_TRADE_SYMBOL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _require_bot():
    if bot is None:
        raise RuntimeError("Bot 尚未初始化，請先呼叫 init_bot()。")
    return bot


def _require_telebot():
    if telebot is None:
        raise RuntimeError(f"telebot 未安裝，無法初始化 Telegram Bot：{_TELEBOT_IMPORT_ERROR}")
    return telebot


def init_bot():
    global AUTHORIZED_USER_ID, bot

    if bot is not None and AUTHORIZED_USER_ID is not None:
        return bot, AUTHORIZED_USER_ID

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    gemini_key = os.getenv("GEMINI_API_KEY")
    my_user_id = os.getenv("TELEGRAM_USER_ID")

    if not bot_token or not gemini_key or not my_user_id:
        raise ValueError("❌ .env 缺少必要 API KEY 或 TELEGRAM_USER_ID")

    AUTHORIZED_USER_ID = int(my_user_id)
    bot = _require_telebot().TeleBot(bot_token)
    initialize_bot_runtime()
    return bot, AUTHORIZED_USER_ID


def initialize_bot_runtime():
    global _runtime_initialized
    if _runtime_initialized:
        return

    bot_instance = _require_bot()
    fubon.init_fubon()
    market.set_fubon_provider(fubon)
    import engine_router as router

    def _deliver_router_alert(message: str):
        if AUTHORIZED_USER_ID is None:
            raise RuntimeError("AUTHORIZED_USER_ID 尚未初始化")
        bot_instance.send_message(AUTHORIZED_USER_ID, message)

    router.set_alert_callback(_deliver_router_alert)
    _runtime_initialized = True


def is_v_turn_active() -> bool:
    return _v_turn_active


def _is_authorized(message) -> bool:
    return AUTHORIZED_USER_ID is not None and getattr(getattr(message, "from_user", None), "id", None) == AUTHORIZED_USER_ID


def _send_or_edit(chat_id, text, message_id=None):
    bot_instance = _require_bot()
    for mode in ("Markdown", None):
        try:
            if message_id is not None:
                bot_instance.edit_message_text(chat_id=chat_id, message_id=message_id, text=text, parse_mode=mode)
            else:
                bot_instance.send_message(chat_id, text, parse_mode=mode)
            return
        except Exception as exc:
            if mode is None:
                raise
            logger.warning(f"Markdown delivery failed, retrying without parse mode: {exc}")


def _append_history_turn(user_text: str, assistant_text: str):
    from src.llm import compact_history

    chat_history = _get_user_chat_history()
    updated_history = list(chat_history) + [
        {"role": "user", "parts": [user_text]},
        {"role": "model", "parts": [assistant_text]},
    ]
    chat_history.clear()
    chat_history.extend(compact_history(updated_history))


def _get_agent_runtime():
    from src.agent import ask_agent, generate_final_report as agent_generate_final_report, reset_history, user_chat_history
    from src.llm import compact_history

    return ask_agent, agent_generate_final_report, reset_history, user_chat_history, compact_history


def _get_user_chat_history():
    try:
        from src.agent import user_chat_history as runtime_history
    except ModuleNotFoundError:
        return user_chat_history
    return runtime_history


def ask_agent(*args, **kwargs):
    real_ask_agent, _, _, _, _ = _get_agent_runtime()
    return real_ask_agent(*args, **kwargs)


def agent_generate_final_report(*args, **kwargs):
    _, real_generate_final_report, _, _, _ = _get_agent_runtime()
    return real_generate_final_report(*args, **kwargs)


def reset_history():
    _, _, real_reset_history, _, _ = _get_agent_runtime()
    return real_reset_history()


def _parse_trade_command(user_input: str):
    payload = user_input.strip()
    if not payload:
        return None

    action = "buy"
    normalized_text = payload
    for candidate_action, pattern in _TRADE_ACTION_PATTERNS:
        match = pattern.search(normalized_text)
        if not match:
            continue
        action = candidate_action
        normalized_text = pattern.sub(" ", normalized_text, count=1).strip()
        break

    shares_match = _TRADE_SHARES_PATTERN.search(normalized_text)
    if not shares_match:
        return None
    shares = float(shares_match.group("shares"))
    normalized_text = (
        normalized_text[:shares_match.start()] + " " + normalized_text[shares_match.end():]
    ).strip()

    price_match = _TRADE_PRICE_PATTERN.search(normalized_text)
    if price_match:
        price = float(price_match.group("price"))
        normalized_text = (
            normalized_text[:price_match.start()] + " " + normalized_text[price_match.end():]
        ).strip()
    else:
        bare_numbers = list(re.finditer(r"\d+(?:\.\d+)?", normalized_text))
        if not bare_numbers:
            return None
        price = float(bare_numbers[-1].group(0))
        normalized_text = (
            normalized_text[:bare_numbers[-1].start()] + " " + normalized_text[bare_numbers[-1].end():]
        ).strip()

    symbol_candidates = [
        token
        for token in _TRADE_SYMBOL_PATTERN.findall(normalized_text)
        if token.upper() not in {"USD", "TWD", "NTD", "BUY", "SELL", "SET"}
    ]
    if not symbol_candidates:
        return None

    return {
        "action": action,
        "symbol": symbol_candidates[0],
        "price": price,
        "shares": shares,
    }


def trigger_nlp_and_callback(symbol, chat_id=None, message_id=None):
    """
    非同步處理：啟動 GPU 運算，完成後主動推送報告。
    """

    def _run():
        try:
            bot_instance = _require_bot()
            logger.info(f"🚀 [情報局] 正在啟動深度收割: {symbol}")
            cmd = [sys.executable, os.path.join(str(PROJECT_ROOT), "nlp_worker.py"), symbol]

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                logger.error(f"❌ [情報局] {symbol} 超時被強制終止")
                if chat_id:
                    bot_instance.send_message(chat_id, f"❌ {symbol} 情報收割超時 (>5min)，請檢查 Ollama 是否正常運行。")
                return

            if process.returncode == 0:
                logger.info(f"✅ [情報局] {symbol} 收割完成")
                if chat_id and message_id:
                    bot_instance.edit_message_text(
                        f"✅ {symbol} 情報收割完畢，正在進行最終決策...",
                        chat_id=chat_id,
                        message_id=message_id,
                    )
                    import engine_router as router
                    strat_data = router.fetch_strat_data(symbol)
                    nlp_data = strat_data.get("nlp_insights", {})
                    generate_final_report(symbol, strat_data, nlp_data, chat_id, message_id)
            else:
                logger.error(f"❌ [情報局] {symbol} 失敗: {stderr[:200]}")
                if chat_id:
                    bot_instance.send_message(chat_id, f"❌ {symbol} 情報收割失敗，請檢查日誌。\n{stderr[:100]}")
        except Exception as exc:
            logger.error(f"🚨 [情報局] 異常: {exc}")
            if chat_id and bot is not None:
                bot.send_message(chat_id, f"🚨 系統異常: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def generate_final_report(symbol, strat_data, nlp_data, chat_id, message_id=None):
    final_text = agent_generate_final_report(symbol, strat_data, nlp_data)
    _send_or_edit(chat_id, final_text, message_id)


def send_pending_trade_followups():
    bot_instance = _require_bot()
    if AUTHORIZED_USER_ID is None:
        raise RuntimeError("AUTHORIZED_USER_ID 尚未初始化")

    followups = portfolio.claim_pending_trade_followups()
    sent_count = 0
    for index, followup in enumerate(followups):
        followup_id = int(followup["id"])
        try:
            prompt_text = portfolio.build_trade_followup_prompt(followup)
            bot_instance.send_message(AUTHORIZED_USER_ID, prompt_text)
            portfolio.mark_trade_followup_prompted(followup_id)
            sent_count += 1
        except Exception:
            portfolio.mark_trade_followup_pending(followup_id)
            for remaining_followup in followups[index + 1:]:
                portfolio.mark_trade_followup_pending(int(remaining_followup["id"]))
            raise
    return sent_count


def handle_deep_analysis(message):
    if not _is_authorized(message):
        return

    bot_instance = _require_bot()
    sent_msg = None

    try:
        import engine_router as router

        parts = message.text.strip().split()
        if len(parts) < 2:
            bot_instance.reply_to(message, "💡 用法: `/analyze <股票代號>`", parse_mode="Markdown")
            return

        symbol = parts[1].upper()
        sent_msg = bot_instance.reply_to(message, f"🔍 正在調閱 {symbol} 雙重視角戰報...\n(GPU 正在運算中，預計 1~2 分鐘，完成後主動通知您)")
        bot_instance.send_chat_action(message.chat.id, "typing")

        strat_data = router.fetch_strat_data(symbol)
        nlp_data = strat_data.get("nlp_insights", {})

        if "error" in nlp_data:
            trigger_nlp_and_callback(symbol, message.chat.id, sent_msg.message_id)
            return

        generate_final_report(symbol, strat_data, nlp_data, message.chat.id, sent_msg.message_id)
    except Exception as exc:
        logger.exception(f"Deep analysis failed for message: {getattr(message, 'text', '')}")
        error_text = format_tool_error(f"⚠️ 分析失敗: {exc}", transient=True)
        if sent_msg is not None:
            _send_or_edit(message.chat.id, error_text, sent_msg.message_id)
        else:
            bot_instance.reply_to(message, error_text)


def reset_memory(message):
    if not _is_authorized(message):
        return

    reset_history()
    _require_bot().reply_to(message, "🧹 推進器記憶體已排空！")


def handle_trade_command(message):
    if not _is_authorized(message):
        return

    bot_instance = _require_bot()

    user_input = message.text.replace("/trade", "").strip()
    if not user_input:
        bot_instance.reply_to(message, "📝 請輸入交易內容，例如：`/trade ONDS $9.8 2股`", parse_mode="Markdown")
        return

    trade_payload = _parse_trade_command(user_input)
    if not trade_payload:
        bot_instance.reply_to(
            message,
            "📝 `/trade` 會直接記帳，請提供明確成交資料，例如：`/trade ONDS $9.8 2股` 或 `/trade 賣出 ONDS $9.8 2股`",
            parse_mode="Markdown",
        )
        return

    sent_msg = bot_instance.reply_to(message, "✍️ 【記帳模式啟動】正在寫入帳本...")
    bot_instance.send_chat_action(message.chat.id, "typing")

    try:
        result = portfolio.execute_position_update(
            trade_payload["symbol"],
            trade_payload["price"],
            trade_payload["shares"],
            action=trade_payload["action"],
            sync_memory=True,
            enforce_pretrade_gate=False,
        )
        history_summary = (
            "[recent confirmed trade]\n"
            f"action={trade_payload['action']} symbol={trade_payload['symbol']} "
            f"price={trade_payload['price']:.4f} shares={trade_payload['shares']:.4f}\n"
            f"{result}"
        )
        _append_history_turn(f"/trade {user_input}", history_summary)
        _send_or_edit(message.chat.id, result, sent_msg.message_id)
    except Exception as exc:
        logger.exception(f"Trade command failed for message: {getattr(message, 'text', '')}")
        _send_or_edit(message.chat.id, format_tool_error(f"⚠️ 記帳失敗: {exc}", transient=True), sent_msg.message_id)


def toggle_v_turn(message):
    global _v_turn_active

    if not _is_authorized(message):
        return

    _v_turn_active = not _v_turn_active
    status = "🟢 已啟動" if _v_turn_active else "🔴 已關閉"
    _require_bot().reply_to(message, f"V 轉監控 {status}")


def handle_unknown_command(message):
    """
    攔截所有未定義的 / 指令，避免送往 AI 浪費 token。
    """
    if not _is_authorized(message):
        return

    help_text = (
        "❓ *未知指令*\n\n"
        "目前支援的指令如下：\n"
        "🔍 `/analyze <代號>` - 深度分析股票 (雙重視角戰報)\n"
        "📝 `/trade <symbol> $<price> <shares>股` - 直接記帳模式\n"
        "🧹 `/reset` - 清空對話記憶\n"
        "🟢 `/vturn` - 切換 V 轉監控狀態"
    )
    _require_bot().reply_to(message, help_text, parse_mode="Markdown")


def _maybe_handle_trade_followup_reply(message) -> bool:
    if not _is_authorized(message):
        return False

    user_text = (getattr(message, "text", "") or "").strip()
    if not user_text:
        return False

    followup = portfolio.get_latest_prompted_trade_followup()
    if not followup:
        return False

    resolution = portfolio.resolve_trade_followup_reply(int(followup["id"]), user_text)
    if resolution is None:
        return False

    _require_bot().reply_to(message, portfolio.format_trade_followup_confirmation(followup, resolution))
    return True


def handle_all_text(message):
    if not _is_authorized(message):
        return

    if _maybe_handle_trade_plan_reply(message):
        return
    if _maybe_handle_trade_followup_reply(message):
        return

    bot_instance = _require_bot()

    user_text = message.text
    mood = "bad_market" if any(word in user_text for word in ["損益", "賠", "慘"]) else "normal"
    sent_msg = bot_instance.reply_to(message, f"【推進器點火】\n{random.choice(WDT_MESSAGES[mood])}")
    bot_instance.send_chat_action(message.chat.id, "typing")

    try:
        final_text = ask_agent(user_text, tools=READ_ONLY_TOOLS, chat_history=_get_user_chat_history())
        _send_or_edit(message.chat.id, final_text, sent_msg.message_id)
    except Exception as exc:
        logger.exception(f"Chat handling failed for message: {user_text}")
        _send_or_edit(message.chat.id, format_tool_error(f"⚠️ 對話處理失敗: {exc}", transient=True), sent_msg.message_id)


def register_handlers():
    global _handlers_registered
    if _handlers_registered:
        return

    bot_instance = _require_bot()
    bot_instance.register_message_handler(handle_deep_analysis, commands=["analyze", "nlp"])
    bot_instance.register_message_handler(reset_memory, commands=["reset"])
    bot_instance.register_message_handler(handle_trade_command, commands=["trade"])
    bot_instance.register_message_handler(toggle_v_turn, commands=["vturn"])
    bot_instance.register_message_handler(handle_unknown_command, func=lambda message: message.text and message.text.startswith("/"))
    bot_instance.register_message_handler(handle_all_text, func=lambda message: True)
    _handlers_registered = True


def run_polling():
    bot_instance = _require_bot()
    while True:
        try:
            logger.info("📡 正在開啟 Infinity Polling...")
            bot_instance.infinity_polling(timeout=20, long_polling_timeout=10)
        except _TELEBOT_API_EXCEPTION as exc:
            if exc.error_code == 409:
                logger.error("🛑 偵測到 409 衝突：已有其他 Bot 實例在運行。將於 15 秒後重試...")
                time.sleep(15)
            else:
                logger.error(f"❌ Telegram API 異常: {exc}")
                time.sleep(5)
        except Exception as exc:
            logger.error(f"🚨 Polling 崩潰: {exc}")
            time.sleep(5)

def send_morning_briefing():
    try:
        bot_instance = _require_bot()
        import engine_briefing as briefing

        if AUTHORIZED_USER_ID is None:
            logger.error("AUTHORIZED_USER_ID 尚未初始化")
            return False

        bot_instance.send_message(AUTHORIZED_USER_ID, briefing.build_morning_briefing())
        return True
    except Exception as exc:
        logger.error(f"Morning briefing push failed: {exc}")
        return False

def send_pending_trade_plan_prompts():
    bot_instance = _require_bot()
    pending = portfolio.claim_pending_trade_plan_prompts()
    sent = 0
    for item in pending:
        if AUTHORIZED_USER_ID is None:
            portfolio.release_trade_plan_prompt(int(item["id"]))
            logger.error("AUTHORIZED_USER_ID 尚未初始化")
            return 0
        try:
            bot_instance.send_message(AUTHORIZED_USER_ID, portfolio.format_trade_plan_prompt(item))
        except Exception:
            portfolio.release_trade_plan_prompt(int(item["id"]))
            raise
        sent += 1
    return sent

def _maybe_handle_trade_plan_reply(message) -> bool:
    plan = portfolio.get_latest_prompted_trade_plan()
    if not plan:
        return False
    resolution = portfolio.resolve_trade_plan_reply(int(plan["id"]), getattr(message, "text", ""))
    if resolution is None:
        return False
    _require_bot().reply_to(message, portfolio.format_trade_plan_confirmation(plan, resolution))
    return True
