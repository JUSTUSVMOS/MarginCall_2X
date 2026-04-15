import logging
import os
import random
import subprocess
import sys
import threading
import time
import telebot

import engine_fundamentals as fundamentals
import engine_market as market
import engine_portfolio as portfolio
import engine_risk as risk
import engine_router as router
import engine_technical as technical
import fubon
from config import PROJECT_ROOT, WDT_MESSAGES, system_prompt
from src.agent import ask_agent, generate_final_report
from src.tools import get_tools


logger = logging.getLogger(__name__)


BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MY_USER_ID = os.getenv("TELEGRAM_USER_ID")

if not BOT_TOKEN or not GEMINI_KEY or not MY_USER_ID:
    raise ValueError("❌ .env 缺少必要 API KEY 或 TELEGRAM_USER_ID")

AUTHORIZED_USER_ID = int(MY_USER_ID)

# 先載入所有工具模組，再從共享 registry 取出可用工具。
_TOOL_MODULES = (portfolio, risk, fundamentals, technical)

bot = telebot.TeleBot(BOT_TOKEN)

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

user_chat_history = []
_handlers_registered = False
_runtime_initialized = False
_v_turn_active = True


def initialize_bot_runtime():
    global _runtime_initialized
    if _runtime_initialized:
        return

    fubon.init_fubon()
    market.set_fubon_provider(fubon)
    router.set_bot(bot)
    _runtime_initialized = True


def is_v_turn_active() -> bool:
    return _v_turn_active


def _is_authorized(message) -> bool:
    return getattr(getattr(message, "from_user", None), "id", None) == AUTHORIZED_USER_ID


def trigger_nlp_and_callback(symbol, chat_id=None, message_id=None):
    """
    非同步處理：啟動 GPU 運算，完成後主動推送報告。
    """

    def _run():
        try:
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
                    bot.send_message(chat_id, f"❌ {symbol} 情報收割超時 (>5min)，請檢查 Ollama 是否正常運行。")
                return

            if process.returncode == 0:
                logger.info(f"✅ [情報局] {symbol} 收割完成")
                if chat_id and message_id:
                    bot.edit_message_text(
                        f"✅ {symbol} 情報收割完畢，正在進行最終決策...",
                        chat_id=chat_id,
                        message_id=message_id,
                    )
                    strat_data = router.fetch_strat_data(symbol)
                    nlp_alpha = strat_data.get("nlp_insights", {})
                    deliver_final_report(symbol, strat_data, nlp_alpha, chat_id, message_id)
            else:
                logger.error(f"❌ [情報局] {symbol} 失敗: {stderr[:200]}")
                if chat_id:
                    bot.send_message(chat_id, f"❌ {symbol} 情報收割失敗，請檢查日誌。\n{stderr[:100]}")
        except Exception as exc:
            logger.error(f"🚨 [情報局] 異常: {exc}")
            if chat_id:
                bot.send_message(chat_id, f"🚨 系統異常: {exc}")

    threading.Thread(target=_run, daemon=True).start()


def deliver_final_report(symbol, strat_data, nlp_alpha, chat_id, message_id=None):
    final_text = generate_final_report(symbol, strat_data, nlp_alpha)
    if message_id:
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text, parse_mode="Markdown")
        except Exception as exc:
            logger.warning(f"Report Markdown parse failed, falling back to plain text: {exc}")
            try:
                bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text)
            except Exception:
                bot.send_message(chat_id, final_text)
    else:
        try:
            bot.send_message(chat_id, final_text, parse_mode="Markdown")
        except Exception as exc:
            logger.warning(f"Report Markdown parse failed (send), falling back to plain text: {exc}")
            bot.send_message(chat_id, final_text)


def handle_deep_analysis(message):
    if not _is_authorized(message):
        return

    parts = message.text.strip().split()
    if len(parts) < 2:
        bot.reply_to(message, "💡 用法: `/analyze <股票代號>`", parse_mode="Markdown")
        return

    symbol = parts[1].upper()
    sent_msg = bot.reply_to(message, f"🔍 正在調閱 {symbol} 雙重視角戰報...\n(GPU 正在運算中，預計 1~2 分鐘，完成後主動通知您)")
    bot.send_chat_action(message.chat.id, "typing")

    strat_data = router.fetch_strat_data(symbol)
    nlp_alpha = strat_data.get("nlp_insights", {})

    if "error" in nlp_alpha:
        trigger_nlp_and_callback(symbol, message.chat.id, sent_msg.message_id)
        return

    deliver_final_report(symbol, strat_data, nlp_alpha, message.chat.id, sent_msg.message_id)


def reset_memory(message):
    if not _is_authorized(message):
        return

    user_chat_history.clear()
    bot.reply_to(message, "🧹 推進器記憶體已排空！對話上下文已重置。")


def handle_trade_command(message):
    if not _is_authorized(message):
        return

    user_input = message.text.replace("/trade", "").strip()
    if not user_input:
        bot.reply_to(message, "📝 請輸入交易內容，例如：`/trade 買入 10 股 NVDA`", parse_mode="Markdown")
        return

    sent_msg = bot.reply_to(message, "✍️ 【記帳模式啟動】正在寫入帳本...")
    bot.send_chat_action(message.chat.id, "typing")

    trade_prompt = system_prompt + "\n\n【⚠️ 記帳指令模式】你現在擁有「更新持倉 (update_position)」的權限。請根據使用者輸入精確執行買入、賣出或校正。"
    result = ask_agent(user_input, tools=FULL_AGENT_TOOLS, system_prompt_override=trade_prompt, allow_retry=False)

    try:
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=result, parse_mode="Markdown")
    except Exception:
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=result)


def toggle_v_turn(message):
    global _v_turn_active

    if not _is_authorized(message):
        return

    _v_turn_active = not _v_turn_active
    status = "🟢 已啟動" if _v_turn_active else "🔴 已關閉"
    bot.reply_to(message, f"V 轉監控 {status}")


def handle_all_text(message):
    if not _is_authorized(message):
        return

    user_text = message.text
    mood = "bad_market" if any(word in user_text for word in ["損益", "賠", "慘"]) else "normal"
    sent_msg = bot.reply_to(message, f"【推進器點火】\n{random.choice(WDT_MESSAGES[mood])}")
    bot.send_chat_action(message.chat.id, "typing")

    final_text = ask_agent(user_text, tools=READ_ONLY_TOOLS, chat_history=user_chat_history)

    try:
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=final_text, parse_mode="Markdown")
    except Exception as exc:
        logger.warning(f"Markdown parse failed, falling back to plain text: {exc}")
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=final_text)


def register_handlers():
    global _handlers_registered
    if _handlers_registered:
        return

    initialize_bot_runtime()
    bot.register_message_handler(handle_deep_analysis, commands=["analyze", "nlp"])
    bot.register_message_handler(reset_memory, commands=["reset"])
    bot.register_message_handler(handle_trade_command, commands=["trade"])
    bot.register_message_handler(toggle_v_turn, commands=["vturn"])
    bot.register_message_handler(handle_all_text, func=lambda message: True)
    _handlers_registered = True


def run_polling():
    while True:
        try:
            logger.info("📡 正在開啟 Infinity Polling...")
            bot.infinity_polling(timeout=20, long_polling_timeout=10)
        except telebot.apihelper.ApiTelegramException as exc:
            if exc.error_code == 409:
                logger.error("🛑 偵測到 409 衝突：已有其他 Bot 實例在運行。將於 15 秒後重試...")
                time.sleep(15)
            else:
                logger.error(f"❌ Telegram API 異常: {exc}")
                time.sleep(5)
        except Exception as exc:
            logger.error(f"🚨 Polling 崩潰: {exc}")
            time.sleep(5)
