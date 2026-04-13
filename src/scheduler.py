import logging
import time
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler

import engine_market as market
import engine_memory as memory
import engine_risk as risk
from src.bot import AUTHORIZED_USER_ID, bot, is_v_turn_active, trigger_nlp_and_callback


logger = logging.getLogger(__name__)


WATCH_LIST = ["NVDA", "TSLA", "AAPL", "MSFT", "ARM"]
_scheduler = None


def daily_nlp_scout():
    for stock in WATCH_LIST:
        trigger_nlp_and_callback(stock)
        time.sleep(20)


def macro_brain_heartbeat(force=False):
    result = memory.sync_market_brain(force=force, max_age_minutes=180)
    logger.info(f"🧠 [MacroHeartbeat] {result['message']}")
    return result


def auto_v_turn_monitor():
    try:
        if not is_v_turn_active():
            return

        now_et = datetime.now(pytz.timezone("US/Eastern"))
        if now_et.weekday() >= 5:
            return
        if not market.is_us_market_open() and now_et.hour < 16:
            return

        report = risk.get_v_turn_confirmation()
        if any(keyword in report for keyword in ["✅ 觸發", "偵測", "📈", "📉"]):
            bot.send_message(AUTHORIZED_USER_ID, report, parse_mode="Markdown")
    except Exception as exc:
        logger.error(f"V-turn monitor failed: {exc}")


def start_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler

    scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Taipei"))

    try:
        macro_brain_heartbeat(force=False)
    except Exception as exc:
        logger.error(f"Initial macro heartbeat failed: {exc}")

    scheduler.add_job(auto_v_turn_monitor, "interval", hours=2, id="auto-v-turn-monitor", replace_existing=True)
    scheduler.add_job(
        macro_brain_heartbeat,
        "interval",
        hours=3,
        id="macro-brain-heartbeat",
        next_run_time=datetime.now(pytz.timezone("Asia/Taipei")),
        replace_existing=True,
    )
    scheduler.add_job(daily_nlp_scout, "interval", hours=4, id="daily-nlp-scout", replace_existing=True)
    scheduler.start()

    _scheduler = scheduler
    return _scheduler
