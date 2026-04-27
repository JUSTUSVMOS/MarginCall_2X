import logging
import importlib
import time
from datetime import datetime
from typing import Callable, Optional

import pytz

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ModuleNotFoundError:
    BackgroundScheduler = None

from config import WATCH_LIST
from src.backup import backup_database

logger = logging.getLogger(__name__)

_scheduler = None
_bot_instance = None
_user_id: Optional[int] = None
_trigger_nlp_cb: Optional[Callable] = None
_is_v_turn_active_cb: Optional[Callable] = None


def setup_dependencies(bot_instance, user_id: int, trigger_cb: Callable, is_v_turn_active_cb: Callable):
    global _bot_instance, _user_id, _trigger_nlp_cb, _is_v_turn_active_cb
    _bot_instance = bot_instance
    _user_id = user_id
    _trigger_nlp_cb = trigger_cb
    _is_v_turn_active_cb = is_v_turn_active_cb


def daily_nlp_scout():
    if not _trigger_nlp_cb:
        return
    for stock in WATCH_LIST:
        _trigger_nlp_cb(stock)
        time.sleep(20)


def daily_portfolio_review():
    """定期執行持倉健檢並更新額葉 (Portfolio Health)"""
    try:
        import engine_portfolio as portfolio
        data = portfolio.refresh_portfolio_health_summary(source="portfolio_review")
        logger.info(f"📊 [PortfolioReview] {data['memory_update']['message']} -> {data['summary']}")
        return data
    except Exception as exc:
        logger.error(f"Daily portfolio review failed: {exc}")
        return None


def fubon_portfolio_sync(source: str = "scheduler"):
    try:
        import engine_portfolio as portfolio

        data = portfolio.sync_fubon_portfolio_state(source=source, sync_memory=True)
        if int(data.get("followup_count") or 0) > 0:
            try:
                from src import bot as bot_runtime

                bot_runtime.send_pending_trade_followups()
            except Exception as exc:
                logger.error(f"Trade follow-up prompt delivery failed: {exc}")
        logger.info(f"🏦 [FubonSync] {data['message']}")
        return data
    except Exception as exc:
        logger.error(f"Fubon portfolio sync failed: {exc}")
        return None


def macro_brain_heartbeat(force=False):
    import engine_memory as memory

    result = memory.sync_market_brain(force=force, max_age_minutes=180)
    logger.info(f"🧠 [MacroHeartbeat] {result['message']}")
    return result


def auto_v_turn_monitor():
    try:
        import engine_market as market
        import engine_risk as risk

        if _is_v_turn_active_cb and not _is_v_turn_active_cb():
            return

        now_et = datetime.now(pytz.timezone("US/Eastern"))
        if now_et.weekday() >= 5:
            return
        if not market.is_us_market_open() and now_et.hour < 16:
            return

        report = risk.build_v_turn_report()
        if any(keyword in report for keyword in ["✅ 觸發", "偵測", "📈", "📉"]):
            if _bot_instance and _user_id:
                _bot_instance.send_message(_user_id, report, parse_mode="Markdown")
    except Exception as exc:
        logger.error(f"V-turn monitor failed: {exc}")


def start_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        return _scheduler
    if BackgroundScheduler is None:
        raise RuntimeError("apscheduler 未安裝，無法啟動排程器。")

    scheduler = BackgroundScheduler(timezone=pytz.timezone("Asia/Taipei"))

    try:
        macro_brain_heartbeat(force=False)
    except Exception as exc:
        logger.error(f"Initial macro heartbeat failed: {exc}")
    daily_portfolio_review()

    scheduler.add_job(auto_v_turn_monitor, "interval", hours=2, id="auto-v-turn-monitor", replace_existing=True)
    scheduler.add_job(
        macro_brain_heartbeat,
        "interval",
        hours=3,
        id="macro-brain-heartbeat",
        next_run_time=datetime.now(pytz.timezone("Asia/Taipei")),
        replace_existing=True,
    )
    scheduler.add_job(
        daily_portfolio_review,
        "cron",
        hour=8,
        minute=5,
        id="daily-portfolio-review",
        replace_existing=True,
    )
    scheduler.add_job(
        backup_database,
        "cron",
        hour=7,
        minute=0,
        id="daily-db-backup",
        replace_existing=True,
    )
    scheduler.add_job(
        fubon_portfolio_sync,
        "cron",
        day_of_week="mon-fri",
        hour="9-13",
        minute="0,30",
        kwargs={"source": "scheduler"},
        id="fubon-portfolio-sync-intraday",
        replace_existing=True,
    )
    scheduler.add_job(
        fubon_portfolio_sync,
        "cron",
        day_of_week="mon-fri",
        hour=13,
        minute=35,
        kwargs={"source": "close_sync"},
        id="fubon-portfolio-sync-close",
        replace_existing=True,
    )
    scheduler.add_job(
        trade_plan_audit_job,
        "cron",
        day_of_week="mon-fri",
        hour="9-13",
        minute="10,40",
        id="trade-plan-audit",
        replace_existing=True,
    )
    scheduler.add_job(daily_nlp_scout, "interval", hours=4, id="daily-nlp-scout", replace_existing=True)
    scheduler.start()

    _scheduler = scheduler
    return _scheduler


def trade_plan_audit_job():
    try:
        import engine_portfolio as portfolio

        backfill = portfolio.sync_trade_plan_backfills()
        audit = portfolio.audit_trade_plan_alerts()
        bot_runtime = importlib.import_module("src.bot")
        prompted_count = bot_runtime.send_pending_trade_plan_prompts()
        logger.info(
            "🧾 [TradePlanAudit] backfill=%s audit=%s prompted=%s",
            backfill,
            audit,
            prompted_count,
        )
        return {
            "backfill": backfill,
            "audit": audit,
            "prompted_count": prompted_count,
        }
    except Exception as exc:
        logger.error(f"Trade plan audit job failed: {exc}")
        return None

def morning_briefing_push():
    from src import bot as bot_runtime
    bot_runtime.send_morning_briefing()
