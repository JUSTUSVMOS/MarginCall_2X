import logging
import sys

from dotenv import load_dotenv

from config import LOG_FILE, PROJECT_ROOT


logger = logging.getLogger(__name__)


def configure_runtime():
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - [%(process)d] - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE)],
    )

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main():
    configure_runtime()

    import engine_portfolio as portfolio
    from src.backup import backup_database
    from src.bot import init_bot, register_handlers, run_polling, trigger_nlp_and_callback, is_v_turn_active
    from src.scheduler import start_scheduler, setup_dependencies

    portfolio.init_db()
    backup_database()
    bot_instance, user_id = init_bot()
    logger.info("🚀 MarginCall Express 已啟動！")
    register_handlers()
    setup_dependencies(bot_instance, user_id, trigger_nlp_and_callback, is_v_turn_active)
    start_scheduler()
    run_polling()


if __name__ == "__main__":
    main()
