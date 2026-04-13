import logging
import sys

from dotenv import load_dotenv

from config import LOG_FILE, PROJECT_ROOT


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

    from src.bot import register_handlers, run_polling
    from src.scheduler import start_scheduler

    print("🚀 MarginCall Express 已啟動！")
    register_handlers()
    start_scheduler()
    run_polling()


if __name__ == "__main__":
    main()
