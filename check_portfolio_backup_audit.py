import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import engine_portfolio
import main as main_module
import src.scheduler as scheduler_runtime
from src import backup as backup_module
from src import database


class PortfolioBackupAuditChecks(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.original_db_file = database.DB_FILE
        database.DB_FILE = self.root / "portfolio.db"

    def tearDown(self):
        database.DB_FILE = self.original_db_file
        self.tempdir.cleanup()

    def test_trade_log_captures_buy_sell_and_set(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("CASH_USD", 1.0, 1000.0, 32000.0, 0),
            )
            conn.commit()

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0):
            engine_portfolio.execute_position_update("AAPL", 100.0, 2.0, action="buy")
            engine_portfolio.execute_position_update("AAPL", 120.0, 1.0, action="sell")
            engine_portfolio.execute_position_update("MSFT", 200.0, 3.0, action="set", locked=1)

        with database.locked_connection() as conn:
            rows = conn.execute(
                "SELECT action, symbol, settle_currency, settle_amount, fx_rate, realized_pnl, cash_before, cash_after, note "
                "FROM trade_log ORDER BY id"
            ).fetchall()

        self.assertEqual(rows[0], ("buy", "AAPL", "CASH_USD", 200.0, 32.0, None, 1000.0, 800.0, None))
        self.assertEqual(rows[1], ("sell", "AAPL", "CASH_USD", 120.0, 32.0, 640.0, 800.0, 920.0, None))
        self.assertEqual(rows[2], ("set", "MSFT", None, None, 32.0, None, None, None, "manual set; locked=1"))

    def test_backup_database_creates_recoverable_copy_and_prunes_old_files(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute("CREATE TABLE sample (value TEXT)")
            conn.execute("INSERT INTO sample VALUES ('ok')")
            conn.commit()

        backup_dir = self.root / "backups"
        backup_dir.mkdir()
        (backup_dir / "portfolio_20240101_000000_000000.db").write_bytes(b"old1")
        (backup_dir / "portfolio_20240102_000000_000000.db").write_bytes(b"old2")

        backup_path = backup_module.backup_database(db_path=database.DB_FILE, backup_dir=backup_dir, max_backups=2)

        self.assertIsNotNone(backup_path)
        with sqlite3.connect(str(backup_path)) as conn:
            value = conn.execute("SELECT value FROM sample").fetchone()[0]
        self.assertEqual(value, "ok")
        self.assertEqual(len(list(backup_dir.glob("portfolio_*.db"))), 2)

    def test_main_runs_startup_backup_after_db_init(self):
        order = []
        fake_bot = object()

        portfolio_module = types.ModuleType("engine_portfolio")
        portfolio_module.init_db = lambda: order.append("init_db")

        backup_stub = types.ModuleType("src.backup")
        backup_stub.backup_database = lambda: order.append("backup_database")

        bot_stub = types.ModuleType("src.bot")
        bot_stub.init_bot = lambda: (order.append("init_bot"), fake_bot, 42)[1:]
        bot_stub.register_handlers = lambda: order.append("register_handlers")
        bot_stub.run_polling = lambda: order.append("run_polling")
        bot_stub.trigger_nlp_and_callback = lambda *args, **kwargs: None
        bot_stub.is_v_turn_active = lambda: True

        scheduler_stub = types.ModuleType("src.scheduler")
        scheduler_stub.setup_dependencies = lambda *args, **kwargs: order.append("setup_dependencies")
        scheduler_stub.start_scheduler = lambda: order.append("start_scheduler")

        with patch.object(main_module, "configure_runtime", side_effect=lambda: order.append("configure_runtime")), patch.object(
            main_module.logger, "info"
        ), patch.dict(
            sys.modules,
            {
                "engine_portfolio": portfolio_module,
                "src.backup": backup_stub,
                "src.bot": bot_stub,
                "src.scheduler": scheduler_stub,
            },
        ):
            main_module.main()

        self.assertEqual(order[:3], ["configure_runtime", "init_db", "backup_database"])

    def test_scheduler_registers_daily_backup_job(self):
        original_scheduler = scheduler_runtime._scheduler
        scheduler_runtime._scheduler = None
        fake_scheduler = SimpleNamespace(running=False, add_job=Mock(), start=Mock())
        try:
            with patch.object(scheduler_runtime, "BackgroundScheduler", return_value=fake_scheduler), patch.object(
                scheduler_runtime, "macro_brain_heartbeat", return_value=None
            ), patch.object(
                scheduler_runtime, "daily_portfolio_review", return_value=None
            ), patch.object(scheduler_runtime, "backup_database", return_value=None) as mock_backup:
                scheduler_runtime.start_scheduler()
        finally:
            scheduler_runtime._scheduler = original_scheduler

        backup_calls = [
            call for call in fake_scheduler.add_job.call_args_list
            if call.args and call.args[0] is mock_backup
        ]
        self.assertEqual(len(backup_calls), 1)
        self.assertEqual(backup_calls[0].kwargs["id"], "daily-db-backup")


if __name__ == "__main__":
    unittest.main()
