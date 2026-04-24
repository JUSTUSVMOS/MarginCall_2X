from pathlib import Path
import sys
import types
import unittest


if "fubon_neo.sdk" not in sys.modules:
    fubon_neo = types.ModuleType("fubon_neo")
    sdk_module = types.ModuleType("fubon_neo.sdk")
    sdk_module.FubonSDK = type("FubonSDK", (), {})
    rest_base_module = types.ModuleType("fubon_neo.fugle_marketdata.rest.base_rest")
    rest_base_module.FugleAPIError = type("FugleAPIError", (Exception,), {})
    sys.modules["fubon_neo"] = fubon_neo
    sys.modules["fubon_neo.sdk"] = sdk_module
    sys.modules["fubon_neo.fugle_marketdata"] = types.ModuleType("fubon_neo.fugle_marketdata")
    sys.modules["fubon_neo.fugle_marketdata.rest"] = types.ModuleType("fubon_neo.fugle_marketdata.rest")
    sys.modules["fubon_neo.fugle_marketdata.rest.base_rest"] = rest_base_module

import engine_portfolio
from src import database


class TradePlanFlowTests(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(__file__).with_name("_task2_trade_plan_flow.sqlite3")
        self.csv_backup = Path(__file__).with_name("_task2_trade_plan_flow.csv")
        self.original_db_file = database.DB_FILE
        self.original_csv_backup = engine_portfolio.CSV_BACKUP
        database.DB_FILE = self.db_path
        engine_portfolio.CSV_BACKUP = str(self.csv_backup)
        self._cleanup_paths()

    def tearDown(self):
        database.DB_FILE = self.original_db_file
        engine_portfolio.CSV_BACKUP = self.original_csv_backup
        self._cleanup_paths()

    def _cleanup_paths(self):
        for path in (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
            self.csv_backup,
        ):
            path.unlink(missing_ok=True)

    def test_sync_trade_plan_backfills_creates_missing_plan_alert_for_live_holding(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("MRVL", 85.2, 30.0, 81792.0, 0),
            )
            conn.commit()

        payload = engine_portfolio.sync_trade_plan_backfills()

        self.assertEqual(payload["missing_plan_count"], 1)
        with database.locked_connection() as conn:
            draft = conn.execute(
                "SELECT symbol, status, source FROM trade_plans WHERE symbol = 'MRVL'"
            ).fetchone()
            alert = conn.execute(
                "SELECT alert_type, status FROM trade_plan_alerts WHERE symbol = 'MRVL'"
            ).fetchone()

        self.assertEqual(draft, ("MRVL", "draft", "manual_backfill"))
        self.assertEqual(alert, ("missing_plan", "open"))
