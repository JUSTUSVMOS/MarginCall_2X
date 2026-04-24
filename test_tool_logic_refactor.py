from pathlib import Path
import unittest

import engine_portfolio
from src import database


class TradePlanPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(__file__).with_name("_task1_trade_plan_test.sqlite3")
        self.csv_backup = Path(__file__).with_name("_task1_trade_plan_test.csv")
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

    def test_init_db_creates_trade_plan_tables(self):
        engine_portfolio.init_db()

        with database.locked_connection() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'trade_plan%'"
                ).fetchall()
            }

        self.assertEqual(tables, {"trade_plans", "trade_plan_events", "trade_plan_alerts"})

    def test_upsert_trade_plan_creates_active_plan_and_event(self):
        engine_portfolio.init_db()

        plan_id = engine_portfolio.upsert_trade_plan(
            symbol="MRVL",
            source="manual_backfill",
            entry_price=85.2,
            stop_loss=80.0,
            take_profit_1=95.0,
            take_profit_2=105.0,
            max_holding_days=60,
            thesis_type="sector_rotation",
            thesis_text="semi rotation re-accelerating",
            thesis_payload={"proxy_symbol": "SOXX", "lookback_days": 10, "underperform_pct": -0.03},
            status="active",
        )

        with database.locked_connection() as conn:
            plan = conn.execute(
                """
                SELECT symbol, status, source, stop_loss, take_profit_1, take_profit_2,
                       max_holding_days, thesis_type, thesis_text
                FROM trade_plans WHERE id = ?
                """,
                (plan_id,),
            ).fetchone()
            event_types = [
                row[0]
                for row in conn.execute(
                    "SELECT event_type FROM trade_plan_events WHERE plan_id = ? ORDER BY id",
                    (plan_id,),
                ).fetchall()
            ]

        fetched_plan = engine_portfolio.get_trade_plan(plan_id)
        active_plan = engine_portfolio.get_active_trade_plan("MRVL")
        active_plans = engine_portfolio.list_active_trade_plans()

        self.assertEqual(
            plan,
            ("MRVL", "active", "manual_backfill", 80.0, 95.0, 105.0, 60, "sector_rotation", "semi rotation re-accelerating"),
        )
        self.assertEqual(event_types, ["plan_created", "plan_activated"])
        self.assertEqual(fetched_plan["id"], plan_id)
        self.assertEqual(fetched_plan["status"], "active")
        self.assertEqual(active_plan["id"], plan_id)
        self.assertEqual([row["id"] for row in active_plans], [plan_id])
