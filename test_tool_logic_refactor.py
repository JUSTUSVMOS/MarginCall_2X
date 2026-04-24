import inspect
import sys
import types
from pathlib import Path
import unittest
from unittest.mock import patch


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

    def test_upsert_trade_plan_helper_signature_no_longer_exposes_dead_select_fragment(self):
        self.assertNotIn("select_fragment", inspect.signature(engine_portfolio._upsert_trade_plan_locked).parameters)
        self.assertNotIn(
            "select_fragment=",
            inspect.getsource(engine_portfolio.upsert_trade_plan),
            "upsert_trade_plan should not pass through a dead select fragment override",
        )

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

    def test_upsert_trade_plan_rejects_active_plan_missing_required_fields(self):
        engine_portfolio.init_db()

        with self.assertRaisesRegex(ValueError, "stop_loss"):
            engine_portfolio.upsert_trade_plan(
                symbol="MRVL",
                source="manual_backfill",
                entry_price=85.2,
                stop_loss=None,
                take_profit_1=95.0,
                take_profit_2=105.0,
                max_holding_days=60,
                thesis_type="sector_rotation",
                thesis_text="semi rotation re-accelerating",
                thesis_payload={"proxy_symbol": "SOXX"},
                status="active",
            )

        with database.locked_connection() as conn:
            persisted = conn.execute("SELECT COUNT(*) FROM trade_plans").fetchone()[0]
            event_count = conn.execute("SELECT COUNT(*) FROM trade_plan_events").fetchone()[0]

        self.assertEqual(persisted, 0)
        self.assertEqual(event_count, 0)

    def test_upsert_trade_plan_preserves_existing_values_and_only_activates_once(self):
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
            thesis_text="initial thesis",
            thesis_payload={"proxy_symbol": "SOXX", "lookback_days": 10},
            status="draft",
        )

        activated_plan_id = engine_portfolio.upsert_trade_plan(
            symbol="MRVL",
            source="manual_refresh",
            entry_price=None,
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            max_holding_days=None,
            thesis_type=None,
            thesis_text="refined thesis",
            thesis_payload=None,
            status="active",
        )
        refreshed_plan_id = engine_portfolio.upsert_trade_plan(
            symbol="MRVL",
            source="manual_refresh",
            entry_price=86.0,
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            max_holding_days=None,
            thesis_type=None,
            thesis_text=None,
            thesis_payload=None,
            status="active",
        )

        self.assertEqual(plan_id, activated_plan_id)
        self.assertEqual(plan_id, refreshed_plan_id)

        plan = engine_portfolio.get_trade_plan(plan_id)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["status"], "active")
        self.assertEqual(plan["source"], "manual_refresh")
        self.assertEqual(plan["entry_price"], 86.0)
        self.assertEqual(plan["stop_loss"], 80.0)
        self.assertEqual(plan["take_profit_1"], 95.0)
        self.assertEqual(plan["take_profit_2"], 105.0)
        self.assertEqual(plan["max_holding_days"], 60)
        self.assertEqual(plan["thesis_type"], "sector_rotation")
        self.assertEqual(plan["thesis_text"], "refined thesis")
        self.assertEqual(plan["thesis_payload_json"], '{"lookback_days": 10, "proxy_symbol": "SOXX"}')

        with database.locked_connection() as conn:
            event_types = [
                row[0]
                for row in conn.execute(
                    "SELECT event_type FROM trade_plan_events WHERE plan_id = ? ORDER BY id",
                    (plan_id,),
                ).fetchall()
            ]

        self.assertEqual(
            event_types,
            ["plan_created", "plan_updated", "plan_activated", "plan_updated"],
        )

    def test_list_active_trade_plans_orders_newest_plan_first_within_symbol(self):
        engine_portfolio.init_db()

        with database.locked_connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_plans (
                    symbol, status, source, entry_price, stop_loss, take_profit_1,
                    take_profit_2, max_holding_days, thesis_type, thesis_text,
                    thesis_payload_json, created_at, updated_at
                ) VALUES (?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "AAPL",
                    "legacy_import",
                    180.0,
                    170.0,
                    195.0,
                    205.0,
                    30,
                    "momentum",
                    "older active plan",
                    '{"window": 5}',
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T00:00:00Z",
                ),
            )
            conn.execute(
                """
                INSERT INTO trade_plans (
                    symbol, status, source, entry_price, stop_loss, take_profit_1,
                    take_profit_2, max_holding_days, thesis_type, thesis_text,
                    thesis_payload_json, created_at, updated_at
                ) VALUES (?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "AAPL",
                    "manual_refresh",
                    182.0,
                    171.0,
                    198.0,
                    208.0,
                    35,
                    "momentum",
                    "newer active plan",
                    '{"window": 10}',
                    "2024-01-02T00:00:00Z",
                    "2024-01-02T00:00:00Z",
                ),
            )
            conn.execute(
                """
                INSERT INTO trade_plans (
                    symbol, status, source, entry_price, stop_loss, take_profit_1,
                    take_profit_2, max_holding_days, thesis_type, thesis_text,
                    thesis_payload_json, created_at, updated_at
                ) VALUES (?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "MSFT",
                    "manual_refresh",
                    410.0,
                    390.0,
                    430.0,
                    450.0,
                    40,
                    "breakout",
                    "msft active plan",
                    '{"window": 7}',
                    "2024-01-03T00:00:00Z",
                    "2024-01-03T00:00:00Z",
                ),
            )
            conn.commit()

        active_plans = engine_portfolio.list_active_trade_plans()

        self.assertEqual(
            [(plan["symbol"], plan["source"]) for plan in active_plans],
            [
                ("AAPL", "manual_refresh"),
                ("AAPL", "legacy_import"),
                ("MSFT", "manual_refresh"),
            ],
        )


class DirectHelperRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(__file__).with_name("_task2_trade_plan_runtime.sqlite3")
        self.csv_backup = Path(__file__).with_name("_task2_trade_plan_runtime.csv")
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

    def test_format_trade_plan_validation_error_does_not_mark_missing_fields_as_data_unavailable(self):
        result = engine_portfolio._format_trade_plan_validation_error(
            {
                "complete": False,
                "missing_fields": ["stop_loss", "take_profit_1"],
            }
        )

        self.assertIn("停損", result)
        self.assertIn("第一止盈", result)
        self.assertNotIn("[資料不可用]", result)

    def test_execute_position_update_rejects_buy_without_complete_trade_plan(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("CASH_USD", 1.0, 1000.0, 32000.0, 0),
            )
            conn.commit()

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0):
            result = engine_portfolio.execute_position_update("AAPL", 100.0, 2.0, action="buy")

        self.assertIn("交易計畫", result)
        self.assertIn("停損", result)

    def test_execute_position_update_buy_persists_active_trade_plan(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("CASH_USD", 1.0, 1000.0, 32000.0, 0),
            )
            conn.commit()

        trade_plan = {
            "stop_loss": 92.0,
            "take_profit_1": 115.0,
            "take_profit_2": 124.0,
            "max_holding_days": 30,
            "thesis_type": "breakout",
            "thesis_text": "earnings revision acceleration",
        }

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0), patch.object(
            engine_portfolio,
            "_apply_pretrade_risk_gate",
            return_value={
                "allowed": True,
                "approved_shares": 2.0,
                "approved_twd_total": 6400.0,
                "message": "",
                "note": None,
            },
        ):
            result = engine_portfolio.execute_position_update(
                "AAPL",
                100.0,
                2.0,
                action="buy",
                trade_plan=trade_plan,
            )

        active_plan = engine_portfolio.get_active_trade_plan("AAPL")

        self.assertIn("✅ 買進成功", result)
        self.assertIsNotNone(active_plan)
        self.assertEqual(active_plan["source"], "bot_trade")
        self.assertEqual(active_plan["status"], "active")
        self.assertEqual(active_plan["entry_price"], 100.0)
        self.assertEqual(active_plan["stop_loss"], 92.0)
        self.assertEqual(active_plan["take_profit_1"], 115.0)
        self.assertEqual(active_plan["take_profit_2"], 124.0)
        self.assertEqual(active_plan["max_holding_days"], 30)
        self.assertEqual(active_plan["thesis_type"], "breakout")
        self.assertEqual(active_plan["thesis_text"], "earnings revision acceleration")

    def test_execute_position_update_allows_buy_without_trade_plan_when_pretrade_gate_disabled(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("CASH_USD", 1.0, 1000.0, 32000.0, 0),
            )
            conn.commit()

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0), patch.object(
            engine_portfolio,
            "_apply_pretrade_risk_gate",
            return_value={
                "allowed": True,
                "approved_shares": 2.0,
                "approved_twd_total": 6400.0,
                "message": "",
                "note": None,
            },
        ):
            result = engine_portfolio.execute_position_update(
                "AAPL",
                100.0,
                2.0,
                action="buy",
                enforce_pretrade_gate=False,
            )

        self.assertIn("✅ 買進成功", result)
        self.assertNotIn("交易計畫未儲存", result)

    def test_execute_position_update_warns_when_gate_disabled_buy_drops_incomplete_trade_plan(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("CASH_USD", 1.0, 1000.0, 32000.0, 0),
            )
            conn.commit()

        trade_plan = {
            "stop_loss": 92.0,
            "take_profit_1": 115.0,
        }

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0), patch.object(
            engine_portfolio,
            "_apply_pretrade_risk_gate",
            return_value={
                "allowed": True,
                "approved_shares": 2.0,
                "approved_twd_total": 6400.0,
                "message": "",
                "note": None,
            },
        ):
            result = engine_portfolio.execute_position_update(
                "AAPL",
                100.0,
                2.0,
                action="buy",
                trade_plan=trade_plan,
                enforce_pretrade_gate=False,
            )

        self.assertIn("✅ 買進成功", result)
        self.assertIn("交易計畫未儲存", result)
        self.assertIsNone(engine_portfolio.get_active_trade_plan("AAPL"))

    def test_execute_position_update_buy_flow_source_does_not_recheck_trade_plan_validation_truthiness(self):
        source = inspect.getsource(engine_portfolio.execute_position_update)

        self.assertNotIn(
            'if trade_plan and trade_plan_validation and trade_plan_validation["complete"]',
            source,
        )
        self.assertNotIn(
            'elif trade_plan and trade_plan_validation and not trade_plan_validation["complete"]',
            source,
        )
