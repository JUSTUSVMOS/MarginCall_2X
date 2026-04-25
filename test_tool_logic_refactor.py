import inspect
import sqlite3
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

    def test_evaluate_trade_plan_alerts_fires_stop_hit_and_dedupes(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("MRVL", 85.2, 5.0, 426.0, 0),
            )
            conn.commit()

        plan_id = engine_portfolio.upsert_trade_plan(
            symbol="MRVL",
            source="manual_backfill",
            entry_price=85.2,
            stop_loss=80.0,
            take_profit_1=95.0,
            take_profit_2=None,
            max_holding_days=30,
            thesis_type="breakout_support",
            thesis_text="breakout should hold prior support",
            thesis_payload=None,
            status="active",
        )

        snapshot = {
            "symbol": "MRVL",
            "market": "US",
            "is_cash": False,
            "is_us_stock": True,
            "shares": 5.0,
            "cost": 85.2,
            "twd_cost": 426.0,
            "current_price": 79.8,
            "market_value_twd": 399.0,
            "pnl_value_twd": -27.0,
            "pnl_percent": -0.0634,
        }

        with patch.object(engine_portfolio, "_build_live_position_snapshots", return_value=[snapshot]):
            engine_portfolio.audit_trade_plan_alerts()
            engine_portfolio.audit_trade_plan_alerts()

        stop_alerts = [
            alert
            for alert in engine_portfolio.get_open_trade_plan_alerts(symbol="MRVL")
            if alert["alert_type"] == "stop_hit"
        ]

        self.assertEqual(len(stop_alerts), 1)
        self.assertEqual(stop_alerts[0]["plan_id"], plan_id)
        self.assertEqual(stop_alerts[0]["payload"]["current_price"], 79.8)
        self.assertEqual(stop_alerts[0]["payload"]["stop_loss"], 80.0)

    def test_evaluate_trade_plan_alerts_marks_monitor_degraded_when_price_refresh_fails(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("AMD", 165.0, 4.0, 660.0, 0),
            )
            conn.commit()

        engine_portfolio.upsert_trade_plan(
            symbol="AMD",
            source="manual_backfill",
            entry_price=165.0,
            stop_loss=150.0,
            take_profit_1=180.0,
            take_profit_2=None,
            max_holding_days=20,
            thesis_type="breakout_support",
            thesis_text="ai leadership should keep trend intact",
            thesis_payload=None,
            status="active",
        )

        with patch.object(
            engine_portfolio,
            "_build_live_position_snapshots",
            side_effect=RuntimeError("price refresh failed"),
        ):
            payload = engine_portfolio.audit_trade_plan_alerts()

        degraded_alerts = [
            alert
            for alert in engine_portfolio.get_open_trade_plan_alerts(symbol="AMD")
            if alert["alert_type"] == "monitor_degraded"
        ]

        self.assertEqual(payload["degraded"], 1)
        self.assertEqual(len(degraded_alerts), 1)
        self.assertEqual(degraded_alerts[0]["payload"]["error"], "price refresh failed")

    def test_upsert_trade_plan_alert_dedupes_per_plan_id_not_symbol(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            plan_ids = []
            for source, created_at in (
                ("legacy_import", "2024-01-01T00:00:00Z"),
                ("manual_refresh", "2024-01-02T00:00:00Z"),
            ):
                cursor = conn.execute(
                    """
                    INSERT INTO trade_plans (
                        symbol, status, source, entry_price, stop_loss, take_profit_1,
                        take_profit_2, max_holding_days, thesis_type, thesis_text,
                        thesis_payload_json, created_at, updated_at
                    ) VALUES (?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "MRVL",
                        source,
                        85.2,
                        80.0,
                        95.0,
                        None,
                        30,
                        "breakout_support",
                        "support should hold",
                        None,
                        created_at,
                        created_at,
                    ),
                )
                plan_ids.append(cursor.lastrowid)
            conn.commit()

        engine_portfolio.upsert_trade_plan_alert(
            symbol="MRVL",
            alert_type="stop_hit",
            severity="critical",
            plan_id=plan_ids[0],
            payload={"current_price": 79.8},
        )
        engine_portfolio.upsert_trade_plan_alert(
            symbol="MRVL",
            alert_type="stop_hit",
            severity="critical",
            plan_id=plan_ids[1],
            payload={"current_price": 79.2},
        )
        engine_portfolio.upsert_trade_plan_alert(
            symbol="MRVL",
            alert_type="stop_hit",
            severity="critical",
            plan_id=plan_ids[0],
            payload={"current_price": 79.5},
        )

        alerts = engine_portfolio.get_open_trade_plan_alerts(symbol="MRVL", alert_type="stop_hit")

        self.assertEqual(len(alerts), 2)
        payload_by_plan = {alert["plan_id"]: alert["payload"] for alert in alerts}
        self.assertEqual(payload_by_plan[plan_ids[0]]["current_price"], 79.5)
        self.assertEqual(payload_by_plan[plan_ids[1]]["current_price"], 79.2)

    def test_resolve_trade_plan_alert_preserves_last_seen_at(self):
        engine_portfolio.init_db()

        with patch.object(engine_portfolio, "_utc_now_iso", return_value="2024-01-01T00:00:00Z"):
            engine_portfolio.upsert_trade_plan_alert(
                symbol="AMD",
                alert_type="monitor_degraded",
                severity="warning",
                payload={"error": "price refresh failed"},
            )

        with patch.object(engine_portfolio, "_utc_now_iso", return_value="2024-01-01T00:05:00Z"):
            resolved = engine_portfolio.resolve_trade_plan_alert(
                symbol="AMD",
                alert_type="monitor_degraded",
            )

        with database.locked_connection() as conn:
            row = conn.execute(
                """
                SELECT status, last_seen_at, resolved_at
                FROM trade_plan_alerts
                WHERE symbol = ? AND alert_type = ?
                """,
                ("AMD", "monitor_degraded"),
            ).fetchone()

        self.assertEqual(resolved, 1)
        self.assertEqual(row[0], "resolved")
        self.assertEqual(row[1], "2024-01-01T00:00:00Z")
        self.assertEqual(row[2], "2024-01-01T00:05:00Z")

    def test_evaluate_trade_plan_alerts_resolves_monitor_degraded_after_snapshot_recovery(self):
        engine_portfolio.init_db()
        plan_id = engine_portfolio.upsert_trade_plan(
            symbol="AMD",
            source="manual_backfill",
            entry_price=165.0,
            stop_loss=150.0,
            take_profit_1=180.0,
            take_profit_2=None,
            max_holding_days=20,
            thesis_type="breakout_support",
            thesis_text="ai leadership should keep trend intact",
            thesis_payload=None,
            status="active",
        )
        engine_portfolio.upsert_trade_plan_alert(
            symbol="AMD",
            alert_type="monitor_degraded",
            severity="warning",
            plan_id=plan_id,
            payload={"error": "price refresh failed"},
        )

        with patch.object(engine_portfolio, "_build_live_position_snapshots", return_value=[]):
            payload = engine_portfolio.audit_trade_plan_alerts()

        self.assertEqual(payload["degraded"], 0)
        self.assertEqual(
            engine_portfolio.get_open_trade_plan_alerts(symbol="AMD", alert_type="monitor_degraded"),
            [],
        )

    def test_audit_trade_plan_alerts_resolves_plan_alerts_when_symbol_missing_from_snapshots(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("MRVL", 85.2, 5.0, 426.0, 0),
            )
            conn.commit()

        plan_id = engine_portfolio.upsert_trade_plan(
            symbol="MRVL",
            source="manual_backfill",
            entry_price=85.2,
            stop_loss=80.0,
            take_profit_1=95.0,
            take_profit_2=None,
            max_holding_days=30,
            thesis_type="breakout_support",
            thesis_text="breakout should hold prior support",
            thesis_payload=None,
            status="active",
        )

        snapshot = {
            "symbol": "MRVL",
            "market": "US",
            "is_cash": False,
            "is_us_stock": True,
            "shares": 5.0,
            "cost": 85.2,
            "twd_cost": 426.0,
            "current_price": 79.8,
            "market_value_twd": 399.0,
            "pnl_value_twd": -27.0,
            "pnl_percent": -0.0634,
        }

        with patch.object(engine_portfolio, "_build_live_position_snapshots", return_value=[snapshot]):
            engine_portfolio.audit_trade_plan_alerts()

        with database.locked_connection() as conn:
            conn.execute("DELETE FROM portfolio WHERE symbol = ?", ("MRVL",))
            conn.commit()

        engine_portfolio.audit_trade_plan_alerts()

        self.assertEqual(engine_portfolio.get_open_trade_plan_alerts(symbol="MRVL", alert_type="stop_hit"), [])
        with database.locked_connection() as conn:
            resolved = conn.execute(
                """
                SELECT status, resolved_at
                FROM trade_plan_alerts
                WHERE plan_id = ? AND alert_type = ?
                """,
                (plan_id, "stop_hit"),
            ).fetchone()

        self.assertEqual(resolved[0], "resolved")
        self.assertIsNotNone(resolved[1])

    def test_evaluate_trade_plan_alerts_fires_take_profit_alerts(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("NVDA", 100.0, 3.0, 300.0, 0),
            )
            conn.commit()

        plan_id = engine_portfolio.upsert_trade_plan(
            symbol="NVDA",
            source="manual_backfill",
            entry_price=100.0,
            stop_loss=92.0,
            take_profit_1=110.0,
            take_profit_2=120.0,
            max_holding_days=30,
            thesis_type="breakout_support",
            thesis_text="trend should keep advancing",
            thesis_payload=None,
            status="active",
        )

        snapshot = {
            "symbol": "NVDA",
            "market": "US",
            "is_cash": False,
            "is_us_stock": True,
            "shares": 3.0,
            "cost": 100.0,
            "twd_cost": 300.0,
            "current_price": 121.5,
            "market_value_twd": 364.5,
            "pnl_value_twd": 64.5,
            "pnl_percent": 0.215,
        }

        with patch.object(engine_portfolio, "_build_live_position_snapshots", return_value=[snapshot]):
            engine_portfolio.audit_trade_plan_alerts()

        alerts = engine_portfolio.get_open_trade_plan_alerts(symbol="NVDA")
        payloads = {alert["alert_type"]: alert["payload"] for alert in alerts if alert["plan_id"] == plan_id}

        self.assertEqual(payloads["tp1_hit"]["target_price"], 110.0)
        self.assertEqual(payloads["tp2_hit"]["target_price"], 120.0)
        self.assertEqual(payloads["tp2_hit"]["current_price"], 121.5)

    def test_evaluate_trade_plan_alerts_fires_holding_expiry_with_payload(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("AMD", 100.0, 4.0, 400.0, 0),
            )
            conn.commit()

        with patch.object(engine_portfolio, "_utc_now_iso", return_value="2024-01-01T00:00:00Z"):
            plan_id = engine_portfolio.upsert_trade_plan(
                symbol="AMD",
                source="manual_backfill",
                entry_price=100.0,
                stop_loss=92.0,
                take_profit_1=110.0,
                take_profit_2=None,
                max_holding_days=10,
                thesis_type="breakout_support",
                thesis_text="trend should stay intact",
                thesis_payload=None,
                status="active",
            )

        snapshot = {
            "symbol": "AMD",
            "market": "US",
            "is_cash": False,
            "is_us_stock": True,
            "shares": 4.0,
            "cost": 100.0,
            "twd_cost": 400.0,
            "current_price": 105.0,
            "market_value_twd": 420.0,
            "pnl_value_twd": 20.0,
            "pnl_percent": 0.05,
        }

        with patch.object(engine_portfolio, "_utc_now_iso", return_value="2024-01-15T00:00:00Z"), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=[snapshot]
        ):
            engine_portfolio.audit_trade_plan_alerts()

        alerts = [
            alert
            for alert in engine_portfolio.get_open_trade_plan_alerts(symbol="AMD")
            if alert["alert_type"] == "holding_expiry"
        ]

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["plan_id"], plan_id)
        self.assertEqual(
            alerts[0]["payload"],
            {
                "symbol": "AMD",
                "held_days": 14,
                "max_days": 10,
                "current_return_pct": 0.05,
            },
        )

    def test_audit_trade_plan_alerts_uses_one_shared_timestamp_for_time_rules(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("AAPL", 100.0, 2.0, 200.0, 0),
                    ("AMD", 100.0, 4.0, 400.0, 0),
                ],
            )
            conn.commit()

        with patch.object(engine_portfolio, "_utc_now_iso", return_value="2024-01-01T00:00:00Z"):
            for symbol in ("AAPL", "AMD"):
                engine_portfolio.upsert_trade_plan(
                    symbol=symbol,
                    source="manual_backfill",
                    entry_price=100.0,
                    stop_loss=92.0,
                    take_profit_1=110.0,
                    take_profit_2=None,
                    max_holding_days=11,
                    thesis_type="breakout_support",
                    thesis_text="trend should stay intact",
                    thesis_payload=None,
                    status="active",
                )

        snapshots = [
            {
                "symbol": "AAPL",
                "market": "US",
                "is_cash": False,
                "is_us_stock": True,
                "shares": 2.0,
                "cost": 100.0,
                "twd_cost": 200.0,
                "current_price": 105.0,
                "market_value_twd": 210.0,
                "pnl_value_twd": 10.0,
                "pnl_percent": 0.05,
            },
            {
                "symbol": "AMD",
                "market": "US",
                "is_cash": False,
                "is_us_stock": True,
                "shares": 4.0,
                "cost": 100.0,
                "twd_cost": 400.0,
                "current_price": 105.0,
                "market_value_twd": 420.0,
                "pnl_value_twd": 20.0,
                "pnl_percent": 0.05,
            },
        ]

        with patch.object(
            engine_portfolio,
            "_utc_now_iso",
            side_effect=["2024-01-11T23:59:59Z", "2024-01-12T00:00:01Z"],
        ) as mocked_now, patch.object(
            engine_portfolio, "resolve_trade_plan_alert", return_value=0
        ), patch.object(
            engine_portfolio, "upsert_trade_plan_alert", return_value=1
        ), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ):
            payload = engine_portfolio.audit_trade_plan_alerts()

        self.assertEqual(mocked_now.call_count, 1)
        self.assertEqual(payload["triggered"], 0)
        self.assertEqual(payload["symbols"], [])
        self.assertEqual(engine_portfolio.get_open_trade_plan_alerts(alert_type="holding_expiry"), [])

    def test_audit_trade_plan_alerts_shares_timestamp_with_time_based_thesis_invalid(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("AAPL", 100.0, 2.0, 200.0, 0),
                    ("AMD", 100.0, 4.0, 400.0, 0),
                ],
            )
            conn.commit()

        thesis_payload = {"invalidation_deadline": "2024-01-11T00:00:00Z", "confirmed": False}
        with patch.object(engine_portfolio, "_utc_now_iso", return_value="2024-01-01T00:00:00Z"):
            for symbol in ("AAPL", "AMD"):
                engine_portfolio.upsert_trade_plan(
                    symbol=symbol,
                    source="manual_backfill",
                    entry_price=100.0,
                    stop_loss=92.0,
                    take_profit_1=110.0,
                    take_profit_2=None,
                    max_holding_days=30,
                    thesis_type="event_driven",
                    thesis_text="catalyst must land before deadline",
                    thesis_payload=thesis_payload,
                    status="active",
                )

        snapshots = [
            {
                "symbol": "AAPL",
                "market": "US",
                "is_cash": False,
                "is_us_stock": True,
                "shares": 2.0,
                "cost": 100.0,
                "twd_cost": 200.0,
                "current_price": 105.0,
                "market_value_twd": 210.0,
                "pnl_value_twd": 10.0,
                "pnl_percent": 0.05,
            },
            {
                "symbol": "AMD",
                "market": "US",
                "is_cash": False,
                "is_us_stock": True,
                "shares": 4.0,
                "cost": 100.0,
                "twd_cost": 400.0,
                "current_price": 105.0,
                "market_value_twd": 420.0,
                "pnl_value_twd": 20.0,
                "pnl_percent": 0.05,
            },
        ]

        with patch.object(
            engine_portfolio,
            "_utc_now_iso",
            side_effect=["2024-01-10T23:59:59Z", "2024-01-11T00:00:01Z"],
        ) as mocked_now, patch.object(
            engine_portfolio, "resolve_trade_plan_alert", return_value=0
        ), patch.object(
            engine_portfolio, "upsert_trade_plan_alert", return_value=1
        ) as mocked_upsert, patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ):
            payload = engine_portfolio.audit_trade_plan_alerts()

        self.assertEqual(mocked_now.call_count, 1)
        mocked_upsert.assert_not_called()
        self.assertEqual(payload["triggered"], 0)
        self.assertEqual(payload["symbols"], [])

    def test_evaluate_trade_plan_alerts_fires_breakout_support_thesis_invalid(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("AAPL", 100.0, 2.0, 200.0, 0),
            )
            conn.commit()

        plan_id = engine_portfolio.upsert_trade_plan(
            symbol="AAPL",
            source="manual_backfill",
            entry_price=100.0,
            stop_loss=85.0,
            take_profit_1=115.0,
            take_profit_2=None,
            max_holding_days=30,
            thesis_type="breakout_support",
            thesis_text="breakout should hold support",
            thesis_payload={"support_level": 95.0},
            status="active",
        )

        snapshot = {
            "symbol": "AAPL",
            "market": "US",
            "is_cash": False,
            "is_us_stock": True,
            "shares": 2.0,
            "cost": 100.0,
            "twd_cost": 200.0,
            "current_price": 94.0,
            "market_value_twd": 188.0,
            "pnl_value_twd": -12.0,
            "pnl_percent": -0.06,
        }

        with patch.object(engine_portfolio, "_build_live_position_snapshots", return_value=[snapshot]):
            engine_portfolio.audit_trade_plan_alerts()

        alerts = [
            alert
            for alert in engine_portfolio.get_open_trade_plan_alerts(symbol="AAPL")
            if alert["alert_type"] == "thesis_invalid"
        ]

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["plan_id"], plan_id)
        self.assertEqual(alerts[0]["payload"]["thesis_type"], "breakout_support")
        self.assertEqual(alerts[0]["payload"]["support_level"], 95.0)
        self.assertEqual(alerts[0]["payload"]["current_price"], 94.0)

    def test_evaluate_trade_plan_thesis_invalid_breakout_support_requires_configured_close_confirmation(self):
        plan = {
            "symbol": "AAPL",
            "thesis_type": "breakout_support",
            "thesis_payload_json": (
                '{"support_level": 95.0, "close_below_count": 2, "grace_rule": "close_below"}'
            ),
        }
        snapshot = {"current_price": 94.0}

        with patch.object(engine_portfolio, "_fetch_recent_closes", return_value=[96.0, 94.5], create=True):
            result = engine_portfolio._evaluate_trade_plan_thesis_invalid_spec(
                plan,
                snapshot,
                now_iso="2024-01-10T00:00:00Z",
            )

        self.assertIsNone(result)

    def test_evaluate_trade_plan_thesis_invalid_mean_reversion_supports_proxy_underperformance_path(self):
        plan = {
            "symbol": "MRVL",
            "entry_price": 100.0,
            "created_at": "2024-01-01T00:00:00Z",
            "thesis_type": "mean_reversion",
            "thesis_payload_json": (
                '{"recovery_window_days": 5, "reference_price": 100.0, "proxy_symbol": "SPY", '
                '"lookback_days": 3, "underperform_pct": -0.05}'
            ),
        }
        snapshot = {"current_price": 101.0}

        with patch.object(engine_portfolio, "_fetch_recent_change", side_effect=[0.01, 0.08]):
            result = engine_portfolio._evaluate_trade_plan_thesis_invalid_spec(
                plan,
                snapshot,
                now_iso="2024-01-10T00:00:00Z",
            )

        self.assertIsNotNone(result)
        self.assertEqual(result["payload"]["thesis_type"], "mean_reversion")
        self.assertEqual(result["payload"]["proxy_symbol"], "SPY")
        self.assertEqual(result["payload"]["relative_performance_pct"], -0.07)
        self.assertEqual(result["payload"]["threshold_pct"], -0.05)
        self.assertEqual(result["payload"]["lookback_days"], 3)

    def test_evaluate_trade_plan_thesis_invalid_earnings_waits_for_event_window_end(self):
        plan = {
            "symbol": "AMD",
            "entry_price": 100.0,
            "created_at": "2024-01-01T00:00:00Z",
            "thesis_type": "earnings",
            "thesis_payload_json": (
                '{"earnings_date": "2024-01-10T00:00:00Z", "review_window_days": 2, '
                '"expected_direction": "up", "reference_price": 100.0}'
            ),
        }
        snapshot = {"current_price": 95.0}

        before_window = engine_portfolio._evaluate_trade_plan_thesis_invalid_spec(
            plan,
            snapshot,
            now_iso="2024-01-11T00:00:00Z",
        )
        after_window = engine_portfolio._evaluate_trade_plan_thesis_invalid_spec(
            plan,
            snapshot,
            now_iso="2024-01-13T00:00:00Z",
        )

        self.assertIsNone(before_window)
        self.assertIsNotNone(after_window)
        self.assertEqual(after_window["payload"]["thesis_type"], "earnings")
        self.assertEqual(after_window["payload"]["expected_direction"], "up")
        self.assertEqual(after_window["payload"]["review_window_days"], 2)

    def test_evaluate_trade_plan_thesis_invalid_event_driven_accepts_catalyst_date_fallback(self):
        plan = {
            "symbol": "NVDA",
            "thesis_type": "event_driven",
            "thesis_payload_json": '{"catalyst_date": "2024-01-05T00:00:00Z", "catalyst_confirmed": false}',
        }
        snapshot = {"current_price": 105.0}

        result = engine_portfolio._evaluate_trade_plan_thesis_invalid_spec(
            plan,
            snapshot,
            now_iso="2024-01-06T00:00:00Z",
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["payload"]["thesis_type"], "event_driven")
        self.assertEqual(result["payload"]["deadline"], "2024-01-05T00:00:00Z")

    def test_evaluate_trade_plan_alerts_sector_rotation_respects_configured_lookback(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("MRVL", 100.0, 2.0, 200.0, 0),
            )
            conn.commit()

        plan_id = engine_portfolio.upsert_trade_plan(
            symbol="MRVL",
            source="manual_backfill",
            entry_price=100.0,
            stop_loss=90.0,
            take_profit_1=115.0,
            take_profit_2=None,
            max_holding_days=30,
            thesis_type="sector_rotation",
            thesis_text="semi leadership should outperform",
            thesis_payload={"proxy_symbol": "SOXX", "lookback_days": 5, "underperform_pct": -0.03},
            status="active",
        )

        snapshot = {
            "symbol": "MRVL",
            "market": "US",
            "is_cash": False,
            "is_us_stock": True,
            "shares": 2.0,
            "cost": 100.0,
            "twd_cost": 200.0,
            "current_price": 98.0,
            "market_value_twd": 196.0,
            "pnl_value_twd": -4.0,
            "pnl_percent": -0.02,
        }

        with patch.object(engine_portfolio, "_fetch_recent_change", side_effect=[-0.08, -0.02]), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=[snapshot]
        ):
            engine_portfolio.audit_trade_plan_alerts()

        alerts = [
            alert
            for alert in engine_portfolio.get_open_trade_plan_alerts(symbol="MRVL")
            if alert["alert_type"] == "thesis_invalid"
        ]

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["plan_id"], plan_id)
        self.assertEqual(alerts[0]["payload"]["lookback_days"], 5)
        self.assertEqual(alerts[0]["payload"]["relative_performance_pct"], -0.06)

    def test_build_trade_plan_status_summary_exposes_briefing_keys_and_decoded_alerts(self):
        engine_portfolio.init_db()
        plan_id = engine_portfolio.upsert_trade_plan(
            symbol="AMD",
            source="manual_backfill",
            entry_price=100.0,
            stop_loss=92.0,
            take_profit_1=110.0,
            take_profit_2=None,
            max_holding_days=30,
            thesis_type="breakout_support",
            thesis_text="trend should stay intact",
            thesis_payload=None,
            status="active",
        )
        engine_portfolio.upsert_trade_plan_alert(
            symbol="AMD",
            alert_type="stop_hit",
            severity="critical",
            plan_id=plan_id,
            payload={"current_price": 89.5},
        )
        engine_portfolio.upsert_trade_plan_alert(
            symbol="NVDA",
            alert_type="missing_plan",
            severity="warning",
            payload={"reason": "live_portfolio_without_plan"},
        )

        summary = engine_portfolio.build_trade_plan_status_summary()

        self.assertEqual(summary["open_alert_count"], 2)
        self.assertEqual(summary["missing_plan_count"], 1)
        self.assertEqual(len(summary["alerts"]), 2)
        stop_alert = next(alert for alert in summary["alerts"] if alert["alert_type"] == "stop_hit")
        self.assertEqual(stop_alert["payload"]["current_price"], 89.5)

    def test_upsert_trade_plan_alert_locked_updates_without_row_factory(self):
        engine_portfolio.init_db()

        with database.locked_connection() as conn:
            cursor = conn.cursor()
            first_id = engine_portfolio._upsert_trade_plan_alert_locked(
                cursor,
                symbol="AMD",
                alert_type="monitor_degraded",
                severity="warning",
                payload={"error": "first"},
            )
            second_id = engine_portfolio._upsert_trade_plan_alert_locked(
                cursor,
                symbol="AMD",
                alert_type="monitor_degraded",
                severity="warning",
                payload={"error": "second"},
            )
            conn.commit()
            payload_json = conn.execute(
                "SELECT payload_json FROM trade_plan_alerts WHERE id = ?",
                (first_id,),
            ).fetchone()[0]

        self.assertEqual(first_id, second_id)
        self.assertEqual(payload_json, '{"error": "second"}')

    def test_init_db_dedupes_legacy_open_trade_plan_alerts_before_creating_unique_index(self):
        with database.locked_connection() as conn:
            conn.execute(
                """
                CREATE TABLE trade_plan_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id INTEGER,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    payload_json TEXT,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    resolved_at TEXT
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO trade_plan_alerts (
                    plan_id, symbol, alert_type, severity, status, payload_json, first_seen_at, last_seen_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (7, "MRVL", "stop_hit", "critical", "open", '{"current_price":79.8}', "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z", None),
                    (7, "MRVL", "stop_hit", "critical", "open", '{"current_price":79.5}', "2024-01-01T00:01:00Z", "2024-01-01T00:01:00Z", None),
                ],
            )
            conn.commit()

        engine_portfolio.init_db()

        with database.locked_connection() as conn:
            remaining = conn.execute(
                """
                SELECT COUNT(*)
                FROM trade_plan_alerts
                WHERE plan_id = ? AND alert_type = ? AND status = 'open'
                """,
                (7, "stop_hit"),
            ).fetchone()[0]
            indexes = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_trade_plan_alerts_open_%'"
                ).fetchall()
            }
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO trade_plan_alerts (
                        plan_id, symbol, alert_type, severity, status, payload_json, first_seen_at, last_seen_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (7, "MRVL", "stop_hit", "critical", "open", '{"current_price":79.1}', "2024-01-01T00:02:00Z", "2024-01-01T00:02:00Z", None),
                )

        self.assertEqual(remaining, 1)
        self.assertEqual(
            indexes,
            {
                "idx_trade_plan_alerts_open_plan_type",
                "idx_trade_plan_alerts_open_symbol_type_null_plan",
            },
        )
