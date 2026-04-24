import ast
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

import config
import engine_memory
import engine_market
import engine_portfolio
import engine_risk
import engine_technical
import engine_router
import fubon
import nlp_worker
from src import backup as backup_module
from src import database


ROOT = Path(__file__).resolve().parent


def _parse_module(path: Path):
    return ast.parse(path.read_text(encoding="utf-8"))


def _get_function_node(path: Path, name: str) -> ast.FunctionDef:
    for node in _parse_module(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{path.name} is missing function {name}")


def _has_tool_decorator(node: ast.FunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "tool":
            return True
    return False


def _assert_wrapper_delegates(testcase: unittest.TestCase, path: Path, wrapper_name: str, helper_name: str):
    helper = _get_function_node(path, helper_name)
    wrapper = _get_function_node(path, wrapper_name)

    testcase.assertFalse(_has_tool_decorator(helper), f"{path.name}:{helper_name} should stay undecorated")
    testcase.assertTrue(_has_tool_decorator(wrapper), f"{path.name}:{wrapper_name} should stay tool-decorated")
    testcase.assertEqual(len(wrapper.body), 2 if isinstance(wrapper.body[0], ast.Expr) else 1)

    return_stmt = wrapper.body[-1]
    testcase.assertIsInstance(return_stmt, ast.Return, f"{path.name}:{wrapper_name} should end with return helper(...)")
    testcase.assertIsInstance(return_stmt.value, ast.Call, f"{path.name}:{wrapper_name} should call {helper_name}(...)")
    testcase.assertIsInstance(return_stmt.value.func, ast.Name)
    testcase.assertEqual(return_stmt.value.func.id, helper_name)


def _make_ohlcv_frame(periods: int, close_start: float, close_step: float, volume_start: int = 1000, freq: str = "D"):
    index = pd.date_range("2024-01-01", periods=periods, freq=freq)
    closes = [close_start + i * close_step for i in range(periods)]
    opens = [price - 0.5 for price in closes]
    highs = [price + 1 for price in closes]
    lows = [price - 1 for price in closes]
    volumes = [volume_start + (i * 50) for i in range(periods)]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=index,
    )


def _prices_from_returns(returns):
    series = 100 * np.cumprod(1 + np.asarray(returns, dtype=float))
    return pd.DataFrame({"Close": series})


class ToolLogicExtractionSourceTests(unittest.TestCase):
    def test_tool_wrappers_delegate_to_extracted_helpers(self):
        expected = {
            "engine_market.py": {
                "resolve_symbol_identity": "build_symbol_identity_report",
                "get_live_price": "fetch_live_price",
                "get_us_realtime_insight": "build_realtime_insight",
                "get_market_sentiment": "build_sentiment_report",
                "get_stock_news": "build_stock_news_report",
                "get_fundamental_data": "build_fundamental_report",
                "get_technical_analysis": "build_technical_report",
                "get_mean_reversion_signal": "build_mean_reversion_report",
                "get_pairs_trade_signal": "build_pairs_trade_report",
                "get_factor_snapshot": "build_factor_snapshot_report",
                "get_nlp_signal_ic": "build_nlp_signal_ic_report",
                "get_candidate_alpha_panel": "build_candidate_alpha_report",
                "get_market_movers": "build_movers_report",
                "get_market_history": "build_market_history_report",
            },
            "engine_risk.py": {
                "get_v_turn_confirmation": "build_v_turn_report",
                "get_capital_flow_matrix": "build_capital_flow_report",
            },
            "engine_portfolio.py": {
                "get_exchange_rate": "fetch_exchange_rate",
                "update_position": "execute_position_update",
                "get_portfolio_raw_data": "build_portfolio_detailed_raw_data",
                "get_portfolio_analytics": "build_portfolio_analytics_report",
                "get_portfolio_beta_attribution": "build_portfolio_beta_report",
                "get_portfolio_risk_overlay": "build_portfolio_risk_overlay_report",
                "get_portfolio_rebalance_plan": "build_portfolio_rebalance_report",
                "get_risk_parity_weights": "build_risk_parity_report",
                "calculate_position_size": "build_position_size_report",
                "calculate_pnl": "calculate_position_pnl",
            },
            "engine_technical.py": {
                "calculate_indicator": "evaluate_indicator_formula",
            },
            "fubon.py": {
                "get_market_trades": "build_market_trades_report",
                "get_price_volumes": "build_price_volumes_report",
                "get_historical_stats": "build_historical_stats_report",
                "get_txo_sentiment": "build_txo_sentiment_report",
                "get_quote_and_orderbook": "build_quote_and_orderbook_report",
                "get_market_hot_stocks": "build_market_hot_stocks_report",
                "get_intraday_trend": "build_intraday_trend_report",
            },
        }

        for filename, mapping in expected.items():
            path = ROOT / filename
            for wrapper_name, helper_name in mapping.items():
                _assert_wrapper_delegates(self, path, wrapper_name, helper_name)

    def test_internal_python_callers_use_direct_helpers(self):
        router_source = (ROOT / "engine_router.py").read_text(encoding="utf-8")
        scheduler_source = (ROOT / "src" / "scheduler.py").read_text(encoding="utf-8")
        sentinel_source = (ROOT / "data_sentinel.py").read_text(encoding="utf-8")
        market_source = (ROOT / "engine_market.py").read_text(encoding="utf-8")

        self.assertIn("market.build_technical_report(symbol)", router_source)
        self.assertIn("market.build_realtime_insight(symbol)", router_source)
        self.assertIn("market.build_sentiment_report()", router_source)
        self.assertIn("market.fetch_live_price(symbol)", router_source)
        self.assertIn("risk.build_v_turn_report()", scheduler_source)
        self.assertNotIn("risk.get_v_turn_confirmation()", scheduler_source)
        self.assertIn("fubon.build_market_hot_stocks_report()", sentinel_source)
        self.assertIn("_fubon_provider.build_historical_stats_report(symbol)", market_source)


class DirectHelperRuntimeTests(unittest.TestCase):
    def setUp(self):
        engine_router._nlp_ic_cache["entries"].clear()
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db_file = database.DB_FILE
        database.DB_FILE = Path(self.tempdir.name) / "tool-refactor-test.db"
        self.original_fubon_ready = fubon.fubon_ready
        self.original_fubon_sdk = fubon.fubon_sdk

    def tearDown(self):
        database.DB_FILE = self.original_db_file
        fubon.fubon_ready = self.original_fubon_ready
        fubon.fubon_sdk = self.original_fubon_sdk
        self.tempdir.cleanup()

    def _install_fubon_sdk(self, stock_client=None, futopt_client=None):
        fubon.fubon_ready = True
        fubon.fubon_sdk = SimpleNamespace(
            marketdata=SimpleNamespace(
                rest_client=SimpleNamespace(stock=stock_client, futopt=futopt_client)
            )
        )

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
            event = conn.execute(
                "SELECT event_type FROM trade_plan_events WHERE plan_id = ? ORDER BY id",
                (plan_id,),
            ).fetchall()
        fetched_plan = engine_portfolio.get_trade_plan(plan_id)
        active_plan = engine_portfolio.get_active_trade_plan("MRVL")
        active_plans = engine_portfolio.list_active_trade_plans()

        self.assertEqual(
            plan,
            ("MRVL", "active", "manual_backfill", 80.0, 95.0, 105.0, 60, "sector_rotation", "semi rotation re-accelerating"),
        )
        self.assertEqual([row[0] for row in event], ["plan_created", "plan_activated"])
        self.assertEqual(fetched_plan["id"], plan_id)
        self.assertEqual(fetched_plan["status"], "active")
        self.assertEqual(active_plan["id"], plan_id)
        self.assertEqual([row["id"] for row in active_plans], [plan_id])

    def test_get_fubon_account_snapshot_uses_single_login_for_inventory_and_cash(self):
        account = object()
        inventory_item = SimpleNamespace(
            stock_no="2330",
            today_qty=100,
            odd=SimpleNamespace(today_qty=20),
        )
        unrealized_item = SimpleNamespace(
            stock_no="2330",
            today_qty=120,
            cost_price=600.0,
        )
        accounting = SimpleNamespace(
            inventories=Mock(return_value=SimpleNamespace(is_success=True, data=[inventory_item])),
            unrealized_gains_and_loses=Mock(return_value=SimpleNamespace(is_success=True, data=[unrealized_item])),
            bank_remain=Mock(return_value=SimpleNamespace(is_success=True, data=SimpleNamespace(available_balance=345678))),
        )
        login = Mock(return_value=SimpleNamespace(is_success=True, data=[account]))

        fubon.fubon_ready = True
        fubon.fubon_sdk = SimpleNamespace(apikey_login=login, accounting=accounting)

        snapshot = fubon.get_fubon_account_snapshot()

        self.assertEqual(
            snapshot,
            {"success": True, "inventories": {"2330": {"shares": 120, "cost": 600.0}}, "cash_twd": 345678, "error": None},
        )
        login.assert_called_once()

    def test_get_fubon_account_snapshot_fails_closed_on_inventory_subcall_failure(self):
        account = object()
        accounting = SimpleNamespace(
            inventories=Mock(return_value=SimpleNamespace(is_success=False, message="inventories unavailable", data=[])),
            unrealized_gains_and_loses=Mock(return_value=SimpleNamespace(is_success=True, data=[])),
            bank_remain=Mock(return_value=SimpleNamespace(is_success=True, data=SimpleNamespace(available_balance=345678))),
        )
        login = Mock(return_value=SimpleNamespace(is_success=True, data=[account]))

        fubon.fubon_ready = True
        fubon.fubon_sdk = SimpleNamespace(apikey_login=login, accounting=accounting)

        snapshot = fubon.get_fubon_account_snapshot()

        self.assertFalse(snapshot["success"])
        self.assertIn("inventories", snapshot["error"])

    def test_get_fubon_account_snapshot_fails_closed_on_unrealized_subcall_failure(self):
        account = object()
        inventory_item = SimpleNamespace(
            stock_no="2330",
            today_qty=100,
            odd=SimpleNamespace(today_qty=0),
        )
        accounting = SimpleNamespace(
            inventories=Mock(return_value=SimpleNamespace(is_success=True, data=[inventory_item])),
            unrealized_gains_and_loses=Mock(
                return_value=SimpleNamespace(is_success=False, message="unrealized unavailable", data=[])
            ),
            bank_remain=Mock(return_value=SimpleNamespace(is_success=True, data=SimpleNamespace(available_balance=345678))),
        )
        login = Mock(return_value=SimpleNamespace(is_success=True, data=[account]))

        fubon.fubon_ready = True
        fubon.fubon_sdk = SimpleNamespace(apikey_login=login, accounting=accounting)

        snapshot = fubon.get_fubon_account_snapshot()

        self.assertFalse(snapshot["success"])
        self.assertIn("unrealized", snapshot["error"])

    def test_execute_position_update_runs_without_tool_wrapper(self):
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
            return_value={"allowed": True, "approved_shares": 2.0, "approved_twd_total": 6400.0, "message": "", "note": None},
        ):
            result = engine_portfolio.execute_position_update("AAPL", 100.0, 2.0, action="buy")

        self.assertIn("✅ 買進成功", result)
        with database.locked_connection() as conn:
            aapl = conn.execute("SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = 'AAPL'").fetchone()
            cash = conn.execute("SELECT shares, twd_cost FROM portfolio WHERE symbol = 'CASH_USD'").fetchone()

        self.assertEqual(aapl, (100.0, 2.0, 6400.0))
        self.assertEqual(cash, (800.0, 25600.0))

    def test_update_position_tool_refreshes_portfolio_health_after_success(self):
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
            return_value={"allowed": True, "approved_shares": 2.0, "approved_twd_total": 6400.0, "message": "", "note": None},
        ), patch.object(
            engine_portfolio, "refresh_portfolio_health_summary", return_value={"summary": "ok"}
        ) as mock_refresh:
            result = engine_portfolio.update_position("AAPL", 100.0, 2.0, action="buy")

        self.assertIn("✅ 買進成功", result)
        mock_refresh.assert_called_once_with(source="portfolio_trade")

    def test_update_trade_followup_status_raises_for_missing_followup(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            cursor = conn.cursor()

            with self.assertRaises(ValueError):
                engine_portfolio._update_trade_followup_status(cursor, 999999, status="done")

            conn.execute(
                """
                INSERT INTO trade_log (symbol, action, price, shares)
                VALUES (?, ?, ?, ?)
                """,
                ("AAPL", "buy", 100.0, 2.0),
            )
            trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, prompt_text, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trade_log_id, "AAPL", "buy", "pending", "pending", "check trade", 0),
            )
            followup_id = conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0]

            engine_portfolio._update_trade_followup_status(
                cursor,
                followup_id,
                status="done",
                prompt_state="responded",
                user_reason="approved",
                skipped=1,
                responded_at="2024-01-01T00:00:00Z",
            )
            conn.commit()

            followup = conn.execute(
                "SELECT status, prompt_state, user_reason, skipped, responded_at FROM trade_followups WHERE id = ?",
                (followup_id,),
            ).fetchone()

        self.assertEqual(followup, ("done", "responded", "approved", 1, "2024-01-01T00:00:00Z"))

    def test_execute_position_update_records_trade_audit_rows(self):
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
            return_value={"allowed": True, "approved_shares": 2.0, "approved_twd_total": 6400.0, "message": "", "note": None},
        ):
            buy_result = engine_portfolio.execute_position_update("AAPL", 100.0, 2.0, action="buy")
            sell_result = engine_portfolio.execute_position_update("AAPL", 120.0, 1.0, action="sell")
            set_result = engine_portfolio.execute_position_update("MSFT", 200.0, 3.0, action="set", locked=1)

        self.assertIn("✅ 買進成功", buy_result)
        self.assertIn("✅ 賣出成功", sell_result)
        self.assertIn("✅ 校正成功", set_result)

        with database.locked_connection() as conn:
            rows = conn.execute(
                "SELECT action, symbol, settle_currency, settle_amount, fx_rate, realized_pnl, cash_before, cash_after, note "
                "FROM trade_log ORDER BY id"
            ).fetchall()

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], ("buy", "AAPL", "CASH_USD", 200.0, 32.0, None, 1000.0, 800.0, None))
        self.assertEqual(rows[1], ("sell", "AAPL", "CASH_USD", 120.0, 32.0, 640.0, 800.0, 920.0, None))
        self.assertEqual(rows[2], ("set", "MSFT", None, None, 32.0, None, None, None, "manual set; locked=1"))

    def test_sync_fubon_portfolio_state_records_inferred_add_and_snapshot_context(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 500.0, 100.0, 50000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        snapshot_payload = {"spy_change_1d": 0.0123, "risk_state": "🔴 警戒", "nlp_alpha": -0.2}
        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": True, "inventories": {"2330": {"shares": 150.0, "cost": 520.0}}, "cash_twd": 85000.0},
        ), patch.object(
            engine_portfolio, "_build_sync_decision_snapshot", return_value=snapshot_payload
        ):
            result = engine_portfolio.sync_fubon_portfolio_state(source="portfolio_query")

        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["events"][0]["action"], "sync_buy")
        self.assertAlmostEqual(result["events"][0]["shares_delta"], 50.0)

        with database.locked_connection() as conn:
            position = conn.execute(
                "SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = '2330'"
            ).fetchone()
            cash = conn.execute(
                "SELECT shares, twd_cost FROM portfolio WHERE symbol = 'CASH_TWD'"
            ).fetchone()
            audit = conn.execute(
                "SELECT action, symbol, price, shares, note, decision_snapshot FROM trade_log ORDER BY id"
            ).fetchone()

        self.assertEqual(position, (520.0, 150.0, 78000.0))
        self.assertEqual(cash, (85000.0, 85000.0))
        self.assertEqual(audit[0], "sync_buy")
        self.assertEqual(audit[1], "2330")
        self.assertAlmostEqual(audit[2], 560.0)
        self.assertAlmostEqual(audit[3], 50.0)
        self.assertIn("inferred average add", audit[4])
        self.assertEqual(json.loads(audit[5])["nlp_alpha"], -0.2)

    def test_sync_fubon_portfolio_state_creates_pending_followup_for_sync_buy(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 500.0, 100.0, 50000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": True, "inventories": {"2330": {"shares": 150.0, "cost": 520.0}}, "cash_twd": 85000.0},
        ), patch.object(
            engine_portfolio, "_build_sync_decision_snapshot", return_value={"risk_state": "🔴 警戒"}
        ):
            result = engine_portfolio.sync_fubon_portfolio_state(source="scheduler")

        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["followup_count"], 1)

        with database.locked_connection() as conn:
            followup = conn.execute(
                "SELECT trade_log_id, symbol, action, status, prompt_state, skipped FROM trade_followups ORDER BY id"
            ).fetchone()
            trade_log_id = conn.execute("SELECT id FROM trade_log WHERE action = 'sync_buy' ORDER BY id").fetchone()[0]

        self.assertEqual(followup, (trade_log_id, "2330", "sync_buy", "pending", "pending", 0))

    def test_sync_fubon_portfolio_state_skips_followup_for_portfolio_query_source(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 500.0, 100.0, 50000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": True, "inventories": {"2330": {"shares": 150.0, "cost": 520.0}}, "cash_twd": 85000.0},
        ), patch.object(
            engine_portfolio, "_build_sync_decision_snapshot", return_value={"risk_state": "🔴 警戒"}
        ):
            result = engine_portfolio.sync_fubon_portfolio_state(source="portfolio_query")

        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["followup_count"], 0)

        with database.locked_connection() as conn:
            followup_count = conn.execute("SELECT COUNT(*) FROM trade_followups").fetchone()[0]

        self.assertEqual(followup_count, 0)

    def test_build_trade_followup_weekly_report_splits_planned_vs_unplanned(self):
        engine_portfolio.init_db()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        planned_time = (now - timedelta(days=5)).isoformat().replace("+00:00", "Z")
        unplanned_time = (now - timedelta(days=4)).isoformat().replace("+00:00", "Z")
        planned_sell_time = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
        unplanned_sell_time = (now - timedelta(days=1)).isoformat().replace("+00:00", "Z")

        with database.locked_connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_log (
                    timestamp, symbol, action, price, shares, decision_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (planned_time, "AAPL", "sync_buy", 100.0, 10.0, json.dumps({"nlp_alpha": 0.2})),
            )
            planned_trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, user_reason, target_price, stop_price, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (planned_trade_log_id, "AAPL", "sync_buy", "resolved", "resolved", "rotation", 120.0, 95.0, 0),
            )
            conn.execute(
                """
                INSERT INTO trade_log (
                    timestamp, symbol, action, price, shares, settle_currency, settle_amount, fx_rate, realized_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (planned_sell_time, "AAPL", "sell", 110.0, 10.0, "CASH_USD", 1100.0, 1.0, 100.0),
            )

            conn.execute(
                """
                INSERT INTO trade_log (
                    timestamp, symbol, action, price, shares, decision_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (unplanned_time, "TSLA", "sync_buy", 200.0, 5.0, json.dumps({"nlp_alpha": -0.1})),
            )
            unplanned_trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, skipped
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (unplanned_trade_log_id, "TSLA", "sync_buy", "resolved", "resolved", 1),
            )
            conn.execute(
                """
                INSERT INTO trade_log (
                    timestamp, symbol, action, price, shares, settle_currency, settle_amount, fx_rate, realized_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (unplanned_sell_time, "TSLA", "sell", 190.0, 5.0, "CASH_USD", 950.0, 1.0, -50.0),
            )
            conn.commit()

        report = engine_portfolio.build_trade_followup_weekly_report(days=7)

        self.assertIn("有計畫", report)
        self.assertIn("無計畫", report)
        self.assertIn("勝率", report)
        self.assertIn("平均報酬", report)
        self.assertIn("有計畫: 1 筆", report)
        self.assertIn("無計畫: 1 筆", report)
        self.assertIn("10.0%", report)
        self.assertIn("-5.0%", report)

    def test_sync_fubon_portfolio_state_records_sync_sell_without_polluting_closed_trade_stats(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 500.0, 100.0, 50000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": True, "inventories": {}, "cash_twd": 120000.0},
        ), patch.object(
            engine_portfolio, "_build_sync_decision_snapshot", return_value={"risk_state": "🟡 整理"}
        ):
            result = engine_portfolio.sync_fubon_portfolio_state(source="scheduler")

        self.assertEqual(result["event_count"], 1)
        self.assertEqual(result["events"][0]["action"], "sync_sell")

        with database.locked_connection() as conn:
            position = conn.execute("SELECT * FROM portfolio WHERE symbol = '2330'").fetchone()
            audit = conn.execute(
                "SELECT action, symbol, price, shares, note FROM trade_log ORDER BY id"
            ).fetchone()
            manual_sell_count = conn.execute(
                "SELECT COUNT(*) FROM trade_log WHERE action = 'sell'"
            ).fetchone()[0]

        self.assertIsNone(position)
        self.assertEqual(audit[0], "sync_sell")
        self.assertEqual(audit[1], "2330")
        self.assertAlmostEqual(audit[2], 500.0)
        self.assertAlmostEqual(audit[3], 100.0)
        self.assertIn("execution price unavailable", audit[4])
        self.assertEqual(manual_sell_count, 0)

    def test_sync_fubon_portfolio_state_skips_reconcile_on_broker_fetch_failure(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 500.0, 100.0, 50000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": False, "inventories": {}, "cash_twd": None, "error": "login failed"},
        ), patch.object(engine_portfolio, "_build_sync_decision_snapshot", return_value={"risk_state": "n/a"}):
            result = engine_portfolio.sync_fubon_portfolio_state(source="scheduler")

        self.assertFalse(result["synced"])
        self.assertIn("login failed", result["message"])
        with database.locked_connection() as conn:
            position = conn.execute(
                "SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = '2330'"
            ).fetchone()
            trade_count = conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]

        self.assertEqual(position, (500.0, 100.0, 50000.0))
        self.assertEqual(trade_count, 0)

    def test_sync_fubon_portfolio_state_prefetches_decision_snapshot_before_db_lock(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 500.0, 100.0, 50000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        lock_state = {"held": False}

        class TrackingLock:
            def __enter__(self_inner):
                lock_state["held"] = True

            def __exit__(self_inner, exc_type, exc, tb):
                lock_state["held"] = False

        def build_snapshot(symbol):
            self.assertFalse(lock_state["held"])
            return {"symbol": symbol}

        with patch.object(engine_portfolio, "db_lock", TrackingLock()), patch.object(
            fubon, "fubon_ready", True
        ), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": True, "inventories": {"2330": {"shares": 150.0, "cost": 520.0}}, "cash_twd": 85000.0},
        ), patch.object(
            engine_portfolio, "_build_sync_decision_snapshot", side_effect=build_snapshot
        ):
            result = engine_portfolio.sync_fubon_portfolio_state(source="scheduler")

        self.assertTrue(result["synced"])
        self.assertEqual(result["event_count"], 1)

    def test_sync_fubon_portfolio_state_requires_explicit_success_flag(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 500.0, 100.0, 50000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"inventories": {}, "cash_twd": None, "error": "missing success flag"},
        ), patch.object(engine_portfolio, "_build_sync_decision_snapshot", return_value={"risk_state": "n/a"}):
            result = engine_portfolio.sync_fubon_portfolio_state(source="scheduler")

        self.assertFalse(result["synced"])
        self.assertIn("missing success flag", result["message"])
        with database.locked_connection() as conn:
            position = conn.execute(
                "SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = '2330'"
            ).fetchone()
            trade_count = conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]

        self.assertEqual(position, (500.0, 100.0, 50000.0))
        self.assertEqual(trade_count, 0)

    def test_sync_fubon_portfolio_state_ignores_small_cost_rounding_noise(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 520.0, 100.0, 52000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": True, "inventories": {"2330": {"shares": 100.0, "cost": 520.00001}}, "cash_twd": 90000.0},
        ), patch.object(engine_portfolio, "_build_sync_decision_snapshot", return_value={"risk_state": "n/a"}):
            result = engine_portfolio.sync_fubon_portfolio_state(source="scheduler")

        self.assertTrue(result["synced"])
        self.assertEqual(result["event_count"], 0)
        with database.locked_connection() as conn:
            trade_count = conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]
        self.assertEqual(trade_count, 0)

    def test_sync_fubon_portfolio_state_uses_single_snapshot_for_alias_resolution(self):
        first_rows = [("2330", 500.0, 100.0, 50000.0, 0)]
        second_rows = [
            ("2330", 500.0, 100.0, 50000.0, 0),
            ("2330.TW", 510.0, 100.0, 51000.0, 0),
        ]
        cursor = Mock()
        cursor.fetchall.side_effect = [first_rows, second_rows]
        conn = Mock()
        conn.cursor.return_value = cursor
        conn.close.return_value = None
        conn.commit.return_value = None

        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": True, "inventories": {"2330": {"shares": 150.0, "cost": 520.0}}, "cash_twd": 85000.0},
        ), patch.object(engine_portfolio, "get_connection", return_value=conn), patch.object(
            engine_portfolio, "_build_sync_decision_snapshot", return_value={"risk_state": "n/a"}
        ), patch.object(engine_portfolio, "_record_trade_log", return_value=101) as mock_log, patch.object(
            engine_portfolio, "_upsert_portfolio_row"
        ) as mock_upsert:
            result = engine_portfolio.sync_fubon_portfolio_state(source="portfolio_query")

        self.assertTrue(result["synced"])
        self.assertEqual(result["event_count"], 1)
        self.assertEqual(
            result["events"],
            [{"symbol": "2330", "action": "sync_buy", "shares_delta": 50.0, "price": 560.0}],
        )
        self.assertEqual(mock_log.call_args.kwargs["symbol"], "2330")
        self.assertTrue(any(call.args[1] == "2330" for call in mock_upsert.call_args_list))

    def test_sync_fubon_portfolio_state_skips_ambiguous_alias_collisions(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                [
                    ("2330", 500.0, 100.0, 50000.0, 0),
                    ("2330.TW", 510.0, 100.0, 51000.0, 0),
                    ("CASH_TWD", 1.0, 90000.0, 90000.0, 0),
                ],
            )
            conn.commit()

        with patch.object(fubon, "fubon_ready", True), patch.object(
            fubon,
            "get_fubon_account_snapshot",
            return_value={"success": True, "inventories": {"2330": {"shares": 150.0, "cost": 520.0}}, "cash_twd": 85000.0},
        ), patch.object(engine_portfolio, "_build_sync_decision_snapshot", return_value={"risk_state": "n/a"}):
            result = engine_portfolio.sync_fubon_portfolio_state(source="scheduler")

        self.assertTrue(result["synced"])
        self.assertEqual(result["event_count"], 0)
        self.assertIn("2330", result.get("skipped_aliases", []))
        with database.locked_connection() as conn:
            rows = conn.execute(
                "SELECT symbol, cost, shares, twd_cost FROM portfolio WHERE symbol IN ('2330', '2330.TW') ORDER BY symbol"
            ).fetchall()
            trade_count = conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]

        self.assertEqual(rows, [("2330", 500.0, 100.0, 50000.0), ("2330.TW", 510.0, 100.0, 51000.0)])
        self.assertEqual(trade_count, 0)

    def test_build_portfolio_raw_data_triggers_fubon_sync_before_render(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("2330", 520.0, 150.0, 78000.0, 0),
            )
            conn.commit()

        with patch.object(fubon, "fubon_ready", True), patch.object(
            engine_portfolio, "sync_fubon_portfolio_state", return_value={"event_count": 0, "events": []}
        ) as mock_sync, patch.object(engine_portfolio, "get_symbol_name", return_value="TSMC"):
            result = engine_portfolio.build_portfolio_raw_data()

        mock_sync.assert_called_once_with(source="portfolio_query", sync_memory=False)
        self.assertIn("2330|150.0sh|cost=520.0|TW", result)

    def test_execute_position_update_applies_scaled_pretrade_gate(self):
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
                "approved_shares": 1.0,
                "approved_twd_total": 3200.0,
                "message": "⚠️ 風控縮倉：AAPL 由 2.0000 股縮至 1.0000 股 (單一持股上限 15.0% NAV)。",
                "note": "risk_gate:單一持股上限 15.0% NAV; requested_shares=2.0000; approved_shares=1.0000",
            },
        ):
            result = engine_portfolio.execute_position_update("AAPL", 100.0, 2.0, action="buy")

        self.assertIn("⚠️ 風控縮倉", result)
        self.assertIn("✅ 買進成功", result)
        with database.locked_connection() as conn:
            aapl = conn.execute("SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = 'AAPL'").fetchone()
            cash = conn.execute("SELECT shares, twd_cost FROM portfolio WHERE symbol = 'CASH_USD'").fetchone()
            note = conn.execute("SELECT note FROM trade_log WHERE action = 'buy'").fetchone()[0]

        self.assertEqual(aapl, (100.0, 1.0, 3200.0))
        self.assertEqual(cash, (900.0, 28800.0))
        self.assertIn("approved_shares=1.0000", note)

    def test_execute_position_update_records_confirmed_buy_even_when_risk_gate_blocks(self):
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
                "allowed": False,
                "approved_shares": 0.0,
                "approved_twd_total": 0.0,
                "message": "❌ 風控拒單：Technology 曝險已達產業上限 30.0% of NAV。",
                "note": None,
            },
        ):
            result = engine_portfolio.execute_position_update(
                "ONDS",
                9.8,
                2.0,
                action="buy",
                enforce_pretrade_gate=False,
            )

        self.assertIn("⚠️ 成交後風控警告", result)
        self.assertIn("✅ 買進成功", result)
        with database.locked_connection() as conn:
            onds = conn.execute("SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = 'ONDS'").fetchone()
            cash = conn.execute("SELECT shares, twd_cost FROM portfolio WHERE symbol = 'CASH_USD'").fetchone()
            note = conn.execute("SELECT note FROM trade_log WHERE action = 'buy'").fetchone()[0]

        self.assertEqual(onds, (9.8, 2.0, 627.2))
        self.assertEqual(cash, (980.4, 31372.8))
        self.assertIn("post_trade_warning", note)

    def test_execute_position_update_keeps_full_confirmed_buy_when_gate_would_scale(self):
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
                "approved_shares": 1.0,
                "approved_twd_total": 3200.0,
                "message": "⚠️ 風控縮倉：AAPL 由 2.0000 股縮至 1.0000 股 (單一持股上限 15.0% NAV)。",
                "note": "risk_gate:單一持股上限 15.0% NAV; requested_shares=2.0000; approved_shares=1.0000",
            },
        ):
            result = engine_portfolio.execute_position_update(
                "AAPL",
                100.0,
                2.0,
                action="buy",
                enforce_pretrade_gate=False,
            )

        self.assertIn("⚠️ 成交後風控警告", result)
        self.assertIn("原始 2.0000 股入帳", result)
        with database.locked_connection() as conn:
            aapl = conn.execute("SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = 'AAPL'").fetchone()
            cash = conn.execute("SELECT shares, twd_cost FROM portfolio WHERE symbol = 'CASH_USD'").fetchone()
            note = conn.execute("SELECT note FROM trade_log WHERE action = 'buy'").fetchone()[0]

        self.assertEqual(aapl, (100.0, 2.0, 6400.0))
        self.assertEqual(cash, (800.0, 25600.0))
        self.assertIn("post_trade_warning", note)

    def test_execute_position_update_sell_recreates_missing_cash_row(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("AAPL", 100.0, 2.0, 6400.0, 0),
            )
            conn.commit()

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0):
            result = engine_portfolio.execute_position_update("AAPL", 120.0, 1.0, action="sell")

        self.assertIn("✅ 賣出成功", result)
        with database.locked_connection() as conn:
            cash = conn.execute("SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = 'CASH_USD'").fetchone()

        self.assertEqual(cash, (32.0, 120.0, 3840.0))

    def test_pretrade_risk_gate_blocks_averaging_down_when_throttled(self):
        snapshots = [
            {"symbol": "AAPL", "is_cash": False, "market_value_twd": 8000.0, "pnl_value_twd": -1200.0},
            {"symbol": "CASH_USD", "is_cash": True, "market_value_twd": 50000.0, "pnl_value_twd": 0.0},
        ]
        overlay = {
            "current_nav_twd": 58000.0,
            "trade_mode_label": "🟠 Risk-Off",
            "trade_mode": "risk_off",
            "allow_new_longs": True,
            "allow_average_down": False,
            "governor_message": "回撤超過 5%，新單砍半且禁止攤平虧損部位。",
            "recommended_gross_scale": 0.5,
            "gross_exposure_twd": 8000.0,
            "target_beta_band": [0.4, 0.7],
            "current_beta_to_nav": 0.3,
        }

        with patch.object(engine_portfolio, "_load_portfolio_rows", return_value=[]), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ), patch.object(
            engine_portfolio, "compute_portfolio_risk_overlay", return_value=overlay
        ):
            gate = engine_portfolio._apply_pretrade_risk_gate("AAPL", "buy", 2.0, 6400.0)

        self.assertFalse(gate["allowed"])
        self.assertIn("禁止攤平虧損部位", gate["message"])

    def test_pretrade_risk_gate_scales_order_to_single_name_cap(self):
        snapshots = [
            {"symbol": "CASH_USD", "is_cash": True, "market_value_twd": 100000.0, "pnl_value_twd": 0.0},
        ]
        overlay = {
            "current_nav_twd": 100000.0,
            "trade_mode_label": "🟢 Normal",
            "trade_mode": "normal",
            "allow_new_longs": True,
            "allow_average_down": True,
            "governor_message": "回撤仍在可接受區間。",
            "recommended_gross_scale": 1.0,
            "gross_exposure_twd": 0.0,
            "target_beta_band": [0.8, 1.1],
            "current_beta_to_nav": 0.2,
        }

        with patch.object(engine_portfolio, "_load_portfolio_rows", return_value=[]), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ), patch.object(
            engine_portfolio, "compute_portfolio_risk_overlay", return_value=overlay
        ), patch.object(
            engine_portfolio.market, "get_asset_profile", return_value={"asset_type": "Value_Holding", "sector": "Unknown"}
        ), patch.object(
            engine_portfolio, "_estimate_symbol_beta", return_value={"symbol": "AAPL", "benchmark": "SPY", "beta": 0.8, "observations": 60}
        ):
            gate = engine_portfolio._apply_pretrade_risk_gate("AAPL", "buy", 200.0, 20000.0)

        self.assertTrue(gate["allowed"])
        self.assertAlmostEqual(gate["approved_shares"], 150.0, places=2)
        self.assertAlmostEqual(gate["approved_twd_total"], 15000.0, places=2)
        self.assertIn("單一持股上限 15.0% NAV", gate["message"])

    def test_get_asset_profile_enriches_etf_tracking_index_and_bucket(self):
        class StubTicker:
            info = {
                "quoteType": "ETF",
                "longName": "Invesco QQQ Trust",
                "shortName": "Invesco QQQ Trust, Series 1",
                "sector": None,
                "industry": None,
                "category": "Large Growth",
                "fundFamily": "Invesco",
                "legalType": "Exchange Traded Fund",
                "currency": "USD",
                "longBusinessSummary": "To maintain the correspondence with the stocks in the NASDAQ-100 Index.",
            }

        profile = None
        with patch.object(engine_market, "get_ticker", return_value=StubTicker()):
            profile = engine_market.get_asset_profile("QQQ")

        self.assertTrue(profile["is_etf"])
        self.assertEqual(profile["quote_type"], "ETF")
        self.assertEqual(profile["tracking_index"], "NASDAQ-100")
        self.assertEqual(profile["concentration_bucket"], "Technology")
        self.assertEqual(profile["asset_type"], "Tech_Momentum")
        self.assertEqual(profile["lookup_symbol"], "QQQ")

    def test_get_asset_profile_uses_lookup_normalizer_for_taiwan_etf_codes(self):
        class StubTicker:
            info = {
                "quoteType": "ETF",
                "longName": "台灣指數 ETF",
                "shortName": "台灣指數 ETF",
                "sector": None,
                "industry": None,
                "category": "Large Blend",
                "fundFamily": "Test",
                "legalType": "Exchange Traded Fund",
                "currency": "TWD",
                "longBusinessSummary": "Tracks the Taiwan Blue Chip Index.",
            }

        with patch.object(engine_market, "_normalize_lookup_symbol", return_value="00981A.TW"), patch.object(
            engine_market, "get_ticker", return_value=StubTicker()
        ) as mock_ticker:
            profile = engine_market.get_asset_profile("00981A")

        mock_ticker.assert_called_once_with("00981A.TW")
        self.assertEqual(profile["lookup_symbol"], "00981A.TW")
        self.assertTrue(profile["is_etf"])

    def test_get_asset_profile_uses_local_registry_for_taiwan_active_etf_without_llm(self):
        class StubTicker:
            info = {
                "quoteType": "ETF",
                "longName": "Uni-President Asset Management Corp - UPAMC Taiwan Growth Active ETF",
                "shortName": "UPAMC Taiwan Growth Active ETF",
                "sector": None,
                "industry": None,
                "category": "Large Growth",
                "fundFamily": "Uni-President",
                "legalType": "Exchange Traded Fund",
                "currency": "TWD",
                "longBusinessSummary": "An actively managed Taiwan growth ETF.",
            }

        with patch.object(engine_market, "get_ticker", return_value=StubTicker()), patch(
            "src.llm.quick_call", side_effect=AssertionError("LLM fallback should not run")
        ):
            profile = engine_market.get_asset_profile("00981A")

        self.assertEqual(profile["fund_family"], "統一投信")
        self.assertEqual(profile["strategy_type"], "Active Taiwan Growth ETF")
        self.assertEqual(profile["concentration_bucket"], "Technology")
        self.assertEqual(profile["asset_type"], "Tech_Momentum")

    def test_get_asset_profile_uses_local_registry_for_us_active_growth_etf(self):
        class StubTicker:
            info = {
                "quoteType": "ETF",
                "longName": "Fidelity Blue Chip Growth ETF",
                "shortName": "Fidelity Blue Chip Growth ETF",
                "sector": None,
                "industry": None,
                "category": "Large Growth",
                "fundFamily": "Fidelity Investments",
                "legalType": "Exchange Traded Fund",
                "currency": "USD",
                "longBusinessSummary": "An actively managed blue chip growth ETF.",
            }

        with patch.object(engine_market, "get_ticker", return_value=StubTicker()):
            profile = engine_market.get_asset_profile("FBCG")

        self.assertEqual(profile["strategy_type"], "Blue Chip Growth Catalyst Strategy")
        self.assertEqual(profile["fund_family"], "Fidelity Investments")
        self.assertEqual(profile["concentration_bucket"], "Technology")
        self.assertEqual(profile["asset_type"], "Tech_Momentum")

    def test_pretrade_risk_gate_uses_etf_bucket_and_explains_limit_source(self):
        snapshots = [
            {"symbol": "XLK", "is_cash": False, "market_value_twd": 31000.0, "pnl_value_twd": 1200.0},
            {"symbol": "CASH_USD", "is_cash": True, "market_value_twd": 69000.0, "pnl_value_twd": 0.0},
        ]
        overlay = {
            "current_nav_twd": 100000.0,
            "trade_mode_label": "🟡 Soft Throttle",
            "trade_mode": "soft_throttle",
            "allow_new_longs": True,
            "allow_average_down": False,
            "governor_message": "回撤超過 3%，新倉位縮到 0.7x。",
            "recommended_gross_scale": 1.0,
            "gross_exposure_twd": 31000.0,
            "target_beta_band": [0.8, 1.1],
            "current_beta_to_nav": 0.2,
        }

        def _profile(symbol):
            if symbol == "XLK":
                return {
                    "asset_type": "Tech_Momentum",
                    "sector": "Unknown",
                    "is_etf": True,
                    "tracking_index": "Technology Select Sector",
                    "concentration_bucket": "Technology",
                }
            return {
                "asset_type": "Tech_Momentum",
                "sector": "Unknown",
                "is_etf": True,
                "tracking_index": "NASDAQ-100",
                "concentration_bucket": "Technology",
            }

        with patch.object(engine_portfolio, "_load_portfolio_rows", return_value=[]), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ), patch.object(
            engine_portfolio, "compute_portfolio_risk_overlay", return_value=overlay
        ), patch.object(
            engine_portfolio.market, "get_asset_profile", side_effect=_profile
        ):
            gate = engine_portfolio._apply_pretrade_risk_gate("QQQ", "buy", 10.0, 5000.0)

        self.assertFalse(gate["allowed"])
        self.assertIn("Technology 曝險已達集中上限 25.0% of NAV", gate["message"])
        self.assertIn("來源: 🟡 Soft Throttle 檔位 + Tech_Momentum tighten", gate["message"])
        self.assertIn("ETF 指數: NASDAQ-100", gate["message"])

    def test_build_position_size_report_caps_by_available_capital(self):
        with patch.object(engine_portfolio, "build_portfolio_analysis", return_value={"total_current": 100_000.0}), patch.object(
            engine_portfolio, "fetch_exchange_rate", return_value=32.0
        ), patch.object(
            engine_technical.IndicatorCalculator, "HIGH", return_value=[101.0, 102.0]
        ), patch.object(
            engine_technical.IndicatorCalculator, "LOW", return_value=[99.0, 100.0]
        ), patch.object(
            engine_technical.IndicatorCalculator, "CLOSE", return_value=[100.0, 100.0]
        ), patch.object(
            engine_technical.IndicatorCalculator, "ATR", return_value=[0.5]
        ):
            result = engine_portfolio.build_position_size_report("AAPL", risk_pct=2.0)

        self.assertIn("ATR 倉位計算", result)
        self.assertIn("建議股數: 31 股", result)
        self.assertIn("已受總資金上限限制", result)

    def test_build_portfolio_analytics_report_summarizes_closed_book_metrics(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_log (
                    timestamp, symbol, action, price, shares, settle_currency,
                    settle_amount, fx_rate, realized_pnl
                ) VALUES (?, ?, 'sell', ?, ?, 'CASH_USD', ?, ?, ?)
                """,
                ("2026-01-02T14:30:00Z", "AAPL", 110.0, 1.0, 110.0, 1.0, 10.0),
            )
            conn.execute(
                """
                INSERT INTO trade_log (
                    timestamp, symbol, action, price, shares, settle_currency,
                    settle_amount, fx_rate, realized_pnl
                ) VALUES (?, ?, 'sell', ?, ?, 'CASH_USD', ?, ?, ?)
                """,
                ("2026-01-06T14:30:00Z", "AAPL", 100.0, 1.0, 100.0, 1.0, -10.0),
            )
            conn.commit()

        analytics = engine_portfolio.compute_portfolio_analytics()
        report = engine_portfolio.build_portfolio_analytics_report()

        self.assertEqual(analytics["closed_trade_count"], 2)
        self.assertAlmostEqual(analytics["win_rate"], 0.5)
        self.assertEqual(analytics["profit_factor"], 1.0)
        self.assertGreater(analytics["max_drawdown"], 0.0)
        self.assertIn("Portfolio Quant Analytics", report)
        self.assertIn("Win Rate: 50.0%", report)
        self.assertIn("Profit Factor: 1.00", report)

    def test_build_portfolio_beta_report_estimates_current_holdings_exposure(self):
        def _prices_from_returns(returns):
            series = 100 * np.cumprod(1 + np.asarray(returns, dtype=float))
            return pd.DataFrame({"Close": series})

        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("AAPL", 100.0, 10.0, 1000.0, 0),
            )
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("MSFT", 50.0, 20.0, 1000.0, 0),
            )
            conn.commit()

        bench_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.012, -0.01] * 8)
        aapl_returns = 0.002 + (1.5 * bench_returns)
        msft_returns = -0.001 + (0.5 * bench_returns)
        histories = {
            "SPY": _prices_from_returns(bench_returns),
            "AAPL": _prices_from_returns(aapl_returns),
            "MSFT": _prices_from_returns(msft_returns),
        }
        last_prices = {"AAPL": 110.0, "MSFT": 55.0, "SPY": 100.0}

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol
                self.fast_info = {"last_price": last_prices.get(symbol)}

            def history(self, period="1d", interval="1d"):
                return histories[self.symbol].copy()

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=1.0), patch.object(
            engine_portfolio, "get_ticker", side_effect=lambda symbol, **_kwargs: StubTicker(symbol)
        ):
            attribution = engine_portfolio.compute_portfolio_beta_attribution({"AAPL": 0.5, "MSFT": 0.5}, benchmark="SPY")
            report = engine_portfolio.build_portfolio_beta_report(benchmark="SPY")

        self.assertAlmostEqual(attribution["portfolio_beta"], 1.0, places=1)
        self.assertAlmostEqual(attribution["coverage_weight"], 1.0, places=2)
        self.assertGreater(attribution["positions"]["AAPL"]["beta"], attribution["positions"]["MSFT"]["beta"])
        self.assertIn("Portfolio Beta: 1.00", report)
        self.assertIn("AAPL: 權重 50.0%", report)
        self.assertIn("MSFT: 權重 50.0%", report)

    def test_build_portfolio_risk_overlay_report_throttles_on_drawdown_and_risk_state(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("2330.TW", 100.0, 10.0, 1000.0, 0),
            )
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("CASH_TWD", 1.0, 200.0, 200.0, 0),
            )
            conn.executemany(
                """
                INSERT INTO portfolio_nav_history (
                    timestamp, nav_twd, total_cost_twd, gross_exposure_twd, cash_twd, pnl_pct, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("2026-01-02T00:00:00Z", 1000.0, 1000.0, 800.0, 200.0, 0.0, "seed"),
                    ("2026-01-06T00:00:00Z", 1200.0, 1000.0, 1000.0, 200.0, 20.0, "seed"),
                ],
            )
            conn.commit()

        class StubTicker:
            def __init__(self, price):
                self.fast_info = {"last_price": price}

            def history(self, period="1d", interval="1d"):
                return pd.DataFrame({"Close": [90.0]})

        with patch.object(fubon, "fubon_ready", False), patch.object(
            engine_portfolio, "get_ticker", return_value=StubTicker(90.0)
        ), patch.object(
            engine_portfolio,
            "compute_portfolio_beta_attribution",
            return_value={"portfolio_beta": 1.1, "methodology": "beta-ok"},
        ), patch.object(
            engine_portfolio,
            "_estimate_portfolio_volatility",
            return_value={"nav_vol_annual": 0.14, "observations": 60, "coverage_weight": 1.0, "skipped": {}},
        ), patch(
            "engine_risk.get_global_risk_snapshot",
            return_value={"state": "🔴 警戒", "riskScore": 66},
        ):
            overlay = engine_portfolio.compute_portfolio_risk_overlay()
            report = engine_portfolio.build_portfolio_risk_overlay_report()

        self.assertEqual(overlay["trade_mode"], "defensive")
        self.assertAlmostEqual(overlay["size_multiplier"], 0.25, places=2)
        self.assertAlmostEqual(overlay["recommended_gross_scale"], 0.25, places=2)
        self.assertGreater(overlay["hedge_notional_twd"], 0.0)
        self.assertIn("Portfolio Risk Overlay", report)
        self.assertIn("Defense Only", report)

    def test_fetch_live_price_runs_without_tool_wrapper(self):
        ticker = SimpleNamespace(info={"currentPrice": 123.456}, history=lambda period: pd.DataFrame())
        with patch.object(engine_market, "FMP_KEY", None), patch.object(
            engine_market, "_fubon_provider", None
        ), patch.object(
            engine_market, "get_ticker", return_value=ticker
        ):
            result = engine_market.fetch_live_price("AAPL")

        self.assertEqual(result, "123.46 (來源: YF)")

    def test_build_realtime_insight_runs_without_tool_wrapper(self):
        intraday = _make_ohlcv_frame(periods=12, close_start=100, close_step=0.5, freq="5min")
        calls = pd.DataFrame({"volume": [100, 200]})
        puts = pd.DataFrame({"volume": [50, 75]})

        class StubTicker:
            info = {
                "bid": 101.0,
                "ask": 101.5,
                "bidSize": 200,
                "askSize": 100,
                "averageVolume": 1_000_000,
                "regularMarketVolume": 500_000,
            }
            options = ["2030-01-17"]

            def history(self, period, interval=None):
                return intraday.copy()

            def option_chain(self, _date):
                return SimpleNamespace(calls=calls, puts=puts)

        with patch.object(engine_market, "get_ticker", return_value=StubTicker()):
            result = engine_market.build_realtime_insight("NVDA")

        self.assertIn("NVDA 美股即時戰情", result)
        self.assertIn("P/C Ratio", result)

    def test_build_sentiment_report_runs_without_tool_wrapper(self):
        history = _make_ohlcv_frame(periods=10, close_start=100, close_step=1.0)
        ticker = SimpleNamespace(history=lambda period: history.copy())
        with patch.object(engine_market, "get_ticker", return_value=ticker):
            result = engine_market.build_sentiment_report()

        self.assertIn("全球宏觀資金流向雷達", result)
        self.assertIn("標普500", result)

    def test_build_sentiment_report_batches_download_and_reuses_short_cache(self):
        indicator_symbols = list(engine_market._SENTIMENT_INDICATORS.keys())
        history = _make_ohlcv_frame(periods=10, close_start=100, close_step=1.0)
        batch_data = pd.concat({symbol: history.copy() for symbol in indicator_symbols}, axis=1)
        engine_market._SENTIMENT_BATCH_CACHE["entries"].clear()

        mock_download = Mock(return_value=batch_data)
        with patch.object(engine_market, "get_download", mock_download), patch.object(
            engine_market,
            "get_ticker",
            side_effect=AssertionError("batch fetch should satisfy all sentiment symbols"),
        ):
            first = engine_market.build_sentiment_report()
            second = engine_market.build_sentiment_report()

        engine_market._SENTIMENT_BATCH_CACHE["entries"].clear()
        self.assertIn("全球宏觀資金流向雷達", first)
        self.assertEqual(first, second)
        self.assertEqual(mock_download.call_count, 1)

    def test_build_technical_report_runs_without_tool_wrapper(self):
        history = _make_ohlcv_frame(periods=90, close_start=100, close_step=0.8)
        ticker = SimpleNamespace(
            history=lambda period, interval="1d": history.copy(),
            info={"fiftyTwoWeekHigh": 180.0, "fiftyTwoWeekLow": 90.0},
        )
        with patch.object(engine_market, "_fubon_provider", None), patch.object(engine_market, "get_ticker", return_value=ticker):
            result = engine_market.build_technical_report("AAPL")

        self.assertIn("AAPL 美股全武裝分析", result)
        self.assertIn("RSI(14)", result)

    def test_mean_reversion_indicator_and_report_surface_signal(self):
        base = 100 + (np.sin(np.linspace(0, 10 * np.pi, 120)) * 3)
        prices = base.copy()
        prices[-1] -= 6.0
        history = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.0,
                "Low": prices - 1.0,
                "Close": prices,
                "Volume": np.linspace(1000, 2000, len(prices)),
            },
            index=pd.date_range("2024-01-01", periods=len(prices), freq="D"),
        )
        ticker = SimpleNamespace(
            history=lambda period, interval="1d": history.copy(),
            info={"fiftyTwoWeekHigh": 110.0, "fiftyTwoWeekLow": 90.0},
        )

        signal = engine_technical.IndicatorCalculator().MEAN_REVERSION(prices, lookback=40)
        with patch.object(
            engine_market,
            "get_mtf_confluence",
            return_value={
                "rsi_by_timeframe": {"weekly": 55.0, "daily": 48.0, "intraday_1h": 42.0},
                "signal_label": "⚪ 中性",
                "signal_reliability": "NORMAL",
            },
        ), patch.object(engine_market, "get_ticker", return_value=ticker):
            report = engine_market.build_mean_reversion_report("AAPL", lookback=40)
            tech_report = engine_market.build_technical_report("AAPL")

        self.assertIsNotNone(signal["zscore"])
        self.assertLess(signal["zscore"], 0)
        self.assertIsNotNone(signal["half_life_days"])
        self.assertIn("均值回歸信號", report)
        self.assertIn("Z-Score", report)
        self.assertIn("均值回歸:", tech_report)

    def test_build_pairs_trade_report_surfaces_cointegration_signal(self):
        index = pd.date_range("2024-01-01", periods=180, freq="B")
        rng = np.random.default_rng(42)
        pair_b = 100.0 + np.cumsum(rng.normal(0.2, 0.6, len(index)))
        spread = np.zeros(len(index))
        for i in range(1, len(index)):
            spread[i] = (0.55 * spread[i - 1]) + rng.normal(0.0, 0.3)
        spread[-1] += 1.5
        pair_a = 5 + (1.25 * pair_b) + spread
        histories = {
            "NVDA": pd.DataFrame({"Close": pair_a}, index=index),
            "AMD": pd.DataFrame({"Close": pair_b}, index=index),
        }

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="1d", interval="1d"):
                return histories[self.symbol].copy()

        with patch.object(engine_market, "get_ticker", side_effect=lambda symbol, **_kwargs: StubTicker(symbol)):
            payload = engine_market.compute_pair_trade_signal("NVDA", "AMD", lookback=120)
            report = engine_market.build_pairs_trade_report("NVDA", "AMD", lookback=120)

        self.assertTrue(payload["cointegrated"])
        self.assertGreater(payload["spread_zscore"], 2.0)
        self.assertIn("配對/協整監控", report)
        self.assertIn("ADF p-value", report)

    def test_build_factor_snapshot_report_surfaces_raw_factor_metrics(self):
        history = _make_ohlcv_frame(periods=300, close_start=100, close_step=0.4)
        ticker = SimpleNamespace(
            history=lambda period, interval="1d": history.copy(),
            info={
                "trailingPE": 20.0,
                "priceToBook": 5.0,
                "returnOnEquity": 0.22,
                "grossMargins": 0.44,
                "debtToEquity": 35.0,
                "marketCap": 900_000_000_000,
            },
        )

        with patch.object(engine_market, "get_ticker", return_value=ticker):
            payload = engine_market.compute_factor_snapshot("AAPL")
            report = engine_market.build_factor_snapshot_report("AAPL")

        self.assertAlmostEqual(payload["earnings_yield"], 0.05, places=3)
        self.assertIsNotNone(payload["momentum_12_1"])
        self.assertIn("單股因子快照", report)
        self.assertIn("未做截面標準化", report)

    def test_build_nlp_signal_ic_report_uses_persisted_signals(self):
        nlp_worker.init_nlp_db()
        signal_values = np.linspace(-0.9, 0.9, 24)
        sundays = pd.date_range("2025-01-05", periods=len(signal_values), freq="7D")
        with database.locked_connection() as conn:
            for ts, signal in zip(sundays, signal_values):
                conn.execute(
                    """
                    INSERT INTO nlp_insights (
                        symbol, timestamp, nlp_alpha, alpha_retail, alpha_macro,
                        alpha_official, total_items, summary_text, insight_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("NVDA", ts.strftime("%Y-%m-%d %H:%M:%S"), float(signal), 0.0, 0.0, 0.0, 1, "{}", "TEST"),
                )
            conn.commit()

        prices = []
        current = 100.0
        for signal in signal_values:
            current *= 1.001
            prices.append(current)
            current *= 1 + (0.03 * float(signal))
            prices.append(current)
            prices.extend([current, current, current])
        history = pd.DataFrame({"Close": prices}, index=pd.bdate_range("2025-01-06", periods=len(prices)))
        ticker = SimpleNamespace(history=lambda period, interval="1d": history.copy())

        with patch.object(engine_market, "get_ticker", return_value=ticker):
            payload = engine_market.compute_nlp_signal_ic("NVDA", horizon_days=1, lookback_signals=24)
            report = engine_market.build_nlp_signal_ic_report("NVDA", horizon_days=1, lookback_signals=24)

        self.assertEqual(payload["signal_quality"], "strong")
        self.assertGreater(payload["ic_full_sample"], 0.8)
        self.assertIn("NLP Alpha IC 追蹤", report)
        self.assertIn("順向 edge", report)

    def test_build_candidate_alpha_report_ranks_cross_sectional_panel(self):
        snapshots = {
            "AAPL": {"mean_reversion": {"zscore": -1.4, "half_life_days": 6.0, "reversion_candidate": True}},
            "TSLA": {"mean_reversion": {"zscore": 0.8, "half_life_days": 25.0, "reversion_candidate": False}},
        }
        factors = {
            "AAPL": {"momentum_12_1": 0.22, "reversal_1m": 0.04, "quality_raw": 0.24, "earnings_yield": 0.05, "book_price": 0.20},
            "TSLA": {"momentum_12_1": 0.08, "reversal_1m": -0.02, "quality_raw": 0.08, "earnings_yield": 0.02, "book_price": 0.05},
        }
        ic_payloads = {
            "AAPL": {"signal_quality": "strong", "ic_rolling_mean": 0.06, "directionality": "positive"},
            "TSLA": {"signal_quality": "weak", "ic_rolling_mean": 0.02, "directionality": "positive"},
        }

        with patch("engine_risk.get_global_risk_snapshot", return_value={"state": "🟡 整理"}), patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "🟡 Soft Throttle", "recommended_gross_scale": 0.7, "size_multiplier": 0.7},
        ), patch.object(
            engine_router,
            "fetch_nlp_alpha",
            side_effect=lambda symbol: {"nlp_alpha": {"AAPL": 0.7, "TSLA": 0.2}[symbol]},
        ), patch.object(
            engine_router,
            "_build_alpha_confidence_overlay",
            side_effect=lambda symbol, *_args, **_kwargs: {
                "effective_alpha": {"AAPL": 0.55, "TSLA": 0.12}[symbol],
                "combined_multiplier": {"AAPL": 0.8, "TSLA": 0.6}[symbol],
            },
        ), patch.object(
            engine_market, "get_asset_profile", side_effect=lambda symbol: {
                "symbol": symbol,
                "asset_type": "Tech_Momentum",
                "sector": "Technology",
                "industry": "Software" if symbol == "AAPL" else "Automotive",
            }
        ), patch.object(
            engine_market, "compute_factor_snapshot", side_effect=lambda symbol: {"symbol": symbol, **factors[symbol]}
        ), patch.object(
            engine_market, "build_technical_snapshot", side_effect=lambda symbol: snapshots[symbol]
        ), patch.object(
            engine_market, "compute_nlp_signal_ic", side_effect=lambda symbol, **_kwargs: ic_payloads[symbol]
        ), patch.object(
            engine_market, "_compute_liquidity_proxy", side_effect=lambda symbol, period="6mo": ((15.0, 3_000_000.0) if symbol == "AAPL" else (14.0, 1_500_000.0))
        ), patch.object(
            engine_portfolio,
            "compute_portfolio_beta_attribution",
            side_effect=lambda holdings, **_kwargs: {
                "positions": {
                    next(iter(holdings)): {
                        "beta": 1.0 if next(iter(holdings)) == "AAPL" else 1.4,
                        "idio_vol": 0.22 if next(iter(holdings)) == "AAPL" else 0.45,
                    }
                }
            },
        ):
            payload = engine_market.compute_candidate_alpha_panel(["AAPL", "TSLA"])
            report = engine_market.build_candidate_alpha_report(["AAPL", "TSLA"])

        self.assertEqual(payload["rows"][0]["symbol"], "AAPL")
        self.assertGreater(payload["rows"][0]["final_alpha_score"], payload["rows"][1]["final_alpha_score"])
        self.assertIn("expected_return_bps", payload["rows"][0])
        self.assertIn("Candidate Alpha Panel", report)
        self.assertIn("AAPL", report)

    def test_compute_candidate_alpha_panel_parallelizes_symbols_and_reuses_ic_payload(self):
        class RecordingExecutor:
            instances = []

            def __init__(self, max_workers):
                self.max_workers = max_workers
                self.submitted = []
                RecordingExecutor.instances.append(self)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                future = Future()
                self.submitted.append(args[0])
                try:
                    future.set_result(fn(*args, **kwargs))
                except Exception as exc:
                    future.set_exception(exc)
                return future

        profile_map = {
            "AAPL": {"asset_type": "Value_Holding", "sector": "Technology", "industry": "Consumer Electronics"},
            "TSLA": {"asset_type": "Tech_Momentum", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers"},
        }
        ic_map = {
            "AAPL": {"signal_quality": "strong", "ic_rolling_mean": 0.12, "directionality": "positive"},
            "TSLA": {"signal_quality": "weak", "ic_rolling_mean": 0.04, "directionality": "positive"},
        }
        overlay_ic_payloads = {}
        beta_cache_keys = []
        benchmark_series = pd.Series(
            [0.01, -0.005, 0.012],
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        def overlay(symbol, nlp_data, risk_snapshot=None, portfolio_overlay=None, ic_payload=None):
            overlay_ic_payloads[symbol] = ic_payload
            return {
                "effective_alpha": nlp_data["nlp_alpha"],
                "combined_multiplier": 1.0,
            }

        def beta_attribution(holdings, **kwargs):
            beta_cache_keys.append(set(kwargs.get("series_cache", {}).keys()))
            symbol = next(iter(holdings))
            return {
                "positions": {
                    symbol: {
                        "beta": 1.0 if symbol == "AAPL" else 1.2,
                        "idio_vol": 0.2 if symbol == "AAPL" else 0.3,
                    }
                }
            }

        with patch.object(engine_risk, "get_global_risk_snapshot", return_value={"state": "🟢 風險開"}), patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "normal", "recommended_gross_scale": 1.0},
        ), patch.object(
            engine_market, "get_asset_profile", side_effect=lambda symbol: profile_map[symbol]
        ), patch.object(
            engine_router, "fetch_nlp_alpha", side_effect=lambda symbol: {"nlp_alpha": {"AAPL": 0.5, "TSLA": 0.2}[symbol]}
        ), patch.object(
            engine_market, "compute_nlp_signal_ic", side_effect=lambda symbol, **_kwargs: dict(ic_map[symbol])
        ) as mock_ic, patch.object(
            engine_router, "_build_alpha_confidence_overlay", side_effect=overlay
        ), patch.object(
            engine_market,
            "compute_factor_snapshot",
            return_value={"momentum_12_1": 0.1, "reversal_1m": -0.01, "quality_raw": 0.08, "earnings_yield": 0.02, "book_price": 0.05},
        ), patch.object(
            engine_market,
            "build_technical_snapshot",
            return_value={"mean_reversion": {"zscore": 0.5, "half_life_days": 15.0, "reversion_candidate": False}},
        ), patch.object(
            engine_portfolio, "compute_portfolio_beta_attribution", side_effect=beta_attribution
        ), patch.object(
            engine_market, "_compute_liquidity_proxy", return_value=(12.0, 2_000_000.0)
        ), patch.object(
            engine_portfolio, "_load_daily_return_series", return_value=("SPY", benchmark_series, None)
        ), patch.object(
            engine_market, "ThreadPoolExecutor", RecordingExecutor
        ):
            payload = engine_market.compute_candidate_alpha_panel(["AAPL", "TSLA"])

        self.assertEqual(mock_ic.call_count, 2)
        self.assertEqual(len(payload["rows"]), 2)
        self.assertEqual(RecordingExecutor.instances[0].submitted, ["AAPL", "TSLA"])
        self.assertEqual(RecordingExecutor.instances[0].max_workers, 2)
        self.assertEqual(overlay_ic_payloads["AAPL"], ic_map["AAPL"])
        self.assertEqual(overlay_ic_payloads["TSLA"], ic_map["TSLA"])
        self.assertTrue(all(("SPY", "6mo") in keys for keys in beta_cache_keys))

    def test_build_risk_parity_report_prefers_lower_vol_assets(self):
        low_vol = _prices_from_returns([0.004, -0.003, 0.004, -0.002, 0.003, -0.003] * 30)
        high_vol = _prices_from_returns([0.03, -0.025, 0.02, -0.02, 0.028, -0.024] * 30)
        histories = {"TLT": low_vol, "NVDA": high_vol}

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="1d", interval="1d"):
                return histories[self.symbol].copy()

        with patch.object(engine_portfolio, "get_ticker", side_effect=lambda symbol, **_kwargs: StubTicker(symbol)):
            payload = engine_portfolio.compute_inverse_vol_weights(["TLT", "NVDA"], lookback=60)
            report = engine_portfolio.build_risk_parity_report("TLT,NVDA", lookback=60)

        self.assertGreater(payload["weights"]["TLT"], payload["weights"]["NVDA"])
        self.assertAlmostEqual(sum(payload["weights"].values()), 1.0, places=3)
        self.assertIn("Inverse-Vol Risk Parity Proxy", report)

    def test_parse_symbol_input_uses_market_lookup_for_numeric_taiwan_symbols(self):
        with patch.object(
            engine_portfolio.market,
            "_normalize_lookup_symbol",
            side_effect=lambda symbol: {"8069": "8069.TWO"}.get(symbol, symbol),
        ):
            parsed = engine_portfolio._parse_symbol_input("8069, AAPL, 8069")

        self.assertEqual(parsed, ["8069.TWO", "AAPL"])

    def test_estimate_portfolio_volatility_limits_low_weight_tail(self):
        symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
        weights = [0.35, 0.20, 0.15, 0.10, 0.08, 0.06, 0.03, 0.03]
        holdings = dict(zip(symbols, weights))
        history = _prices_from_returns([0.01, -0.008, 0.009, -0.007, 0.008, -0.006] * 30)

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="1d", interval="1d"):
                return history.copy()

        with patch.object(engine_portfolio.market, "_normalize_lookup_symbol", side_effect=lambda symbol: symbol), patch.object(
            engine_portfolio,
            "get_ticker",
            side_effect=lambda symbol, **_kwargs: StubTicker(symbol),
        ) as mock_get_ticker:
            payload = engine_portfolio._estimate_portfolio_volatility(holdings, invested_ratio=1.0, lookback=60)

        self.assertEqual(mock_get_ticker.call_count, 5)
        self.assertEqual(payload["selected_symbol_count"], 5)
        self.assertEqual(payload["requested_symbol_count"], 8)
        self.assertAlmostEqual(payload["coverage_weight"], 0.88, places=2)
        self.assertIn("FFF", payload["skipped"])
        self.assertEqual(payload["skipped"]["FFF"], "低權重，未納入快速波動估算")

    def test_portfolio_rebalance_plan_respects_sector_caps(self):
        snapshots = [
            {"symbol": "AAPL", "is_cash": False, "market_value_twd": 10000.0},
            {"symbol": "CASH_TWD", "is_cash": True, "market_value_twd": 90000.0},
        ]
        panel = {
            "generated_at": "2026-01-01 00:00:00",
            "rows": [
                {
                    "symbol": "NVDA",
                    "asset_type": "Tech_Momentum",
                    "sector": "Technology",
                    "expected_return_bps": 120.0,
                    "forecast_confidence": 0.8,
                    "final_alpha_score": 1.2,
                },
                {
                    "symbol": "AAPL",
                    "asset_type": "Tech_Momentum",
                    "sector": "Technology",
                    "expected_return_bps": 90.0,
                    "forecast_confidence": 0.7,
                    "final_alpha_score": 0.8,
                },
                {
                    "symbol": "GLD",
                    "asset_type": "Macro_Hedge",
                    "sector": "Unknown",
                    "expected_return_bps": 70.0,
                    "forecast_confidence": 0.6,
                    "final_alpha_score": 0.6,
                },
            ],
        }

        with patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={
                "trade_mode": "normal",
                "trade_mode_label": "🟢 Normal",
                "recommended_gross_scale": 0.6,
                "gross_exposure_ratio": 0.1,
                "current_nav_twd": 100000.0,
                "primary_constraint": "目前無強制降風險",
            },
        ), patch.object(
            engine_portfolio.market, "compute_candidate_alpha_panel", return_value=panel
        ), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ), patch.object(
            engine_portfolio, "_load_portfolio_rows", return_value=[]
        ):
            payload = engine_portfolio.compute_portfolio_rebalance_plan(
                symbols=["NVDA", "AAPL", "GLD"],
                candidate_panel=panel,
                snapshots=snapshots,
            )
            report = engine_portfolio.build_portfolio_rebalance_report("NVDA,AAPL,GLD")

        self.assertLessEqual(payload["sector_allocations"]["Technology"], 0.30)
        self.assertLessEqual(payload["target_weights"]["NVDA"], 0.13)
        self.assertIn("Portfolio Rebalance Proposal", report)
        self.assertIn("NVDA", report)

    def test_portfolio_rebalance_plan_normalizes_raw_numeric_holdings(self):
        snapshots = [
            {"symbol": "2330", "is_cash": False, "market_value_twd": 10000.0},
            {"symbol": "CASH_TWD", "is_cash": True, "market_value_twd": 90000.0},
        ]
        panel = {
            "generated_at": "2026-01-01 00:00:00",
            "rows": [
                {
                    "symbol": "2330.TW",
                    "asset_type": "Tech_Momentum",
                    "sector": "Technology",
                    "expected_return_bps": 80.0,
                    "forecast_confidence": 0.6,
                    "final_alpha_score": 0.7,
                }
            ],
        }

        with patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={
                "trade_mode": "normal",
                "trade_mode_label": "🟢 Normal",
                "recommended_gross_scale": 0.1,
                "gross_exposure_ratio": 0.1,
                "current_nav_twd": 100000.0,
                "primary_constraint": "目前無強制降風險",
            },
        ), patch.object(
            engine_portfolio.market, "_normalize_lookup_symbol", side_effect=lambda symbol: "2330.TW" if symbol == "2330" else symbol
        ), patch.object(
            engine_portfolio.market, "compute_candidate_alpha_panel", return_value=panel
        ), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ), patch.object(
            engine_portfolio, "_load_portfolio_rows", return_value=[]
        ):
            payload = engine_portfolio.compute_portfolio_rebalance_plan(
                symbols=["2330"],
                candidate_panel=panel,
                snapshots=snapshots,
            )

        self.assertEqual([row["symbol"] for row in payload["recommendations"]], ["2330.TW"])
        self.assertEqual(payload["recommendations"][0]["action"], "hold")

    def test_portfolio_rebalance_plan_reserves_gross_for_trust_holdings(self):
        snapshots = [
            {"symbol": "0050_TRUST", "is_cash": False, "market_value_twd": 60000.0},
            {"symbol": "CASH_TWD", "is_cash": True, "market_value_twd": 40000.0},
        ]
        panel = {
            "generated_at": "2026-01-01 00:00:00",
            "rows": [
                {
                    "symbol": "NVDA",
                    "asset_type": "Tech_Momentum",
                    "sector": "Technology",
                    "expected_return_bps": 120.0,
                    "forecast_confidence": 0.8,
                    "final_alpha_score": 1.1,
                }
            ],
        }

        with patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={
                "trade_mode": "kill_switch",
                "trade_mode_label": "💀 Kill Switch",
                "recommended_gross_scale": 0.0,
                "gross_exposure_ratio": 0.6,
                "current_nav_twd": 100000.0,
                "primary_constraint": "drawdown governor 💀 Kill Switch",
            },
        ), patch.object(
            engine_portfolio.market, "_normalize_lookup_symbol", side_effect=lambda symbol: "0050.TW" if symbol == "0050" else symbol
        ), patch.object(
            engine_portfolio.market, "get_asset_profile", return_value={"sector": "ETF", "asset_type": "Value_Holding"}
        ), patch.object(
            engine_portfolio.market, "compute_candidate_alpha_panel", return_value=panel
        ), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ), patch.object(
            engine_portfolio, "_load_portfolio_rows", return_value=[]
        ):
            payload = engine_portfolio.compute_portfolio_rebalance_plan(
                symbols=["NVDA"],
                candidate_panel=panel,
                snapshots=snapshots,
            )

        self.assertAlmostEqual(payload["protected_gross_ratio"], 0.6, places=2)
        self.assertAlmostEqual(payload["allocated_gross_ratio"], 0.6, places=2)
        self.assertTrue(any(item["symbol"] == "ACCUMULATION_ONLY" for item in payload["blocked_by_risk"]))
        self.assertFalse(any(row["symbol"] == "NVDA" and row["action"] == "buy" for row in payload["recommendations"]))

    def test_beta_attribution_uses_market_lookup_for_numeric_taiwan_symbols(self):
        bench = _prices_from_returns([0.008, -0.004, 0.007, -0.003, 0.006] * 40)
        stock = _prices_from_returns([0.012, -0.006, 0.011, -0.004, 0.009] * 40)
        histories = {"2330.TW": bench, "8069.TWO": stock}

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="1d", interval="1d"):
                return histories[self.symbol].copy()

        with patch.object(
            engine_portfolio.market,
            "_normalize_lookup_symbol",
            side_effect=lambda symbol: {"2330": "2330.TW", "8069": "8069.TWO"}.get(symbol, symbol),
        ), patch.object(
            engine_portfolio,
            "get_ticker",
            side_effect=lambda symbol, **_kwargs: StubTicker(symbol),
        ):
            payload = engine_portfolio.compute_portfolio_beta_attribution({"8069": 1.0}, benchmark="2330")

        self.assertAlmostEqual(payload["coverage_weight"], 1.0, places=3)
        self.assertIn("8069.TWO", payload["positions"])

    def test_build_movers_report_runs_without_tool_wrapper(self):
        gainers = [{"symbol": "AAA", "price": "10.0", "changesPercentage": "5.0"}]
        losers = [{"symbol": "BBB", "price": "8.0", "changesPercentage": "-4.0"}]

        class StubResponse:
            def __init__(self, payload):
                self.payload = payload

            def json(self):
                return self.payload

        with patch.object(engine_market, "FMP_KEY", "demo"), patch.object(
            engine_market.requests,
            "get",
            side_effect=[StubResponse(gainers), StubResponse(losers)],
        ):
            result = engine_market.build_movers_report()

        self.assertIn("領漲榜", result)
        self.assertIn("AAA", result)
        self.assertIn("BBB", result)

    def test_evaluate_indicator_formula_runs_without_tool_wrapper(self):
        with patch.object(engine_technical.IndicatorCalculator, "calculate", return_value="單一數值: 12.3"):
            result = engine_technical.evaluate_indicator_formula("RSI(CLOSE('AAPL','1d'), 14)[-1]")

        self.assertEqual(result, "單一數值: 12.3")

    def test_build_capital_flow_report_runs_without_tool_wrapper(self):
        history = _make_ohlcv_frame(periods=8, close_start=100, close_step=1.0)
        hist_data = {
            "^SOX": history,
            "XLU": history.assign(Close=[100, 99, 98, 97, 96, 95, 94, 93]),
            "HG=F": history,
            "GC=F": history.assign(Close=[100, 100, 101, 101, 102, 102, 103, 103]),
            "^TNX": history,
            "TLT": history.assign(Close=[100, 101, 102, 103, 104, 105, 106, 107]),
            "DX-Y.NYB": history,
            "TWD=X": history,
            "JPY=X": history.assign(Close=[100, 101, 102, 103, 104, 105, 106, 107]),
            "^VIX": history,
        }

        with patch.object(engine_risk, "get_download", return_value=hist_data):
            result = engine_risk.build_capital_flow_report()

        self.assertIn("Capital Flow Matrix", result)

    def test_build_v_turn_report_runs_without_tool_wrapper(self):
        engine_risk.init_market_db()
        splg = _make_ohlcv_frame(periods=30, close_start=100, close_step=1.0)
        rsp = _make_ohlcv_frame(periods=30, close_start=100, close_step=1.1)
        hyg = _make_ohlcv_frame(periods=30, close_start=100, close_step=0.4)
        lqd = _make_ohlcv_frame(periods=30, close_start=100, close_step=0.1)
        oil = _make_ohlcv_frame(periods=30, close_start=70, close_step=0.2)
        hist_data = {
            "SPLG": splg,
            "RSP": rsp,
            "HYG": hyg,
            "LQD": lqd,
            "CL=F": oil,
        }
        vix_hist = _make_ohlcv_frame(periods=10, close_start=15, close_step=-0.1, freq="15min")
        vix3m_hist = _make_ohlcv_frame(periods=10, close_start=20, close_step=0.0, freq="15min")
        vvix_hist = _make_ohlcv_frame(periods=10, close_start=100, close_step=0.2, freq="15min")
        spy_hist = _make_ohlcv_frame(periods=10, close_start=500, close_step=1.0, freq="5min")

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period, interval=None):
                lookup = {
                    ("^VIX", "2d", "15m"): vix_hist,
                    ("^VIX3M", "2d", "15m"): vix3m_hist,
                    ("^VVIX", "2d", "15m"): vvix_hist,
                    ("SPY", "2d", "5m"): spy_hist,
                    ("^VIX", "5d", None): vix_hist,
                    ("^VIX3M", "5d", None): vix3m_hist,
                    ("^VVIX", "5d", None): vvix_hist,
                }
                return lookup[(self.symbol, period, interval)].copy()

        with patch.object(engine_risk, "get_download", return_value=hist_data), patch.object(
            engine_risk, "get_ticker", side_effect=lambda symbol: StubTicker(symbol)
        ), patch.object(engine_risk, "calculate_buying_pressure", return_value=0.2):
            result = engine_risk.build_v_turn_report()

        self.assertIn("V 轉戰報", result)
        self.assertIn("Day 1 低點", result)
        with database.locked_connection() as conn:
            row = conn.execute("SELECT day1_price FROM v_turn_state WHERE id = 1").fetchone()
        self.assertIsNotNone(row)

    def test_build_market_trades_report_runs_without_tool_wrapper(self):
        stock_client = SimpleNamespace(
            intraday=SimpleNamespace(
                trades=lambda symbol, limit=20: {
                    "data": [
                        {"price": 52.3, "size": 10, "time": 1710000000000000},
                        {"price": 52.1, "size": 6, "time": 1710000060000000},
                    ]
                }
            )
        )
        self._install_fubon_sdk(stock_client=stock_client)

        result = fubon.build_market_trades_report("2330.TW", limit=1)

        self.assertIn("2330 最近 2 筆成交明細", result)
        self.assertIn("價: 52.3", result)

    def test_build_price_volumes_report_runs_without_tool_wrapper(self):
        stock_client = SimpleNamespace(
            intraday=SimpleNamespace(
                volumes=lambda symbol: {
                    "data": [
                        {"price": 101.0, "volume": 30},
                        {"price": 99.5, "volume": 10},
                    ]
                }
            )
        )
        self._install_fubon_sdk(stock_client=stock_client)

        result = fubon.build_price_volumes_report("2330")

        self.assertIn("2330 分價量表", result)
        self.assertIn("101.00", result)

    def test_build_historical_stats_report_runs_without_tool_wrapper(self):
        stock_client = SimpleNamespace(
            historical=SimpleNamespace(
                stats=lambda symbol: {
                    "name": "台積電",
                    "week52High": 700,
                    "week52Low": 500,
                    "closePrice": 650,
                }
            )
        )
        self._install_fubon_sdk(stock_client=stock_client)

        result = fubon.build_historical_stats_report("2330")

        self.assertIn("台積電", result)
        self.assertIn("目前位階: 75.0%", result)

    def test_build_txo_sentiment_report_runs_without_tool_wrapper(self):
        futopt_client = SimpleNamespace(
            snapshot=SimpleNamespace(
                actives=lambda market, trade: {
                    "data": [
                        {"symbol": "TXO17000C4", "volume": 120},
                        {"symbol": "TXO17000P4", "volume": 60},
                    ]
                }
            )
        )
        self._install_fubon_sdk(futopt_client=futopt_client)

        result = fubon.build_txo_sentiment_report()

        self.assertIn("P/C Ratio: 0.50", result)
        self.assertIn("Call 總量: 120 | Put 總量: 60", result)

    def test_build_quote_and_orderbook_report_runs_without_tool_wrapper(self):
        stock_client = SimpleNamespace(
            intraday=SimpleNamespace(
                quote=lambda symbol: {
                    "closePrice": 88.6,
                    "bids": [{"price": 88.5, "size": 12}],
                    "asks": [{"price": 88.7, "size": 20}],
                }
            )
        )
        self._install_fubon_sdk(stock_client=stock_client)

        result = fubon.build_quote_and_orderbook_report("2317.TW")

        self.assertIn("2317 即時報價與五檔觀測", result)
        self.assertIn("賣5: 價格 88.7", result)
        self.assertIn("買1: 價格 88.5", result)

    def test_build_market_hot_stocks_report_runs_without_tool_wrapper(self):
        stock_client = SimpleNamespace(
            snapshot=SimpleNamespace(
                actives=lambda market, trade: {
                    "data": [{"symbol": "2330", "name": "台積電", "closePrice": 915}]
                },
                movers=lambda market, direction, change: {
                    "data": [{"symbol": "3017", "name": "奇鋐", "changePercent": 6.5}]
                },
            )
        )
        self._install_fubon_sdk(stock_client=stock_client)

        result = fubon.build_market_hot_stocks_report()

        self.assertIn("成交值排行榜", result)
        self.assertIn("2330 台積電", result)
        self.assertIn("3017 奇鋐", result)

    def test_build_intraday_trend_report_runs_without_tool_wrapper(self):
        candles = {
            "data": [
                {"date": "2024-01-01T09:00:00", "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2, "volume": 100},
                {"date": "2024-01-01T09:05:00", "open": 10.2, "high": 10.6, "low": 10.1, "close": 10.4, "volume": 120},
                {"date": "2024-01-01T09:10:00", "open": 10.4, "high": 10.8, "low": 10.3, "close": 10.7, "volume": 140},
            ]
        }
        stock_client = SimpleNamespace(
            intraday=SimpleNamespace(
                candles=lambda symbol, timeframe='5': candles
            )
        )
        self._install_fubon_sdk(stock_client=stock_client)

        result = fubon.build_intraday_trend_report("0050")

        self.assertIn("0050 盤中短線趨勢", result)
        self.assertIn("近期平均價", result)


class RiskAndConfigHardeningTests(unittest.TestCase):
    def setUp(self):
        self.original_fx_cache = dict(engine_portfolio._fx_cache)

    def tearDown(self):
        engine_portfolio._fx_cache.update(self.original_fx_cache)

    def test_score_risk_multiplier_uses_fixed_dix_offset(self):
        _, low_score_plain, _ = engine_risk._score_risk_multiplier(1.6)
        _, low_score_dix, low_offset = engine_risk._score_risk_multiplier(1.6, dix_support_active=True)
        _, high_score_plain, _ = engine_risk._score_risk_multiplier(2.4)
        _, high_score_dix, high_offset = engine_risk._score_risk_multiplier(2.4, dix_support_active=True)

        self.assertEqual(low_offset, -engine_risk.DIX_SUPPORT_OFFSET_POINTS)
        self.assertEqual(high_offset, -engine_risk.DIX_SUPPORT_OFFSET_POINTS)
        self.assertEqual(low_score_plain - low_score_dix, engine_risk.DIX_SUPPORT_OFFSET_POINTS)
        self.assertEqual(high_score_plain - high_score_dix, engine_risk.DIX_SUPPORT_OFFSET_POINTS)

    def test_format_global_risk_snapshot_surfaces_dix_offset(self):
        snapshot = {
            "riskScore": 55,
            "state": "🔴 警戒",
            "reasons": ["🟢 暗池吸籌，大戶提供下檔支撐"],
            "signals": {
                "yieldCurve10Y2Y": -0.3,
                "fedFundsRate": 5.25,
                "dixPr": 0.92,
                "dixSupportOffset": -12,
                "gexBillions": -1.8,
                "sentimentLabel": "Bearish",
                "sentimentScore": -0.6,
                "spx": 5100.0,
                "spx20Ma": 5200.0,
                "spx200Ma": 5300.0,
                "sectorBreadth50": 45.5,
                "sectorBreadth200": 27.3,
                "sectorBreadthState": "weak",
                "spySpot": 510.2,
                "spyGammaFlipLevel": 515.0,
                "spyMaxPain": 505.0,
                "spyBelowGammaFlip": True,
                "spyCurrentIv": 28.0,
                "spyRealizedVol30d": 33.5,
                "spyVrp": -5.5,
                "spyVolSignal": "⚠️ 波動低估",
                "spyTltCorr60d": 0.22,
            },
        }

        report = engine_risk.format_global_risk_snapshot(snapshot)

        self.assertIn("DIX 抵扣: -12", report)
        self.assertIn("Breadth: 50MA 45.5% | 200MA 27.3% (weak)", report)
        self.assertIn("Gamma Levels: Spot 510.2 | Flip 515.0 | Max Pain 505.0 | 低於 Flip", report)
        self.assertIn("SPY Vol Context: IV 28.0% | RV30 33.5% | VRP -5.5pt (⚠️ 波動低估)", report)

    def test_safe_float_rejects_non_finite_values(self):
        self.assertIsNone(engine_risk._safe_float(float("nan")))
        self.assertIsNone(engine_risk._safe_float(float("inf")))
        self.assertIsNone(engine_risk._safe_float(float("-inf")))
        self.assertEqual(engine_risk._safe_float("10.567", 2), 10.57)

    def test_get_market_breadth_counts_sector_participation(self):
        payload = {}
        for idx, etf in enumerate(engine_risk.SECTOR_ETFS):
            close_step = 0.5 if idx < 8 else -0.3
            payload[etf] = _make_ohlcv_frame(periods=220, close_start=100 + idx, close_step=close_step)

        with patch.object(engine_risk, "get_download", return_value=payload):
            breadth = engine_risk.get_market_breadth()

        self.assertEqual(breadth["total_sectors"], 11)
        self.assertAlmostEqual(breadth["pct_above_200ma"], 72.7, places=1)
        self.assertAlmostEqual(breadth["pct_above_50ma"], 72.7, places=1)
        self.assertEqual(breadth["breadth_signal"], "healthy")

    def test_get_spy_gex_levels_surfaces_flip_and_max_pain(self):
        calls = pd.DataFrame(
            {
                "strike": [100.0, 105.0],
                "impliedVolatility": [0.20, 0.22],
                "openInterest": [220.0, 180.0],
            }
        )
        puts = pd.DataFrame(
            {
                "strike": [110.0, 115.0],
                "impliedVolatility": [0.25, 0.28],
                "openInterest": [400.0, 150.0],
            }
        )

        class StubTicker:
            options = ["2030-01-17"]

            def history(self, period):
                return pd.DataFrame({"Close": [105.0]})

            def option_chain(self, _exp):
                return SimpleNamespace(calls=calls, puts=puts)

        class StubRateTicker:
            def history(self, period):
                return pd.DataFrame({"Close": [4.0]})

        with patch.object(
            engine_risk,
            "get_ticker",
            side_effect=lambda symbol, **_kwargs: StubRateTicker() if symbol == "^TNX" else StubTicker(),
        ), patch.object(engine_risk, "calculate_gamma", return_value=1.0):
            profile = engine_risk.get_spy_gex_levels()

        self.assertIsNotNone(profile["total_gex_billions"])
        self.assertGreater(profile["gamma_flip_level"], 105.0)
        self.assertLessEqual(profile["gamma_flip_level"], 110.0)
        self.assertTrue(profile["below_flip"])
        self.assertIsNotNone(profile["max_pain"])

    def test_build_global_risk_snapshot_surfaces_new_context_signals(self):
        frame = pd.DataFrame(
            [
                {
                    "SPX": 5200.0,
                    "SPX_10MA": 5100.0,
                    "SPX_20MA": 5000.0,
                    "SPX_200MA": 4900.0,
                    "dix_PR": 0.4,
                    "DXY_Z": 0.0,
                    "TNX_Z": 0.0,
                    "VIX_Z": 0.0,
                    "SKEW_PR": 0.0,
                    "gex": 1_000_000_000,
                }
            ]
        )

        with patch.object(engine_risk, "fetch_all_market_data", return_value=frame), patch.object(
            engine_risk.MacroEngine, "get_macro_dashboard", return_value={"Yield_Curve_10Y2Y": 0.2, "Fed_Funds_Rate": 4.5}
        ), patch.object(
            engine_risk, "get_market_sentiment_score", return_value=(0.0, "Neutral")
        ), patch.object(
            engine_risk, "_get_spx_trend_snapshot", return_value=(18.0, "ranging")
        ), patch.object(
            engine_risk, "get_market_breadth", return_value={"pct_above_50ma": 35.0, "pct_above_200ma": 25.0, "breadth_signal": "weak"}
        ), patch.object(
            engine_risk,
            "get_spy_gex_levels",
            return_value={
                "total_gex_billions": 0.6,
                "gamma_flip_level": 510.0,
                "max_pain": 505.0,
                "spot": 500.0,
                "below_flip": True,
                "above_flip": False,
            },
        ), patch.object(
            engine_risk.market,
            "build_option_volatility_context",
            return_value={
                "current_iv": 28.0,
                "realized_vol_30d": 33.0,
                "vrp": -5.0,
                "iv_vs_rv_percentile": 20.0,
                "signal": "⚠️ 波動低估",
            },
        ), patch.object(
            engine_risk, "get_rolling_correlations", return_value={"spyTltCorr60d": 0.25, "spyGldCorr60d": 0.1, "spyDxyCorr60d": -0.05}
        ):
            snapshot = engine_risk._build_global_risk_snapshot()

        self.assertEqual(snapshot["signals"]["sectorBreadth200"], 25.0)
        self.assertEqual(snapshot["signals"]["spyGammaFlipLevel"], 510.0)
        self.assertEqual(snapshot["signals"]["spyVrp"], -5.0)
        self.assertEqual(snapshot["signals"]["spyTltCorr60d"], 0.25)
        self.assertGreater(snapshot["riskMultiplier"], 1.0)

    def test_fetch_exchange_rate_uses_cached_rate_on_parse_error(self):
        engine_portfolio._fx_cache.update({"rate": 31.88, "timestamp": 0})

        class BadTicker:
            fast_info = {"last_price": "bad-rate"}

            def history(self, period):
                return pd.DataFrame()

        with patch.object(engine_portfolio, "get_ticker", return_value=BadTicker()):
            rate = engine_portfolio.fetch_exchange_rate()

        self.assertEqual(rate, 31.88)
        self.assertEqual(engine_portfolio._fx_cache["rate"], 31.88)

    def test_fetch_live_price_falls_back_to_yfinance_after_fmp_parse_error(self):
        class BadResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return [{"price": "not-a-number"}]

        ticker = SimpleNamespace(
            info={"currentPrice": 123.456},
            history=lambda period="1d": pd.DataFrame(),
        )

        with patch.object(engine_market, "FMP_KEY", "demo"), patch.object(
            engine_market, "is_us_market_open", return_value=True
        ), patch.object(
            engine_market.requests, "get", return_value=BadResponse()
        ), patch.object(
            engine_market, "get_ticker", side_effect=lambda *args, **kwargs: ticker
        ):
            price = engine_market.fetch_live_price("AAPL")

        self.assertEqual(price, "123.46 (來源: YF)")

    def test_load_system_prompt_prefers_local_override_then_default_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_path = root / "system_prompt.txt"
            local_path = root / "system_prompt.local.txt"
            prompt_path.write_text("tracked prompt", encoding="utf-8")
            local_path.write_text("local prompt", encoding="utf-8")

            self.assertEqual(
                config._load_system_prompt(prompt_path=prompt_path, local_prompt_path=local_path),
                "local prompt",
            )

            local_path.unlink()

            self.assertEqual(
                config._load_system_prompt(prompt_path=prompt_path, local_prompt_path=local_path),
                "tracked prompt",
            )

    def test_backup_database_creates_sqlite_copy_and_prunes_old_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            db_path = root / "portfolio.db"
            backup_dir = root / "backups"
            backup_dir.mkdir()

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("CREATE TABLE sample (value TEXT)")
                conn.execute("INSERT INTO sample VALUES ('ok')")
                conn.commit()

            (backup_dir / "portfolio_20240101_000000_000000.db").write_bytes(b"old1")
            (backup_dir / "portfolio_20240102_000000_000000.db").write_bytes(b"old2")

            backup_path = backup_module.backup_database(db_path=db_path, backup_dir=backup_dir, max_backups=2)

            self.assertIsNotNone(backup_path)
            self.assertTrue(backup_path.exists())
            with sqlite3.connect(str(backup_path)) as conn:
                value = conn.execute("SELECT value FROM sample").fetchone()[0]
            self.assertEqual(value, "ok")
            self.assertEqual(len(list(backup_dir.glob("portfolio_*.db"))), 2)


class TechnicalSignalUpgradeTests(unittest.TestCase):
    def test_indicator_calculator_supports_divergence_adx_and_obv(self):
        calc = engine_technical.IndicatorCalculator()
        price = pd.Series([10.0, 9.0, 10.5, 8.2, 10.0, 7.4, 11.0]).to_numpy()
        rsi = pd.Series([42.0, 24.0, 48.0, 27.0, 45.0, 33.0, 55.0]).to_numpy()

        divergence = calc.DIVERGENCE(price, rsi, lookback=7, order=1)
        adx = calc.ADX(
            pd.Series(range(101, 141)).to_numpy() + 1,
            pd.Series(range(101, 141)).to_numpy() - 1,
            pd.Series(range(101, 141)).to_numpy(),
        )
        obv = calc.OBV(
            pd.Series([10.0, 11.0, 10.0, 12.0]).to_numpy(),
            pd.Series([100.0, 120.0, 80.0, 150.0]).to_numpy(),
        )

        self.assertTrue(divergence["bullish_divergence"])
        self.assertEqual(adx["trend_regime"], "trending")
        self.assertEqual(obv.tolist(), [0.0, 120.0, 40.0, 190.0])

    def test_build_realtime_insight_includes_vwap_and_dual_anchor(self):
        intraday = _make_ohlcv_frame(periods=12, close_start=100, close_step=0.5, freq="5min")
        calls = pd.DataFrame({"volume": [100, 200]})
        puts = pd.DataFrame({"volume": [50, 75]})

        class StubTicker:
            info = {
                "bid": 101.0,
                "ask": 101.5,
                "bidSize": 200,
                "askSize": 100,
                "averageVolume": 1_000_000,
                "regularMarketVolume": 500_000,
            }
            options = ["2030-01-17"]

            def history(self, period, interval=None):
                return intraday.copy()

            def option_chain(self, _date):
                return SimpleNamespace(calls=calls, puts=puts)

        with patch.object(engine_market, "get_ticker", return_value=StubTicker()):
            result = engine_market.build_realtime_insight("NVDA")

        self.assertIn("VWAP:", result)
        self.assertIn("雙錨點:", result)
        self.assertIn("波動定價:", result)

    def test_get_mtf_confluence_detects_strong_oversold(self):
        def fake_close(_symbol, interval):
            return {
                "1wk": [45.0],
                "1d": [25.0],
                "1h": [20.0],
            }[interval]

        with patch.object(engine_market.IndicatorCalculator, "CLOSE", side_effect=fake_close), patch.object(
            engine_market.IndicatorCalculator, "RSI", side_effect=lambda values: values
        ):
            mtf = engine_market.get_mtf_confluence("AAPL")

        self.assertEqual(mtf["confluence_signal"], "strong_oversold")
        self.assertEqual(mtf["signal_label"], "🟢 強超賣共振")
        self.assertEqual(mtf["signal_reliability"], "HIGH")

    def test_build_technical_report_surfaces_new_signals(self):
        snapshot = {
            "current_price": 105.0,
            "high_52w": 180.0,
            "low_52w": 90.0,
            "ma20": 101.2,
            "ma60": 98.6,
            "rsi": {"value": 32.5, "state": "⚖️中性"},
            "macd": {"dif": 1.25, "histogram": 0.48, "state": "📈多頭增強"},
            "kdj": {"k": 28.0, "d": 24.0, "j": 36.0},
            "bbands": {"upper": 110.0, "lower": 94.0},
            "adx": {"value": 29.5, "plus_di": 24.0, "minus_di": 18.0, "trend_regime": "trending"},
            "divergence": {
                "label": "🟢 底背離",
                "details": "底背離：價格創低但指標抬高",
                "bullish_divergence": True,
                "bearish_divergence": False,
            },
            "obv": {"label": "上升", "signal": "📈 價漲量增，趨勢健康", "obv_ma20": 2500.0},
            "mtf_rsi": {
                "rsi_by_timeframe": {"weekly": 45.0, "daily": 25.0, "intraday_1h": 20.0},
                "signal_label": "🟢 強超賣共振",
                "signal_reliability": "HIGH",
            },
        }

        with patch.object(engine_market, "_fubon_provider", None), patch.object(
            engine_market, "build_technical_snapshot", return_value=snapshot
        ):
            result = engine_market.build_technical_report("AAPL")

        self.assertIn("RSI 背離: 🟢 底背離", result)
        self.assertIn("ADX(14): 29.50", result)
        self.assertIn("OBV 趨勢: 上升", result)
        self.assertIn("多時間框 RSI: W:45.0 | D:25.0 | H1:20.0 -> 🟢 強超賣共振 (HIGH)", result)

    def test_fetch_strat_data_escalates_hard_alert_only_with_bearish_divergence(self):
        fake_ticker = SimpleNamespace(history=lambda period, interval=None: pd.DataFrame({"Close": [10.0, 9.5]}))
        captured_alerts = []
        technical_snapshot = {
            "divergence": {"label": "🔴 頂背離", "bearish_divergence": True},
            "adx": {"value": 31.2, "trend_regime": "trending"},
            "obv": {"signal": "📉 價跌量弱，空方主導"},
            "mtf_rsi": {"signal_label": "🔴 強過熱共振", "confluence_strength": 2, "signal_reliability": "HIGH"},
        }

        with patch.object(engine_router.market, "normalize_ticker", return_value="TEST"), patch.object(
            engine_router.market, "get_asset_profile", return_value={"asset_type": "Tech_Momentum"}
        ), patch.object(engine_router, "fetch_nlp_alpha", return_value={"nlp_alpha": -0.2}), patch.object(
            engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
        ), patch.object(engine_router, "get_ticker", return_value=fake_ticker), patch.object(
            engine_router.risk, "calculate_buying_pressure", return_value=-0.95
        ), patch.object(
            engine_router.risk, "get_global_risk_snapshot", return_value={"state": "🟡 整理", "riskScore": 42}
        ), patch.object(
            engine_router.market,
            "compute_nlp_signal_ic",
            return_value={"signal_quality": "strong", "directionality": "positive", "ic_rolling_mean": 0.08},
        ), patch(
            "engine_portfolio.compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "🟢 Normal", "size_multiplier": 1.0, "recommended_gross_scale": 1.0},
        ), patch.object(
            engine_router.market, "build_technical_snapshot", return_value=technical_snapshot
        ), patch.object(
            engine_router.market, "build_technical_report", return_value="TECH"
        ), patch.object(
            engine_router.market, "build_realtime_insight", return_value="P/C Ratio: 1.60"
        ), patch.object(
            engine_router.market, "build_option_volatility_context", return_value={"summary": "N/A", "signal": "⚪ 無期權波動資料"}
        ), patch.object(engine_router, "_alert_callback", lambda msg: captured_alerts.append(msg)):
            data = engine_router.fetch_strat_data("test")

        self.assertEqual(data["leading_indicators"]["rsi_divergence"], "🔴 頂背離")
        self.assertEqual(len(captured_alerts), 1)
        self.assertIn("硬體中斷", captured_alerts[0])

    def test_fetch_strat_data_adds_vol_context_and_signal_reliability(self):
        fake_ticker = SimpleNamespace(history=lambda period, interval=None: pd.DataFrame({"Close": [10.0, 9.5]}))
        technical_snapshot = {
            "divergence": {"label": "⚪ 無明顯背離", "bearish_divergence": False},
            "adx": {"value": 18.4, "trend_regime": "ranging"},
            "obv": {"signal": "⚪ 量價中性"},
            "mtf_rsi": {"signal_label": "🟢 強超賣共振", "confluence_strength": 2, "signal_reliability": "HIGH"},
        }

        with patch.object(engine_router.market, "normalize_ticker", return_value="TEST"), patch.object(
            engine_router.market, "get_asset_profile", return_value={"asset_type": "Tech_Momentum"}
        ), patch.object(engine_router, "fetch_nlp_alpha", return_value={"nlp_alpha": -0.2}), patch.object(
            engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
        ), patch.object(engine_router, "get_ticker", return_value=fake_ticker), patch.object(
            engine_router.risk, "calculate_buying_pressure", return_value=-0.95
        ), patch.object(
            engine_router.risk, "get_global_risk_snapshot", return_value={"state": "🟡 整理", "riskScore": 42}
        ), patch.object(
            engine_router.market,
            "compute_nlp_signal_ic",
            return_value={"signal_quality": "strong", "directionality": "positive", "ic_rolling_mean": 0.08},
        ), patch(
            "engine_portfolio.compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "🟢 Normal", "size_multiplier": 1.0, "recommended_gross_scale": 1.0},
        ), patch.object(
            engine_router.market, "build_technical_snapshot", return_value=technical_snapshot
        ), patch.object(
            engine_router.market, "build_technical_report", return_value="TECH"
        ), patch.object(
            engine_router.market, "build_realtime_insight", return_value="P/C Ratio: 1.60"
        ), patch.object(
            engine_router.market,
            "build_option_volatility_context",
            return_value={"summary": "ATM IV 40.0% | RV30 25.0% | VRP +15.0pt (🔥 恐慌定價)", "signal": "🔥 恐慌定價", "vrp": 15.0},
        ), patch.object(engine_router, "_alert_callback", None):
            data = engine_router.fetch_strat_data("test")

        self.assertEqual(data["leading_indicators"]["signal_reliability"], "HIGH")
        self.assertEqual(data["leading_indicators"]["mtf_rsi_signal"], "🟢 強超賣共振")
        self.assertIn("恐慌避險定價", data["leading_indicators"]["pc_context"])
        self.assertEqual(
            data["metrics"]["option_volatility"]["summary"],
            "ATM IV 40.0% | RV30 25.0% | VRP +15.0pt (🔥 恐慌定價)",
        )

    def test_fetch_strat_data_scales_positive_alpha_with_ic_regime_and_drawdown(self):
        fake_ticker = SimpleNamespace(history=lambda period, interval=None: pd.DataFrame({"Close": [10.0, 10.2]}))
        technical_snapshot = {
            "divergence": {"label": "⚪ 無明顯背離", "bearish_divergence": False},
            "adx": {"value": 18.4, "trend_regime": "ranging"},
            "obv": {"signal": "⚪ 量價中性"},
            "mtf_rsi": {"signal_label": "🟢 強超賣共振", "confluence_strength": 2, "signal_reliability": "HIGH"},
        }

        with patch.object(engine_router.market, "normalize_ticker", return_value="TEST"), patch.object(
            engine_router.market, "get_asset_profile", return_value={"asset_type": "Tech_Momentum"}
        ), patch.object(
            engine_router, "fetch_nlp_alpha", return_value={"nlp_alpha": 0.8}
        ), patch.object(
            engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
        ), patch.object(
            engine_router, "get_ticker", return_value=fake_ticker
        ), patch.object(
            engine_router.risk, "calculate_buying_pressure", return_value=0.2
        ), patch.object(
            engine_router.risk, "get_global_risk_snapshot", return_value={"state": "🔴 警戒", "riskScore": 66}
        ), patch.object(
            engine_router.market,
            "compute_nlp_signal_ic",
            return_value={"signal_quality": "weak", "directionality": "positive", "ic_rolling_mean": 0.03},
        ), patch(
            "engine_portfolio.compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "🟠 Risk-Off", "size_multiplier": 0.5, "recommended_gross_scale": 0.4},
        ), patch.object(
            engine_router.market, "build_technical_snapshot", return_value=technical_snapshot
        ), patch.object(
            engine_router.market, "build_technical_report", return_value="TECH"
        ), patch.object(
            engine_router.market, "build_realtime_insight", return_value="P/C Ratio: 0.80"
        ), patch.object(
            engine_router.market, "build_option_volatility_context", return_value={"summary": "N/A", "signal": "⚪ 無期權波動資料"}
        ), patch.object(engine_router, "_alert_callback", None):
            data = engine_router.fetch_strat_data("test")

        self.assertEqual(data["nlp_insights"]["nlp_alpha"], 0.8)
        self.assertAlmostEqual(data["leading_indicators"]["alpha_adjusted"], 0.195, places=3)
        self.assertEqual(data["leading_indicators"]["portfolio_trade_mode"], "🟠 Risk-Off")
        self.assertEqual(data["leading_indicators"]["risk_state"], "🔴 警戒")
        self.assertIn("新增多單需縮倉", " ".join(data["nlp_insights"]["alpha_overlay"]["reasons"]))

    def test_fetch_strat_data_routes_nlp_alerts_through_callback(self):
        captured_alerts = []

        with patch.object(engine_router.market, "normalize_ticker", return_value="TEST"), patch.object(
            engine_router.market, "get_asset_profile", return_value={"asset_type": "Unknown"}
        ), patch.object(
            engine_router, "fetch_nlp_alpha", return_value={
                "nlp_alpha": -0.82,
                "signal_pack": {
                    "sec_detail": ["subpoena disclosed"],
                    "macro_detail": ["demand outlook cut"],
                    "divergence": "⚠️ 官方偏空 vs 散戶追價",
                    "nuclear_alert": True,
                },
                "semantic_summary": "SEC filing pressure remains elevated.",
            }
        ), patch.object(
            engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
        ), patch.object(
            engine_router.risk, "get_global_risk_snapshot", return_value={"state": "🟡 整理", "riskScore": 42}
        ), patch.object(
            engine_router.market,
            "compute_nlp_signal_ic",
            return_value={"signal_quality": "weak", "directionality": "positive", "ic_rolling_mean": 0.03},
        ), patch(
            "engine_portfolio.compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "🟢 Normal", "size_multiplier": 1.0, "recommended_gross_scale": 1.0},
        ), patch.object(
            engine_router.market, "fetch_live_price", return_value="123.45 (來源: YF)"
        ), patch.object(
            engine_router.market, "get_stock_news", return_value="NEWS"
        ), patch.object(
            engine_router, "_alert_callback", lambda msg: captured_alerts.append(msg)
        ):
            data = engine_router.fetch_strat_data("test")

        self.assertEqual(data["nlp_insights"]["nlp_alpha"], -0.82)
        self.assertEqual(len(captured_alerts), 1)
        self.assertIn("NLP 核心預警", captured_alerts[0])
        self.assertIn("subpoena disclosed", captured_alerts[0])

    def test_get_cached_nlp_signal_ic_reuses_recent_payload(self):
        engine_router._nlp_ic_cache["entries"].clear()

        with patch.object(
            engine_router.market,
            "compute_nlp_signal_ic",
            return_value={"signal_quality": "strong", "directionality": "positive", "ic_rolling_mean": 0.08},
        ) as mock_ic:
            first = engine_router._get_cached_nlp_signal_ic("AAPL")
            second = engine_router._get_cached_nlp_signal_ic("AAPL")

        self.assertEqual(first["signal_quality"], "strong")
        self.assertEqual(second["signal_quality"], "strong")
        self.assertEqual(mock_ic.call_count, 1)

    def test_get_strat_context_reuses_shared_portfolio_overlay(self):
        engine_router._nlp_ic_cache["entries"].clear()

        with patch.object(engine_router, "detect_symbols", return_value=["AAPL", "MSFT"]), patch.object(
            engine_router.market, "normalize_ticker", side_effect=lambda symbol: symbol.upper()
        ), patch.object(
            engine_router.market, "get_asset_profile", return_value={"asset_type": "Unknown"}
        ), patch.object(
            engine_router, "fetch_nlp_alpha", return_value={"nlp_alpha": 0.1}
        ), patch.object(
            engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
        ), patch.object(
            engine_router.risk, "get_global_risk_snapshot", return_value={"state": "🟡 整理", "riskScore": 42}
        ) as mock_risk_snapshot, patch(
            "engine_portfolio.compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "🟢 Normal", "size_multiplier": 1.0, "recommended_gross_scale": 1.0},
        ) as mock_overlay, patch.object(
            engine_router.market,
            "compute_nlp_signal_ic",
            return_value={"signal_quality": "weak", "directionality": "positive", "ic_rolling_mean": 0.03},
        ) as mock_ic, patch.object(
            engine_router.market, "fetch_live_price", return_value="123.45 (來源: YF)"
        ), patch.object(
            engine_router.market, "get_stock_news", return_value="NEWS"
        ):
            context = engine_router.get_strat_context("AAPL MSFT")

        self.assertIn("AAPL", context)
        self.assertIn("MSFT", context)
        self.assertEqual(mock_overlay.call_count, 1)
        self.assertEqual(mock_risk_snapshot.call_count, 1)
        self.assertEqual(mock_ic.call_count, 2)


class FrontalLobePatchRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        temp_root = Path(self.tempdir.name)
        self.patchers = [
            patch.object(engine_memory, "BRAIN_DIR", temp_root),
            patch.object(engine_memory, "BRAIN_FILE", temp_root / "commit.json"),
            patch.object(engine_memory, "FRONTAL_LOBE_FILE", temp_root / "frontal-lobe.md"),
            patch.object(engine_memory, "EMOTION_FILE", temp_root / "emotion-log.json"),
            patch.object(engine_memory, "MARKET_REGIME_FILE", temp_root / "market-regime.md"),
            patch.object(engine_memory, "HEARTBEAT_FILE", temp_root / "heartbeat.json"),
            patch.object(engine_memory, "SNAPSHOT_FILE", temp_root / "snapshot.json"),
        ]
        for patcher in self.patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.tempdir.cleanup()

    def test_update_lobe_section_preserves_legacy_note_context(self):
        brain = engine_memory.Brain()
        brain.state["frontalLobe"] = (
            "Bearish on NVDA while 900 acts like resistance. "
            "If 880 breaks, trim exposure and wait for confirmation."
        )

        result = brain.update_lobe_section(
            "Portfolio Health",
            "NAV: NT$1,000,000 | PnL: +3.2% | Top3 集中度: 68%",
            source="portfolio_review",
        )

        sections = engine_memory.parse_frontal_lobe_note(brain.state["frontalLobe"])

        self.assertTrue(result["success"])
        self.assertIn("Bearish", sections["Market View"])
        self.assertIn("900", sections["Core Levels"])
        self.assertEqual(sections["Portfolio Health"], "NAV: NT$1,000,000 | PnL: +3.2% | Top3 集中度: 68%")
        self.assertIn("trim exposure", sections["Next Round"])


if __name__ == "__main__":
    unittest.main()
