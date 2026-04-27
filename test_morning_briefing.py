import builtins
import datetime
import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd


class MorningBriefingTests(unittest.TestCase):
    def _fresh_import(self, module_name: str):
        original = sys.modules.pop(module_name, None)
        try:
            return importlib.import_module(module_name)
        finally:
            if original is not None and module_name not in sys.modules:
                sys.modules[module_name] = original

    def test_engine_briefing_imports_without_engine_risk_dependency(self):
        original_module = sys.modules.pop("engine_briefing", None)
        original_risk_module = sys.modules.pop("engine_risk", None)
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "engine_risk":
                raise ModuleNotFoundError("No module named 'scipy'")
            return original_import(name, globals, locals, fromlist, level)

        try:
            with patch.object(builtins, "__import__", side_effect=guarded_import):
                briefing = importlib.import_module("engine_briefing")
        finally:
            sys.modules.pop("engine_briefing", None)
            sys.modules.pop("engine_risk", None)
            if original_module is not None:
                sys.modules["engine_briefing"] = original_module
            if original_risk_module is not None:
                sys.modules["engine_risk"] = original_risk_module

        self.assertTrue(callable(briefing.build_morning_briefing))

    def test_build_morning_briefing_prioritizes_trade_plan_alerts_over_noise(self):
        briefing = self._fresh_import("engine_briefing")
        fake_risk = types.SimpleNamespace(get_global_risk_snapshot=lambda: {"state": "🟡 整理", "riskScore": 42})

        with patch.object(
            briefing.portfolio,
            "get_open_trade_plan_alerts",
            return_value=[
                {"symbol": "MRVL", "alert_type": "stop_hit", "severity": "critical", "payload": {"current_price": 79.8}},
                {"symbol": "TSLA", "alert_type": "holding_expiry", "severity": "medium", "payload": {}},
            ],
        ), patch.object(
            briefing.market,
            "get_market_calendar_events",
            return_value=[
                {"symbol": "AMD", "event_type": "earnings", "starts_at": "2026-04-24T20:00:00Z", "label": "AMD earnings"}
            ],
        ), patch.object(briefing, "risk", fake_risk), patch.object(
            briefing.portfolio,
            "build_trade_plan_status_summary",
            return_value={"open_alert_count": 2, "missing_plan_count": 0, "alerts": []},
        ), patch.object(briefing.portfolio, "get_current_portfolio_symbols", return_value=["MRVL", "AMD", "TSLA"]):
            report = briefing.build_morning_briefing()

        action_lines = [line for line in report.splitlines() if line[:2] in {"1.", "2.", "3."}]
        self.assertEqual(
            action_lines,
            [
                "1. MRVL stop_hit - review or exit immediately",
                "2. TSLA holding_expiry - time-box has been exceeded",
                "3. AMD AMD earnings today - avoid adding risk before the event",
            ],
        )

    def test_build_morning_briefing_surfaces_monitor_degraded_explicitly(self):
        briefing = self._fresh_import("engine_briefing")
        fake_risk = types.SimpleNamespace(get_global_risk_snapshot=lambda: {"state": "🟢 風險開", "riskScore": 20})

        with patch.object(
            briefing.portfolio,
            "get_open_trade_plan_alerts",
            return_value=[
                {"symbol": "AMD", "alert_type": "monitor_degraded", "severity": "high", "payload": {"error": "price refresh failed"}}
            ],
        ), patch.object(briefing.market, "get_market_calendar_events", return_value=[]), patch.object(
            briefing, "risk", fake_risk
        ), patch.object(
            briefing.portfolio,
            "build_trade_plan_status_summary",
            return_value={"open_alert_count": 1, "missing_plan_count": 0, "alerts": []},
        ), patch.object(briefing.portfolio, "get_current_portfolio_symbols", return_value=["AMD"]):
            report = briefing.build_morning_briefing()

        self.assertIn("1. AMD monitor_degraded - price refresh failed", report)
        self.assertIn("一句話：AMD monitor_degraded - price refresh failed", report)

    def test_start_scheduler_registers_morning_briefing_push(self):
        import src.scheduler as scheduler_runtime

        recorded_jobs = []

        class FakeScheduler:
            def __init__(self, *args, **kwargs):
                self.running = False

            def add_job(self, func, *args, **kwargs):
                recorded_jobs.append((func, args, kwargs))

            def start(self):
                self.running = True

        original_scheduler = scheduler_runtime._scheduler
        scheduler_runtime._scheduler = None
        try:
            with patch.object(scheduler_runtime, "BackgroundScheduler", FakeScheduler), patch.object(
                scheduler_runtime, "macro_brain_heartbeat", return_value=None
            ), patch.object(scheduler_runtime, "daily_portfolio_review", return_value=None):
                scheduler_runtime.start_scheduler()
        finally:
            scheduler_runtime._scheduler = original_scheduler

        self.assertTrue(
            any(
                func is scheduler_runtime.morning_briefing_push and kwargs.get("id") == "morning-briefing-push"
                for func, _, kwargs in recorded_jobs
            )
        )

    def test_get_market_calendar_events_handles_explicit_symbols_and_emits_earnings_event(self):
        import engine_market

        earnings_dates = pd.DataFrame(
            {"EPS Estimate": [1.23]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-04-24T06:00:00Z")]),
        )
        fake_ticker = MagicMock(earnings_dates=earnings_dates)
        now = datetime.datetime(2026, 4, 24, 1, 30, 0, tzinfo=datetime.timezone.utc)

        with patch.object(engine_market, "get_ticker", return_value=fake_ticker) as mock_get_ticker, patch.object(
            engine_market.datetime, "datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = now
            events = engine_market.get_market_calendar_events(symbols=["amd"], days=1)

        self.assertEqual(mock_get_ticker.call_args_list[0].args[0], "AMD")
        self.assertEqual(
            events,
            [
                {
                    "symbol": "AMD",
                    "event_type": "earnings",
                    "starts_at": "2026-04-24T06:00:00+00:00",
                    "label": "AMD earnings",
                }
            ],
        )


if __name__ == '__main__':
    unittest.main()
