import unittest
from unittest.mock import patch, MagicMock
import engine_briefing

class MorningBriefingTests(unittest.TestCase):
    def test_build_morning_briefing_prioritizes_trade_plan_alerts_over_noise(self):
        with patch("engine_briefing.portfolio.get_open_trade_plan_alerts", return_value=[
            {"symbol": "MRVL", "alert_type": "stop_hit", "severity": "critical", "payload": {"current_price": 79.8}}
        ]), patch("engine_briefing.market.get_market_calendar_events", return_value=[
            {"symbol": "AMD", "event_type": "earnings", "starts_at": "2026-04-24T20:00:00Z", "label": "AMD earnings"}
        ]), patch("engine_briefing.risk.get_global_risk_snapshot", return_value={"state": "🟡 整理", "riskScore": 42}), patch(
            "engine_briefing.portfolio.build_trade_plan_status_summary",
            return_value={"open_alert_count": 1, "missing_plan_count": 0, "alerts": []},
        ), patch("engine_briefing.portfolio.get_current_portfolio_symbols", return_value=["MRVL", "AMD"]):
            report = engine_briefing.build_morning_briefing()

        self.assertIn("MRVL", report)
        self.assertIn("stop", report.lower())
        self.assertIn("AMD", report)

    def test_build_morning_briefing_surfaces_degraded_monitoring(self):
        with patch("engine_briefing.portfolio.get_open_trade_plan_alerts", return_value=[
            {"symbol": "AMD", "alert_type": "monitor_degraded", "severity": "high", "payload": {"error": "price refresh failed"}}
        ]), patch("engine_briefing.market.get_market_calendar_events", return_value=[]), patch(
            "engine_briefing.risk.get_global_risk_snapshot", return_value={"state": "🟢 風險開", "riskScore": 20}
        ), patch("engine_briefing.portfolio.build_trade_plan_status_summary", return_value={"open_alert_count": 1, "missing_plan_count": 0, "alerts": []}), patch("engine_briefing.portfolio.get_current_portfolio_symbols", return_value=["AMD"]):
            report = engine_briefing.build_morning_briefing()

        self.assertIn("monitor_degraded", report)
        self.assertIn("price refresh failed", report)

if __name__ == '__main__':
    unittest.main()
