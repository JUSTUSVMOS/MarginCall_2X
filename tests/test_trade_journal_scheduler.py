"""tests/test_trade_journal_scheduler.py — Tracked tests for trade journal scheduler jobs.

Verifies that:
- start_scheduler registers 'trade-journal-settlement' and 'weekly-trade-journal' jobs.
- trade_journal_checkpoint_job delegates to engine_journal.settle_due_trade_outcomes.
- weekly_trade_journal_job delegates to engine_journal.build_weekly_attribution_report.
"""

import unittest
from unittest.mock import MagicMock, patch

import src.scheduler as scheduler


class TestTradeJournalSchedulerRegistration(unittest.TestCase):
    @patch("src.scheduler.BackgroundScheduler")
    def test_start_scheduler_registers_trade_journal_settlement_job(self, scheduler_cls):
        scheduler_obj = MagicMock()
        scheduler_obj.running = False
        scheduler_cls.return_value = scheduler_obj
        with (
            patch("src.scheduler.macro_brain_heartbeat"),
            patch("src.scheduler.daily_portfolio_review"),
        ):
            scheduler._scheduler = None
            scheduler.start_scheduler()
        added_ids = [call.kwargs["id"] for call in scheduler_obj.add_job.call_args_list]
        self.assertIn("trade-journal-settlement", added_ids)

    @patch("src.scheduler.BackgroundScheduler")
    def test_start_scheduler_registers_weekly_trade_journal_job(self, scheduler_cls):
        scheduler_obj = MagicMock()
        scheduler_obj.running = False
        scheduler_cls.return_value = scheduler_obj
        with (
            patch("src.scheduler.macro_brain_heartbeat"),
            patch("src.scheduler.daily_portfolio_review"),
        ):
            scheduler._scheduler = None
            scheduler.start_scheduler()
        added_ids = [call.kwargs["id"] for call in scheduler_obj.add_job.call_args_list]
        self.assertIn("weekly-trade-journal", added_ids)

    @patch("src.scheduler.BackgroundScheduler")
    def test_start_scheduler_registers_both_journal_jobs(self, scheduler_cls):
        """Both journal job IDs must be present in a single start_scheduler call."""
        scheduler_obj = MagicMock()
        scheduler_obj.running = False
        scheduler_cls.return_value = scheduler_obj
        with (
            patch("src.scheduler.macro_brain_heartbeat"),
            patch("src.scheduler.daily_portfolio_review"),
        ):
            scheduler._scheduler = None
            scheduler.start_scheduler()
        added_ids = [call.kwargs["id"] for call in scheduler_obj.add_job.call_args_list]
        self.assertIn("trade-journal-settlement", added_ids)
        self.assertIn("weekly-trade-journal", added_ids)

    @patch("src.scheduler.BackgroundScheduler")
    def test_start_scheduler_uses_plan_aligned_trade_journal_timings(self, scheduler_cls):
        """Trade journal jobs must keep the plan-aligned non-conflicting schedule."""
        scheduler_obj = MagicMock()
        scheduler_obj.running = False
        scheduler_cls.return_value = scheduler_obj
        with (
            patch("src.scheduler.macro_brain_heartbeat"),
            patch("src.scheduler.daily_portfolio_review"),
        ):
            scheduler._scheduler = None
            scheduler.start_scheduler()

        settlement_call = next(
            call for call in scheduler_obj.add_job.call_args_list
            if call.kwargs.get("id") == "trade-journal-settlement"
        )
        weekly_call = next(
            call for call in scheduler_obj.add_job.call_args_list
            if call.kwargs.get("id") == "weekly-trade-journal"
        )

        self.assertEqual(settlement_call.kwargs.get("day_of_week"), "tue-sat")
        self.assertEqual(settlement_call.kwargs.get("hour"), 7)
        self.assertEqual(settlement_call.kwargs.get("minute"), 15)
        self.assertEqual(weekly_call.kwargs.get("day_of_week"), "sun")
        self.assertEqual(weekly_call.kwargs.get("hour"), 18)
        self.assertEqual(weekly_call.kwargs.get("minute"), 0)


class TestTradeJournalCheckpointJob(unittest.TestCase):
    @patch("engine_journal.settle_due_trade_outcomes", return_value={"settled": 2, "errors": 0})
    def test_calls_settle_due_trade_outcomes(self, settle_mock):
        result = scheduler.trade_journal_checkpoint_job()
        settle_mock.assert_called_once()
        self.assertEqual(result["settled"], 2)

    @patch("engine_journal.settle_due_trade_outcomes", side_effect=RuntimeError("db error"))
    def test_returns_none_on_exception(self, settle_mock):
        result = scheduler.trade_journal_checkpoint_job()
        self.assertIsNone(result)

    @patch("engine_journal.settle_due_trade_outcomes", return_value={"settled": 0, "errors": 0})
    def test_returns_result_dict(self, settle_mock):
        result = scheduler.trade_journal_checkpoint_job()
        self.assertIsInstance(result, dict)


class TestWeeklyTradeJournalJob(unittest.TestCase):
    def tearDown(self):
        scheduler._bot_instance = None
        scheduler._user_id = None

    @patch("engine_journal.build_weekly_attribution_report", return_value={
        "as_of": "2025-01-15",
        "resolved_checkpoints": 3,
        "avg_actual_return_pct": 1.5,
        "avg_beta_component_pct": 0.8,
        "avg_sector_component_pct": 0.4,
        "avg_timing_component_pct": 0.3,
        "beta_coverage_count": 3,
        "sector_coverage_count": 2,
    })
    def test_calls_build_weekly_attribution_report(self, report_mock):
        result = scheduler.weekly_trade_journal_job()
        report_mock.assert_called_once()
        self.assertIsNotNone(result)

    @patch("engine_journal.build_weekly_attribution_report", side_effect=RuntimeError("report error"))
    def test_returns_none_on_exception(self, report_mock):
        result = scheduler.weekly_trade_journal_job()
        self.assertIsNone(result)

    @patch("engine_journal.build_weekly_attribution_report", return_value={
        "as_of": "2025-01-15",
        "resolved_checkpoints": 0,
        "avg_actual_return_pct": 0.0,
        "avg_beta_component_pct": 0.0,
        "avg_sector_component_pct": 0.0,
        "avg_timing_component_pct": 0.0,
        "beta_coverage_count": 0,
        "sector_coverage_count": 0,
    })
    def test_returns_report_dict(self, report_mock):
        result = scheduler.weekly_trade_journal_job()
        self.assertIsInstance(result, dict)

    @patch("engine_journal.build_weekly_attribution_report", return_value={
        "as_of": "2025-01-15",
        "resolved_checkpoints": 3,
        "avg_actual_return_pct": 1.5,
        "avg_beta_component_pct": 0.8,
        "avg_sector_component_pct": 0.4,
        "avg_timing_component_pct": 0.3,
        "beta_coverage_count": 3,
        "sector_coverage_count": 2,
    })
    def test_sends_message_via_bot_when_configured(self, report_mock):
        scheduler._bot_instance = MagicMock()
        scheduler._user_id = 12345

        result = scheduler.weekly_trade_journal_job()

        self.assertEqual(result["resolved_checkpoints"], 3)
        scheduler._bot_instance.send_message.assert_called_once()
        user_id, message = scheduler._bot_instance.send_message.call_args[0]
        self.assertEqual(user_id, 12345)
        self.assertIn("Weekly Attribution Report", message)
        self.assertIn("Beta coverage: 3/3", message)


if __name__ == "__main__":
    unittest.main()
