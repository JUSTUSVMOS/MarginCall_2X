import unittest

from src.tool_loop_guard import ToolLoopGuard
from src.result_budget import (
    SINGLE_RESULT_CAP,
    PER_TURN_BUDGET,
    cap_history_text,
    cap_single_result,
    enforce_turn_budget,
)


class ToolLoopGuardTests(unittest.TestCase):
    def test_tool_loop_guard_warns_on_near_duplicate_args(self):
        guard = ToolLoopGuard(max_calls_per_tool=3, similarity_threshold=0.6)
        guard.record_call("web_search", {"query": "tesla delivery miss q1"}, "found prior result")

        warning = guard.check_should_warn("web_search", {"query": "tesla delivery miss q1 april"})

        self.assertIsNotNone(warning)
        self.assertIn("web_search", warning)
        self.assertIn("prior result", warning)

    def test_tool_loop_guard_formats_usage_warning_after_limit(self):
        guard = ToolLoopGuard(max_calls_per_tool=2, similarity_threshold=0.7)
        guard.record_call("get_market_data", {"symbol": "TSLA"}, "snapshot 1")
        guard.record_call("get_market_data", {"symbol": "TSLA"}, "snapshot 2")

        reminder = guard.format_warning_for_prompt()

        self.assertIsNotNone(reminder)
        self.assertIn("get_market_data", reminder)
        self.assertIn("2", reminder)

    def test_format_warning_includes_near_duplicate_triggers(self):
        guard = ToolLoopGuard(max_calls_per_tool=5, similarity_threshold=0.6)
        guard.record_call("web_search", {"query": "tesla delivery miss q1"}, "previous search result")

        # trigger near-duplicate detection (read-only)
        near_warn = guard.check_should_warn("web_search", {"query": "tesla delivery miss q1 april"})
        self.assertIsNotNone(near_warn)

        # Only recording an actual call should increment the near-duplicate counters
        guard.record_call("web_search", {"query": "tesla delivery miss q1 april"}, "new result")

        reminder = guard.format_warning_for_prompt()
        self.assertIsNotNone(reminder)
        self.assertIn("web_search", reminder)

    def test_check_should_warn_does_not_mutate_counts(self):
        guard = ToolLoopGuard(max_calls_per_tool=5, similarity_threshold=0.6)
        guard.record_call("web_search", {"query": "tesla delivery miss q1"}, "previous search result")

        # First check should warn but not increment near_duplicate_counts yet
        near_warn1 = guard.check_should_warn("web_search", {"query": "tesla delivery miss q1 april"})
        self.assertIsNotNone(near_warn1)

        # Check again; should still be read-only
        near_warn2 = guard.check_should_warn("web_search", {"query": "tesla delivery miss q1 april"})
        self.assertIsNotNone(near_warn2)

        # No near-duplicate warning should be recorded until record_call is called
        self.assertEqual(guard.near_duplicate_counts.get("web_search", 0), 0)

        # Now record the actual call; this should increment the count
        guard.record_call("web_search", {"query": "tesla delivery miss q1 april"}, "new result")
        self.assertEqual(guard.near_duplicate_counts.get("web_search", 0), 1)



class ResultBudgetTests(unittest.TestCase):
    def test_cap_single_result_keeps_head_and_tail(self):
        oversized = "HEAD-" + ("x" * (SINGLE_RESULT_CAP + 50)) + "-TAIL"

        capped = cap_single_result(oversized, "demo_tool")

        self.assertLessEqual(len(capped), SINGLE_RESULT_CAP + 200)
        self.assertIn("HEAD-", capped)
        self.assertIn("-TAIL", capped)
        self.assertIn("結果過長", capped)

    def test_enforce_turn_budget_caps_largest_result_first(self):
        results = [
            {"name": "small_tool", "content": "a" * 200, "tool_call_id": "1"},
            {"name": "big_tool", "content": "b" * (PER_TURN_BUDGET + 500), "tool_call_id": "2"},
        ]

        trimmed = enforce_turn_budget(results)

        self.assertEqual(trimmed[0]["content"], "a" * 200)
        self.assertIn("結果過長", trimmed[1]["content"])

    def test_cap_history_text_trims_tool_like_payload(self):
        oversized = "payload-" + ("z" * (SINGLE_RESULT_CAP + 400))

        capped = cap_history_text(oversized)

        self.assertLess(len(capped), len(oversized))
        self.assertIn("payload-", capped)


if __name__ == "__main__":
    unittest.main()
