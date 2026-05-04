import unittest

from src.tool_loop_guard import ToolLoopGuard


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


if __name__ == "__main__":
    unittest.main()
