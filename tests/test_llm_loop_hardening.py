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


if __name__ == "__main__":
    unittest.main()
