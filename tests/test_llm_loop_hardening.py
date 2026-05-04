import unittest
from unittest.mock import patch

from src import llm
from src.tool_loop_guard import ToolLoopGuard
from src.result_budget import (
    SINGLE_RESULT_CAP,
    PER_TURN_BUDGET,
    cap_history_text,
    cap_single_result,
    cap_to_length,
    enforce_turn_budget,
)
from src.tools import _REGISTRY, tool


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

        self.assertLessEqual(len(capped), SINGLE_RESULT_CAP)
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

    def test_enforce_turn_budget_does_not_mutate_input(self):
        # prepare input and keep a copy
        import copy

        results = [
            {"name": "t1", "content": "x" * (SINGLE_RESULT_CAP + 200)},
            {"name": "t2", "content": "y" * (SINGLE_RESULT_CAP + 200)},
        ]
        original = copy.deepcopy(results)

        trimmed = enforce_turn_budget(results)

        # original input must not be mutated
        self.assertEqual(results, original)
        # returned list may be different objects
        self.assertIsNot(results, trimmed)
        self.assertIsInstance(trimmed, list)

    def test_enforce_turn_budget_compacts_multiple_under_single_cap(self):
        # create several items each under SINGLE_RESULT_CAP but collectively over PER_TURN_BUDGET
        piece_len = SINGLE_RESULT_CAP - 1
        count = (PER_TURN_BUDGET // piece_len) + 2
        results = [
            {"name": f"tool{i}", "content": "x" * piece_len} for i in range(count)
        ]

        total_before = sum(len(r["content"]) for r in results)
        self.assertGreater(total_before, PER_TURN_BUDGET)

        trimmed = enforce_turn_budget(results)
        total_after = sum(len(str(r.get("content", ""))) for r in trimmed)

        # must fit within per-turn budget
        self.assertLessEqual(total_after, PER_TURN_BUDGET)
        # ensure at least one item was shortened
        self.assertTrue(any(len(str(r.get("content", ""))) < piece_len for r in trimmed))

    def test_00_enforce_turn_budget_converges_when_all_items_at_final_min_keep(self):
        # Many items exactly at final_min_keep (20) to reproduce degenerate case
        piece_len = 20
        count = (PER_TURN_BUDGET // piece_len) + 3
        results = [
            {"name": f"t{i}", "content": "x" * piece_len} for i in range(count)
        ]

        total_before = sum(len(r["content"]) for r in results)
        self.assertGreater(total_before, PER_TURN_BUDGET)

        trimmed = enforce_turn_budget(results)
        total_after = sum(len(str(r.get("content", ""))) for r in trimmed)

        # must fit within per-turn budget
        self.assertLessEqual(total_after, PER_TURN_BUDGET)

    def test_enforce_turn_budget_converges_on_many_small_items(self):
        # Many items just above min_keep (101) used to fail convergence where
        # the second pass could only cut 1 char per item. Ensure we actually
        # reach PER_TURN_BUDGET.
        piece_len = 101
        count = (PER_TURN_BUDGET // piece_len) + 3
        results = [
            {"name": f"t{i}", "content": "x" * piece_len} for i in range(count)
        ]

        total_before = sum(len(r["content"]) for r in results)
        self.assertGreater(total_before, PER_TURN_BUDGET)

        trimmed = enforce_turn_budget(results)
        total_after = sum(len(str(r.get("content", ""))) for r in trimmed)

        # must fit within per-turn budget
        self.assertLessEqual(total_after, PER_TURN_BUDGET)

    def test_cap_to_length_never_exceeds_max_len(self):
        content = "x" * 200
        capped = cap_to_length(content, "demo_tool", 100)
        self.assertLessEqual(len(capped), 100)

    def test_enforce_turn_budget_second_pass_does_not_bloat_total(self):
        # create many modest-sized items so second-pass trimming is needed
        piece_len = 150
        count = (PER_TURN_BUDGET // piece_len) + 5
        results = [
            {"name": f"t{i}", "content": "x" * piece_len} for i in range(count)
        ]

        total_before = sum(len(r["content"]) for r in results)
        self.assertGreater(total_before, PER_TURN_BUDGET)

        trimmed = enforce_turn_budget(results)
        total_after = sum(len(str(r.get("content", ""))) for r in trimmed)

        # must fit within per-turn budget
        self.assertLessEqual(total_after, PER_TURN_BUDGET)
        # must not increase the total size
        self.assertLessEqual(total_after, total_before)

    def test_cap_history_text_trims_tool_like_payload(self):
        oversized = "payload-" + ("z" * (SINGLE_RESULT_CAP + 400))

        capped = cap_history_text(oversized)

        self.assertLess(len(capped), len(oversized))
        self.assertIn("payload-", capped)


class OpenRouterLoopTests(unittest.TestCase):
    def test_execute_single_tool_call_logs_dropped_args(self):
        def only_symbol(symbol: str) -> str:
            return f"ok:{symbol}"

        tool_call = {
            "id": "call_1",
            "function": {"name": "only_symbol", "arguments": '{"symbol": "TSLA", "extra": "ignore"}'},
        }

        with patch.object(llm.logger, "warning") as warning_mock:
            result = llm._execute_single_tool_call(tool_call, {"only_symbol": only_symbol})

        self.assertEqual(result["content"], "ok:TSLA")
        warning_mock.assert_called_once()
        self.assertIn("Dropping unsupported args", warning_mock.call_args[0][0])

    def test_execute_openai_tool_calls_uses_registry_modes_for_parallel_reads(self):
        execution_log = []
        submitted = []

        @tool()
        def runtime_read_a(symbol: str) -> str:
            execution_log.append(("read", symbol))
            return f"read-a:{symbol}"

        @tool()
        def runtime_read_b(symbol: str) -> str:
            execution_log.append(("read", symbol))
            return f"read-b:{symbol}"

        @tool(mode="write")
        def runtime_write(note: str) -> str:
            execution_log.append(("write", note))
            return f"write:{note}"

        self.addCleanup(_REGISTRY.pop, "runtime_read_a", None)
        self.addCleanup(_REGISTRY.pop, "runtime_read_b", None)
        self.addCleanup(_REGISTRY.pop, "runtime_write", None)

        class FakeFuture:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        class RecordingExecutor:
            def __init__(self, max_workers=None):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                submitted.append(args[0]["function"]["name"])
                return FakeFuture(fn(*args, **kwargs))

        tool_calls = [
            {"id": "call_1", "function": {"name": "runtime_read_a", "arguments": '{"symbol": "TSLA"}'}},
            {"id": "call_2", "function": {"name": "runtime_read_b", "arguments": '{"symbol": "NVDA"}'}},
            {"id": "call_3", "function": {"name": "runtime_write", "arguments": '{"note": "persist"}'}},
        ]

        with patch.object(llm.concurrent.futures, "ThreadPoolExecutor", RecordingExecutor), patch.object(
            llm.concurrent.futures, "as_completed", side_effect=lambda futures: list(futures)
        ):
            results = llm._execute_openai_tool_calls(tool_calls, [runtime_read_a, runtime_read_b, runtime_write])

        self.assertEqual(submitted, ["runtime_read_a", "runtime_read_b"])
        self.assertEqual([item["tool_call_id"] for item in results], ["call_1", "call_2", "call_3"])
        self.assertEqual(execution_log[-1], ("write", "persist"))

    def test_chat_with_tools_injects_loop_warning_before_next_openrouter_round(self):
        responses = iter(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "only_symbol_tool", "arguments": '{"symbol": "ARKK"}'},
                        }
                    ],
                },
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {"name": "only_symbol_tool", "arguments": '{"symbol": "ARKK"}'},
                        }
                    ],
                },
                {"content": {"text": "loop handled"}},
            ]
        )
        captured_messages = []

        def only_symbol_tool(symbol: str) -> str:
            return f"ok:{symbol}"

        def fake_openrouter(model_name, messages, temperature=0.3, tools=None, timeout_seconds=60):
            captured_messages.append(list(messages))
            return next(responses)

        with patch.object(llm, "_call_openrouter", side_effect=fake_openrouter):
            result = llm.chat_with_tools(
                "analyze arkk",
                tools=[only_symbol_tool],
                models=["minimax/minimax-m2.5:free"],
                history=[],
            )

        flattened = "\n".join(
            message.get("content", "")
            for batch in captured_messages
            for message in batch
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        )
        self.assertEqual(result, "loop handled")
        self.assertIn("very similar", flattened)

    def test_chat_with_tools_logs_tool_count_mismatch(self):
        responses = iter(
            [
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "only_symbol_tool", "arguments": '{"symbol": "ARKK"}'},
                        }
                    ],
                },
                {"content": {"text": "loop handled"}},
            ]
        )

        def only_symbol_tool(symbol: str) -> str:
            return f"ok:{symbol}"

        def fake_openrouter(model_name, messages, temperature=0.3, tools=None, timeout_seconds=60):
            return next(responses)

        with patch.object(llm, "_call_openrouter", side_effect=fake_openrouter), patch.object(
            llm, "_execute_openai_tool_calls", return_value=[]
        ), patch.object(llm.logger, "error") as error_mock:
            result = llm.chat_with_tools(
                "analyze arkk",
                tools=[only_symbol_tool],
                models=["minimax/minimax-m2.5:free"],
                history=[],
            )

        self.assertEqual(result, "loop handled")
        error_mock.assert_called_once()
        self.assertIn("Tool execution mismatch", error_mock.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
