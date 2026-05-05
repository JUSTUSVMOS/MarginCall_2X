import unittest
from unittest.mock import patch

from google.genai import types

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


class FakeChat:
    def __init__(self, response_text: str, history):
        self.response_text = response_text
        self._history = list(history)

    def send_message(self, part):
        self._history.extend(
            [
                types.Content(role="user", parts=[types.Part(text=part.text)]),
                types.Content(role="model", parts=[types.Part(text=self.response_text)]),
            ]
        )
        return type("Response", (), {"text": self.response_text})()

    def get_history(self):
        return list(self._history)


class FakeChats:
    def __init__(self, response_text: str):
        self.response_text = response_text

    def create(self, model, config, history):
        return FakeChat(self.response_text, history)


class FakeClient:
    def __init__(self, response_text: str):
        self.chats = FakeChats(response_text)


class FakeHttpxResponse:
    def __init__(self, status_code: int, payload, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class FakeHttpxClient:
    def __init__(self, response: FakeHttpxResponse):
        self.response = response
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers=None, json=None):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return self.response


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


class LlmCompactionTests(unittest.TestCase):
    def setUp(self):
        self.original_client = llm._client
        self.addCleanup(setattr, llm, "_client", self.original_client)
        llm.reset_dead_engines()
        self.addCleanup(llm.reset_dead_engines)

    def test_full_compact_history_uses_quick_call_summary(self):
        history = [
            {"role": "user", "parts": ["u1"]},
            {"role": "tool", "parts": ["tool-1 full payload"]},
            {"role": "user", "parts": ["u2"]},
            {"role": "tool", "parts": ["tool-2 full payload"]},
            {"role": "user", "parts": ["u3"]},
            {"role": "tool", "parts": ["tool-3 full payload"]},
            {"role": "user", "parts": ["u4"]},
            {"role": "tool", "parts": ["tool-4 full payload"]},
            {"role": "user", "parts": ["u5"]},
            {"role": "tool", "parts": ["tool-5 full payload"]},
            {"role": "user", "parts": ["u6"]},
            {"role": "tool", "parts": ["tool-6 full payload"]},
            {"role": "user", "parts": ["u7"]},
        ]

        with patch.object(llm, "quick_call", return_value="## 原始問題\nold summary"):
            compacted = llm._full_compact_history([llm._normalize_history_item(item) for item in history])

        self.assertTrue(compacted[0].parts[0].text.startswith("[history summary]"))
        self.assertIn("## 原始問題", compacted[0].parts[0].text)

    def test_full_compact_history_falls_back_when_quick_call_returns_none(self):
        history = [
            {"role": "user", "parts": ["u1"]},
            {"role": "tool", "parts": ["tool-1 full payload"]},
            {"role": "user", "parts": ["u2"]},
            {"role": "tool", "parts": ["tool-2 full payload"]},
            {"role": "user", "parts": ["u3"]},
            {"role": "tool", "parts": ["tool-3 full payload"]},
            {"role": "user", "parts": ["u4"]},
            {"role": "tool", "parts": ["tool-4 full payload"]},
            {"role": "user", "parts": ["u5"]},
            {"role": "tool", "parts": ["tool-5 full payload"]},
            {"role": "user", "parts": ["u6"]},
            {"role": "tool", "parts": ["tool-6 full payload"]},
            {"role": "user", "parts": ["u7"]},
        ]

        with patch.object(llm, "quick_call", return_value=None):
            compacted = llm._full_compact_history([llm._normalize_history_item(item) for item in history])

        self.assertTrue(compacted[0].parts[0].text.startswith("[history summary]"))
        self.assertIn("- user:", compacted[0].parts[0].text)

    def test_full_compact_history_logs_when_quick_call_returns_empty_summary(self):
        history = [
            {"role": "user", "parts": ["u1"]},
            {"role": "tool", "parts": ["tool-1 full payload"]},
            {"role": "user", "parts": ["u2"]},
            {"role": "tool", "parts": ["tool-2 full payload"]},
            {"role": "user", "parts": ["u3"]},
            {"role": "tool", "parts": ["tool-3 full payload"]},
            {"role": "user", "parts": ["u4"]},
            {"role": "tool", "parts": ["tool-4 full payload"]},
            {"role": "user", "parts": ["u5"]},
            {"role": "tool", "parts": ["tool-5 full payload"]},
            {"role": "user", "parts": ["u6"]},
            {"role": "tool", "parts": ["tool-6 full payload"]},
            {"role": "user", "parts": ["u7"]},
        ]

        with patch.object(llm, "quick_call", return_value=""), patch.object(llm.logger, "info") as mock_info:
            compacted = llm._full_compact_history([llm._normalize_history_item(item) for item in history])

        self.assertTrue(compacted[0].parts[0].text.startswith("[history summary]"))
        self.assertIn("- user:", compacted[0].parts[0].text)
        mock_info.assert_called_once()
        self.assertIn("falling back to naive compaction", mock_info.call_args[0][0])

    def test_normalize_history_item_serializes_function_payloads_as_json(self):
        normalized = llm._normalize_history_item(
            {
                "role": "tool",
                "parts": [
                    {
                        "function_call": {
                            "name": "lookup",
                            "arguments": {"symbol": "TSLA", "limit": 5},
                        }
                    }
                ],
            }
        )

        text = normalized.parts[0].text
        self.assertTrue(text.startswith("[function_call] "))
        self.assertIn('"name": "lookup"', text)
        self.assertIn('"symbol": "TSLA"', text)
        self.assertIn('"limit": 5', text)

    def test_chat_with_tools_preserves_summary_history_on_gemini_path(self):
        llm._client = FakeClient("gemini ok")
        history = [
            types.Content(role="user", parts=[types.Part(text=f"older question {index}")])
            if index % 2 == 0
            else types.Content(role="model", parts=[types.Part(text=f"older answer {index}")])
            for index in range(14)
        ]

        with patch.object(llm, "quick_call", return_value="## 原始問題\ncarry forward"):
            result = llm.chat_with_tools(
                "next step",
                tools=[],
                history=history,
                models=["gemini-2.5-flash"],
                timeout_seconds=11,
                max_timeouts=2,
            )

        self.assertEqual(result, "gemini ok")
        self.assertTrue(history[0].parts[0].text.startswith("[history summary]"))



class OpenRouterLoopTests(unittest.TestCase):

    def test_call_openrouter_falls_back_to_choice_content_when_message_missing(self):
        client = FakeHttpxClient(
            FakeHttpxResponse(
                200,
                {"choices": [{"content": [{"type": "text", "text": "hello from content"}]}]},
            )
        )

        with patch.object(llm, "_openrouter_key", "test-key"), patch.object(
            llm.httpx, "Client", return_value=client
        ), patch.object(llm, "mark_dead") as mock_mark_dead:
            result = llm._call_openrouter(
                "openrouter/test-model",
                [{"role": "user", "content": "hi"}],
                timeout_seconds=5,
            )

        self.assertEqual(result, {"content": [{"type": "text", "text": "hello from content"}]})
        mock_mark_dead.assert_not_called()

    def test_execute_single_tool_call_accepts_dict_arguments(self):
        @tool()
        def runtime_read(symbol: str) -> str:
            return f"read:{symbol}"

        self.addCleanup(_REGISTRY.pop, "runtime_read", None)

        result = llm._execute_single_tool_call(
            {
                "id": "call_dict_args",
                "function": {
                    "name": "runtime_read",
                    "arguments": {"symbol": "TSLA"},
                },
            },
            {"runtime_read": runtime_read},
        )

        self.assertEqual(result["tool_call_id"], "call_dict_args")
        self.assertEqual(result["name"], "runtime_read")
        self.assertEqual(result["content"], "read:TSLA")

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

    def test_chat_with_tools_appends_tool_results_before_loop_warning(self):
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

        third_round = captured_messages[2]
        assistant_index = next(
            i for i, message in enumerate(third_round)
            if any(call.get("id") == "call_2" for call in message.get("tool_calls", []))
        )
        tool_index = next(
            i for i, message in enumerate(third_round)
            if message.get("role") == "tool" and message.get("tool_call_id") == "call_2"
        )
        warning_index = next(
            i for i, message in enumerate(third_round)
            if message.get("role") == "user" and "very similar" in message.get("content", "")
        )

        self.assertEqual(result, "loop handled")
        self.assertLess(assistant_index, tool_index)
        self.assertLess(tool_index, warning_index)

    def test_chat_with_tools_avoids_duplicate_same_turn_loop_warnings(self):
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

        third_round = captured_messages[2]
        same_turn_warnings = [
            message for message in third_round
            if message.get("role") == "user" and "only_symbol_tool" in message.get("content", "")
        ]

        self.assertEqual(result, "loop handled")
        self.assertEqual(len(same_turn_warnings), 1)



if __name__ == "__main__":
    unittest.main()
