import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

from src import llm


def _timeout_from_config(config):
    if config is None:
        return None
    http_options = getattr(config, "http_options", None)
    return getattr(http_options, "timeout", None) if http_options is not None else None


class FakeModels:
    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.calls = []
        self.timeouts = []

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        self.timeouts.append(_timeout_from_config(config))
        behavior = self.behaviors[model]
        if isinstance(behavior, Exception):
            raise behavior
        return SimpleNamespace(text=behavior)


class FakeChat:
    def __init__(self, model, behavior, config, history):
        self.model = model
        self.behavior = behavior
        self.config = config
        self.create_timeout = _timeout_from_config(config)
        self.send_timeouts = []
        self._history = list(history or [])

    def send_message(self, user_text, config=None):
        effective_config = config or self.config
        self.send_timeouts.append(_timeout_from_config(effective_config))
        if isinstance(self.behavior, Exception):
            raise self.behavior

        self._history = list(self._history) + [
            types.Content(role="user", parts=[types.Part(text=llm._normalize_part(user_text))]),
            types.Content(role="model", parts=[types.Part(text=self.behavior)]),
        ]
        return SimpleNamespace(text=self.behavior)

    def get_history(self):
        return list(self._history)


class FakeChats:
    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.created = []

    def create(self, model, config, history):
        chat = FakeChat(model, self.behaviors[model], config, history)
        self.created.append(chat)
        return chat


class FakeClient:
    def __init__(self, model_behaviors=None, chat_behaviors=None):
        self.models = FakeModels(model_behaviors or {})
        self.chats = FakeChats(chat_behaviors or {})


class LlmTimeoutRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.original_client = llm._client
        llm.reset_dead_engines()

    def tearDown(self):
        llm._client = self.original_client
        llm.reset_dead_engines()

    def test_quick_call_uses_transport_timeout_and_falls_back_after_timeout(self):
        first_model = "gemini-3.1-flash-lite-preview"
        second_model = "gemma-4-31b-it"
        llm._client = FakeClient(
            model_behaviors={
                first_model: TimeoutError("request timed out"),
                second_model: "fallback ok",
            }
        )

        result = llm.quick_call(
            "hello",
            models=[first_model, second_model],
            timeout_seconds=7,
        )

        self.assertEqual(result, "fallback ok")
        self.assertEqual(llm._client.models.calls, [first_model, second_model])
        self.assertEqual(llm._client.models.timeouts, [7, 7])
        self.assertEqual(llm.get_alive_models([first_model, second_model]), [second_model])

    def test_chat_with_tools_timeout_fallback_preserves_history_without_worker_threads(self):
        first_model = "gemini-3.1-flash-lite-preview"
        second_model = "gemma-4-31b-it"
        llm._client = FakeClient(
            chat_behaviors={
                first_model: TimeoutError("request timed out"),
                second_model: "工具對話完成",
            }
        )
        history = [
            types.Content(role="user", parts=[types.Part(text="old question")]),
            types.Content(role="model", parts=[types.Part(text="old answer")]),
        ]

        with patch.object(llm.threading, "Thread", side_effect=AssertionError("worker thread should not be used")):
            result = llm.chat_with_tools(
                "ping",
                tools=[],
                history=history,
                models=[first_model, second_model],
                timeout_seconds=9,
                max_timeouts=2,
            )

        self.assertEqual(result, "工具對話完成")
        self.assertEqual([chat.model for chat in llm._client.chats.created], [first_model, second_model])
        self.assertEqual([chat.create_timeout for chat in llm._client.chats.created], [9, 9])
        self.assertEqual(llm._client.chats.created[1].send_timeouts, [9])
        self.assertEqual(llm.get_alive_models([first_model, second_model]), [second_model])
        self.assertEqual([item.role for item in history], ["user", "model", "user", "model"])
        self.assertEqual(history[-2].parts[0].text, "ping")
        self.assertEqual(history[-1].parts[0].text, "工具對話完成")

    def test_chat_with_tools_can_switch_to_openrouter_and_normalize_history(self):
        first_model = "gemini-3.1-flash-lite-preview"
        second_model = "minimax/minimax-m2.5:free"
        llm._client = FakeClient(chat_behaviors={first_model: TimeoutError("request timed out")})
        history = [
            {"role": "user", "parts": ["older question"]},
            {"role": "model", "parts": ["older answer"]},
        ]
        captured = {}

        def fake_openrouter(model_name, messages, temperature=0.3, tools=None, timeout_seconds=60):
            captured["model"] = model_name
            captured["messages"] = messages
            captured["timeout"] = timeout_seconds
            return {"content": {"text": "openrouter ok"}}

        with patch.object(llm.threading, "Thread", side_effect=AssertionError("worker thread should not be used")), patch.object(
            llm, "_call_openrouter", side_effect=fake_openrouter
        ):
            result = llm.chat_with_tools(
                "next step",
                tools=[],
                history=history,
                models=[first_model, second_model],
                timeout_seconds=11,
                max_timeouts=2,
            )

        self.assertEqual(result, "openrouter ok")
        self.assertEqual(captured["model"], second_model)
        self.assertEqual(captured["timeout"], 11)
        self.assertEqual(
            [message["role"] for message in captured["messages"]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(captured["messages"][0]["content"], "older question")
        self.assertEqual(captured["messages"][1]["content"], "older answer")
        self.assertEqual(captured["messages"][2]["content"], "next step")
        self.assertTrue(all(hasattr(item, "role") for item in history))
        self.assertEqual(history[-1].parts[0].text, "openrouter ok")

    def test_chat_with_tools_returns_timeout_message_after_repeated_timeouts(self):
        first_model = "gemini-3.1-flash-lite-preview"
        second_model = "gemma-4-31b-it"
        llm._client = FakeClient(
            chat_behaviors={
                first_model: TimeoutError("request timed out"),
                second_model: TimeoutError("request timed out"),
            }
        )

        result = llm.chat_with_tools(
            "ping",
            tools=[],
            history=[],
            models=[first_model, second_model],
            timeout_seconds=5,
            max_timeouts=2,
            timeout_message="timeout hit",
        )

        self.assertEqual(result, "timeout hit")


if __name__ == "__main__":
    unittest.main()
