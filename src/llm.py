import logging
import os
import threading
import time
from typing import Dict, List, Optional, Set

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


AVAILABLE_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemma-4-31b-it",
    "gemini-flash-latest",
]

TEMPORARY_ERROR_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "503",
    "UNAVAILABLE",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
)

_api_key = os.getenv("GEMINI_API_KEY")
_client = genai.Client(api_key=_api_key) if _api_key else None
_dead_engines: Dict[str, float] = {}
_lock = threading.Lock()


def is_configured() -> bool:
    return _client is not None


def reset_dead_engines() -> None:
    with _lock:
        _dead_engines.clear()


def mark_dead(model: str, cooldown_seconds: int = 120) -> None:
    with _lock:
        _dead_engines[model] = time.time() + cooldown_seconds


def get_alive_models(models: Optional[List[str]] = None) -> List[str]:
    candidates = list(models or AVAILABLE_MODELS)
    now = time.time()
    alive: List[str] = []
    seen: Set[str] = set()
    with _lock:
        for model_name in candidates:
            if model_name in seen:
                continue
            seen.add(model_name)
            if _dead_engines.get(model_name, 0) < now:
                alive.append(model_name)
    return alive


def is_temporary_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return any(marker in message for marker in TEMPORARY_ERROR_MARKERS)


def _build_config(system_instruction: str, temperature: float, tools=None):
    kwargs = {"temperature": temperature}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if tools is not None:
        kwargs["tools"] = tools
    return types.GenerateContentConfig(**kwargs)


def quick_call(
    prompt: str,
    models: Optional[List[str]] = None,
    system_instruction: str = "",
    temperature: float = 0.1,
) -> Optional[str]:
    if not _client:
        return None

    for model_name in get_alive_models(models):
        try:
            response = _client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=_build_config(system_instruction, temperature),
            )
            if response.text:
                return response.text
        except Exception as exc:
            if is_temporary_error(exc):
                logger.warning(f"[LLM] {model_name} temp failure: {exc}")
                mark_dead(model_name)
                continue
            logger.error(f"[LLM] {model_name} fatal: {exc}")
            return None

    return None


def chat_with_tools(
    user_text: str,
    tools: list,
    system_instruction: str = "",
    history=None,
    temperature: float = 0.3,
    timeout_seconds: int = 30,
    max_timeouts: int = 2,
    models: Optional[List[str]] = None,
    unavailable_message: str = "🚀 所有推進器皆暫時熄火，請稍後再試。",
    timeout_message: Optional[str] = None,
) -> str:
    if not _client:
        return "⚠️ 未設定 GEMINI_API_KEY。"

    timeout_count = 0
    for model_name in get_alive_models(models):
        if timeout_count >= max_timeouts:
            break
        try:
            chat = _client.chats.create(
                model=model_name,
                config=_build_config(system_instruction, temperature, tools),
                history=history,
            )

            response_container = []
            exception_container = []

            def _thread_task():
                try:
                    response_container.append(chat.send_message(user_text))
                except Exception as exc:
                    exception_container.append(exc)

            llm_thread = threading.Thread(target=_thread_task)
            llm_thread.start()
            llm_thread.join(timeout=timeout_seconds)

            if llm_thread.is_alive():
                logger.warning(f"[LLM] {model_name} timeout ({timeout_seconds}s)")
                mark_dead(model_name)
                timeout_count += 1
                if timeout_message:
                    return timeout_message
                continue

            if exception_container:
                raise exception_container[0]

            if not response_container:
                continue

            response = response_container[0]
            if history is not None:
                new_history = chat.get_history()
                history.clear()
                history.extend(new_history[-20:])

            return response.text if response.text else "大腦空白。"
        except Exception as exc:
            if is_temporary_error(exc):
                logger.warning(f"[LLM] {model_name} temp failure: {exc}")
                mark_dead(model_name)
                if "DEADLINE_EXCEEDED" in str(exc).upper():
                    timeout_count += 1
                continue
            logger.error(f"[LLM] {model_name} fatal: {exc}")
            return f"⚠️ 模型異常: {str(exc)[:100]}"

    return unavailable_message
