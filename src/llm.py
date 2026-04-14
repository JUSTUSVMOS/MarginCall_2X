import logging
import os
import threading
import time
from typing import Dict, List, Optional, Set

from google import genai
from google.genai import types


logger = logging.getLogger(__name__)


# 定義哪些模型支援工具呼叫 (Function Calling)
TOOL_SUPPORTED_MODELS = {
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
}

# 重活：主對話 (需要 tool calling + 深度推理)
HEAVY_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
]

# 輕活：symbol detection、情緒打分、asset 分類
LIGHT_MODELS = [
    "gemma-4-31b-it",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-lite-latest",
]

AVAILABLE_MODELS = HEAVY_MODELS

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


def mark_dead(model: str, exc: Optional[Exception] = None) -> None:
    """極速冷卻邏輯，防止熄火門檻過低"""
    msg = str(exc).upper() if exc else ""
    if "503" in msg or "UNAVAILABLE" in msg:
        cooldown = 5   # 塞車很快會恢復
    elif "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        cooldown = 30  # 配額限制通常按分鐘計
    elif "DEADLINE_EXCEEDED" in msg:
        cooldown = 15  # 逾時通常是暫時的
    else:
        cooldown = 60  # 預設

    with _lock:
        _dead_engines[model] = time.time() + cooldown
        logger.info(f"[LLM] {model} is marked dead for {cooldown}s (Reason: {msg[:30]}...)")


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
    
    # 如果全部都死了，強行復活最後一個保底模型 (Gemma 3 27b)
    if not alive and candidates:
        last_resort = candidates[-1]
        logger.warning(f"[LLM] All models dead! Force resurrecting {last_resort}")
        return [last_resort]
        
    return alive


def is_temporary_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return any(marker in message for marker in TEMPORARY_ERROR_MARKERS)


def _build_config(system_instruction: str, temperature: float, tools=None, thinking_level: Optional[str] = None):
    kwargs = {"temperature": temperature}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if tools is not None:
        kwargs["tools"] = tools
    
    # 支援 Gemini 3 系列的 thinking_level
    if thinking_level:
        kwargs["thinking_config"] = types.ThinkingConfig(
            include_thoughts=False, # 預設不回傳思考過程以節省 token
            thinking_level=thinking_level
        )
        
    return types.GenerateContentConfig(**kwargs)


def quick_call(
    prompt: str,
    models: Optional[List[str]] = None,
    system_instruction: str = "",
    temperature: float = 0.1,
    thinking_level: Optional[str] = None,
) -> Optional[str]:
    if not _client:
        return None

    for model_name in get_alive_models(models):
        try:
            response = _client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=_build_config(system_instruction, temperature, thinking_level=thinking_level),
            )
            if response.text:
                return response.text
        except Exception as exc:
            if is_temporary_error(exc):
                logger.warning(f"[LLM] {model_name} temp failure: {exc}")
                mark_dead(model_name, exc)
                continue
            logger.error(f"[LLM] {model_name} fatal: {exc}")
            return None

    return None


def supports_tools(model_name: str) -> bool:
    return any(m in model_name for m in TOOL_SUPPORTED_MODELS)


def chat_with_tools(
    user_text: str,
    tools: list,
    system_instruction: str = "",
    history=None,
    temperature: float = 0.3,
    timeout_seconds: int = 30,
    max_timeouts: int = 2,
    models: Optional[List[str]] = None,
    thinking_level: Optional[str] = "medium",
    unavailable_message: str = "🚀 所有推進器皆暫時熄火，請稍後再試。",
    timeout_message: Optional[str] = None,
) -> str:
    if not _client:
        return "⚠️ 未設定 GEMINI_API_KEY。"

    # 取得候選模型清單
    candidates = get_alive_models(models)
    
    # 邏輯 A: 優先嘗試支援工具的模型
    tool_enabled_candidates = [m for m in candidates if supports_tools(m)]
    
    # 邏輯 B: 如果需要工具但沒有可用模型，則退而求其次使用不帶工具的模型 (Gemma)
    final_candidates = tool_enabled_candidates if tool_enabled_candidates else candidates
    use_tools = tools if tool_enabled_candidates else None
    
    if not final_candidates:
        return unavailable_message

    if not tool_enabled_candidates and tools:
        logger.warning("[LLM] No tool-supported models alive. Dropping tools and falling back to text-only mode.")

    timeout_count = 0
    for model_name in final_candidates:
        if timeout_count >= max_timeouts:
            break
        try:
            # 判斷該模型是否支援 thinking
            actual_thinking = thinking_level if "gemini-3" in model_name else None
            
            chat = _client.chats.create(
                model=model_name,
                config=_build_config(system_instruction, temperature, use_tools, thinking_level=actual_thinking),
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
                mark_dead(model_name, Exception("DEADLINE_EXCEEDED"))
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
                mark_dead(model_name, exc)
                if "DEADLINE_EXCEEDED" in str(exc).upper():
                    timeout_count += 1
                continue
            logger.error(f"[LLM] {model_name} fatal: {exc}")
            return f"⚠️ 模型異常: {str(exc)[:100]}"

    return unavailable_message

