import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Set

import httpx
from google import genai
from google.genai import types


logger = logging.getLogger(__name__)

MICROCOMPACT_KEEP_FULL_TOOL_RESULTS = 3
RECENT_HISTORY_WINDOW = 12
MAX_HISTORY_MESSAGES = 40
SOFT_HISTORY_CHAR_LIMIT = 12000


# 定義哪些模型支援工具呼叫 (Function Calling)
TOOL_SUPPORTED_MODELS = {
    "gemini-3.1-flash-lite-preview",
    "minimax/minimax-m2.5:free",
    "openai/gpt-oss-120b:free",
}

# 重活：主對話 (需要 tool calling + 深度推理)
HEAVY_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "minimax/minimax-m2.5:free",
    "openai/gpt-oss-120b:free",
]

# 輕活：symbol detection、情緒打分、asset 分類
LIGHT_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "openai/gpt-oss-120b:free",
]

AVAILABLE_MODELS = HEAVY_MODELS

AVAILABLE_MODELS = HEAVY_MODELS

TEMPORARY_ERROR_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "503",
    "UNAVAILABLE",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
    "TIMEOUT",
)

_api_key = os.getenv("GEMINI_API_KEY")
_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_client = genai.Client(api_key=_api_key) if _api_key else None
_dead_engines: Dict[str, float] = {}
_lock = threading.Lock()


def is_configured() -> bool:
    return _client is not None or _openrouter_key is not None


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
    elif "DEADLINE_EXCEEDED" in msg or "TIMEOUT" in msg:
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
    
    # 如果全部都死了，強行復活最後一個保底模型
    if not alive and candidates:
        last_resort = candidates[-1]
        logger.warning(f"[LLM] All models dead! Force resurrecting {last_resort}")
        return [last_resort]
        
    return alive


def is_temporary_error(exc: Exception) -> bool:
    message = str(exc).upper()
    return any(marker in message for marker in TEMPORARY_ERROR_MARKERS)


def _convert_to_openai_tools(tools: list) -> Optional[List[Dict]]:
    """將 Python 函式列表轉換為 OpenAI/OpenRouter 相容的 tools 格式"""
    if not tools:
        return None
    
    openai_tools = []
    for tool in tools:
        try:
            # 嘗試抓取 docstring 作為描述
            desc = tool.__doc__ or f"Execute {tool.__name__} function"
            
            # 這裡我們簡化處理，假設大部份工具目前不帶複雜參數或由 LLM 自行決定
            # 若要更精確，需解析 inspect.signature
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": desc.split('\n')[0].strip(), # 只取第一行
                    "parameters": {
                        "type": "object",
                        "properties": {}, # 簡化：由模型根據 docstring 推斷
                        "required": []
                    }
                }
            })
        except Exception as e:
            logger.warning(f"Failed to convert tool {tool}: {e}")
            
    return openai_tools if openai_tools else None


def _call_openrouter(model_name: str, messages: List[Dict], temperature: float = 0.3, tools=None) -> Optional[str]:
    """呼叫 OpenRouter API"""
    if not _openrouter_key:
        logger.error("[LLM] OPENROUTER_API_KEY not found.")
        return None
    
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {_openrouter_key}",
            "HTTP-Referer": "https://github.com/margincaller/MarginCall_2X",
            "X-Title": "MarginCall_2X Bot",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        
        # 注入工具定義
        if tools:
            openai_tools = _convert_to_openai_tools(tools)
            if openai_tools:
                data["tools"] = openai_tools
                data["tool_choice"] = "auto"

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, headers=headers, json=data)
            if resp.status_code == 200:
                result = resp.json()
                choice = result["choices"][0]
                message = choice.get("message", {})
                
                # 處理工具調用 (OpenRouter 可能回傳 tool_calls)
                if "tool_calls" in message:
                    # 目前系統主要由 Gemini 原生處理工具循環，
                    # 這裡我們先回傳一個提示標記，或嘗試簡化回傳。
                    # 為了相容性，這裡先回傳內容或首個調用。
                    return message.get("content") or f"[tool_calls detected: {message['tool_calls'][0]['function']['name']}]"
                
                return message.get("content")
            else:
                logger.warning(f"[LLM] OpenRouter {model_name} failed: {resp.status_code} {resp.text}")
                if resp.status_code in (429, 503, 502, 504):
                    mark_dead(model_name, Exception(f"HTTP {resp.status_code}"))
                return None
    except Exception as e:
        logger.error(f"[LLM] OpenRouter {model_name} error: {e}")
        mark_dead(model_name, e)
        return None


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


def _normalize_part(part) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if "text" in part:
            return str(part["text"])
        if "function_call" in part:
            return f"[function_call] {part['function_call']}"
        if "function_response" in part:
            return f"[function_response] {part['function_response']}"
        return str(part)

    text = getattr(part, "text", None)
    if text:
        return str(text)
    function_call = getattr(part, "function_call", None)
    if function_call:
        return f"[function_call] {function_call}"
    function_response = getattr(part, "function_response", None)
    if function_response:
        return f"[function_response] {function_response}"
    return str(part)


def _normalize_history_item(item) -> types.Content:
    if isinstance(item, dict):
        role = item.get("role", "user")
        parts = item.get("parts", [])
    elif isinstance(item, types.Content):
        role = item.role
        parts = item.parts
    else:
        role = getattr(item, "role", "user")
        parts = getattr(item, "parts", [])

    if not isinstance(parts, list):
        parts = [parts]

    normalized_parts = [types.Part(text=_normalize_part(part)) for part in parts]
    return types.Content(role=role, parts=normalized_parts)


def _is_tool_like_message(item: types.Content) -> bool:
    role = str(item.role).lower()
    if "tool" in role:
        return True
    return any(
        (p.text and (p.text.startswith("[function_response]") or p.text.startswith("[function_call]")))
        for p in item.parts
    )


def _estimate_history_chars(history: List[types.Content]) -> int:
    return sum(len(p.text) for item in history for p in item.parts if p.text)


def _microcompact_history(history: List[types.Content]) -> List[types.Content]:
    tool_indexes = [idx for idx, item in enumerate(history) if _is_tool_like_message(item)]
    keep_full = set(tool_indexes[-MICROCOMPACT_KEEP_FULL_TOOL_RESULTS:])
    compacted = []
    for idx, item in enumerate(history):
        if idx in keep_full or not _is_tool_like_message(item):
            compacted.append(item)
            continue
        compacted.append(types.Content(role=item.role, parts=[types.Part(text="[content truncated]")]))
    return compacted


def _full_compact_history(history: List[types.Content]) -> List[types.Content]:
    if len(history) <= RECENT_HISTORY_WINDOW:
        return history[-MAX_HISTORY_MESSAGES:]

    head = history[:-RECENT_HISTORY_WINDOW]
    tail = history[-RECENT_HISTORY_WINDOW:]
    summary_lines = []
    for item in head[-12:]:
        text_parts = [p.text for p in item.parts if p.text]
        text = " ".join(text_parts)[:160]
        summary_lines.append(f"- {item.role}: {text}")
    summary = types.Content(role="user", parts=[types.Part(text="[history summary]\n" + "\n".join(summary_lines))])
    return [summary] + tail


def compact_history(history) -> List[types.Content]:
    normalized = [_normalize_history_item(item) for item in history]
    compacted = _microcompact_history(normalized)

    if _estimate_history_chars(compacted) > SOFT_HISTORY_CHAR_LIMIT or len(compacted) > MAX_HISTORY_MESSAGES:
        compacted = _full_compact_history(compacted)

    if len(compacted) > MAX_HISTORY_MESSAGES:
        compacted = compacted[-MAX_HISTORY_MESSAGES:]

    return compacted


def quick_call(
    prompt: str,
    models: Optional[List[str]] = None,
    system_instruction: str = "",
    temperature: float = 0.1,
    thinking_level: Optional[str] = None,
    max_503_retries: int = 2,
) -> Optional[str]:
    # 取得候選模型清單
    candidates = get_alive_models(models)
    
    if not candidates:
        candidates = get_alive_models(LIGHT_MODELS)
    
    if not candidates:
        return None

    for model_name in candidates:
        # OpenRouter 模型判斷
        if "/" in model_name:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            res = _call_openrouter(model_name, messages, temperature)
            if res:
                return res
            continue

        # Gemini 原生模型路徑
        if not _client:
            continue
            
        actual_thinking = thinking_level if "gemini-3" in model_name else None
        for attempt in range(1 + max_503_retries):
            try:
                response = _client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=_build_config(system_instruction, temperature, thinking_level=actual_thinking),
                )
                if response.text:
                    return response.text
                break
            except Exception as exc:
                msg = str(exc).upper()
                if ("503" in msg or "UNAVAILABLE" in msg) and attempt < max_503_retries:
                    logger.info(f"[LLM] {model_name} 503, retry {attempt+1}/{max_503_retries} in 3s...")
                    time.sleep(3)
                    continue
                elif is_temporary_error(exc):
                    logger.warning(f"[LLM] {model_name} temp failure: {exc}")
                    mark_dead(model_name, exc)
                    break
                else:
                    logger.error(f"[LLM] {model_name} fatal: {exc}")
                    break 

    return None


def supports_tools(model_name: str) -> bool:
    return model_name in TOOL_SUPPORTED_MODELS


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
    # 取得候選模型清單
    candidates = get_alive_models(models)
    
    # 邏輯 A: 優先嘗試支援工具的模型
    tool_enabled_candidates = [m for m in candidates if supports_tools(m)]
    
    # 邏輯 B: 如果需要工具但沒有可用模型，則退而求其次使用不帶工具的模型
    final_candidates = tool_enabled_candidates if (tool_enabled_candidates or not tools) else candidates
    use_tools = tools if tool_enabled_candidates else None
    
    if not final_candidates:
        return unavailable_message

    timeout_count = 0
    for model_name in final_candidates:
        if timeout_count >= max_timeouts:
            break
            
        # --- OpenRouter 路徑 (目前透過 _call_openrouter 支援工具定義) ---
        if "/" in model_name:
            # 轉換歷史紀錄為 OpenRouter 格式
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            
            if history:
                for h in history:
                    # 重要：OpenRouter 不認識 'model' 角色，必須映射為 'assistant'
                    role = "assistant" if h.role == "model" else h.role
                    messages.append({"role": role, "content": _normalize_part(h.parts[0])})
            
            messages.append({"role": "user", "content": user_text})
            
            # 傳遞工具定義
            res = _call_openrouter(model_name, messages, temperature, tools=use_tools)
            if res:
                # 寫回歷史紀錄
                if history is not None:
                    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
                    history.append(types.Content(role="model", parts=[types.Part(text=res)]))
                return res
            continue

        # --- Gemini 原生路徑 ---
        if not _client:
            continue

        try:
            actual_thinking = thinking_level if "gemini-3" in model_name else None
            chat = _client.chats.create(
                model=model_name,
                config=_build_config(system_instruction, temperature, use_tools, thinking_level=actual_thinking),
                history=history,
            )

            retries_503 = 0
            while retries_503 <= 2:
                try:
                    response_container = []
                    exception_container = []

                    def _thread_task():
                        try:
                            part = types.Part(text=user_text)
                            response_container.append(chat.send_message(part))
                        except Exception as exc:
                            exception_container.append(exc)

                    llm_thread = threading.Thread(target=_thread_task)
                    llm_thread.start()
                    llm_thread.join(timeout=timeout_seconds)

                    if llm_thread.is_alive():
                        logger.warning(f"[LLM] {model_name} timeout ({timeout_seconds}s)")
                        mark_dead(model_name, Exception("TIMEOUT"))
                        timeout_count += 1
                        break

                    if exception_container:
                        raise exception_container[0]

                    if not response_container:
                        break

                    response = response_container[0]
                    if history is not None:
                        new_history = chat.get_history()
                        history.clear()
                        history.extend(compact_history(new_history))

                    return response.text if response.text else "大腦空白。"
                except Exception as exc:
                    msg = str(exc).upper()
                    if ("503" in msg or "UNAVAILABLE" in msg) and retries_503 < 2:
                        retries_503 += 1
                        logger.info(f"[LLM] {model_name} 503, retry {retries_503}/2 in 3s...")
                        time.sleep(3)
                        continue
                    elif is_temporary_error(exc):
                        logger.warning(f"[LLM] {model_name} temp failure: {exc}")
                        mark_dead(model_name, exc)
                        break
                    else:
                        logger.error(f"[LLM] {model_name} fatal: {exc}")
                        return f"⚠️ 模型異常: {str(exc)[:100]}"
        except Exception as outer_exc:
            if is_temporary_error(outer_exc):
                logger.warning(f"[LLM] {model_name} session init failed: {outer_exc}")
                mark_dead(model_name, outer_exc)
                continue
            else:
                logger.error(f"[LLM] {model_name} session fatal: {outer_exc}")
                return f"⚠️ 模型會話異常: {str(outer_exc)[:100]}"

    return unavailable_message

