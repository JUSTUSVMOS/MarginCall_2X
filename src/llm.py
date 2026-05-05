import concurrent.futures
import json
import inspect
import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Set, get_args, get_origin

import httpx
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.result_budget import cap_history_text, cap_single_result, enforce_turn_budget
from src.tool_loop_guard import ToolLoopGuard
from src.tools import _REGISTRY

load_dotenv()

logger = logging.getLogger(__name__)

MICROCOMPACT_KEEP_FULL_TOOL_RESULTS = 3
RECENT_HISTORY_WINDOW = 12
MAX_HISTORY_MESSAGES = 40
SOFT_HISTORY_CHAR_LIMIT = 12000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 45

COMPACTION_PROMPT = """你是一個對話壓縮器。把以下對話歷史壓縮成結構化摘要，保留所有重要數據。

格式要求：
## 原始問題
（用戶最初問什麼）

## 關鍵發現
（已經確認的事實、數字、結論）

## 已使用的數據源
（呼叫了哪些 tools，拿到什麼）

## 待解決問題
（還沒回答的部分）

## 下一步
（接下來應該做什麼）

---
對話歷史：
{history_text}
"""


# 定義哪些模型支援工具呼叫 (Function Calling)
TOOL_SUPPORTED_MODELS = {
    "gemini-3.1-flash-lite-preview",
    "minimax/minimax-m2.5:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/elephant-alpha",
    "google/gemma-4-31b-it:free",
}

# 重活：主對話 (需要 tool calling + 深度推理)
HEAVY_MODELS = [
    "gemma-4-31b-it",
    "gemini-3.1-flash-lite-preview",
    # "minimax/minimax-m2.5:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/elephant-alpha",
    "google/gemma-4-31b-it:free",
    "deepseek/deepseek-v3.2",
]

# 最終戰報彙整 (不需 tools，純文字推理)
REPORT_MODELS = [
    # "gemini-3.1-flash-lite-preview",
    # "openai/gpt-oss-120b:free",
    # "minimax/minimax-m2.5:free",
    "gemma-4-31b-it",
    "google/gemma-4-31b-it:free",
    "gemma-4-26b-it",
    "google/gemma-4-26b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/elephant-alpha",
]

# 輕活：symbol detection、情緒打分、asset 分類
LIGHT_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "openai/gpt-oss-120b:free",
    "gemma-4-31b-it",
    "gemma-4-26b-it",
    "google/gemma-4-31b-it:free",
]

AVAILABLE_MODELS = HEAVY_MODELS + ["gemma-4-31b-it", "gemma-3-27b-it"]

TEMPORARY_ERROR_MARKERS = (
    "429",
    "RESOURCE_EXHAUSTED",
    "503",
    "UNAVAILABLE",
    "INTERNAL",
    "DEADLINE_EXCEEDED",
    "TIMEOUT",
    "TIMED OUT",
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
    elif "DEADLINE_EXCEEDED" in msg or "TIMEOUT" in msg or "TIMED OUT" in msg:
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
            logger.info(f"🤖 [LLM_SELECT] Attempting quick_call with: {model_name}")
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
    if is_timeout_error(exc):
        return True
    message = str(exc).upper()
    return any(marker in message for marker in TEMPORARY_ERROR_MARKERS)


def is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return True

    message = str(exc).upper()
    return (
        "TIMEOUT" in message
        or "TIMED OUT" in message
        or "DEADLINE_EXCEEDED" in message
    )


def _annotation_to_openai_schema(annotation) -> Dict:
    if annotation is inspect.Signature.empty:
        return {"type": "string"}

    origin = get_origin(annotation)
    args = [arg for arg in get_args(annotation) if arg is not type(None)]

    if args and origin is not None:
        if origin in (list, List):
            item_annotation = args[0] if args else str
            return {"type": "array", "items": _annotation_to_openai_schema(item_annotation)}
        if origin in (dict, Dict):
            return {"type": "object"}
        if len(args) == 1:
            return _annotation_to_openai_schema(args[0])

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    return {"type": "string"}


def _normalize_openrouter_content(content) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content and isinstance(content["content"], str):
            return content["content"]
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        normalized_parts = []
        for item in content:
            normalized = _normalize_openrouter_content(item)
            if normalized:
                normalized_parts.append(normalized)
        merged = "\n".join(normalized_parts).strip()
        return merged or None
    return str(content)


def _build_http_options(timeout_seconds: Optional[int]) -> Optional[types.HttpOptions]:
    if timeout_seconds is None:
        return None

    try:
        timeout_value = int(timeout_seconds)
    except (TypeError, ValueError):
        return None

    if timeout_value <= 0:
        return None

    return types.HttpOptions(timeout=timeout_value)


def _convert_to_openai_tools(tools: list) -> Optional[List[Dict]]:
    """將 Python 函式列表轉換為 OpenAI/OpenRouter 相容的 tools 格式"""
    if not tools:
        return None
    
    openai_tools = []
    for tool in tools:
        try:
            # 嘗試抓取 docstring 作為描述
            desc = tool.__doc__ or f"Execute {tool.__name__} function"
            signature = inspect.signature(tool)
            properties = {}
            required = []

            for name, param in signature.parameters.items():
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                param_schema = _annotation_to_openai_schema(param.annotation)
                properties[name] = param_schema
                if param.default is inspect.Signature.empty:
                    required.append(name)

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": desc.split('\n')[0].strip(), # 只取第一行
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
            })
        except Exception as e:
            logger.warning(f"Failed to convert tool {tool}: {e}")
            
    return openai_tools if openai_tools else None


def _call_openrouter(
    model_name: str,
    messages: List[Dict],
    temperature: float = 0.3,
    tools=None,
    timeout_seconds: int = 60,
) -> Optional[Dict]:
    """
    呼叫 OpenRouter API 並回傳完整的訊息物件。
    """
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
        
        # 注入工具定義 (OpenAI 格式)
        if tools:
            openai_tools = _convert_to_openai_tools(tools)
            if openai_tools:
                data["tools"] = openai_tools
                data["tool_choice"] = "auto"

        with httpx.Client(timeout=float(timeout_seconds)) as client:
            resp = client.post(url, headers=headers, json=data)
            if resp.status_code == 200:
                result = resp.json()
                return _extract_openrouter_message(result)
            else:
                logger.warning(f"[LLM] OpenRouter {model_name} failed: {resp.status_code} {resp.text}")
                if resp.status_code in (429, 503, 502, 504):
                    mark_dead(model_name, Exception(f"HTTP {resp.status_code}"))
                return None
    except Exception as e:
        logger.error(f"[LLM] OpenRouter {model_name} error: {e}")
        mark_dead(model_name, e)
        return None


def _extract_openrouter_message(result: Dict) -> Optional[Dict]:
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        logger.warning("[LLM] OpenRouter response missing choices payload")
        return None

    choice = choices[0]
    if not isinstance(choice, dict):
        logger.warning("[LLM] OpenRouter response returned non-dict choice payload")
        return None

    message = choice.get("message")
    if isinstance(message, dict):
        return message

    fallback_content = choice.get("content")
    if fallback_content is None and choice.get("text") is not None:
        fallback_content = choice.get("text")

    if fallback_content is None:
        logger.warning("[LLM] OpenRouter response missing message/content payload")
        return None

    fallback_message = {"content": fallback_content}
    for key in ("role", "tool_calls", "refusal"):
        if key in choice:
            fallback_message[key] = choice[key]
    return fallback_message


def _get_tool_mode(tool_name: str) -> str:
    entry = _REGISTRY.get(tool_name)
    if not entry:
        return "write"
    return str(entry.get("mode", "write"))


def _execute_single_tool_call(tc: Dict, tool_map: Dict[str, Callable]) -> Dict:
    call_id = tc.get("id")
    func_name = tc.get("function", {}).get("name")
    raw_args = tc.get("function", {}).get("arguments", "{}")
    logger.info(f"🛠️ [OpenRouter_Tool] Executing {func_name}...")
    try:
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            args = json.loads(raw_args)
        else:
            raise ValueError(f"Tool arguments for {func_name} must be a JSON object.")
        if not isinstance(args, dict):
            raise ValueError(f"Tool arguments for {func_name} must be a JSON object.")
        func = tool_map.get(func_name)
        if func is None:
            logger.warning(f"⚠️ [OpenRouter_Tool] Tool {func_name} not found in tool_map")
            return {"role": "tool", "tool_call_id": call_id, "name": func_name, "content": f"Error: Tool {func_name} not found."}

        signature = inspect.signature(func)
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
        filtered_args = args
        if not accepts_var_kwargs:
            allowed_names = set(signature.parameters.keys())
            filtered_args = {key: value for key, value in args.items() if key in allowed_names}
            dropped_args = sorted(set(args.keys()) - allowed_names)
            if dropped_args:
                logger.warning(
                    f"⚠️ [OpenRouter_Tool] Dropping unsupported args for {func_name}: {', '.join(dropped_args)}"
                )
        result = func(**filtered_args)
        logger.info(f"✅ [OpenRouter_Tool] {func_name} completed")
        return {"role": "tool", "tool_call_id": call_id, "name": func_name, "content": str(result)}
    except Exception as exc:
        logger.error(f"❌ [OpenRouter_Tool] {func_name} failed: {exc}")
        return {"role": "tool", "tool_call_id": call_id, "name": func_name, "content": f"Error: {exc}"}


def _execute_openai_tool_calls(tool_calls: List[Dict], tools: List[Callable], max_concurrent: int = 4) -> List[Dict]:
    """
    執行 OpenRouter 回傳的工具調用指令，並格式化為 OpenAI 的 tool 回傳格式。
    """
    tool_map = {tool.__name__: tool for tool in tools}
    ordered_results: List[Dict] = []
    pending_reads: List[Dict] = []

    def flush_reads() -> None:
        nonlocal ordered_results, pending_reads
        if not pending_reads:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
            futures = [pool.submit(_execute_single_tool_call, tc, tool_map) for tc in pending_reads]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        order = {tc.get("id"): index for index, tc in enumerate(pending_reads)}
        results.sort(key=lambda item: order.get(item.get("tool_call_id"), 999))
        ordered_results.extend(results)
        pending_reads = []

    for tc in tool_calls:
        tool_name = tc.get("function", {}).get("name", "")
        if _get_tool_mode(tool_name) == "read":
            pending_reads.append(tc)
            continue
        flush_reads()
        ordered_results.append(_execute_single_tool_call(tc, tool_map))

    flush_reads()
    final_order = {tc.get("id"): index for index, tc in enumerate(tool_calls)}
    ordered_results.sort(key=lambda item: final_order.get(item.get("tool_call_id"), 999))
    return ordered_results


def _build_config(
    system_instruction: str,
    temperature: float,
    tools=None,
    thinking_level: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
):
    kwargs = {"temperature": temperature}
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if tools is not None:
        kwargs["tools"] = tools
    http_options = _build_http_options(timeout_seconds)
    if http_options is not None:
        kwargs["http_options"] = http_options
    
    # 支援 Gemini 3 系列的 thinking_level
    if thinking_level:
        kwargs["thinking_config"] = types.ThinkingConfig(
            include_thoughts=True, # 改為 True，避免模型只思考不講話導致大腦空白
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
            return f"[function_call] {_serialize_history_payload(part['function_call'])}"
        if "function_response" in part:
            return f"[function_response] {_serialize_history_payload(part['function_response'])}"
        return str(part)

    text = getattr(part, "text", None)
    if text:
        return str(text)
    function_call = getattr(part, "function_call", None)
    if function_call:
        return f"[function_call] {_serialize_history_payload(function_call)}"
    function_response = getattr(part, "function_response", None)
    if function_response:
        return f"[function_response] {_serialize_history_payload(function_response)}"
    return str(part)


def _serialize_history_payload(payload) -> str:
    if isinstance(payload, str):
        return payload
    if hasattr(payload, "to_json_dict"):
        payload = payload.to_json_dict()
    elif hasattr(payload, "model_dump"):
        payload = payload.model_dump(exclude_none=True)
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(payload)


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

    normalized_parts = []
    for part in parts:
        text = _normalize_part(part)
        if text.startswith("[function_response]") or text.startswith("[function_call]"):
            text = cap_history_text(text)
        normalized_parts.append(types.Part(text=text))
    return types.Content(role=role, parts=normalized_parts)


def _history_item_to_openrouter_message(item) -> Dict[str, str]:
    normalized = _normalize_history_item(item)
    role = "assistant" if normalized.role == "model" else normalized.role
    content = "\n".join(part.text for part in normalized.parts if part.text).strip()
    return {"role": role, "content": content}


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


def _naive_full_compact_history(history: List[types.Content]) -> List[types.Content]:
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


def _full_compact_history(history: List[types.Content]) -> List[types.Content]:
    if len(history) <= RECENT_HISTORY_WINDOW:
        return history[-MAX_HISTORY_MESSAGES:]

    head = history[:-RECENT_HISTORY_WINDOW]
    tail = history[-RECENT_HISTORY_WINDOW:]
    history_text = "\n".join(
        f"[{item.role}] {' '.join(p.text for p in item.parts if p.text)[:300]}"
        for item in head[-20:]
    )
    summary_text = quick_call(
        COMPACTION_PROMPT.format(history_text=history_text),
        models=LIGHT_MODELS,
        temperature=0.1,
        timeout_seconds=15,
    )
    if not summary_text:
        logger.info("[LLM] quick_call returned empty summary during compaction; falling back to naive compaction")
        return _naive_full_compact_history(history)

    summary = types.Content(
        role="user",
        parts=[types.Part(text="[history summary]\n" + summary_text.strip())],
    )
    compacted = [summary] + tail
    return compacted[-MAX_HISTORY_MESSAGES:]


def compact_history(history) -> List[types.Content]:
    normalized = [_normalize_history_item(item) for item in history]
    compacted = _microcompact_history(normalized)

    if len(compacted) > RECENT_HISTORY_WINDOW:
        compacted = _full_compact_history(compacted)
    elif _estimate_history_chars(compacted) > SOFT_HISTORY_CHAR_LIMIT or len(compacted) > MAX_HISTORY_MESSAGES:
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
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> Optional[str]:
    # 取得候選模型清單
    candidates = get_alive_models(models)
    
    if not candidates:
        candidates = get_alive_models(LIGHT_MODELS)
    
    if not candidates:
        return None

    for model_name in candidates:
        logger.info(f"🤖 [LLM_SELECT] Attempting quick_call with: {model_name}")
        # OpenRouter 模型判斷
        if "/" in model_name:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
             
            res = _call_openrouter(model_name, messages, temperature, timeout_seconds=timeout_seconds)
            if res:
                return _normalize_openrouter_content(res.get("content"))
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
                    config=_build_config(
                        system_instruction,
                        temperature,
                        thinking_level=actual_thinking,
                        timeout_seconds=timeout_seconds,
                    ),
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
    
    # 邏輯 B: 只有真的要用 tools 時，才限制為支援工具的模型。
    # 否則保留完整候選集，避免 timeout/fallback 時被不必要地卡死在少數模型。
    if tools:
        final_candidates = tool_enabled_candidates or candidates
        use_tools = tools if tool_enabled_candidates else None
    else:
        final_candidates = candidates
        use_tools = None
    
    if not final_candidates:
        return unavailable_message

    timeout_count = 0
    for model_name in final_candidates:
        logger.info(f"🤖 [LLM_SELECT] Attempting chat_with_tools with: {model_name}")
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
                    message = _history_item_to_openrouter_message(h)
                    if message["content"]:
                        messages.append(message)
             
            messages.append({"role": "user", "content": user_text})
             
            # 傳遞工具定義
            res_message = _call_openrouter(
                model_name,
                messages,
                temperature,
                tools=use_tools,
                timeout_seconds=timeout_seconds,
            )
            logger.debug("OpenRouter initial response: %s", res_message)
             
            if res_message:
                tool_calls = res_message.get("tool_calls")
                content = _normalize_openrouter_content(res_message.get("content"))
                
                # 建立一個迴圈，讓模型可以連續呼叫工具 (最多 15 次)
                loop_guard = ToolLoopGuard(max_calls_per_tool=3, similarity_threshold=0.7)
                loop_count = 0
                while tool_calls and use_tools and loop_count < 15:
                    loop_count += 1
                    messages.append(res_message)
                    per_call_warnings = []
                    warned_tools = set()

                    for tc in tool_calls:
                        func_name = tc.get("function", {}).get("name", "")
                        raw_args = tc.get("function", {}).get("arguments", "{}")
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:
                            parsed_args = {}
                        warning = loop_guard.check_should_warn(func_name, parsed_args if isinstance(parsed_args, dict) else {})
                        if warning:
                            per_call_warnings.append({"role": "user", "content": warning})
                            warned_tools.add(func_name)

                    tool_results = _execute_openai_tool_calls(tool_calls, use_tools)
                    if len(tool_results) != len(tool_calls):
                        logger.error(
                            f"[LLM] Tool execution mismatch: {len(tool_calls)} calls vs {len(tool_results)} results"
                        )
                    for tc, result in zip(tool_calls, tool_results):
                        raw_args = tc.get("function", {}).get("arguments", "{}")
                        try:
                            parsed_args = json.loads(raw_args)
                        except Exception:
                            parsed_args = {}
                        loop_guard.record_call(
                            tc.get("function", {}).get("name", ""),
                            parsed_args if isinstance(parsed_args, dict) else {},
                            str(result.get("content", ""))[:200],
                        )
                    tool_results = [
                        {**item, "content": cap_single_result(str(item.get("content", "")), str(item.get("name", "unknown")))}
                        for item in tool_results
                    ]
                    tool_results = enforce_turn_budget(tool_results)
                    messages.extend(tool_results)
                    messages.extend(per_call_warnings)

                    usage_warning = loop_guard.format_warning_for_prompt(suppressed_tools=warned_tools)
                    if usage_warning:
                        messages.append({"role": "user", "content": usage_warning})

                    res_message = _call_openrouter(
                        model_name,
                        messages,
                        temperature,
                        tools=use_tools,
                        timeout_seconds=timeout_seconds,
                    )
                    logger.debug("OpenRouter step %s response: %s", loop_count, res_message)
                     
                    if not res_message:
                        break
                    
                    tool_calls = res_message.get("tool_calls")
                    content = _normalize_openrouter_content(res_message.get("content"))
                
                if res_message is None:
                    logger.warning(f"[LLM] {model_name} failed to return final summary after tool execution.")
                    continue
                    
                # 如果依然沒有 content，看看有沒有 reasoning 可以頂替
                if not content and res_message.get("reasoning"):
                    content = res_message.get("reasoning")
                
                if not content:
                    logger.warning(f"[LLM] {model_name} returned empty content.")
                    continue
                
                # 寫回歷史紀錄 (維持原本的 Gemini Content 格式)
                if history is not None and content:
                    updated_history = list(history) + [
                        types.Content(role="user", parts=[types.Part(text=user_text)]),
                        types.Content(role="model", parts=[types.Part(text=content)]),
                    ]
                    history.clear()
                    history.extend(compact_history(updated_history))
                return content
            continue

        # --- Gemini 原生路徑 ---
        if not _client:
            continue

        try:
            actual_thinking = thinking_level if "gemini-3" in model_name else None
            prepared_history = compact_history(history) if history else history
            chat = _client.chats.create(
                model=model_name,
                config=_build_config(
                    system_instruction,
                    temperature,
                    use_tools,
                    thinking_level=actual_thinking,
                    timeout_seconds=timeout_seconds,
                ),
                history=prepared_history,
            )

            retries_503 = 0
            while retries_503 <= 2:
                try:
                    part = types.Part(text=user_text)
                    response = chat.send_message(part)
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
                    elif is_timeout_error(exc):
                        logger.warning(f"[LLM] {model_name} timeout ({timeout_seconds}s): {exc}")
                        mark_dead(model_name, exc)
                        timeout_count += 1
                        break
                    elif is_temporary_error(exc):
                        logger.warning(f"[LLM] {model_name} temp failure: {exc}")
                        mark_dead(model_name, exc)
                        break
                    else:
                        logger.error(f"[LLM] {model_name} fatal: {exc}")
                        return f"⚠️ 模型異常: {str(exc)[:100]}"
        except Exception as outer_exc:
            if is_timeout_error(outer_exc):
                logger.warning(f"[LLM] {model_name} session init timeout ({timeout_seconds}s): {outer_exc}")
                mark_dead(model_name, outer_exc)
                timeout_count += 1
                continue
            elif is_temporary_error(outer_exc):
                logger.warning(f"[LLM] {model_name} session init failed: {outer_exc}")
                mark_dead(model_name, outer_exc)
                continue
            else:
                logger.error(f"[LLM] {model_name} session fatal: {outer_exc}")
                return f"⚠️ 模型會話異常: {str(outer_exc)[:100]}"

    if timeout_count >= max_timeouts and timeout_message:
        return timeout_message

    return unavailable_message
