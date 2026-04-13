import logging
import time
from functools import wraps
from typing import Callable, Dict, List


logger = logging.getLogger(__name__)

_REGISTRY: dict[str, dict] = {}


def _wrap_tool(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = f"args={args}, kwargs={kwargs}"
        logger.info(f"🚀 [TOOL_START] {func.__name__} | {arg_str[:100]}...")
        start_t = time.time()
        try:
            result = func(*args, **kwargs)
            logger.info(f"✅ [TOOL_DONE] {func.__name__} ({time.time() - start_t:.2f}s)")
            return result
        except Exception as exc:
            logger.error(f"❌ [TOOL_ERROR] {func.__name__}: {exc}")
            raise

    return wrapper


def tool(mode: str = "read"):
    def decorator(func: Callable) -> Callable:
        if mode not in {"read", "write"}:
            raise ValueError("tool mode must be 'read' or 'write'")
        wrapped = _wrap_tool(func)
        _REGISTRY[wrapped.__name__] = {"func": wrapped, "mode": mode}
        return wrapped

    return decorator


def get_tools(mode: str = "all") -> List[Callable]:
    if mode not in {"all", "read", "write"}:
        raise ValueError("tool mode must be 'all', 'read', or 'write'")
    if mode == "all":
        return [entry["func"] for entry in _REGISTRY.values()]
    return [entry["func"] for entry in _REGISTRY.values() if entry["mode"] == mode]
