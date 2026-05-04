from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ToolLoopGuard:
    max_calls_per_tool: int = 3
    similarity_threshold: float = 0.7
    tool_calls: Dict[str, List[dict]] = field(default_factory=dict)
    # Track how many times a near-duplicate warning was triggered per tool
    near_duplicate_counts: Dict[str, int] = field(default_factory=dict)

    def record_call(self, tool_name: str, args: dict, result_preview: str) -> None:
        # Keep a single place where preview length is enforced
        history = self.tool_calls.setdefault(tool_name, [])
        # If this recorded call is similar to any previous one, increment near-duplicate counter
        for previous in history:
            if self._args_similar(previous["args"], args or {}):
                self.near_duplicate_counts[tool_name] = self.near_duplicate_counts.get(tool_name, 0) + 1
                break
        history.append(
            {
                "args": args or {},
                "result_preview": (result_preview or "")[:200],
            }
        )

    def check_should_warn(self, tool_name: str, args: dict) -> Optional[str]:
        history = self.tool_calls.get(tool_name, [])
        if len(history) >= self.max_calls_per_tool:
            return (
                f"⚠️ Tool '{tool_name}' has already been called {len(history)} times in this query. "
                "Consider switching tools, changing the query, or answering with the data already gathered."
            )

        for previous in history:
            if self._args_similar(previous["args"], args or {}):
                # Use the stored preview as-is (no second truncation here)
                preview = previous["result_preview"]
                return (
                    f"⚠️ You are about to call '{tool_name}' again with very similar arguments. "
                    f"Previous result preview: {preview}"
                )
        return None

    def format_warning_for_prompt(self, suppressed_tools: Optional[Set[str]] = None) -> Optional[str]:
        suppressed_tools = suppressed_tools or set()
        warnings = []
        # Include tools that exceeded call limits
        for tool_name, calls in self.tool_calls.items():
            if tool_name in suppressed_tools:
                continue
            if len(calls) >= self.max_calls_per_tool:
                warnings.append(f"- {tool_name}: {len(calls)} calls")

        # Include tools that triggered near-duplicate warnings
        for tool_name, count in self.near_duplicate_counts.items():
            if tool_name in suppressed_tools:
                continue
            if count > 0:
                # Avoid duplicate entries
                if not any(w.startswith(f"- {tool_name}:") for w in warnings):
                    warnings.append(f"- {tool_name}: {count} near-duplicate calls detected")

        if not warnings:
            return None
        return "## Tool Usage Warning\n" + "\n".join(warnings)

    def _args_similar(self, left: dict, right: dict) -> bool:
        left_words = self._tokenize(left)
        right_words = self._tokenize(right)
        if not left_words or not right_words:
            return False
        overlap = len(left_words & right_words) / len(left_words | right_words)
        return overlap >= self.similarity_threshold

    def _tokenize(self, payload: dict) -> set[str]:
        return {
            token
            for token in str(payload).lower().replace("{", " ").replace("}", " ").split()
            if len(token) > 2
        }
