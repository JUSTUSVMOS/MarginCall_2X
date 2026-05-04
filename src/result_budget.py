from __future__ import annotations

from typing import Dict, List

SINGLE_RESULT_CAP = 8_000
PER_TURN_BUDGET = 20_000


def cap_single_result(content: str, tool_name: str) -> str:
    content = str(content)
    if len(content) <= SINGLE_RESULT_CAP:
        return content
    head = content[: SINGLE_RESULT_CAP // 2]
    tail = content[-(SINGLE_RESULT_CAP // 4) :]
    return (
        f"{head}\n\n"
        f"... [結果過長，已截斷。tool={tool_name} 原始 {len(content)} 字，保留頭尾] ...\n\n"
        f"{tail}"
    )


def enforce_turn_budget(results: List[Dict]) -> List[Dict]:
    total = sum(len(str(item.get("content", ""))) for item in results)
    if total <= PER_TURN_BUDGET:
        return results

    indexed = sorted(
        enumerate(results),
        key=lambda pair: len(str(pair[1].get("content", ""))),
        reverse=True,
    )
    for index, result in indexed:
        if total <= PER_TURN_BUDGET:
            break
        old_content = str(result.get("content", ""))
        new_content = cap_single_result(old_content, str(result.get("name", "unknown")))
        results[index]["content"] = new_content
        total -= len(old_content) - len(new_content)
    return results


def cap_history_text(content: str) -> str:
    return cap_single_result(content, "history")
