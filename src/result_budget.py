from __future__ import annotations

from typing import Dict, List

SINGLE_RESULT_CAP = 8_000
PER_TURN_BUDGET = 20_000


def cap_to_length(content: str, tool_name: str, max_len: int) -> str:
    """Cap a string to at most max_len characters, preserving head and tail and
    inserting a clear truncation marker. Returns a new string (never mutates).
    """
    content = str(content)
    if len(content) <= max_len:
        return content
    # keep head ~50% and tail ~25% of the allowed length
    head_len = max_len // 2
    tail_len = max_len // 4
    head = content[:head_len]
    tail = content[-tail_len:]
    return (
        f"{head}\n\n"
        f"... [結果過長，已截斷。tool={tool_name} 原始 {len(content)} 字，保留頭尾] ...\n\n"
        f"{tail}"
    )


def cap_single_result(content: str, tool_name: str) -> str:
    return cap_to_length(content, tool_name, SINGLE_RESULT_CAP)


def enforce_turn_budget(results: List[Dict]) -> List[Dict]:
    """Return a new list of shallow-copied result dicts that fits within
    PER_TURN_BUDGET. Does not mutate the input list or its dicts.

    Strategy:
    1. Shallow-copy the per-result dicts.
    2. First pass: apply cap_single_result to the largest items (this is
       a no-op for items already <= SINGLE_RESULT_CAP).
    3. If still over budget, second pass: aggressively trim the largest
       items further (using cap_to_length with a computed target length)
       until the total size is within PER_TURN_BUDGET.
    """
    # shallow-copy container and each per-result dict to avoid mutating caller data
    new_results = [dict(item) for item in results]

    total = sum(len(str(item.get("content", ""))) for item in new_results)
    if total <= PER_TURN_BUDGET:
        return new_results

    # First pass: try the per-item cap
    indexed = sorted(
        enumerate(new_results),
        key=lambda pair: len(str(pair[1].get("content", ""))),
        reverse=True,
    )
    for index, result in indexed:
        if total <= PER_TURN_BUDGET:
            break
        old_content = str(result.get("content", ""))
        new_content = cap_single_result(old_content, str(result.get("name", "unknown")))
        # assign into the shallow-copied dict
        new_results[index]["content"] = new_content
        total -= len(old_content) - len(new_content)

    # Second pass: if still over budget, forcibly trim largest items further
    if total > PER_TURN_BUDGET:
        needed = total - PER_TURN_BUDGET
        # re-sort because sizes changed
        indexed = sorted(
            enumerate(new_results),
            key=lambda pair: len(str(pair[1].get("content", ""))),
            reverse=True,
        )
        for index, result in indexed:
            if needed <= 0:
                break
            old_content = str(result.get("content", ""))
            old_len = len(old_content)
            # compute a conservative minimum to leave for readability
            min_keep = 100
            # maximum we can cut from this item
            max_cut = max(0, old_len - min_keep)
            if max_cut <= 0:
                continue
            cut = min(max_cut, needed)
            target_len = max(min_keep, old_len - cut)
            new_content = cap_to_length(old_content, str(result.get("name", "unknown")), target_len)
            new_results[index]["content"] = new_content
            actual_cut = old_len - len(new_content)
            needed -= actual_cut
            total -= actual_cut

    return new_results


def cap_history_text(content: str) -> str:
    return cap_single_result(content, "history")
