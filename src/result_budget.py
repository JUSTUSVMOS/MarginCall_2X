from __future__ import annotations

from typing import Dict, List

SINGLE_RESULT_CAP = 8_000
PER_TURN_BUDGET = 20_000


def cap_to_length(content: str, tool_name: str, max_len: int) -> str:
    """Cap a string to at most max_len characters, preserving head and tail and
    inserting a clear truncation marker. Returns a new string (never mutates).
    Guarantees the returned string length is <= max_len even for small max_len
    by reserving space for the marker before slicing head/tail. If the marker
    alone exceeds max_len, the marker is truncated to fit.
    """
    content = str(content)
    if len(content) <= max_len:
        return content

    marker = (
        f"\n\n... [結果過長，已截斷。tool={tool_name} 原始 {len(content)} 字，保留頭尾] ...\n\n"
    )
    marker_len = len(marker)

    # If the marker itself doesn't fit, return a truncated marker to respect max_len
    if marker_len >= max_len:
        return marker[:max_len]

    remaining = max_len - marker_len
    # allocate head roughly 2/3 and tail 1/3 of the remaining budget
    head_len = (remaining * 2) // 3
    tail_len = remaining - head_len

    head = content[:head_len] if head_len > 0 else ""
    tail = content[-tail_len:] if tail_len > 0 else ""

    return f"{head}{marker}{tail}"


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
            # actual_cut should never be negative; clamp to zero to avoid
            # accidentally increasing total when cap_to_length returns longer
            # strings for very small targets.
            actual_cut = max(0, old_len - len(new_content))
            needed -= actual_cut
            total -= actual_cut

    # Final fallback: if we're still over budget, allow a last-pass that can
    # reduce items below the conservative min_keep to a smaller final_min_keep
    # so the function always converges to within PER_TURN_BUDGET.
    if total > PER_TURN_BUDGET:
        remaining = total - PER_TURN_BUDGET
        final_min_keep = 20
        # re-sort by current sizes
        indexed = sorted(
            enumerate(new_results),
            key=lambda pair: len(str(pair[1].get("content", ""))),
            reverse=True,
        )
        for index, result in indexed:
            if remaining <= 0:
                break
            old_content = str(result.get("content", ""))
            old_len = len(old_content)
            max_cut2 = max(0, old_len - final_min_keep)
            if max_cut2 <= 0:
                continue
            cut = min(max_cut2, remaining)
            target_len = max(final_min_keep, old_len - cut)
            new_content = cap_to_length(old_content, str(result.get("name", "unknown")), target_len)
            new_results[index]["content"] = new_content
            actual_cut = max(0, old_len - len(new_content))
            remaining -= actual_cut
            total -= actual_cut

    return new_results


def cap_history_text(content: str) -> str:
    return cap_single_result(content, "history")
