# LLM Loop Hardening Design

## Goal

Harden `src/llm.py` so provider tool usage becomes more resilient under long research turns:

1. detect and surface repeated tool-call loops before they spiral
2. parallelize safe read-only tool batches on the OpenRouter path
3. cap oversized tool results before they flood context
4. replace destructive history truncation with structured LLM compaction
5. improve both OpenRouter and Gemini stability without rewriting the whole provider stack

## Current Problems

1. The OpenRouter path in `chat_with_tools()` runs tool calls serially inside a Python `while tool_calls ...` loop.
2. There is no per-query loop guard for repeated or near-duplicate tool calls, so the model can keep retrying similar calls with little feedback.
3. Large tool outputs are appended back into context with no explicit single-result or per-turn budget control on the OpenRouter path.
4. `_full_compact_history()` currently compresses old turns into `- role: first 160 chars`, which destroys important facts, numbers, and unresolved questions.
5. Gemini and OpenRouter do not share the same execution model:
   - OpenRouter uses an explicit Python-side tool loop.
   - Gemini currently uses SDK-native tool handling via `chat.send_message(...)`.

## Goals

1. Add soft loop detection for repeated tool calls within a single query.
2. Parallelize `@tool(mode="read")` calls on the OpenRouter path while preserving ordered tool results.
3. Add shared result-budget helpers that can cap oversized payloads before context bloats.
4. Replace naive full-history compaction with LLM-generated structured summaries.
5. Apply the hardening consistently across both providers where the architecture allows it.

## Non-Goals

1. Do not rewrite Gemini into a manual function-calling runtime in this phase.
2. Do not redesign agent prompts, Brain memory contracts, or portfolio logic.
3. Do not change tool semantics or reclassify read/write ownership outside the existing registry.
4. Do not introduce hard blocks that prevent the model from retrying a tool when retry is genuinely needed.

## Current Context

Relevant repo findings before this spec:

- `src/llm.py` already owns the OpenRouter tool loop, history compaction, and provider fallback behavior.
- `src/tools.py` already provides a shared tool registry with `@tool(mode="read"|"write")` and `get_tools(...)`.
- `src/bot.py` already builds `READ_ONLY_TOOLS` from the registry instead of maintaining a separate hard-coded catalog.
- Existing focused checks already cover OpenRouter normalization, history preservation, timeout fallback, and compaction basics:
  - `test_telegram_stall_regressions.py`
  - `test_agent_runtime.py`
  - `check_llm_timeout_recovery.py`

## Approaches Considered

### 1. Full provider unification

Rewrite Gemini so it also uses a manual Python-side function-calling loop, matching OpenRouter.

Why not recommended:

- It would replace a currently working SDK-native path with a larger orchestration rewrite.
- It expands scope far beyond the reliability issues this change is meant to solve.
- It increases rollout risk for provider fallback and history handling.

### 2. Hybrid targeted hardening (**Recommended**)

Keep provider-specific execution where it already works, but harden shared context behavior and the explicit OpenRouter runtime:

- OpenRouter gets loop detection, read-only parallelism, and result budgets.
- Gemini keeps SDK-native tool handling.
- Both providers get the improved history compaction path and large-context safety helpers.

Why this is recommended:

- It attacks the real weak spots without destabilizing Gemini's native runtime.
- It reuses the repo's existing tool-mode registry instead of introducing another classification source.
- It keeps the largest code changes isolated to `src/llm.py` plus two small helpers.

### 3. Minimal compaction-only patch

Replace `_full_compact_history()` and stop there.

Why not recommended:

- It improves context quality but leaves the repeated-tool and oversized-result issues mostly intact.
- It would only solve the most visible symptom, not the runtime pressure that creates it.

## Recommended Design

### 1. Add a query-scoped tool loop guard

Create `src/tool_loop_guard.py` as a small helper for tracking tool usage inside one `chat_with_tools()` call.

Responsibilities:

1. store tool call history by tool name
2. record compact argument snapshots and short result previews
3. detect suggested-limit overuse
4. detect near-duplicate arguments using token-overlap similarity
5. format warning text that can be injected back into the next model turn

Why this file name:

- `scratchpad.py` would be easy to confuse with Brain memory and other persistent state concepts in this repo.
- `tool_loop_guard.py` makes the scope explicit: this helper is ephemeral and runtime-only.

Behavior rules:

1. warnings are advisory, not blocking
2. similarity uses normalized stringified arguments with a threshold around `0.7`
3. result previews stay short and human-readable
4. guard state lives only for the current query

### 2. Rework OpenRouter tool execution around the registry

Refactor `_execute_openai_tool_calls(...)` in `src/llm.py` into a small orchestration layer:

1. add `_execute_single_tool_call(...)`
2. classify each tool from the existing `src.tools` registry mode
3. batch consecutive `read` tools for concurrent execution
4. keep `write` tools serial
5. restore final tool-result order to match the original `tool_calls` sequence

Execution rules:

1. concurrency only applies to `@tool(mode="read")`
2. unknown or unclassified tools default to serial execution
3. each tool failure becomes its own `"Error: ..."` result, matching current behavior
4. unsupported arguments are still filtered against the Python signature before invocation

This is safer than the proposed hard-coded read-only name list because the repo already has a canonical source of truth for tool mutability.

### 3. Add shared result-budget helpers

Create `src/result_budget.py` with small, focused helpers:

1. `cap_single_result(...)`
2. `enforce_turn_budget(...)`
3. helper(s) for capping large history text before compaction

Design rules:

1. OpenRouter tool results are capped immediately after execution.
2. The per-turn budget is enforced after all tool results are collected but before the next model call.
3. The cap keeps the most informative parts of the payload, favoring head + tail preservation over blind prefix truncation.
4. Gemini uses the same utility family for oversized history/function-response text before compaction logic runs.

This phase does not need a disk-persistence subsystem for large tool results. The existing problem is uncontrolled context growth, not long-term artifact retention.

### 4. Upgrade full-history compaction to structured LLM summarization

Rewrite `_full_compact_history()` so older history is summarized by a fast model through `quick_call(...)`.

The new compaction prompt must preserve:

1. original user question
2. confirmed facts and numerical data
3. tools/data sources already used
4. unresolved gaps
5. recommended next step

Compaction rules:

1. summarize only the older head of the history, keeping the recent tail verbatim
2. wrap the summary as a synthetic history item so downstream logic keeps working
3. if compaction fails or returns empty content, fall back to the current naive summary path
4. do not bind tools for the compaction call

This change applies to both providers because both eventually rely on `compact_history(...)`.

### 5. Provider-specific behavior boundaries

#### OpenRouter

The OpenRouter `while tool_calls ...` loop becomes:

1. append the assistant tool-call message
2. consult `ToolLoopGuard` and inject soft warnings for the next step
3. execute tools with read/write-aware batching
4. record call metadata and short previews in the guard
5. cap individual results and enforce per-turn budget
6. append tool results and any usage-warning reminder
7. call OpenRouter again

#### Gemini

Gemini remains SDK-native:

1. keep `chat.send_message(...)`
2. do not replace it with a manual tool loop in this phase
3. strengthen the shared history path with:
   - smarter full-history compaction
   - capped oversized history content before compaction
   - unchanged timeout/fallback semantics

This means Gemini gains context hardening, but not the exact same pre-tool warning mechanism as OpenRouter.

### 6. Error handling and fallback rules

1. `ToolLoopGuard` never blocks a call outright.
2. Parallel OpenRouter execution is fail-soft: one failed read tool must not abort sibling results.
3. Compaction failure falls back to the current naive history summary rather than dropping history.
4. Existing timeout and model-fallback behavior in `chat_with_tools()` remains intact.
5. Result budgeting must preserve the most recent turns so the active conversation does not become detached from its latest state.

## Testing Strategy

Add or extend focused regression checks rather than introducing a new test harness.

### OpenRouter path

Use `test_telegram_stall_regressions.py` to cover:

1. repeated/similar tool-call warning behavior
2. preserved unsupported-arg filtering
3. ordered results after concurrent read execution
4. final content normalization after loop execution

### Shared compaction behavior

Use `test_agent_runtime.py` to cover:

1. LLM-based full compaction success path
2. fallback to naive compaction when `quick_call(...)` fails
3. capped oversized tool-like history entries remaining structurally valid

### Provider fallback safety

Use `check_llm_timeout_recovery.py` to ensure:

1. Gemini timeout fallback still works
2. OpenRouter fallback still normalizes history correctly
3. the new helpers do not reintroduce worker-thread regressions

### Syntax checks

Run:

1. `python -m py_compile src/llm.py src/agent.py`

## Expected Outcome

After this change:

1. OpenRouter is less likely to spin in repetitive tool loops without noticing.
2. Safe read-only tool batches no longer pay unnecessary serial latency.
3. Oversized tool payloads stop dominating context.
4. Both providers keep higher-value history when compaction is needed.
5. Gemini remains stable because its native SDK tool path is preserved instead of being rewritten.
