# Frontal Lobe Restructuring Design

## Goal

Turn Brain from a fragile labeled-note parser into a durable trading-memory system with clear ownership boundaries:

1. `frontalLobe` stores only the LLM's structured trading thesis.
2. `portfolioHealth` stores auto-computed portfolio state outside the thesis.
3. `get_cognitive_context()` presents those two surfaces separately from macro regime memory.

This spec covers **Phase B only**. It does not include the later risk-engine additive scoring work, alpha decomposition changes, or the stress-test / what-if features.

## Current Problems

1. `update_frontal_lobe(content: str)` pushes the model to hand-author a four-section string contract.
2. `normalize_frontal_lobe_note()` and the `_infer_*` helpers try to repair malformed input after the fact, which produces permanent artifacts such as duplicated labels and mixed-content sections.
3. `Portfolio Health` currently lives inside the frontal-lobe note even though it is system-derived state rather than human trading judgment.
4. `get_cognitive_context()` renders the broken free-text shape directly, so malformed frontal-lobe memory leaks back into the prompt.

## Goals

1. Replace the free-text frontal-lobe write contract with structured tool parameters.
2. Store `state.frontalLobe` as JSON/dict data instead of a labeled string.
3. Move portfolio-health state into its own non-commit-producing state surface.
4. Remove the inference-heavy frontal-lobe normalization stack.
5. Keep persisted legacy brain state readable through lazy migration.
6. Improve LLM context quality by rendering three clear sections: trading thesis, portfolio health, and market regime.

## Non-Goals

1. No attempt to preserve legacy parse artifacts exactly as written.
2. No commit-history rewrite or backfill of older commits.
3. No automatic trade execution or risk-engine redesign in this phase.
4. No dual-write release that keeps the labeled-string frontal-lobe format alive.

## Current Context

Relevant repo findings before this spec:

- `engine_memory.py` still uses `FRONTAL_LOBE_FIELDS`, `FRONTAL_LOBE_SECTION_ALIASES`, `FRONTAL_LOBE_KEYWORDS`, `normalize_frontal_lobe_note()`, `_infer_*()`, and `parse_frontal_lobe_note()` to coerce a free-text note into the current stored shape.
- `src/agent.py` still injects a prompt contract telling the model to write a four-section professional note string.
- Tool schema generation in `src/llm.py` already derives JSON schema from Python signatures, so changing the `update_frontal_lobe(...)` function signature is enough to update function-calling parameters.
- There is no current tracked runtime caller that still writes `Portfolio Health` through `update_lobe_section(...)`, so Phase B can fail fast on that path instead of carrying a compatibility shim.

## Approaches Considered

### 1. Native JSON + lazy migration (**Recommended**)

Change the source of truth immediately:

- `state.frontalLobe` becomes a dict
- `state.portfolioHealth` becomes a separate dict
- legacy string state is migrated lazily on load
- rendered markdown remains human-readable for operators

Why this is recommended:

- It removes the root cause instead of adding more heuristics.
- It keeps prompt/tool contracts aligned with stored data.
- It reduces long-term maintenance by deleting the inference stack instead of preserving it.

Trade-off:

- The memory interfaces and related tests need a coordinated update in one slice.

### 2. Dual-write compatibility release

Keep both the new dict shape and the old labeled string for one release.

Why not recommended:

- It preserves the exact ambiguity that caused the current artifacts.
- It adds conversion logic in both directions and delays deletion of the broken interface.

### 3. Big-bang strict rewrite without migration

Switch to the new dict shape and ignore old string state entirely.

Why not recommended:

- It is cleaner in code, but it unnecessarily discards already persisted high-signal legacy notes.
- It raises rollout risk for no material gain over lazy migration.

## Recommended Design

### 1. Structured frontal-lobe storage

Replace the current string state with a dict:

```python
{
    "market_view": "",
    "core_levels": "",
    "next_round": "",
    "context_note": "",
    "updated_at": None,
}
```

This dict becomes the source of truth for:

- persisted state
- commit deltas
- frontal-lobe no-op detection
- cognitive-context rendering
- `.brain/frontal-lobe.md` rendering

`Portfolio Health` is intentionally absent from this structure.

### 2. Dedicated portfolio-health state

Add a separate state surface:

```python
{
    "nav_twd": None,
    "pnl_pct": None,
    "top3_concentration": None,
    "drawdown_pct": None,
    "risk_state": None,
    "gross_scale": None,
    "updated_at": None,
}
```

This state is system-owned, not LLM-authored.

Introduce `Brain.update_portfolio_health(health_data: dict) -> Dict[str, Any]` with these rules:

1. apply a materiality gate before writing
2. update `state["portfolioHealth"]`
3. refresh `updated_at`
4. save the state
5. do **not** create a brain commit

`risk_state` changes are always material. Small NAV drift alone should not rewrite state.

### 3. New write contract for `update_frontal_lobe`

Change the public write tool from:

- `update_frontal_lobe(content: str)`

to:

- `update_frontal_lobe(market_view: str, core_levels: str, next_round: str, context_note: str = "")`

Design rules:

1. required fields stay required in the function signature
2. `context_note` remains optional
3. `portfolio_health` is removed from the tool entirely
4. placeholder-quality writes are still rejected
5. unchanged structured writes return a successful no-op result

The wrapper tool docstring and `FRONTAL_LOBE_WRITE_GUIDE` should describe parameter purpose and quality expectations, not a labeled multiline template.

### 4. Remove inference-heavy note normalization

Delete the free-text repair stack:

- `normalize_frontal_lobe_note()`
- `_infer_market_view()`
- `_infer_core_levels()`
- `_infer_portfolio_health()`
- `_infer_next_round()`
- `_extract_labeled_value()`
- `_split_sentences()`
- `FRONTAL_LOBE_KEYWORDS`

Simplify the supporting helpers:

- `parse_frontal_lobe_note()` becomes a dict-aware reader for the new structured state
- `_coerce_frontal_lobe_sections()` becomes a thin normalizer around dict/default handling
- `_render_frontal_lobe_note()` renders the stored dict to human-readable text for view files
- `_frontal_lobe_write_is_unchanged()` compares normalized dict content instead of string content
- `_build_frontal_lobe_ref()` builds a compact summary from dict fields

### 5. Lazy migration for legacy state

During `_load()`, if `state["frontalLobe"]` is still a string, migrate it with a dedicated helper:

```python
if isinstance(self.state["frontalLobe"], str):
    self.state["frontalLobe"] = _migrate_legacy_frontal_lobe(self.state["frontalLobe"])
```

Migration rules:

1. read old labeled sections when present
2. map `Market View`, `Core Levels`, `Next Round`, and optional `Context Note`
3. ignore legacy `Portfolio Health`
4. if parsing fails, fall back to empty structured values rather than attempting heuristic inference
5. preserve any resulting structured state going forward through normal saves

This migration happens lazily at read time. Existing commit history is not rewritten.

### 6. Cognitive-context rewrite

`get_cognitive_context()` should render three separate blocks:

1. **Trading Thesis (Frontal Lobe)** — only LLM-authored structured thesis fields
2. **Portfolio Health (Auto)** — system-computed portfolio state
3. **Persistent Macro / Market Regime** — heartbeat-managed regime data

Rendering requirements:

- If the frontal lobe is blank, explicitly say it has not been established yet and instruct the model to log it after analysis.
- Portfolio-health output should be compact and numeric-first.
- Market-regime output should keep the existing durability role, but no longer be visually mixed with malformed frontal-lobe lines.

## Compatibility and Migration

### `update_lobe_section(...)`

Keep the helper for targeted frontal-lobe sections that still belong to the thesis, but reject `Portfolio Health` as an invalid section after Phase B. This is intentional boundary enforcement, not a temporary limitation.

### Persisted view files

Keep writing `.brain/frontal-lobe.md`, but make it a rendered operator view of the structured dict. Human-readable markdown remains useful for inspection, but it is no longer the canonical storage format.

### Tool / prompt surfaces

The following surfaces must be updated in the same slice:

- `engine_memory.py` tool wrapper signature and guide text
- `src/agent.py` prompt contract that currently instructs the LLM to emit a four-section labeled note
- any tests or runtime assertions that assume `update_frontal_lobe` takes a single `content` string

`src/llm.py` does not need custom schema logic; the updated Python signature is sufficient.

## Failure Handling

1. Structured frontal-lobe writes with missing required fields should fail through normal Python/tool schema validation.
2. Low-quality frontal-lobe writes should still return explicit rejection instead of silently persisting placeholders.
3. `update_portfolio_health()` should treat no-op updates as successful unchanged writes, not as errors.
4. Legacy malformed string state should migrate to a safe empty structured shape rather than crashing load.

## Testing Strategy

Add focused regression coverage for:

1. default state initializes `frontalLobe` and `portfolioHealth` as structured dicts
2. legacy string frontal-lobe state migrates on load
3. identical structured frontal-lobe writes do not create commits
4. `update_lobe_section(...)` still updates valid thesis fields, but rejects `Portfolio Health`
5. `update_portfolio_health()` updates state without creating commits
6. small NAV-only drift is treated as unchanged, while `risk_state` changes are material
7. `get_cognitive_context()` renders the new three-block format cleanly
8. tool/runtime schema expectations reflect the new multi-parameter `update_frontal_lobe(...)` signature

Prefer tracked tests under `tests/` for clean-worktree reliability.

## Recommended Implementation Order

1. Add or update tracked regression tests for migration, structured defaults, structured frontal-lobe writes, portfolio-health no-commit updates, and cognitive-context rendering.
2. Refactor `engine_memory.py` data structures and helper methods to make dict-backed frontal-lobe storage the source of truth.
3. Add `update_portfolio_health()` and materiality gating.
4. Update tool wrappers and prompt contract text.
5. Update any runtime/test expectations tied to the old single-string tool signature.
6. Run focused memory and runtime regression checks.

## Expected Outcome

After Phase B:

1. parse artifacts like `Market View: Neutral - Market View:` cannot be produced by the interface itself
2. portfolio-health churn no longer pollutes the brain commit chain
3. the model sees a cleaner prompt context with separate human-thesis and system-state blocks
4. `engine_memory.py` becomes smaller and easier to reason about because the inference stack is removed
