# System Integrity Design

## Goal

Harden the brain persistence layer in `engine_memory.py` so repeated low-signal writes stop polluting the commit chain, market heartbeat updates stop producing noisy HOLD commits, persisted history stays bounded, and fresh brain state starts from a structured default instead of an empty frontal-lobe string.

This spec covers the **A: System Integrity** slice only. It does **not** include the later Frontal Lobe API redesign, Risk/Alpha scoring refactor, or the D2/D3 simulation features.

## Current Problems

1. `update_lobe_section()` always overwrites and commits, even when the normalized section text did not change.
2. `update_market_regime()` treats `signals` changes as commit-worthy changes, so heartbeat refreshes can create noisy market-regime commits even when the summary did not materially change.
3. Brain persistence keeps the full commit chain forever.
4. `_default_state()` starts `frontalLobe` as an empty string, which gives a weak initial state and pushes structure enforcement onto later writes.

## Scope

In scope:

- `engine_memory.py` write-path hardening for frontal-lobe updates
- market-regime commit gating
- commit retention capping
- structured default frontal-lobe content for new state
- regression tests for the above behavior

Out of scope:

- changing `state.frontalLobe` to an object/JSON structure
- changing public tool signatures
- rewriting existing `.brain/commit.json` contents
- risk-engine or alpha-governor redesign

## Design Summary

### 1. Keep the current public API

The public write surfaces stay unchanged for this slice:

- `update_frontal_lobe(content: str)`
- `update_lobe_section(section_name: str, new_content: str, source: str = "system")`
- `update_market_regime(...)`

This keeps A isolated from the later B-stage frontal-lobe redesign.

### 2. Add explicit no-op gates for frontal-lobe writes

Frontal-lobe writes should use a shared internal comparison rule:

- normalize incoming content into the same shape already used for persistence
- compare normalized incoming content to the current stored content
- if there is no material change, return a successful no-op result and skip commit creation

For `update_lobe_section()`, the comparison happens after rebuilding the full normalized note. This avoids section-level whitespace churn producing new commits.

For `update_frontal_lobe()`, the comparison happens after `normalize_frontal_lobe_note()` and placeholder rejection. If the normalized note matches the current stored note, treat the write as unchanged.

### 3. Separate state refresh from commit-worthy change detection

`update_market_regime()` should continue updating runtime state and heartbeat fields, even when the change is not material enough for a new commit.

The commit gate should treat these fields as material:

- `summary`
- `state`
- `riskScore`
- `watchpoints`
- `reasons`

`signals` should still be persisted into `state.marketRegime.signals`, but `signals` changes alone must **not** trigger a `market_regime` commit.

That produces the intended HOLD/no-change behavior:

- summary unchanged + signals drift -> state and heartbeat update, no commit
- summary/state/risk/watchpoints/reasons change -> state update + commit

### 4. Cap persisted history at 200 commits

Introduce a single retention rule in the persistence path:

- define `MAX_COMMITS = 200`
- whenever a new commit is appended, trim the list to the newest 200 entries before saving
- keep `head` pointing at the newest surviving commit

The cap belongs in the commit append/save path, not scattered across callers.

### 5. Use a structured default frontal-lobe template

Keep `state.frontalLobe` as a string for now, but initialize it with a fixed four-section template:

- `Market View:`
- `Core Levels:`
- `Portfolio Health:`
- `Next Round:`

The initial content should be structured and explicit, but neutral in tone. It should not look like a strong trade opinion, and it should not include placeholder phrases that the current quality filters would immediately reject as useless if re-written unchanged.

This improves first-run state without forcing the B-stage JSON/object redesign yet.

## Compatibility and Migration

- Existing persisted brain files remain readable.
- Existing loaded commit chains are not rewritten eagerly.
- The new rules only affect future writes and future saves.
- Legacy frontal-lobe notes keep using the existing normalization path.

## Failure Handling

- Empty or invalid frontal-lobe writes should still raise or return explicit errors consistent with current behavior.
- No-op writes are not errors; they should return a success-shaped result that clearly indicates nothing changed.
- Commit retention must never discard current state, only old history entries.

## Testing Strategy

Add focused regression coverage for:

1. repeated `update_lobe_section()` writes with identical normalized content -> no extra commit
2. repeated `update_frontal_lobe()` writes with identical normalized content -> no extra commit
3. `update_market_regime()` with unchanged summary but different signals -> heartbeat updated, signals updated, no new commit
4. commit-chain growth beyond 200 entries -> only newest 200 survive and `head` remains valid
5. new/default brain state -> structured frontal-lobe template is present instead of an empty string
6. legacy/older persisted state still loads without schema breakage

## Recommended Implementation Order

1. Add regression tests for no-op frontal-lobe writes, HOLD/no-change heartbeat behavior, commit capping, and structured defaults.
2. Implement shared frontal-lobe no-op comparison helpers.
3. Narrow `_market_regime_changed()` to material commit-driving fields while keeping state refresh behavior intact.
4. Add commit retention helper and apply it in the commit/save path.
5. Update `_default_state()` to emit the structured frontal-lobe template.

## Why This Slice Comes First

The later B and C changes will build on these write-path guarantees. Doing A first reduces noise in persisted cognition, prevents runaway history growth, and gives the later frontal-lobe redesign a cleaner, more predictable base to evolve from.
