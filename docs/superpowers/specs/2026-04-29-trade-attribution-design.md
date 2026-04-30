# Trade Attribution & Journal Design

## Problem

The broader cognitive-risk overhaul is blocked on one missing definition: the system promises weekly **Beta / Sector / Timing** attribution, but the current plan does not yet define the exact trade-level inputs, formulas, or module boundaries needed to compute that report correctly.

Today the codebase already has useful pieces:

- `engine_portfolio._record_trade_log()` persists an immutable `trade_log` row per trade.
- `trade_log.decision_snapshot` already stores decision-time context.
- `_build_sync_decision_snapshot()` already captures sector metadata, a sector proxy, and risk context.
- `compute_portfolio_beta_attribution()` already provides a benchmark-beta estimation pattern for current holdings.

What is missing is a bounded attribution design that:

1. Works with the current aggregated portfolio model.
2. Produces mathematically consistent weekly output.
3. Preserves the decision-time snapshot needed for later self-review.
4. Does not require a full lot-history rebuild before the first usable report ships.

## Goals

1. Add a v1 trade journal layer that evaluates new risk-taking decisions after entry.
2. Support deterministic `T+5` and `T+20` outcome checkpoints for `buy` / `add` trades.
3. Produce a weekly report with explicit **Beta / Sector / Timing** methodology.
4. Make missing data visible through coverage metrics instead of silently fabricating values.
5. Reuse existing portfolio and scheduler patterns instead of creating a parallel analytics stack.

## Non-Goals

1. No lot-level fill reconstruction in v1.
2. No closed-trade-only attribution model in v1.
3. No automatic trade execution.
4. No claim of pure execution-timing attribution; the v1 `Timing` bucket is a post-sector residual.
5. No attempt to backfill historical attribution for old trades that lack the required decision-time snapshot.

## Current Context

This design targets a smaller, implementation-ready slice of the larger overhaul: **trade attribution and journal reporting**.

Relevant repo context gathered before this spec:

- The live upstream checkout on `main` is dirty, so implementation must happen in an isolated worktree.
- A clean worktree from `origin/main` does not contain the local ignored `test_*.py` / `check_*.py` files that existed only in the dirty checkout, so the implementation plan must switch baseline verification to tracked checks that really exist on the clean branch.
- `requirements.txt` includes `fubon-neo-fugle-marketdata`, which is not available from the current pip environment, so worktree setup should reuse the existing shared venv instead of assuming a fresh `pip install -r requirements.txt` will succeed.

These environment findings do not change the design itself, but they do affect how the follow-on implementation plan must be written.

## Approaches Considered

### 1. Trade-level proxy attribution (**Recommended**)

Each `buy` / `add` trade stores a benchmark proxy, sector proxy, and entry-time beta proxy. `T+5` and `T+20` checkpoints later resolve the trade outcome and decompose the realized move into **Beta**, **Sector**, and **Timing** components with explicit fallback rules.

Why this is the recommended v1:

- It fits the current aggregated holdings model.
- It keeps the report tied to the actual decision that added risk.
- It yields deterministic output without rebuilding the entire accounting system.

Trade-off:

- The `Timing` bucket is still a residual bucket, not a pure execution-timing estimator.

### 2. Lot-level attribution rebuild

Reconstruct fill history, lot state, and holding windows first, then compute more precise attribution.

Why not v1:

- It is architecturally cleaner but much larger than the requested immediate unblock.
- It would delay the broader A/B/C/D work and force a larger portfolio-accounting refactor first.

### 3. Portfolio-level simplified attribution

Ignore trade-level events and report only portfolio-level Beta / Sector / residual decomposition each week.

Why not v1:

- It loses the core user value: using the user's own trade decisions to evaluate decision quality.
- It cannot answer whether a specific new position or add was good or bad after entry.

## Recommended Design

The recommended v1 keeps attribution tied to individual risk-increasing trades, stores the inputs required for later checkpoint settlement, and derives the weekly report from resolved checkpoint rows instead of from current holdings.

### Scope

This spec covers only **risk-increasing trades**:

- include: `buy`, `add`, broker-synced quantity increases
- exclude: trims, sells, rebalance exits, round-trip realized PnL attribution

The rationale is simple: `T+5` and `T+20` are intended to evaluate whether adding risk was a good decision, not to conflate entry analysis with exit execution.

### Data Model

### Extend `trade_log.decision_snapshot`

`trade_log.decision_snapshot` remains the immutable decision-time capture. v1 extends it so each eligible entry can later be attributed without reverse-engineering the market context.

Required snapshot fields for attribution:

- `captured_at`
- `symbol`
- `lookup_symbol`
- `sector`
- `industry`
- `benchmark_symbol` (`SPY` in v1 unless a future feature explicitly overrides it)
- `sector_proxy_symbol`
- `beta_proxy_period` (`6mo` in v1)
- `beta_proxy_at_entry`
- `risk_state`
- `risk_score`
- `vix`
- `alpha_sec`
- `alpha_macro`
- `alpha_retail`

Compatibility note:

- Existing legacy `decision_snapshot` rows remain readable.
- Journal settlement must treat missing fields as coverage gaps instead of crashing or inventing replacements.

### New table: `trade_outcome_checkpoints`

This table acts as both the deferred-work queue and the durable result store.

Suggested columns:

- `id`
- `trade_log_id`
- `symbol`
- `horizon_label` (`T+5`, `T+20`)
- `due_at`
- `status` (`pending`, `resolved`, `skipped`, `error`)
- `entry_timestamp`
- `entry_price`
- `entry_notional_twd`
- `benchmark_symbol`
- `benchmark_entry_price`
- `sector_proxy_symbol`
- `sector_entry_price`
- `beta_proxy_at_entry`
- `beta_coverage`
- `sector_coverage`
- `resolved_price`
- `benchmark_return_pct`
- `sector_return_pct`
- `actual_return_pct`
- `beta_component_pct`
- `sector_component_pct`
- `timing_component_pct`
- `beta_component_twd`
- `sector_component_twd`
- `timing_component_twd`
- `last_error`
- `retry_count`
- `resolved_at`

Constraints:

- one row per `(trade_log_id, horizon_label)`
- only create rows for eligible `buy` / `add` events

### Component Boundaries

### `engine_portfolio.py`

Responsibilities:

- continue to own trade persistence
- enrich `decision_snapshot` for new risk-increasing trades
- call the journal enqueue hook after the `trade_log` row is committed

This file should not compute weekly attribution reports. It only captures the decision and hands off the deferred analysis work.

### `engine_journal.py`

Responsibilities:

1. enqueue `T+5` / `T+20` checkpoints
2. settle due checkpoints
3. build the weekly attribution report from resolved checkpoint rows

This keeps all journal math in one place and avoids re-deriving trade outcomes from current holdings.

### `src/scheduler.py`

Responsibilities:

- run the due-checkpoint settlement job on schedule
- run the weekly journal report job on Sunday

This scheduler integration is orchestration-only. It must not trigger trade execution or import additional interactive runtime side effects.

### Calculation Rules

### Outcome horizons

For each eligible `buy` / `add` trade:

1. create one `T+5` checkpoint
2. create one `T+20` checkpoint

Settlement resolves each checkpoint using the first tradable price on or after the due trading day.

### Return primitives

For a resolved checkpoint:

- `actual_return_pct = (resolved_price / entry_price) - 1`
- `benchmark_return_pct = benchmark_price_at_due / benchmark_price_at_entry - 1`
- `sector_return_pct = sector_price_at_due / sector_price_at_entry - 1`

### Attribution components

The v1 formulas must add back to the realized outcome.

If the checkpoint has valid benchmark beta coverage and a valid sector proxy:

- `beta_component_pct = beta_proxy_at_entry * benchmark_return_pct`
- `sector_component_pct = sector_return_pct - beta_component_pct`
- `timing_component_pct = actual_return_pct - sector_return_pct`

This ensures:

- `beta_component_pct + sector_component_pct + timing_component_pct = actual_return_pct`

If the checkpoint has valid benchmark beta coverage but no usable sector proxy:

- `beta_component_pct = beta_proxy_at_entry * benchmark_return_pct`
- `sector_component_pct = 0`
- `timing_component_pct = actual_return_pct - beta_component_pct`
- `sector_coverage = 0`

If the checkpoint lacks a usable beta proxy:

- keep the row
- mark `beta_coverage = 0`
- do not fabricate beta decomposition
- report the row under coverage gaps / skipped attribution rows

### Currency aggregation

Each resolved checkpoint also stores approximate TWD contributions:

- `beta_component_twd = entry_notional_twd * beta_component_pct`
- `sector_component_twd = entry_notional_twd * sector_component_pct`
- `timing_component_twd = entry_notional_twd * timing_component_pct`

The weekly report sums the TWD components across included checkpoints and separately reports coverage ratios so partial coverage cannot masquerade as full truth.

### Coverage and Failure Rules

### Coverage rules

- `beta_coverage = 1` only when a valid entry-time beta proxy exists
- `sector_coverage = 1` only when a distinct usable sector proxy exists
- rows with missing coverage remain visible in the journal output
- the weekly report must explicitly show:
  - resolved checkpoint count
  - beta coverage ratio
  - sector coverage ratio
  - skipped / unresolved reasons

### Failure rules

- if a due checkpoint cannot resolve a usable price, keep it pending or mark it `error` with `last_error`
- increment `retry_count`
- let the next scheduler pass retry it
- never silently drop the checkpoint

### Report Semantics

The weekly report is a **decision-quality** report, not a portfolio-level NAV statement.

It should answer:

1. How did my recent risk-increasing trades do after entry?
2. How much of that move was explained by market beta?
3. How much was explained by sector behavior?
4. What residual remained after those two layers?

Important wording rule for v1:

- the user-facing report may still label the residual bucket as `Timing`
- the spec and implementation comments must document that this is a **sector-neutral residual bucket**, which blends timing and selection rather than claiming to isolate pure execution timing

### Validation Strategy

The implementation plan that follows this spec should add tracked checks that exist on the clean worktree branch and should not depend on ignored local files.

At minimum, the follow-on plan must verify:

1. risk-increasing trades enqueue exactly one `T+5` and one `T+20` checkpoint
2. checkpoint settlement computes and persists the resolved return fields correctly
3. attribution components sum back to `actual_return_pct`
4. missing beta / sector coverage is surfaced explicitly
5. the weekly report aggregates only resolved checkpoints and prints the correct coverage counts

## Why This Design Unblocks the Larger Overhaul

This design solves the blocker that prevented the broader overhaul from moving forward:

- it fixes the missing per-trade attribution inputs
- it defines mathematically consistent formulas
- it keeps the scope bounded to a clean v1
- it avoids pretending that local ignored checks are part of the clean baseline

That gives the next implementation plan a stable target: build a tracked, worktree-safe journal system first, then continue the rest of the cognitive-risk overhaul on top of it.
