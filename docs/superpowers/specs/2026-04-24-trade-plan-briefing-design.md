# Trade Plan Monitor + Morning Briefing Design

## Problem

The current system can analyze positions, gate some buy decisions, and persist broker-detected follow-ups, but it still has a major post-entry gap:

1. A position can exist without a complete exit plan.
2. Existing holdings can remain unmanaged because the system does not force a structured backfill.
3. Morning context exists in separate modules, but there is no deterministic orchestrator that answers "what do I need to do today?"
4. Thesis failure is not encoded in a machine-checkable form, so reminders depend too much on ad hoc chat instead of explicit rules.

This design introduces a combined **Trade Plan Monitor** and **Morning Briefing** architecture that closes those gaps without moving the system into auto-execution.

## Goals

1. Require a structured trade plan for new buys before the trade is treated as complete.
2. Force backfill for current holdings that do not yet have an active trade plan.
3. Monitor stop loss, take profit, holding window, and thesis invalidation with deterministic rules.
4. Push actionable alerts to Telegram and surface the same state in query paths.
5. Build a Morning Briefing orchestrator that turns overnight changes, events, trade-plan alerts, and risk state into deterministic action items.
6. Preserve an audit trail so later journal, T+5/T+20 review, and attribution work can build on the same data model.

## Non-Goals

1. No automatic order execution.
2. No full What-If Engine implementation in this spec.
3. No full T+5/T+20 outcome tracker in this spec.
4. No full weekly alpha attribution rebuild in this spec.
5. No lot-level multi-plan accounting in v1.

## Current Context

Relevant capabilities already exist:

- `engine_portfolio.sync_fubon_portfolio_state()` can detect inferred broker trades and capture `decision_snapshot`.
- `trade_followups` already exists for broker-detected post-trade follow-up.
- `compute_portfolio_risk_overlay()` and `engine_risk.get_global_risk_snapshot()` already provide deterministic risk context.
- The bot runtime already supports follow-up prompting and reply handling patterns.
- The system remains proposal-only, which this design preserves.

Relevant gap:

- `trade_followups` is an ingestion and reminder surface, not a durable trade-plan model.
- There is no active-plan table, no alert lifecycle for stop/target/thesis expiry, and no morning orchestrator that consumes those alerts.

## Scope and Phasing

This design intentionally combines two closely related surfaces in one spec, but keeps them separable in implementation.

### Phase A - Trade Plan Core

- Add the active trade-plan data model.
- Support new-buy plan capture.
- Support current-holdings backfill.
- Expose query helpers for plan state.

### Phase B - Monitor and Alerts

- Add scheduled rule checks for stop, target, holding window, and thesis invalidation.
- Persist and deduplicate alerts.
- Push alerts to Telegram.
- Surface alert state in relevant query/report flows.

### Phase C - Morning Briefing

- Build a deterministic orchestrator that consumes overnight changes, events, plan alerts, and risk state.
- Produce prioritized action items.
- Use LLM formatting only after action items are fixed.

### Phase D - Analytics Hooks

- Keep append-only history and stable interfaces so future What-If and Decision Journal work can attach cleanly.
- Do not implement those larger analytics systems in this phase.

## Architecture

The system should be split into four cooperating layers.

### 1. Plan Intake Layer

Responsibilities:

- Create or update a trade plan when a new buy is initiated through the bot.
- Create urgent backfill tasks when a broker-detected buy or an existing holding has no active plan.
- Reuse existing `trade_followups` for broker-detected ingestion, but hand off plan persistence to the new trade-plan subsystem.

Why this split:

- A broker-detected follow-up is not the same thing as a durable trading plan.
- Separating ingestion from the plan model keeps the later journal and analytics layer clean.

### 2. Trade Plan Engine

Responsibilities:

- Store one active plan per symbol.
- Track plan revisions and state transitions.
- Provide deterministic helpers for plan reads, writes, closure, and validation.

V1 design choice:

- **One active plan per symbol** instead of lot-level plans.
- This matches the current aggregated portfolio/accounting model and avoids forcing a larger holdings refactor.

### 3. Monitor Engine

Responsibilities:

- Periodically evaluate active plans against market data, risk state, and time.
- Generate, update, and resolve plan alerts.
- Maintain a stable alert lifecycle shared by Telegram and Morning Briefing.

### 4. Morning Briefing Engine

Responsibilities:

- Pull overnight moves, today's events, open plan alerts, and global risk context.
- Derive deterministic action items.
- Format the final briefing after the action list is fixed.

Key rule:

- The briefing is an **orchestrator**, not a new judgment engine.
- It should summarize what already needs attention rather than invent new discretionary actions.

## Data Model

### `trade_plans`

Durable current-state record for each active symbol plan.

Suggested fields:

- `id`
- `symbol`
- `status` (`draft`, `active`, `closed`, `expired`, `invalidated`)
- `source` (`bot_buy`, `broker_sync`, `manual_backfill`, `plan_revision`)
- `opened_trade_log_id` nullable
- `entry_price`
- `stop_loss`
- `take_profit_1`
- `take_profit_2`
- `max_holding_days`
- `thesis_type`
- `thesis_text`
- `thesis_payload_json`
- `created_at`
- `updated_at`
- `completed_at`

Constraints:

- At most one `active` plan per symbol in v1.
- A plan is not considered complete until `stop_loss`, at least one take-profit level, `max_holding_days`, `thesis_type`, and `thesis_text` are populated.

### `trade_plan_events`

Append-only audit history.

Suggested event types:

- `plan_created`
- `plan_activated`
- `plan_updated`
- `plan_backfill_requested`
- `alert_fired`
- `alert_acknowledged`
- `alert_resolved`
- `plan_closed`

Why this table exists:

- Future journal and attribution work needs an immutable trail of what the user planned, changed, ignored, or resolved.

### `trade_plan_alerts`

Current alert state for reminder and briefing consumption.

Suggested fields:

- `id`
- `plan_id`
- `symbol`
- `alert_type`
- `severity`
- `status` (`open`, `acknowledged`, `resolved`, `suppressed`)
- `payload_json`
- `first_seen_at`
- `last_seen_at`
- `resolved_at`

Alert types in v1:

- `missing_plan`
- `stop_hit`
- `tp1_hit`
- `tp2_hit`
- `holding_expiry`
- `thesis_invalid`
- `monitor_degraded`

Deduplication rule:

- One open alert per `(plan_id, alert_type)` pair.
- Repeated detections update `last_seen_at` and payload instead of creating alert spam.

## Thesis Model

V1 thesis types are fixed and structured. Freeform-only plans are not sufficient.

### Thesis types

1. `mean_reversion`
2. `earnings`
3. `event_driven`
4. `sector_rotation`
5. `breakout_support`

Each plan must include:

- `thesis_text`: the human-readable statement of the trade
- `thesis_payload_json`: machine-checkable parameters used by the monitor

### Example payload expectations

- `mean_reversion`: reference level, expected recovery window, comparison proxy
- `earnings`: earnings date, expected direction, review window
- `event_driven`: catalyst date, catalyst type, invalidation deadline
- `sector_rotation`: proxy symbol, relative-strength window, failure threshold
- `breakout_support`: support or breakout level, grace rule, close-below threshold

## Core Flows

### Flow A - New Buy Through Bot

1. User requests a buy.
2. Existing pre-trade risk gate runs first.
3. If risk gate passes, the system enters the plan gate.
4. The trade is only considered complete after the user provides:
   - stop loss
   - at least one take-profit target
   - max holding days
   - thesis type
   - thesis text
5. The plan is stored as `active`.
6. The buy flow may then continue through the existing proposal/update path.

Interpretation:

- This is a **hard gate** for bot-originated buys.

### Flow B - Broker-Detected Buy

1. `trade_followups` captures the inferred external trade.
2. The intake layer checks whether the symbol already has an active plan.
3. If not, create a `draft` or `missing-plan` backfill record plus a `missing_plan` alert.
4. Push Telegram follow-up until the plan becomes complete.
5. Query surfaces should show the position as unmanaged until resolved.

Interpretation:

- This is a **hard chase**, not a gate, because the trade already happened outside the bot.

### Flow C - Current Holdings Backfill

1. On migration or scheduled scans, compare portfolio holdings against active trade plans.
2. For any symbol with holdings but no active plan, create a backfill request.
3. Prioritize larger or riskier positions first if batching becomes necessary.
4. Keep the symbol flagged across Telegram, morning briefing, and relevant query outputs until completed.

User decision:

- Backfill must be manual. The system should not invent thesis text for the user in v1.

### Flow D - Alert Handling

1. Monitor engine evaluates active plans on schedule.
2. If a rule triggers, create or update the matching alert.
3. Telegram notification is sent for new or escalated alerts.
4. The user can:
   - act on the position
   - acknowledge and revise the plan
   - snooze temporarily
5. Ignoring an alert without updating rationale or parameters is not allowed as a clean resolution path.

## Monitoring Rules

### Stop Loss

- Trigger `stop_hit` when current price crosses the plan stop threshold.
- This is the highest-priority trading alert in v1.

### Take Profit

- Trigger `tp1_hit` or `tp2_hit` when the relevant price threshold is reached.
- These are informational-to-actionable alerts, lower priority than stop loss or thesis invalidation.

### Holding Window

- Trigger `holding_expiry` when the remaining allowed holding time is low or expired.
- The alert payload should include held days, max days, and current return.

### Thesis Invalidation

Rules are deterministic and thesis-type specific.

- `mean_reversion`
  - Trigger invalidation if the expected recovery window expires without the expected response.
  - Trigger invalidation if the symbol persistently underperforms its comparison proxy after the mean-reversion condition should have normalized.

- `earnings`
  - Trigger invalidation if the event window ends and price action contradicts the expected thesis.
  - Trigger invalidation if the earnings catalyst has passed and the position is no longer justified by the original event thesis.

- `event_driven`
  - Trigger invalidation if the catalyst date passes with no confirming outcome or with clearly opposite outcome.

- `sector_rotation`
  - Trigger invalidation if the sector proxy loses relative strength over the configured window or the symbol fails to participate in the expected rotation.

- `breakout_support`
  - Trigger invalidation if price breaks below the protected level according to the configured close-based rule.

## Morning Briefing Design

The Morning Briefing should answer a single question:

> What, if anything, needs action today?

### Inputs

1. Overnight changes for portfolio symbols and relevant benchmark proxies
2. Today's events for portfolio symbols and watchlist
3. Open trade-plan alerts
4. Global risk snapshot

### Orchestration Steps

1. Collect all inputs
2. Normalize them into a briefing payload
3. Rank urgency
4. Produce deterministic action items
5. Pass the payload to a formatter for final human-readable text

### Output Shape

Recommended structure:

1. One-line summary
2. Overnight changes
3. Today's events
4. Portfolio-specific alerts
5. Action items

### Prioritization

Default ordering:

1. `stop_hit`
2. `thesis_invalid`
3. `holding_expiry`
4. major same-day event risk for owned symbols
5. low-importance overnight noise

### Determinism Rule

- LLM may improve phrasing.
- LLM must not invent or reprioritize action items.

## Error Handling

### Market Data or Event Data Failure

- Do not silently suppress monitoring.
- Instead, emit or update a `monitor_degraded` alert.
- Morning Briefing should surface degraded monitoring state explicitly.

### Duplicate Alerts

- Repeated runs must update the same open alert rather than produce notification spam.

### Missing Fields

- An incomplete plan remains incomplete until all required fields exist.
- Query/report surfaces should show incomplete status explicitly.

### User Override

- A user may revise a plan.
- A user may snooze an alert.
- A user may not resolve a critical alert without either taking action or revising the plan.

## Interfaces and File Boundaries

Recommended implementation shape:

- `engine_portfolio.py`
  - schema management
  - core plan helpers
  - alert persistence
  - holdings reconciliation against plan state

- `src/bot.py`
  - Telegram prompting, reply parsing, acknowledgement flow

- `src/scheduler.py`
  - periodic monitor execution
  - briefing job trigger

- `engine_market.py` or a small new helper module
  - event/overnight helper reuse where appropriate

- `engine_briefing.py` or equivalent focused module
  - deterministic Morning Briefing orchestration

Preferred boundary:

- Keep the Morning Briefing engine isolated from the bot transport so briefing logic stays testable without Telegram.

## Validation Strategy

Use focused tests aligned to the modified surfaces.

### Extend existing tests

- `test_tool_logic_refactor.py`
  - schema
  - helper behavior
  - rule evaluation

### Add focused runtime tests

- `test_trade_plan_flow.py`
  - new-buy hard gate
  - current-holdings backfill
  - alert lifecycle
  - unresolved missing-plan behavior

- `test_morning_briefing.py`
  - deterministic action-item selection
  - degraded-monitoring surfacing
  - urgency ordering

### Reuse existing validation style

- Follow repo convention: focused suites, not broad whole-system runs.
- Prefer direct helper tests plus a thin integration layer for bot/scheduler entry points.

## Roadmap After This Spec

The following should remain follow-on work, not part of this implementation phase:

1. What-If Engine
   - historical scenario stress tests
   - swap/rebalance simulations

2. Decision Journal and Outcome Tracking
   - T+5 / T+20 outcome records
   - signal correctness review
   - ignored-stop historical behavior analysis

3. Weekly Attribution Expansion
   - selection alpha
   - timing alpha
   - signal IC review tied back to plan quality

## Recommendation

Implement this design as a phased rollout:

1. Trade Plan Core
2. Monitor and Alerts
3. Morning Briefing

This sequence fixes the highest-value behavioral gap first: positions without a durable, reviewable exit plan.
