# Cognitive Risk & Analytics Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace noisy brain commits and multiplicative risk/alpha coupling with structured memory, additive risk scoring, and portfolio analytics features that can be merged to `main` and pushed safely.

**Architecture:** Deliver this in four mergeable layers: `engine_memory.py` for cognitive-state integrity, `engine_risk.py` + `engine_router.py` + `nlp_worker.py` for scoring decoupling, `engine_portfolio.py` + `engine_journal.py` for trade outcome tracking, and `engine_scenarios.py` for historical replay / swap simulation. Keep backward-compatible read surfaces where possible, but move all decision logic onto structured state and additive scores instead of string parsing and multiplicative governors.

**Tech Stack:** Python, sqlite3 via `src.database`, pandas, APScheduler, existing unittest/check scripts, yfinance / existing market helpers.

---

## Execution notes

- The live checkout at `/home/margincaller/MarginCall_2X` is already dirty on `main` (`.brain/commit.json`, `engine_market.py`, `src/symbols.py`, untracked docs, `graphify-out/`). Do **not** implement directly there.
- Create a fresh worktree from `origin/main`, do all implementation there, and perform the final merge/push from a clean `main` worktree so unrelated local edits are never mixed into this feature.
- Preserve the existing trade-plan / follow-up / scheduler behavior; this plan must not regress `check_followup_proposals.py`, `test_phase_lifecycle_refactor.py`, or `test_refactor_runtime.py`.

## File structure

- **Modify:** `/home/margincaller/MarginCall_2X/engine_memory.py`
  - Add commit dedupe, commit-cap trimming, structured frontal-lobe state, portfolio-health side channel, reset helpers, and meaningful-context filtering.
- **Modify:** `/home/margincaller/MarginCall_2X/engine_portfolio.py`
  - Extend decision snapshots, move portfolio-health writes out of frontal-lobe commits, add journal/outcome schema, and expose portfolio inputs used by the scenario engine.
- **Modify:** `/home/margincaller/MarginCall_2X/.gitignore`
  - Unignore the new permanent journal/scenario regression files so they can be committed.
- **Modify:** `/home/margincaller/MarginCall_2X/engine_risk.py`
  - Replace multiplicative `riskMultiplier` scoring with additive `riskScore` contributions and explicit score components.
- **Modify:** `/home/margincaller/MarginCall_2X/engine_router.py`
  - Make alpha governor IC-only, expose independent NLP dimensions, and stop using regime/drawdown multipliers inside alpha.
- **Modify:** `/home/margincaller/MarginCall_2X/nlp_worker.py`
  - Produce/store independent `alpha_sec` / `alpha_macro` / `alpha_retail` outputs and stop relying on composite `nlp_alpha` as the primary decision signal.
- **Modify:** `/home/margincaller/MarginCall_2X/engine_market.py`
  - Update candidate confidence / forecast calibration to consume additive risk state and independent alpha dimensions.
- **Modify:** `/home/margincaller/MarginCall_2X/src/agent.py`
  - Update the frontal-lobe write contract and final-report text so the LLM sees structured memory + separate alpha dimensions.
- **Modify:** `/home/margincaller/MarginCall_2X/src/bot.py`
  - Import the new journal/scenario tool modules so runtime registration can expose them.
- **Modify:** `/home/margincaller/MarginCall_2X/src/scheduler.py`
  - Schedule T+5/T+20 checkpoint processing and the weekly attribution report.
- **Create:** `/home/margincaller/MarginCall_2X/engine_journal.py`
  - Build trade-outcome checkpoint generation, checkpoint settlement, and weekly attribution reporting.
- **Create:** `/home/margincaller/MarginCall_2X/engine_scenarios.py`
  - Build historical stress-test and swap what-if simulation helpers/tools.
- **Modify:** `/home/margincaller/MarginCall_2X/test_brain_memory.py`
  - Cover dedupe, commit cap, reset behavior, structured frontal-lobe state, and context filtering.
- **Modify:** `/home/margincaller/MarginCall_2X/check_risk_quality_refactor.py`
  - Lock in additive risk score expectations and component-level assertions.
- **Modify:** `/home/margincaller/MarginCall_2X/check_alpha_signal_pipeline.py`
  - Lock in independent alpha dimensions, IC-only alpha scaling, and event-lane behavior.
- **Modify:** `/home/margincaller/MarginCall_2X/check_risk_overlays.py`
  - Verify router overlays stop compounding regime/drawdown multipliers into alpha.
- **Modify:** `/home/margincaller/MarginCall_2X/check_candidate_constructor.py`
  - Update candidate-construction expectations to use dimension-first alpha payloads.
- **Modify:** `/home/margincaller/MarginCall_2X/check_followup_proposals.py`
  - Update follow-up proposal checks so they no longer depend on `riskMultiplier` / legacy `nlp_alpha`.
- **Modify:** `/home/margincaller/MarginCall_2X/check_technical_signal_upgrades.py`
  - Update technical/risk tests that still stub the old `nlp_alpha`-only payload.
- **Modify:** `/home/margincaller/MarginCall_2X/test_phase_lifecycle_refactor.py`
  - Update runtime assumptions affected by the new memory reset / report payloads.
- **Modify:** `/home/margincaller/MarginCall_2X/test_refactor_runtime.py`
  - Register any new `@tool()` exports from `engine_journal.py` / `engine_scenarios.py`.
- **Create:** `/home/margincaller/MarginCall_2X/check_trade_journal.py`
  - Cover T+5/T+20 scheduling, outcome capture, and weekly attribution output.
- **Create:** `/home/margincaller/MarginCall_2X/check_scenario_engine.py`
  - Cover historical stress replay and A→B swap simulation.

---

### Task 1: Isolate the worktree and freeze the baseline

**Files:**
- Test: baseline commands only

- [ ] **Step 1: Create a clean feature worktree from `origin/main`.**

Run:

```bash
cd /home/margincaller/MarginCall_2X && \
git fetch origin && \
git worktree add .worktrees/cognitive-risk-analytics -b feature/cognitive-risk-analytics origin/main
```

Expected: a clean worktree at `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics`.

- [ ] **Step 2: Run the focused baseline suite inside the worktree.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest \
  test_brain_memory.py \
  check_risk_quality_refactor.py \
  check_alpha_signal_pipeline.py \
  check_risk_overlays.py \
  check_followup_proposals.py \
  check_candidate_constructor.py \
  check_technical_signal_upgrades.py \
  test_phase_lifecycle_refactor.py \
  test_refactor_runtime.py
```

Expected: current baseline passes before any refactor starts.

- [ ] **Step 3: Record the baseline in the branch before editing.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && git --no-pager status --short
```

Expected: no local edits in the new worktree.

---

### Task 2: Stop duplicate brain writes and migrate memory to structured state

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_memory.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Modify: `/home/margincaller/MarginCall_2X/src/agent.py`
- Modify: `/home/margincaller/MarginCall_2X/test_brain_memory.py`
- Modify: `/home/margincaller/MarginCall_2X/test_phase_lifecycle_refactor.py`
- Modify: `/home/margincaller/MarginCall_2X/check_followup_proposals.py`
- Test: `/home/margincaller/MarginCall_2X/test_brain_memory.py`

- [ ] **Step 1: Write the failing tests for dedupe, commit cap, structured state, and reset behavior.**

```python
def test_update_lobe_section_skips_identical_write(self):
    brain = memory.Brain()
    self.assertTrue(
        brain.update_frontal_lobe(
            market_view="Neutral - CPI is the next catalyst while breadth remains mixed.",
            core_levels=["SPX 5200 support", "SPX 5250 resistance"],
            portfolio_health="Keep leverage light until event risk clears.",
            next_round="If CPI cools, add risk; if 5200 breaks, cut exposure.",
            context_note="seed",
        )["success"]
    )
    before = len(brain.commits)
    result = brain.update_lobe_section("Market View", "Neutral - CPI is the next catalyst while breadth remains mixed.")
    self.assertTrue(result["success"])
    self.assertEqual(len(brain.commits), before)


def test_market_regime_same_summary_skips_hold_commit_even_if_signals_move(self):
    brain = memory.Brain()
    first = brain.update_market_regime(
        summary="風險維持整理盤，等待事件突破。",
        regime="🟡 整理",
        risk_score=34,
        signals={"spx": 5200.0, "gexBillions": 1.2},
        source="unit_test",
    )
    second = brain.update_market_regime(
        summary="風險維持整理盤，等待事件突破。",
        regime="🟡 整理",
        risk_score=34,
        signals={"spx": 5203.0, "gexBillions": 1.1},
        source="unit_test",
    )
    self.assertTrue(first["changed"])
    self.assertFalse(second["changed"])


def test_commit_chain_is_capped_at_200(self):
    brain = memory.Brain()
    for idx in range(205):
        brain.update_emotion("neutral", f"tick-{idx}")
    self.assertEqual(len(brain.commits), 200)
    self.assertEqual(brain.head, brain.commits[-1]["hash"])


def test_reset_brain_preserves_market_regime_and_uses_structured_defaults(self):
    brain = memory.Brain()
    brain.update_market_regime("macro steady", regime="🟢 多頭", risk_score=12, source="unit_test")
    brain.reset_brain_state(preserve_market_regime=True)
    snapshot = brain.get_brain_snapshot(max_age_minutes=999999)
    self.assertEqual(snapshot["state"]["marketRegime"]["state"], "🟢 多頭")
    self.assertEqual(snapshot["state"]["portfolioHealth"]["summary"], "")
    self.assertEqual(snapshot["state"]["frontalLobe"]["marketView"], "")
    self.assertEqual(snapshot["recentCommits"], [])
```

- [ ] **Step 2: Run the focused test file and confirm the new assertions fail first.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest test_brain_memory.py
```

Expected: FAIL because the structured `update_frontal_lobe` signature, commit trimming, and `reset_brain_state()` do not exist yet.

- [ ] **Step 3: Replace string-only frontal-lobe storage with structured state, and prevent redundant commits.**

```python
# engine_memory.py
MAX_COMMITS = 200


def _default_frontal_lobe() -> Dict[str, Any]:
    return {
        "marketView": "",
        "coreLevels": [],
        "nextRound": "",
        "contextNote": "",
        "updatedAt": None,
    }


def _default_portfolio_health() -> Dict[str, Any]:
    return {"summary": "", "source": "", "updatedAt": None}


def _default_state() -> Dict[str, Any]:
    return {
        "frontalLobe": _default_frontal_lobe(),
        "portfolioHealth": _default_portfolio_health(),
        "emotion": "neutral",
        "marketRegime": _default_market_regime(),
        "heartbeat": _default_heartbeat(),
    }


def _load(self):
    if not BRAIN_FILE.exists():
        self._persist_views()
        return
    with open(BRAIN_FILE, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    raw_state = data.get("state", {})
    self.state = _merge_defaults(_default_state(), raw_state)
    if isinstance(raw_state.get("frontalLobe"), str):
        legacy_sections = parse_legacy_frontal_lobe_note(raw_state.get("frontalLobe", ""))
        self.state["frontalLobe"] = self._coerce_structured_frontal_lobe(legacy_sections)
        if legacy_sections.get("Portfolio Health"):
            self.state["portfolioHealth"] = {
                "summary": legacy_sections["Portfolio Health"],
                "source": "legacy_migration",
                "updatedAt": _utc_now_iso(),
            }
    self.commits = []
    for commit in data.get("commits", []):
        normalized = self._normalize_loaded_commit(commit)
        if normalized:
            self.commits.append(normalized)
    self.head = data.get("head") or (self.commits[-1]["hash"] if self.commits else None)
    self._persist_views()


def _trim_commit_chain(self) -> None:
    if len(self.commits) <= MAX_COMMITS:
        return
    self.commits = self.commits[-MAX_COMMITS:]
    self.commits[0]["parent_hash"] = None
    self.head = self.commits[-1]["hash"]


def _create_commit(
    self,
    commit_type: str,
    summary: str,
    delta: Dict[str, Any] | None = None,
    expected_head: Optional[str] = None,
    **kwargs,
) -> bool:
    if expected_head is not None and expected_head != self.head:
        return False
    if expected_head is not None and self._read_persisted_head() != expected_head:
        return False
    timestamp = _utc_now_iso()
    commit = {
        "hash": generate_commit_hash(
            {"type": commit_type, "summary": summary, "delta": delta or {}, "parent_hash": self.head, "timestamp": timestamp}
        ),
        "parent_hash": self.head,
        "timestamp": timestamp,
        "type": commit_type,
        "summary": summary,
        "delta": delta or {},
        **kwargs,
    }
    self.commits.append(commit)
    self.head = commit["hash"]
    self._trim_commit_chain()
    self._save()
    return True


def _render_frontal_lobe_sections(self) -> Dict[str, str]:
    frontal = self.state.get("frontalLobe") or _default_frontal_lobe()
    return {
        "Market View": frontal.get("marketView", ""),
        "Core Levels": ", ".join(frontal.get("coreLevels") or []),
        "Portfolio Health": (self.state.get("portfolioHealth") or {}).get("summary", ""),
        "Next Round": frontal.get("nextRound", ""),
        "Context Note": frontal.get("contextNote", ""),
    }


def _coerce_structured_frontal_lobe(self, sections: Dict[str, str]) -> Dict[str, Any]:
    return {
        "marketView": sections.get("Market View", "").strip(),
        "coreLevels": [item.strip() for item in sections.get("Core Levels", "").split(",") if item.strip()],
        "nextRound": sections.get("Next Round", "").strip(),
        "contextNote": sections.get("Context Note", "").strip(),
        "updatedAt": _utc_now_iso(),
    }


def parse_legacy_frontal_lobe_note(content: str) -> Dict[str, str]:
    normalized = content.replace("：", ":")
    sections = {"Market View": "", "Core Levels": "", "Portfolio Health": "", "Next Round": "", "Context Note": ""}
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        label, value = line.split(":", 1)
        if label.strip() in sections:
            sections[label.strip()] = value.strip()
    return sections


def _market_regime_changed(self, new_market: Dict[str, Any]) -> bool:
    current = self.state.get("marketRegime", _default_market_regime())
    current_summary = (current.get("summary") or "").strip()
    new_summary = (new_market.get("summary") or "").strip()
    current_state = current.get("state")
    new_state = new_market.get("state")
    current_risk = current.get("riskScore")
    new_risk = new_market.get("riskScore")
    if current_summary == new_summary and current_state == new_state and current_risk == new_risk:
        return False
    keys = ["summary", "state", "riskScore", "watchpoints", "reasons"]
    return any(_to_comparable(current.get(key)) != _to_comparable(new_market.get(key)) for key in keys)


def update_lobe_section(self, section_name: str, new_content: str, source: str = "system") -> Dict[str, Any]:
    current_value = self._render_frontal_lobe_sections().get(section_name, "")
    if new_content.strip() == current_value.strip():
        return {"success": True, "message": f"Frontal lobe section '{section_name}' unchanged."}
    sections = self._render_frontal_lobe_sections()
    sections[section_name] = new_content.strip()
    self.state["frontalLobe"] = self._coerce_structured_frontal_lobe(sections)
    committed = self._create_commit(
        "frontal_lobe_patch",
        f"🧠 {section_name.upper()} AUTO-UPDATE: {_shorten(new_content, 80)}",
        delta={section_name.lower().replace(' ', '_'): new_content.strip()},
        key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
        frontal_lobe_ref=self._build_frontal_lobe_ref(),
        source=source,
    )
    return {"success": committed, "message": f"Frontal lobe section '{section_name}' updated."}


def update_portfolio_health(self, summary: str, source: str = "system") -> Dict[str, Any]:
    self.state["portfolioHealth"] = {"summary": summary.strip(), "source": source, "updatedAt": _utc_now_iso()}
    self._save()
    return {"success": True, "message": "Portfolio health updated successfully"}


def reset_brain_state(self, preserve_market_regime: bool = True) -> Dict[str, Any]:
    preserved_market = copy.deepcopy(self.state.get("marketRegime", _default_market_regime()))
    self.state = _default_state()
    if preserve_market_regime:
        self.state["marketRegime"] = preserved_market
    self.commits = []
    self.head = None
    self._save()
    return {"success": True, "message": "Brain state reset successfully"}
```

- [ ] **Step 4: Replace the frontal-lobe tool contract and move portfolio-health auto-refresh off the commit chain.**

```python
# engine_memory.py
@tool(mode="write")
def update_frontal_lobe(
    market_view: str,
    core_levels: list[str],
    portfolio_health: str,
    next_round: str,
    context_note: str = "",
) -> str:
    res = _global_brain.update_frontal_lobe(
        market_view=market_view,
        core_levels=core_levels,
        portfolio_health=portfolio_health,
        next_round=next_round,
        context_note=context_note,
    )
    return res["message"]


# engine_memory.py
def update_frontal_lobe(
    self,
    content: str | None = None,
    *,
    market_view: str = "",
    core_levels: list[str] | None = None,
    portfolio_health: str = "",
    next_round: str = "",
    context_note: str = "",
) -> Dict[str, Any]:
    if isinstance(content, str) and content.strip():
        normalized_sections = parse_legacy_frontal_lobe_note(content)
    else:
        normalized_sections = {
            "Market View": market_view.strip(),
            "Core Levels": ", ".join(item.strip() for item in (core_levels or []) if str(item).strip()),
            "Portfolio Health": portfolio_health.strip(),
            "Next Round": next_round.strip(),
            "Context Note": context_note.strip(),
        }
    parsed_market_view = normalized_sections.get("Market View", "").strip()
    parsed_core_levels = [item.strip() for item in normalized_sections.get("Core Levels", "").split(",") if item.strip()]
    parsed_portfolio_health = normalized_sections.get("Portfolio Health", "").strip()
    parsed_next_round = normalized_sections.get("Next Round", "").strip()
    parsed_context_note = normalized_sections.get("Context Note", "").strip()
    if self._is_placeholder_content("\n".join(f"{k}: {v}" for k, v in normalized_sections.items() if v)):
        return {"success": False, "message": "Rejected: content is too vague to persist."}
    snapshot_head = self.head
    self.state["frontalLobe"] = self._coerce_structured_frontal_lobe(normalized_sections)
    self.state["portfolioHealth"] = {"summary": parsed_portfolio_health, "source": "frontal_lobe_write", "updatedAt": _utc_now_iso()}
    committed = self._create_commit(
        "frontal_lobe",
        self._build_frontal_lobe_commit_summary(self._render_frontal_lobe_sections()),
        delta={
            "market_view": parsed_market_view,
            "core_levels": parsed_core_levels,
            "portfolio_health": parsed_portfolio_health,
            "next_round": parsed_next_round,
            "context_note": parsed_context_note,
        },
        key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
        frontal_lobe_ref=self._build_frontal_lobe_ref(),
        source="frontal_lobe_write",
        expected_head=snapshot_head,
    )
    return {"success": committed, "message": "Frontal lobe updated successfully" if committed else "Rejected: concurrent frontal lobe update detected."}


# engine_portfolio.py
def refresh_portfolio_health_summary(source: str = "portfolio_review") -> Dict[str, Any]:
    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    nav_snapshot = record_portfolio_nav_snapshot(source=source, snapshots=snapshots) if snapshots else {"error": "無有效持倉"}
    analysis = build_portfolio_analysis(snapshots=snapshots)
    overlay = compute_portfolio_risk_overlay(snapshots=snapshots)
    memory_update = memory._global_brain.update_portfolio_health(analysis["summary"], source=source)
    return {**analysis, "risk_overlay": overlay, "nav_snapshot": nav_snapshot, "memory_update": memory_update}


# src/agent.py
brain_context += "\n\n## Frontal Lobe Write Contract\n"
brain_context += memory.get_frontal_lobe_write_guide()


# engine_memory.py
def _render_frontal_lobe_markdown(self) -> str:
    frontal = self.state.get("frontalLobe") or _default_frontal_lobe()
    portfolio_health = self.state.get("portfolioHealth") or _default_portfolio_health()
    lines = [
        f"Market View: {frontal.get('marketView', '')}",
        f"Core Levels: {', '.join(frontal.get('coreLevels') or [])}",
        f"Portfolio Health: {portfolio_health.get('summary', '')}",
        f"Next Round: {frontal.get('nextRound', '')}",
    ]
    if frontal.get("contextNote"):
        lines.append(f"Context Note: {frontal['contextNote']}")
    return "\n".join(lines)


def get_frontal_lobe(self) -> str:
    return self._render_frontal_lobe_markdown()


def _build_frontal_lobe_ref(self, content: Optional[Dict[str, Any]] = None) -> str:
    frontal = content or self.state.get("frontalLobe") or _default_frontal_lobe()
    return " | ".join(
        part for part in [
            _shorten(frontal.get("marketView"), 80),
            f"Next: {_shorten(frontal.get('nextRound'), 80)}" if frontal.get("nextRound") else "",
        ]
        if part
    )


def _normalize_loaded_commit(self, commit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    legacy_state = _merge_defaults(_default_state(), commit.get("stateAfter", {}))
    legacy_note_source = legacy_state.get("frontalLobe", "")
    if isinstance(legacy_note_source, str):
        self.state["frontalLobe"] = self._coerce_structured_frontal_lobe(parse_legacy_frontal_lobe_note(legacy_note_source))
    return {
        "hash": commit.get("hash") or generate_commit_hash(commit),
        "parent_hash": commit.get("parent_hash") or commit.get("parentHash"),
        "timestamp": commit.get("timestamp") or _utc_now_iso(),
        "type": commit.get("type", "unknown"),
        "summary": commit.get("summary") or commit.get("message") or "",
        "key_signals": commit.get("key_signals") or "",
        "frontal_lobe_ref": self._build_frontal_lobe_ref(),
        "source": commit.get("source") or "",
        "delta": commit.get("delta") or {},
    }


def _persist_views(self):
    FRONTAL_LOBE_FILE.write_text("# Frontal Lobe\n\n" + self._render_frontal_lobe_markdown(), encoding="utf-8")


def get_frontal_lobe_write_guide() -> str:
    return (
        "Call update_frontal_lobe with named fields only:\n"
        "- market_view: one-sentence thesis\n"
        "- core_levels: list of support/resistance/MA levels\n"
        "- portfolio_health: sizing / drawdown / concentration summary\n"
        "- next_round: explicit if/then plan\n"
        "- context_note: optional extra context\n"
        "Do not write a free-form four-section paragraph."
    )
```

Explicit removal: delete `normalize_frontal_lobe_note()` after migrating its legacy parsing responsibilities into a narrow `parse_legacy_frontal_lobe_note()` adapter used only during load/migration.

- [ ] **Step 5: Update the cognitive-context builder to compose structured fields and skip low-value HOLD/no-change entries.**

```python
def get_cognitive_context(self, max_age_minutes: int = 180) -> str:
    frontal = self.state.get("frontalLobe") or _default_frontal_lobe()
    portfolio_health = self.state.get("portfolioHealth") or _default_portfolio_health()
    market = self.get_market_regime(max_age_minutes=max_age_minutes)
    watchpoints_text = "\n".join(f"  - {item}" for item in _coerce_text_items(market.get("watchpoints"))) or "  - 無"
    reasons_text = "\n".join(f"  - {item}" for item in _coerce_text_items(market.get("reasons"))[:5]) or "  - 無"
    recent_commits = [
        commit for commit in reversed(self.commits)
        if "RISK HOLD" not in str(commit.get("summary") or "")
        and str(commit.get("source") or "") != "portfolio_review"
    ][:5]
    recent_lines = "\n".join(f"  - {commit['summary']}" for commit in recent_commits) or "  - 無近期重大變化"
    return (
        "\n\n## Current Brain State\n"
        f"- Emotion: {self.state['emotion']}\n"
        f"- Market View: {frontal.get('marketView') or '空白 (首次運行)'}\n"
        f"- Core Levels: {', '.join(frontal.get('coreLevels') or []) or '無'}\n"
        f"- Portfolio Health: {portfolio_health.get('summary') or '無'}\n"
        f"- Next Round: {frontal.get('nextRound') or '無'}\n"
        f"- Context Note: {frontal.get('contextNote') or '無'}\n"
        "\n## Persistent Macro / Market Regime\n"
        f"- Regime: {market.get('state') or '未初始化'}\n"
        f"- Risk Score: {market.get('riskScore') if market.get('riskScore') is not None else 'N/A'}\n"
        f"- Summary: {market.get('summary') or '尚未建立'}\n"
        f"- Watchpoints:\n{watchpoints_text}\n"
        f"- Reasons:\n{reasons_text}\n"
        "\n## Recent Meaningful Commits\n"
        f"{recent_lines}\n"
    )
```

- [ ] **Step 6: Reset `.brain/commit.json` to the new default shape while preserving the current market regime payload.**

Run after code is in place:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python - <<'PY'
import json
import engine_memory as memory
brain = memory.Brain()
preserved_market_regime = dict(brain.state.get("marketRegime") or {})
brain.reset_brain_state(preserve_market_regime=True)
assert brain.state["marketRegime"] == preserved_market_regime
print(json.dumps(brain.get_brain_snapshot(max_age_minutes=999999)["state"], ensure_ascii=False, indent=2))
PY
```

Expected: `frontalLobe` and `portfolioHealth` are structured JSON objects, `marketRegime` is preserved, and `commits` is empty.

- [ ] **Step 7: Re-run the memory/runtime tests and commit the layer.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest test_brain_memory.py test_phase_lifecycle_refactor.py check_followup_proposals.py
```

Commit:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add engine_memory.py engine_portfolio.py src/agent.py test_brain_memory.py test_phase_lifecycle_refactor.py \
  check_followup_proposals.py .brain/commit.json && \
git commit -m "refactor: structure brain state and trim noisy commits"
```

---

### Task 3: Replace multiplicative risk scoring with additive `riskScore`

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_risk.py`
- Modify: `/home/margincaller/MarginCall_2X/check_risk_quality_refactor.py`
- Test: `/home/margincaller/MarginCall_2X/check_risk_quality_refactor.py`

- [ ] **Step 1: Add failing tests that assert additive components instead of `riskMultiplier` math.**

```python
def test_additive_risk_score_caps_at_100_and_flags_danger_at_60(self):
    snapshot = engine_risk._finalize_risk_snapshot(
        base_score=118,
        active_components={"volatility_break": 18, "gamma_flip_risk": 14, "tail_risk": 10},
        dix_support_active=False,
    )
    self.assertEqual(snapshot["riskScore"], 100)
    self.assertEqual(snapshot["state"], "💀 系統風險")
    self.assertEqual(snapshot["riskComponents"]["volatility_break"], 18)


def test_dix_support_remains_a_fixed_negative_offset(self):
    snapshot = engine_risk._finalize_risk_snapshot(
        base_score=44,
        active_components={"volatility": 18},
        dix_support_active=True,
    )
    self.assertEqual(snapshot["scoreAdjustments"]["dixSupport"], -engine_risk.DIX_SUPPORT_OFFSET_POINTS)
    self.assertEqual(snapshot["riskScore"], 32)
```

- [ ] **Step 2: Run the risk-focused checks to see them fail.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest check_risk_quality_refactor.py
```

Expected: FAIL because `_finalize_risk_snapshot()` and `riskComponents` do not exist yet.

- [ ] **Step 3: Implement additive scoring with explicit components and 60+ danger threshold.**

```python
# engine_risk.py
RISK_COMPONENT_POINTS = {
    "yield_curve_inversion": 8,
    "high_rates": 5,
    "liquidity_shock": 16,
    "volatility_break": 18,
    "tail_risk": 10,
    "weak_breadth": 6,
    "negative_sentiment": 7,
    "gamma_flip_risk": 5,
    "below_10ma": 4,
    "below_20ma": 9,
    "below_200ma": 18,
}


def _build_global_risk_summary(score: int, state: str, reasons) -> str:
    if score >= 60:
        lead = "系統風險進入高警戒，先以防守和流動性管理為優先。"
    elif score >= 45:
        lead = "市場進入警戒帶，偏向控槓桿、降追價、等確認。"
    elif score >= 30:
        lead = "市場處於整理盤，適合等待更清楚的方向再擴大部位。"
    else:
        lead = "市場仍維持偏多結構，但要留意短線波動升溫。"
    top_reasons = "；".join(reasons[:3]) if reasons else "目前主要風險指標穩定。"
    return f"{lead} 當前 regime：{state}，風險分數 {score}。核心觀察：{top_reasons}"


def _finalize_risk_snapshot(base_score: int, active_components: Dict[str, int], dix_support_active: bool) -> Dict[str, Any]:
    gross_score = min(100, max(0, int(base_score)))
    dix_offset = -DIX_SUPPORT_OFFSET_POINTS if dix_support_active and gross_score > 0 else 0
    score = max(0, min(100, gross_score + dix_offset))
    state = "🟢 多頭" if score < 25 else "🟡 整理" if score < 45 else "🔴 警戒" if score < 60 else "💀 系統風險"
    return {
        "grossRiskScore": gross_score,
        "riskScore": score,
        "state": state,
        "riskComponents": active_components,
        "scoreAdjustments": {"dixSupport": dix_offset},
    }


def _build_global_risk_snapshot() -> Dict[str, Any]:
    components: Dict[str, int] = {}
    reasons = []
    if yc is not None and yc < 0:
        components["yield_curve_inversion"] = RISK_COMPONENT_POINTS["yield_curve_inversion"]
        reasons.append(f"⚠️ 殖利率曲線倒掛 ({yc:.2f}) - 衰退隱憂")
    if latest.get('VIX_Z', 0) > 2.0 or final_gex < 0:
        components["volatility_break"] = RISK_COMPONENT_POINTS["volatility_break"]
        reasons.append(f"🔴 波動率失控 / 負 Gamma ({final_gex:.2f}B)")
    if latest.get('SKEW_PR', 0) > 0.90:
        components["tail_risk"] = RISK_COMPONENT_POINTS["tail_risk"]
        reasons.append("🟠 尾部風險升溫")
    if ma200 > 0 and spx < ma200:
        components["below_200ma"] = RISK_COMPONENT_POINTS["below_200ma"]
        reasons.append("🚨 [Trigger] 熊市區間：跌破 200MA 均線！")
    base_score = sum(components.values())
    finalized = _finalize_risk_snapshot(base_score, components, dix_support_active=dix_support_active)
    return {
        "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        **finalized,
        "summary": _build_global_risk_summary(finalized["riskScore"], finalized["state"], reasons),
        "reasons": reasons or ["🟢 指標目前健康"],
        "signals": {
            "yieldCurve10Y2Y": _safe_float(yc, 3),
            "dixPr": _safe_float(latest.get('dix_PR', 0), 2),
            "gexBillions": _safe_float(final_gex, 2),
            "spx": _safe_float(spx, 1),
            "spx20Ma": _safe_float(ma20, 1),
            "spx200Ma": _safe_float(ma200, 1),
            "vixZ": _safe_float(latest.get('VIX_Z', 0), 2),
        },
    }
```

- [ ] **Step 4: Update the risk-check expectations to assert components, not multipliers.**

```python
self.assertIn("volatility_break", snapshot["riskComponents"])
self.assertNotIn("riskMultiplier", snapshot)
self.assertGreaterEqual(snapshot["riskScore"], 60)
```

- [ ] **Step 5: Re-run the risk checks and commit the layer.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest check_risk_quality_refactor.py
```

Commit:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add engine_risk.py check_risk_quality_refactor.py && \
git commit -m "refactor: switch global risk snapshot to additive scoring"
```

---

### Task 4: Decouple alpha quality from market risk and expose independent NLP dimensions

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/nlp_worker.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_router.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_market.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Modify: `/home/margincaller/MarginCall_2X/src/agent.py`
- Modify: `/home/margincaller/MarginCall_2X/check_alpha_signal_pipeline.py`
- Modify: `/home/margincaller/MarginCall_2X/check_risk_overlays.py`
- Modify: `/home/margincaller/MarginCall_2X/check_candidate_constructor.py`
- Modify: `/home/margincaller/MarginCall_2X/check_followup_proposals.py`
- Modify: `/home/margincaller/MarginCall_2X/check_technical_signal_upgrades.py`
- Test: `/home/margincaller/MarginCall_2X/check_alpha_signal_pipeline.py`, `/home/margincaller/MarginCall_2X/check_risk_overlays.py`

- [ ] **Step 1: Write failing tests for IC-only alpha scaling and independent dimensions.**

```python
def test_fetch_strat_data_exposes_independent_alpha_dimensions(self):
    with patch.object(engine_router, "fetch_nlp_alpha", return_value={
        "alpha_sec": -0.8,
        "alpha_macro": -0.3,
        "alpha_retail": 0.4,
        "signal_pack": {"divergence": "無"},
    }):
        data = engine_router.fetch_strat_data("TEST")
    self.assertEqual(data["leading_indicators"]["alpha_sec"], -0.8)
    self.assertEqual(data["leading_indicators"]["alpha_macro"], -0.3)
    self.assertEqual(data["leading_indicators"]["alpha_retail"], 0.4)


def test_alpha_governor_uses_only_ic_multiplier(self):
    overlay = engine_router._build_alpha_confidence_overlay(
        "TEST",
        {"alpha_sec": 0.6, "alpha_macro": 0.2, "alpha_retail": -0.1},
        risk_snapshot={"state": "💀 系統風險", "riskScore": 75},
        portfolio_overlay={"size_multiplier": 0.25},
        ic_payload={"signal_quality": "weak", "directionality": "positive", "ic_rolling_mean": 0.03},
    )
    self.assertEqual(overlay["combined_multiplier"], 0.75)
    self.assertNotIn("regime_multiplier", overlay)
    self.assertNotIn("drawdown_multiplier", overlay)
```

- [ ] **Step 2: Run the alpha/risk overlay tests and confirm they fail.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest check_alpha_signal_pipeline.py check_risk_overlays.py
```

Expected: FAIL because the runtime still expects composite `nlp_alpha` and regime/drawdown multipliers.

- [ ] **Step 3: Make `nlp_worker.py` store/report separate dimensions as the canonical signal output.**

```python
# nlp_worker.py
def _build_signal_pack(
    *,
    sec_dir: str,
    a_sec: float,
    sec_detail: list[str],
    mac_dir: str,
    a_mac: float,
    macro_detail: list[str],
    ret_dir: str,
    a_retail: float,
    retail_detail: list[str],
    divergence_alert: str,
    nuclear_confirmed: bool,
    groups: dict,
    effective_counts: dict,
    alpha_sec: float,
    alpha_macro: float,
    alpha_retail: float,
    must_mention_events: list[str],
) -> dict:
    return {
        "sec_stance": sec_dir,
        "sec_detail": sec_detail,
        "macro_stance": mac_dir,
        "macro_detail": macro_detail,
        "retail_stance": ret_dir,
        "retail_detail": retail_detail,
        "alpha_sec": round(alpha_sec, 4),
        "alpha_macro": round(alpha_macro, 4),
        "alpha_retail": round(alpha_retail, 4),
        "divergence": divergence_alert,
        "nuclear_alert": nuclear_confirmed,
        "must_mention_events": must_mention_events,
        "source_counts": effective_counts,
    }


signal_pack = _build_signal_pack(
    sec_dir=sec_dir,
    a_sec=a_sec,
    sec_detail=categorized_tags["SEC"],
    mac_dir=mac_dir,
    a_mac=a_mac,
    macro_detail=categorized_tags["Macro"],
    ret_dir=ret_dir,
    a_retail=a_retail,
    retail_detail=categorized_tags["Retail"],
    divergence_alert=divergence_alert,
    nuclear_confirmed=nuclear_confirmed,
    groups=groups,
    effective_counts=effective_counts,
    alpha_sec=a_sec,
    alpha_macro=a_mac,
    alpha_retail=a_retail,
    must_mention_events=macro_selection["must_mention_events"],
)
save_to_db(stock, nlp_alpha, a_retail, a_mac, a_sec, total, storage_payload, "TRINITY_V2")
```

Compatibility rule: keep persisting raw/composite `nlp_alpha` for audit and historical research, but treat it as deprecated archival data. New decision paths must read `alpha_sec`, `alpha_macro`, and `alpha_retail` first.

- [ ] **Step 4: Make `engine_router.py` return dimension-first payloads and an IC-only overlay.**

```python
def fetch_nlp_alpha(symbol: str) -> dict:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT nlp_alpha, alpha_retail, alpha_macro, alpha_official, summary_text, timestamp
                FROM nlp_insights WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1
                """,
                (symbol,),
            )
            row = cursor.fetchone()
        finally:
            conn.close()
    signal_pack, semantic_summary = _decode_nlp_summary_payload(row[4])
    return {
        "nlp_alpha": _safe_round(row[0], 4),
        "alpha_sec": _safe_round(row[3], 4),
        "alpha_macro": _safe_round(row[2], 4),
        "alpha_retail": _safe_round(row[1], 4),
        "alpha_dimensions": {
            "sec": _safe_round(row[3], 4),
            "macro": _safe_round(row[2], 4),
            "retail": _safe_round(row[1], 4),
        },
        "signal_pack": signal_pack,
        "semantic_summary": semantic_summary,
        "timestamp": row[5],
        "legacy_nlp_alpha": _safe_round(row[0], 4),
    }


def _build_alpha_confidence_overlay(
    symbol: str,
    nlp_data: dict,
    risk_snapshot: dict | None = None,
    portfolio_overlay: dict | None = None,
    ic_payload: dict | None = None,
) -> dict:
    ic_multiplier = {"strong": 1.0, "weak": 0.75, "noise": 0.55, "unknown": 0.9}.get(ic_quality, 0.9)
    adjusted_dimensions = {
        key: round(float(value) * ic_multiplier, 4)
        for key, value in {
            "alpha_sec": nlp_data.get("alpha_sec"),
            "alpha_macro": nlp_data.get("alpha_macro"),
            "alpha_retail": nlp_data.get("alpha_retail"),
        }.items()
        if isinstance(value, (int, float))
    }
    return {
        "combined_multiplier": round(ic_multiplier, 4),
        "adjusted_dimensions": adjusted_dimensions,
        "ic_quality": ic_quality,
        "ic_rolling_mean": _safe_round(ic_mean, 4),
        "directionality": directionality,
        "summary": f"IC-only governor x{ic_multiplier:.2f}",
        "reasons": reasons,
    }
```

- [ ] **Step 5: Update downstream consumers so risk stays in `riskScore`, not inside alpha math.**

```python
# engine_portfolio.py
def _fetch_sync_nlp_dimensions(symbol: str, lookup_symbol: str) -> Dict[str, float | None]:
    candidates = [normalize_ticker(symbol), lookup_symbol, lookup_symbol.replace(".TW", "").replace(".TWO", "") if lookup_symbol else None]
    for candidate in [item for item in candidates if item]:
        payload = router.fetch_nlp_alpha(candidate)
        if not payload.get("error"):
            return {
                "alpha_sec": payload.get("alpha_sec"),
                "alpha_macro": payload.get("alpha_macro"),
                "alpha_retail": payload.get("alpha_retail"),
            }
    return {"alpha_sec": None, "alpha_macro": None, "alpha_retail": None}


# engine_router.py
def fetch_strat_data(symbol: str, *, risk_snapshot: dict | None = None, portfolio_overlay: dict | None = None, alpha_ic_payload: dict | None = None) -> dict:
    nlp_data = fetch_nlp_alpha(symbol)
    risk_snapshot = risk_snapshot or risk.get_global_risk_snapshot()
    portfolio_overlay = portfolio_overlay or engine_portfolio.compute_portfolio_risk_overlay()
    alpha_overlay = _build_alpha_confidence_overlay(
        symbol,
        nlp_data,
        risk_snapshot=risk_snapshot,
        portfolio_overlay=portfolio_overlay,
        ic_payload=alpha_ic_payload,
    )
    data["leading_indicators"].update(
        {
            "alpha_sec": alpha_overlay["adjusted_dimensions"].get("alpha_sec"),
            "alpha_macro": alpha_overlay["adjusted_dimensions"].get("alpha_macro"),
            "alpha_retail": alpha_overlay["adjusted_dimensions"].get("alpha_retail"),
            "alpha_governor": alpha_overlay.get("summary"),
            "risk_score": risk_snapshot.get("riskScore"),
            "risk_state": risk_snapshot.get("state"),
        }
    )


# engine_market.py
def _calibrate_candidate_forecast(row: dict, risk_state: str, portfolio_overlay: dict) -> dict:
    risk_score = float(row.get("risk_score") or 0.0)
    alpha_sec = float(row.get("alpha_sec") or 0.0)
    alpha_macro = float(row.get("alpha_macro") or 0.0)
    alpha_retail = float(row.get("alpha_retail") or 0.0)
    alpha_strength = max(abs(alpha_sec), abs(alpha_macro), abs(alpha_retail))
    confidence = 0.35 + (0.35 * min(1.0, alpha_strength))
    if alpha_sec >= 0 and risk_score >= 60:
        confidence -= 0.18
    elif alpha_sec >= 0 and risk_score >= 45:
        confidence -= 0.10
    forecast_confidence = round(float(max(0.1, min(0.95, confidence))), 4)


def compute_nlp_signal_ic(symbol: str, horizon_days: int = 5, lookback_signals: int = 120) -> dict:
    query = """
        SELECT timestamp, alpha_official, alpha_macro, alpha_retail, nlp_alpha
        FROM nlp_insights
        WHERE symbol IN ({placeholders})
        ORDER BY timestamp DESC
        LIMIT ?
    """
    signals["signal"] = signals["alpha_official"].fillna(signals["alpha_macro"]).fillna(signals["alpha_retail"]).fillna(signals["nlp_alpha"])
```

```python
# src/agent.py
nlp_block = f"""
- SEC Alpha: {leading.get('alpha_sec', 'N/A')}
- Macro Alpha: {leading.get('alpha_macro', 'N/A')}
- Retail Alpha: {leading.get('alpha_retail', 'N/A')}
- Alpha Governor: {alpha_overlay.get('summary', 'N/A')}
- Risk State: {portfolio_overlay.get('risk_state', 'N/A')}
""".strip()
```

Also update `check_candidate_constructor.py`, `check_followup_proposals.py`, and `check_technical_signal_upgrades.py` in this same task so repo checks no longer assert `riskMultiplier` or composite `nlp_alpha` as the live decision input.

- [ ] **Step 6: Re-run the alpha/risk overlay checks and commit the layer.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest check_alpha_signal_pipeline.py check_risk_overlays.py
```

Commit:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add nlp_worker.py engine_router.py engine_market.py engine_portfolio.py src/agent.py \
  check_alpha_signal_pipeline.py check_risk_overlays.py check_candidate_constructor.py \
  check_followup_proposals.py check_technical_signal_upgrades.py && \
git commit -m "refactor: decouple alpha dimensions from risk scaling"
```

---

### Task 5: Build trade-outcome checkpoints and weekly attribution

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/.gitignore`
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Create: `/home/margincaller/MarginCall_2X/engine_journal.py`
- Create: `/home/margincaller/MarginCall_2X/check_trade_journal.py`
- Test: `/home/margincaller/MarginCall_2X/check_trade_journal.py`

- [ ] **Step 1: Write failing tests for T+5 / T+20 queueing, checkpoint settlement, and weekly attribution output.**

```python
def test_trade_log_insert_enqueues_t5_and_t20_checkpoints(self):
    trade_log_id = engine_portfolio.record_manual_trade_for_test(
        symbol="MRVL",
        action="buy",
        price=85.2,
        shares=30,
        decision_snapshot={"risk_state": "🟡 整理", "alpha_sec": 0.3},
        timestamp="2026-04-01T14:30:00Z",
    )
    with database.locked_connection() as conn:
        rows = conn.execute(
            "SELECT horizon_label, due_at FROM trade_outcome_checkpoints WHERE trade_log_id = ? ORDER BY horizon_label",
            (trade_log_id,),
        ).fetchall()
    self.assertEqual([row[0] for row in rows], ["T+20", "T+5"])


def test_weekly_attribution_report_breaks_out_beta_sector_and_timing(self):
    report = engine_journal.build_weekly_attribution_report(as_of="2026-04-26T12:00:00Z")
    self.assertIn("Beta 貢獻", report)
    self.assertIn("Sector 貢獻", report)
    self.assertIn("選股 Alpha", report)
    self.assertIn("Timing", report)


def test_settle_due_trade_outcomes_records_returns_and_excess(self):
    trade_log_id = engine_portfolio.record_manual_trade_for_test(
        symbol="MRVL",
        action="buy",
        price=100.0,
        shares=10,
        decision_snapshot={"risk_state": "🟡 整理", "alpha_sec": 0.2},
        timestamp="2026-04-01T14:30:00Z",
    )
    with patch.object(engine_journal, "_load_price_on_or_after", side_effect=[110.0, 102.0, 100.0, 101.0, 103.0]), patch.object(
        engine_journal, "_lookup_sector", return_value="Software"
    ):
        result = engine_journal.settle_due_trade_outcomes(as_of="2026-04-10T00:00:00Z")
    self.assertEqual(result["processed"], 1)
    with database.locked_connection() as conn:
        row = conn.execute(
            "SELECT return_pct, benchmark_return_pct, excess_return_pct FROM trade_outcome_checkpoints WHERE trade_log_id = ? AND horizon_label = 'T+5'",
            (trade_log_id,),
        ).fetchone()
    self.assertEqual(row, (0.1, 0.02, 0.08))
```

- [ ] **Step 2: Run the journal tests and verify they fail before implementation.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest check_trade_journal.py
```

Expected: FAIL because `trade_outcome_checkpoints`, `engine_journal.py`, and helper entry points do not exist yet.

- [ ] **Step 3: Extend the trade schema and enrich decision snapshots with the fields required for attribution.**

```python
# .gitignore
!check_trade_journal.py
!check_scenario_engine.py
```

```python
# engine_portfolio.py
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS trade_outcome_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_log_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        horizon_label TEXT NOT NULL,
        due_at TEXT NOT NULL,
        observed_at TEXT,
        price_at_due REAL,
        return_pct REAL,
        benchmark_return_pct REAL,
        excess_return_pct REAL,
        evaluation_json TEXT,
        UNIQUE(trade_log_id, horizon_label),
        FOREIGN KEY(trade_log_id) REFERENCES trade_log(id) ON DELETE CASCADE
    )
    """
)


def record_manual_trade_for_test(
    *, symbol: str, action: str, price: float, shares: float, decision_snapshot: Dict[str, Any], timestamp: str
) -> int:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            trade_log_id = _record_trade_log(
                cursor,
                symbol=normalize_ticker(symbol),
                action=action,
                price=price,
                shares=shares,
                decision_snapshot=decision_snapshot,
            )
            cursor.execute("UPDATE trade_log SET timestamp = ? WHERE id = ?", (timestamp, trade_log_id))
            conn.commit()
            committed_trade_log_id = trade_log_id
        finally:
            conn.close()
    import engine_journal as journal
    journal.enqueue_trade_outcome_checkpoints(committed_trade_log_id, normalize_ticker(symbol), timestamp)
    return committed_trade_log_id


def _maybe_enqueue_trade_outcomes(trade_log_id: int, symbol: str, action: str, trade_timestamp: str) -> None:
    if action not in {"buy", "sync_buy"}:
        return
    import engine_journal as journal
    journal.enqueue_trade_outcome_checkpoints(trade_log_id, normalize_ticker(symbol), trade_timestamp)


def execute_position_update(symbol: str, action: str, price: float, shares: float):
    decision_snapshot = _build_sync_decision_snapshot(symbol)
    trade_timestamp = _utc_now_iso()
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            trade_log_id = _record_trade_log(
                cursor,
                symbol=normalize_ticker(symbol),
                action=action,
                price=price,
                shares=shares,
                decision_snapshot=decision_snapshot,
            )
            conn.commit()
        finally:
            conn.close()
    _maybe_enqueue_trade_outcomes(trade_log_id, symbol, action, trade_timestamp)
    return f"ok:{trade_log_id}"


def sync_fubon_portfolio_state(source: str = "scheduler", sync_memory: bool = False) -> Dict[str, Any]:
    created_trade_logs: List[tuple[int, str, str]] = []
    trade_timestamp = _utc_now_iso()
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            trade_log_id = _record_trade_log(
                cursor,
                symbol=normalized,
                action="sync_buy",
                price=float(data["cost_price"]),
                shares=float(data["today_qty"]),
                decision_snapshot=decision_cache.get(normalize_ticker(normalized)),
            )
            created_trade_logs.append((trade_log_id, normalized, "sync_buy"))
            conn.commit()
        finally:
            conn.close()
    # enqueue only after the DB lock is released
    for trade_log_id, symbol, action in created_trade_logs:
        _maybe_enqueue_trade_outcomes(trade_log_id, symbol, action, trade_timestamp)
    return result


def _build_sync_decision_snapshot(symbol: str) -> Dict[str, Any]:
    lookup_symbol = _resolve_sync_lookup_symbol(symbol)
    payload = router.fetch_nlp_alpha(lookup_symbol or symbol)
    overlay = compute_portfolio_risk_overlay()
    snapshot["alpha_sec"] = payload.get("alpha_sec")
    snapshot["alpha_macro"] = payload.get("alpha_macro")
    snapshot["alpha_retail"] = payload.get("alpha_retail")
    snapshot["portfolio_gross_scale"] = overlay.get("recommended_gross_scale")
    snapshot["portfolio_trade_mode"] = overlay.get("trade_mode_label")
    snapshot["sector_exposure"] = _build_sector_exposure_snapshot()
    return snapshot


def _build_sector_exposure_snapshot() -> Dict[str, float]:
    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    total_value = sum(float(item.get("market_value_twd") or 0.0) for item in snapshots) or 1.0
    exposure: Dict[str, float] = {}
    for item in snapshots:
        sector = str(item.get("sector") or "Unknown")
        exposure[sector] = exposure.get(sector, 0.0) + (float(item.get("market_value_twd") or 0.0) / total_value)
    return {key: round(value, 4) for key, value in sorted(exposure.items())}


def get_portfolio_holdings_snapshot() -> Dict[str, float]:
    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    total_value = sum(float(item.get("market_value_twd") or 0.0) for item in snapshots) or 1.0
    return {
        normalize_ticker(str(item["symbol"])): round(float(item.get("market_value_twd") or 0.0) / total_value, 6)
        for item in snapshots
        if float(item.get("market_value_twd") or 0.0) > 0
    }


def get_portfolio_nav_snapshot() -> Dict[str, float]:
    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    nav_twd = sum(float(item.get("market_value_twd") or 0.0) for item in snapshots)
    return {"nav_twd": round(nav_twd, 2)}
```

- [ ] **Step 4: Create `engine_journal.py` for checkpoint creation/settlement and weekly attribution.**

```python
# engine_journal.py
CHECKPOINT_HORIZONS = (("T+5", 5), ("T+20", 20))  # trading days


def _shift_trading_days(base_ts: pd.Timestamp, days: int) -> pd.Timestamp:
    current = pd.Timestamp(base_ts).normalize()
    remaining = days
    while remaining > 0:
        current += pd.Timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current


def enqueue_trade_outcome_checkpoints(trade_log_id: int, symbol: str, trade_timestamp: str) -> None:
    base_ts = pd.Timestamp(trade_timestamp, tz="UTC")
    with database.locked_connection() as conn:
        for label, days in CHECKPOINT_HORIZONS:
            due_at = _shift_trading_days(base_ts, days).strftime("%Y-%m-%d")
            conn.execute(
                """
                INSERT OR IGNORE INTO trade_outcome_checkpoints (trade_log_id, symbol, horizon_label, due_at)
                VALUES (?, ?, ?, ?)
                """,
                (trade_log_id, symbol, label, due_at),
            )
        conn.commit()


def _load_trade_entry_context(trade_meta: Dict[str, Any]) -> tuple[float, str]:
    entry_price = float(trade_meta.get("price") or 0.0)
    return entry_price, "SPY"


def _load_trade_entry_benchmark_price(trade_meta: Dict[str, Any], benchmark_symbol: str) -> float:
    trade_timestamp = trade_meta.get("timestamp") or pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return _load_price_on_or_after(benchmark_symbol, trade_timestamp)


def _load_price_on_or_after(symbol: str, timestamp_or_date: str) -> float:
    history = get_ticker(symbol).history(start=str(timestamp_or_date)[:10], period="7d", interval="1d")
    return float(history["Close"].dropna().iloc[0])


def _estimate_sector_excess_return(snapshot_json: str | None, symbol: str, due_at: str, return_pct: float) -> float:
    snapshot = json.loads(snapshot_json) if snapshot_json else {}
    sector_exposure = snapshot.get("sector_exposure") or {}
    primary_sector = max(sector_exposure, key=sector_exposure.get) if sector_exposure else _lookup_sector(symbol)
    sector_proxy = {"Semiconductors": "SOXX", "Software": "IGV"}.get(primary_sector, "SPY")
    entry_sector_price = _load_price_on_or_after(sector_proxy, snapshot.get("captured_at") or due_at)
    due_sector_price = _load_price_on_or_after(sector_proxy, due_at)
    sector_return_pct = ((due_sector_price / entry_sector_price) - 1.0) if entry_sector_price else 0.0
    return round(return_pct - sector_return_pct, 4)


def _lookup_sector(symbol: str) -> str:
    return market.get_asset_profile(symbol).get("sector", "Unknown")


def _evaluate_trade_outcome(trade_meta: Dict[str, Any], snapshot_json: str | None, symbol: str, due_at: str) -> Dict[str, Any]:
    entry_price, benchmark_symbol = _load_trade_entry_context(trade_meta)
    price_at_due = _load_price_on_or_after(symbol, due_at)
    benchmark_price = _load_price_on_or_after(benchmark_symbol, due_at)
    entry_benchmark_price = _load_trade_entry_benchmark_price(trade_meta, benchmark_symbol)
    return_pct = ((price_at_due / entry_price) - 1.0) if entry_price else 0.0
    benchmark_return_pct = ((benchmark_price / entry_benchmark_price) - 1.0) if entry_benchmark_price else 0.0
    sector_excess_return_pct = _estimate_sector_excess_return(snapshot_json, symbol, due_at, return_pct)
    return {
        "price_at_due": round(price_at_due, 4),
        "return_pct": round(return_pct, 4),
        "benchmark_return_pct": round(benchmark_return_pct, 4),
        "excess_return_pct": round(return_pct - benchmark_return_pct, 4),
        "sector_excess_return_pct": sector_excess_return_pct,
    }


def settle_due_trade_outcomes(as_of: str | None = None) -> Dict[str, Any]:
    cutoff = pd.Timestamp(as_of or pd.Timestamp.utcnow(), tz="UTC")
    processed = 0
    with database.locked_connection() as conn:
        due_rows = conn.execute(
            """
            SELECT id, trade_log_id, symbol, horizon_label, due_at
            FROM trade_outcome_checkpoints
            WHERE observed_at IS NULL AND due_at <= ?
            ORDER BY due_at, id
            """,
            (cutoff.strftime("%Y-%m-%d"),),
        ).fetchall()
    evaluated_rows = []
    with database.locked_connection() as conn:
        trade_rows = {
            row[0]: {"decision_snapshot": row[1], "price": row[2], "timestamp": row[3]}
            for row in conn.execute(
                "SELECT id, decision_snapshot, price, timestamp FROM trade_log WHERE id IN ({})".format(",".join("?" for _ in due_rows)),
                [row[1] for row in due_rows],
            ).fetchall()
        } if due_rows else {}
    for checkpoint_id, trade_log_id, symbol, horizon_label, due_at in due_rows:
        trade_meta = trade_rows.get(trade_log_id, {})
        evaluation = _evaluate_trade_outcome(trade_meta, trade_meta.get("decision_snapshot"), symbol, due_at)
        evaluated_rows.append((checkpoint_id, evaluation))
    with database.locked_connection() as conn:
        for checkpoint_id, evaluation in evaluated_rows:
            conn.execute(
                """
                UPDATE trade_outcome_checkpoints
                SET observed_at = ?, price_at_due = ?, return_pct = ?, benchmark_return_pct = ?,
                    excess_return_pct = ?, evaluation_json = ?
                WHERE id = ?
                """,
                (
                    cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    evaluation["price_at_due"],
                    evaluation["return_pct"],
                    evaluation["benchmark_return_pct"],
                    evaluation["excess_return_pct"],
                    json.dumps(evaluation, ensure_ascii=False, sort_keys=True),
                    checkpoint_id,
                ),
            )
            processed += 1
        conn.commit()
    return {"processed": processed, "as_of": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")}


def _load_recent_checkpoint_frame(as_of: str | None = None) -> pd.DataFrame:
    cutoff = pd.Timestamp(as_of or pd.Timestamp.utcnow(), tz="UTC")
    with database.locked_connection() as conn:
        frame = pd.read_sql(
            """
            SELECT horizon_label, return_pct, benchmark_return_pct, excess_return_pct,
                   json_extract(evaluation_json, '$.sector_excess_return_pct') AS sector_excess_return_pct
            FROM trade_outcome_checkpoints
            WHERE observed_at IS NOT NULL AND observed_at >= ?
            """,
            conn,
            params=((cutoff - pd.Timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),),
        )
    return frame


def _compute_weekly_attribution_components(as_of: str | None = None) -> Dict[str, float]:
    checkpoint_df = _load_recent_checkpoint_frame(as_of=as_of)
    beta_contribution = float(checkpoint_df["benchmark_return_pct"].sum())
    sector_contribution = float(checkpoint_df["sector_excess_return_pct"].sum()) if "sector_excess_return_pct" in checkpoint_df else 0.0
    selection_contribution = float(checkpoint_df["excess_return_pct"].sum()) - sector_contribution
    timing_contribution = float((checkpoint_df["return_pct"] - checkpoint_df["benchmark_return_pct"]).clip(lower=-0.1, upper=0.1).sum())
    t5 = checkpoint_df[checkpoint_df["horizon_label"] == "T+5"]
    t20 = checkpoint_df[checkpoint_df["horizon_label"] == "T+20"]
    return {
        "beta_contribution": beta_contribution,
        "sector_contribution": sector_contribution,
        "selection_contribution": selection_contribution,
        "timing_contribution": timing_contribution,
        "t5_hit_rate": float((t5["excess_return_pct"] > 0).mean()) if not t5.empty else 0.0,
        "t20_hit_rate": float((t20["excess_return_pct"] > 0).mean()) if not t20.empty else 0.0,
    }


def build_weekly_attribution_report(as_of: str | None = None) -> str:
    totals = _compute_weekly_attribution_components(as_of=as_of)
    return (
        "【週報】\n"
        f"- Beta 貢獻: {totals['beta_contribution']:+,.0f}\n"
        f"- Sector 貢獻: {totals['sector_contribution']:+,.0f}\n"
        f"- 選股 Alpha: {totals['selection_contribution']:+,.0f}\n"
        f"- Timing: {totals['timing_contribution']:+,.0f}\n"
        f"- T+5 命中率: {totals['t5_hit_rate']:.1%}\n"
        f"- T+20 命中率: {totals['t20_hit_rate']:.1%}"
    )


@tool()
def get_trade_journal_weekly_report() -> str:
    return build_weekly_attribution_report()
```

- [ ] **Step 5: Re-run the journal tests and commit the layer.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest check_trade_journal.py check_quant_desk_upgrades.py
```

Commit:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add .gitignore engine_portfolio.py engine_journal.py check_trade_journal.py && \
git commit -m "feat: add trade outcome checkpoints and weekly attribution"
```

---

### Task 6: Add historical stress replay and A→B what-if simulation

**Files:**
- Create: `/home/margincaller/MarginCall_2X/engine_scenarios.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Create: `/home/margincaller/MarginCall_2X/check_scenario_engine.py`
- Test: `/home/margincaller/MarginCall_2X/check_scenario_engine.py`

- [ ] **Step 1: Write failing tests for stress replay and swap simulation.**

```python
def test_stress_test_replays_weighted_event_drawdown(self):
    with patch.object(engine_scenarios.engine_portfolio, "get_portfolio_holdings_snapshot", return_value={"AMD": 0.6, "AVGO": 0.4}), patch.object(
        engine_scenarios.engine_portfolio, "get_portfolio_nav_snapshot", return_value={"nav_twd": 1_000_000}
    ), patch.object(
        engine_scenarios, "_load_price_on_or_after", side_effect=[100.0, 88.0, 100.0, 93.0]
    ):
        report = engine_scenarios.run_portfolio_stress_test(event_key="carry_trade_2024_08")
    self.assertEqual(report["event_key"], "carry_trade_2024_08")
    self.assertAlmostEqual(report["portfolio_loss_pct"], -0.1, places=4)
    self.assertEqual(report["largest_detractors"][0]["symbol"], "AMD")


def test_swap_simulation_recalculates_beta_and_concentration(self):
    with patch.object(engine_scenarios.engine_portfolio, "get_portfolio_holdings_snapshot", return_value={"AMD": 0.5, "MSFT": 0.5}), patch.object(
        engine_scenarios.engine_portfolio, "compute_portfolio_beta_attribution", side_effect=[{"portfolio_beta": 0.92}, {"portfolio_beta": 0.98}]
    ), patch.object(
        engine_scenarios, "_lookup_sector", side_effect=lambda symbol: {"AMD": "Semiconductors", "AVGO": "Semiconductors", "MSFT": "Software"}[symbol]
    ):
        result = engine_scenarios.simulate_position_swap("AMD", "AVGO", replace_fraction=1.0)
    self.assertIn("current", result)
    self.assertIn("proposed", result)
    self.assertIn("delta", result)
    self.assertEqual(result["proposed"]["portfolio_beta"], 0.98)
```

- [ ] **Step 2: Run the scenario tests and confirm they fail first.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest check_scenario_engine.py
```

Expected: FAIL because `engine_scenarios.py` and the new tools do not exist yet.

- [ ] **Step 3: Implement deterministic historical event replay using fixed event windows.**

```python
# engine_scenarios.py
HISTORICAL_EVENTS = {
    "covid_2020_03": ("2020-02-19", "2020-03-23"),
    "rates_2022_01": ("2022-01-03", "2022-01-31"),
    "carry_trade_2024_08": ("2024-08-01", "2024-08-12"),
    "tariff_panic_2025_04": ("2025-04-01", "2025-04-18"),
}


def _load_price_on_or_after(symbol: str, timestamp_or_date: str) -> float:
    history = get_ticker(symbol).history(start=str(timestamp_or_date)[:10], period="7d", interval="1d")
    return float(history["Close"].dropna().iloc[0])


# In tests, always stub _load_price_on_or_after() or get_ticker() so replay math is deterministic and never depends on live network responses.


def run_portfolio_stress_test(event_key: str) -> Dict[str, Any]:
    start_date, end_date = HISTORICAL_EVENTS[event_key]
    holdings = engine_portfolio.get_portfolio_holdings_snapshot()
    weighted_loss_pct, weighted_loss_twd, detractors = _replay_event_window(holdings, start_date, end_date)
    hedge_note = "Consider XLP or TLT as a tail hedge if the modeled loss breaches the current defense threshold."
    return {
        "event_key": event_key,
        "portfolio_loss_pct": round(weighted_loss_pct, 4),
        "portfolio_loss_twd": round(weighted_loss_twd, 0),
        "largest_detractors": detractors[:5],
        "hedge_note": hedge_note,
    }


def _replay_event_window(holdings: Dict[str, float], start_date: str, end_date: str) -> tuple[float, float, list[dict[str, float]]]:
    detractors = []
    weighted_loss_pct = 0.0
    nav_twd = engine_portfolio.get_portfolio_nav_snapshot().get("nav_twd", 0.0)
    for symbol, weight in holdings.items():
        start_price = _load_price_on_or_after(symbol, start_date)
        end_price = _load_price_on_or_after(symbol, end_date)
        drawdown = ((end_price / start_price) - 1.0) if start_price else 0.0
        weighted_loss_pct += weight * drawdown
        detractors.append({"symbol": symbol, "drawdown_pct": round(drawdown, 4), "weight": round(weight, 4)})
    detractors.sort(key=lambda item: item["drawdown_pct"])
    return weighted_loss_pct, weighted_loss_pct * nav_twd, detractors


@tool()
def stress_test_portfolio(event_key: str = "carry_trade_2024_08") -> str:
    return json.dumps(run_portfolio_stress_test(event_key), ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Implement the A→B swap simulator using existing holdings + beta attribution helpers.**

```python
def _swap_holdings(holdings: Dict[str, float], from_symbol: str, to_symbol: str, replace_fraction: float) -> Dict[str, float]:
    proposed = dict(holdings)
    trimmed_weight = proposed.get(from_symbol, 0.0) * replace_fraction
    proposed[from_symbol] = max(0.0, proposed.get(from_symbol, 0.0) - trimmed_weight)
    proposed[to_symbol] = proposed.get(to_symbol, 0.0) + trimmed_weight
    return proposed


def _max_weight(holdings: Dict[str, float]) -> float:
    return max(holdings.values()) if holdings else 0.0


def _lookup_sector(symbol: str) -> str:
    return market.get_asset_profile(symbol).get("sector", "Unknown")


def _sector_weight(holdings: Dict[str, float], sector_name: str) -> float:
    return sum(weight for symbol, weight in holdings.items() if _lookup_sector(symbol) == sector_name)


def _summarize_holdings(holdings: Dict[str, float], portfolio_beta: float) -> Dict[str, float]:
    return {
        "portfolio_beta": round(portfolio_beta, 4),
        "max_position_weight": round(_max_weight(holdings), 4),
        "semi_exposure": round(_sector_weight(holdings, "Semiconductors"), 4),
    }


def simulate_position_swap(from_symbol: str, to_symbol: str, replace_fraction: float = 1.0) -> Dict[str, Any]:
    holdings = engine_portfolio.get_portfolio_holdings_snapshot()
    proposed = _swap_holdings(holdings, from_symbol, to_symbol, replace_fraction=replace_fraction)
    current_beta = engine_portfolio.compute_portfolio_beta_attribution(holdings)["portfolio_beta"]
    proposed_beta = engine_portfolio.compute_portfolio_beta_attribution(proposed)["portfolio_beta"]
    return {
        "current": _summarize_holdings(holdings, current_beta),
        "proposed": _summarize_holdings(proposed, proposed_beta),
        "delta": {
            "portfolio_beta": round(proposed_beta - current_beta, 4),
            "max_position_weight": round(_max_weight(proposed) - _max_weight(holdings), 4),
            "semi_exposure": round(_sector_weight(proposed, "Semiconductors") - _sector_weight(holdings, "Semiconductors"), 4),
        },
    }


@tool()
def simulate_position_swap_tool(from_symbol: str, to_symbol: str, replace_fraction: float = 1.0) -> str:
    return json.dumps(simulate_position_swap(from_symbol, to_symbol, replace_fraction=replace_fraction), ensure_ascii=False, indent=2)
```

- [ ] **Step 5: Re-run the scenario tests and commit the layer.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest check_scenario_engine.py
```

Commit:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add engine_scenarios.py engine_portfolio.py check_scenario_engine.py && \
git commit -m "feat: add portfolio stress replay and swap simulation"
```

---

### Task 7: Wire the new jobs/tools, run integration verification, merge to `main`, and push

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/src/bot.py`
- Modify: `/home/margincaller/MarginCall_2X/src/scheduler.py`
- Modify: `/home/margincaller/MarginCall_2X/test_refactor_runtime.py`
- Modify: `/home/margincaller/MarginCall_2X/test_phase_lifecycle_refactor.py`
- Test: focused runtime + full targeted suite

- [ ] **Step 1: Add scheduler hooks for due checkpoints and Sunday attribution.**

```python
# src/scheduler.py
def trade_outcome_checkpoint_job():
    import engine_journal as journal
    result = journal.settle_due_trade_outcomes()
    logger.info("📒 [TradeOutcomeJob] %s", result)
    return result


def weekly_trade_journal_job():
    import engine_journal as journal
    report = journal.build_weekly_attribution_report()
    logger.info("📊 [WeeklyAttribution] generated")
    return report


scheduler.add_job(
    trade_outcome_checkpoint_job,
    "cron",
    day_of_week="tue-sat",
    hour=5,
    minute=10,
    id="trade-outcome-checkpoint",
    replace_existing=True,
)
scheduler.add_job(
    weekly_trade_journal_job,
    "cron",
    day_of_week="sun",
    hour=9,
    minute=0,
    id="weekly-trade-journal",
    replace_existing=True,
)
```

- [ ] **Step 2: Import the new tool modules in `src/bot.py`, then register them in the runtime expectations.**

```python
# src/bot.py
import engine_journal  # noqa: F401
import engine_scenarios  # noqa: F401


# test_refactor_runtime.py
"engine_journal.py": {
    "get_trade_journal_weekly_report": "read",
},
"engine_scenarios.py": {
    "stress_test_portfolio": "read",
    "simulate_position_swap_tool": "read",
},


# test_phase_lifecycle_refactor.py
def test_scheduler_registers_trade_outcome_and_weekly_journal_jobs(self):
    fake_scheduler = MagicMock()
    with patch.object(scheduler_module, "BackgroundScheduler", return_value=fake_scheduler), patch.object(
        scheduler_module, "macro_brain_heartbeat", return_value={}
    ), patch.object(
        scheduler_module, "daily_portfolio_review", return_value={}
    ):
        scheduler_module.start_scheduler()
    added_ids = [call.kwargs["id"] for call in fake_scheduler.add_job.call_args_list]
    self.assertIn("trade-outcome-checkpoint", added_ids)
    self.assertIn("weekly-trade-journal", added_ids)


def test_bot_imports_new_tool_modules_before_tool_loading(self):
    bot_source = (ROOT / "src" / "bot.py").read_text(encoding="utf-8")
    self.assertIn("import engine_journal", bot_source)
    self.assertIn("import engine_scenarios", bot_source)
```

- [ ] **Step 3: Run the full targeted verification suite for memory, risk, alpha, journal, scenarios, and the existing trade-plan/briefing flow.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest \
  test_brain_memory.py \
  check_risk_quality_refactor.py \
  check_alpha_signal_pipeline.py \
  check_risk_overlays.py \
  check_followup_proposals.py \
  check_candidate_constructor.py \
  check_technical_signal_upgrades.py \
  check_quant_desk_upgrades.py \
  check_trade_journal.py \
  check_scenario_engine.py \
  test_phase_lifecycle_refactor.py \
  test_refactor_runtime.py && \
/home/margincaller/MarginCall_2X/venv/bin/python -m py_compile \
  engine_memory.py engine_risk.py engine_router.py engine_market.py engine_portfolio.py \
  engine_journal.py engine_scenarios.py nlp_worker.py src/agent.py src/bot.py src/scheduler.py
```

Expected: all targeted tests pass and the touched modules compile cleanly.

- [ ] **Step 4: Commit the final runtime/scheduler wiring layer.**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add src/bot.py src/scheduler.py test_refactor_runtime.py test_phase_lifecycle_refactor.py && \
git commit -m "feat: wire journal and scenario runtime hooks"
```

- [ ] **Step 5: Merge the feature into a clean `main` worktree, re-run the targeted suite on the merge commit, then push.**

Run:

```bash
cd /home/margincaller/MarginCall_2X && \
git fetch origin && \
git worktree add .worktrees/main-clean -b release-cognitive-risk origin/main && \
cd .worktrees/main-clean && \
git merge --no-ff feature/cognitive-risk-analytics -m "merge: cognitive risk analytics overhaul" && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest \
  test_brain_memory.py \
  check_risk_quality_refactor.py \
  check_alpha_signal_pipeline.py \
  check_risk_overlays.py \
  check_followup_proposals.py \
  check_candidate_constructor.py \
  check_technical_signal_upgrades.py \
  check_quant_desk_upgrades.py \
  check_trade_journal.py \
  check_scenario_engine.py \
  test_phase_lifecycle_refactor.py \
  test_refactor_runtime.py && \
/home/margincaller/MarginCall_2X/venv/bin/python -m py_compile \
  engine_memory.py engine_risk.py engine_router.py engine_market.py engine_portfolio.py \
  engine_journal.py engine_scenarios.py nlp_worker.py src/agent.py src/bot.py src/scheduler.py && \
git push origin HEAD:main
```

Expected: merge happens from a clean `main` tree without carrying the unrelated local edits from the primary checkout.
