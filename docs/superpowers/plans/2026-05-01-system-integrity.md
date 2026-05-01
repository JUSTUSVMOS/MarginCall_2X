# System Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden `engine_memory.py` so frontal-lobe writes skip redundant commits, market heartbeat refreshes stop emitting signal-only HOLD commits, persisted brain history is capped at 200 commits, and new brain state starts from a structured frontal-lobe template.

**Architecture:** Keep the current public memory tool surface intact and add internal guardrails inside `engine_memory.py`: a structured default frontal-lobe template, shared no-op detection for frontal-lobe writes, a market-regime commit gate that ignores signal-only churn, and centralized commit-history trimming. Drive every behavior through focused TDD in `test_brain_memory.py`, then run the documented memory-focused checks from the handoff docs.

**Tech Stack:** Python, `unittest`, `pathlib`, JSON persistence, existing `engine_memory.py` helpers and Brain class.

---

## Execution notes

- Run this work in a dedicated git worktree branched from current `main`.
- Touch only `/home/margincaller/MarginCall_2X/engine_memory.py` and `/home/margincaller/MarginCall_2X/test_brain_memory.py`.
- Do **not** rewrite any existing `.brain/commit.json` contents by hand; tests already redirect persistence into a temp directory.
- Follow focused validation from `/home/margincaller/agent/margincall-hygiene/references/dev-workflow.md` for `engine_memory.py`: `test_brain_memory.py` and `test_agent_runtime.py`.

## File structure

- **Modify:** `/home/margincaller/MarginCall_2X/engine_memory.py`
  - Add the structured frontal-lobe default template.
  - Add shared frontal-lobe no-op detection helpers.
  - Narrow market-regime commit detection to material fields.
  - Add commit-history trimming with `MAX_COMMITS = 200`.
- **Modify:** `/home/margincaller/MarginCall_2X/test_brain_memory.py`
  - Add regression coverage for default state, no-op frontal-lobe writes, signal-only heartbeat refreshes, and commit-history trimming.

---

### Task 1: Structured defaults and frontal-lobe no-op gates

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/test_brain_memory.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_memory.py`
- Test: `/home/margincaller/MarginCall_2X/test_brain_memory.py`

- [ ] **Step 1: Write the failing tests for structured defaults and redundant frontal-lobe writes**

```python
class BrainMemoryTests(unittest.TestCase):
    def test_default_state_starts_with_structured_frontal_lobe_template(self):
        brain = memory.Brain()

        frontal = brain.get_frontal_lobe()

        self.assertNotEqual(frontal, "")
        self.assertTrue(frontal.startswith("Market View: "))
        self.assertIn("Core Levels: ", frontal)
        self.assertIn("Portfolio Health: ", frontal)
        self.assertIn("Next Round: ", frontal)
        self.assertEqual(brain.commits, [])

    def test_update_lobe_section_skips_identical_normalized_content(self):
        brain = memory.Brain()
        self.assertTrue(brain.update_frontal_lobe(VALID_NOTE)["success"])
        commit_count = len(brain.commits)

        result = brain.update_lobe_section(
            "Market View",
            "  Neutral - CPI is the next catalyst while breadth remains mixed.  ",
            source="unit_test",
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["unchanged"])
        self.assertEqual(result["message"], "Frontal lobe section 'Market View' unchanged.")
        self.assertEqual(len(brain.commits), commit_count)
        self.assertEqual(brain.get_frontal_lobe(), VALID_NOTE)

    def test_update_frontal_lobe_skips_identical_normalized_content(self):
        brain = memory.Brain()
        first = brain.update_frontal_lobe(VALID_NOTE)
        self.assertTrue(first["success"])
        commit_count = len(brain.commits)

        second = brain.update_frontal_lobe(
            "Market View: Neutral - CPI is the next catalyst while breadth remains mixed.\n"
            "Core Levels: Watch SPX 5200 support and 5250 resistance.\n"
            "Portfolio Health: Keep leverage light until event risk clears.\n"
            "Next Round: If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure.\n"
        )

        self.assertTrue(second["success"])
        self.assertTrue(second["unchanged"])
        self.assertEqual(second["message"], "Frontal lobe unchanged; skipped commit.")
        self.assertEqual(len(brain.commits), commit_count)
        self.assertEqual(brain.get_frontal_lobe(), VALID_NOTE)

    def test_update_frontal_lobe_rejects_placeholder_quality_note(self):
        brain = memory.Brain()
        default_note = brain.get_frontal_lobe()
        result = brain.update_frontal_lobe(
            "Market View: 觀望 - no clear thesis, waiting for CPI.\n"
            "Core Levels: 尚未建立\n"
            "Portfolio Health: 暫無明確評估\n"
            "Next Round: waiting for confirmation."
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Rejected: content is too vague to persist.")
        self.assertEqual(brain.get_frontal_lobe(), default_note)
        self.assertEqual(brain.commits, [])
```

- [ ] **Step 2: Run the focused test selection and verify it fails**

Run:

```bash
cd /home/margincaller/MarginCall_2X && \
./venv/bin/python -m unittest \
  test_brain_memory.BrainMemoryTests.test_default_state_starts_with_structured_frontal_lobe_template \
  test_brain_memory.BrainMemoryTests.test_update_lobe_section_skips_identical_normalized_content \
  test_brain_memory.BrainMemoryTests.test_update_frontal_lobe_skips_identical_normalized_content \
  test_brain_memory.BrainMemoryTests.test_update_frontal_lobe_rejects_placeholder_quality_note
```

Expected:

- `test_default_state_starts_with_structured_frontal_lobe_template` fails because `brain.get_frontal_lobe()` is currently `""`
- the no-op tests fail because `update_lobe_section()` and `update_frontal_lobe()` still create commits and do not return `unchanged`
- `test_update_frontal_lobe_rejects_placeholder_quality_note` will fail until the assertion is updated away from `""`

- [ ] **Step 3: Implement the structured default template and shared no-op detection**

```python
# near the top-level constants in engine_memory.py
MAX_COMMITS = 200

DEFAULT_FRONTAL_LOBE_TEMPLATE = (
    "Market View: Neutral - No durable market thesis has been logged yet.\n"
    "Core Levels: No key support or resistance levels have been logged yet.\n"
    "Portfolio Health: Exposure review pending; keep sizing disciplined until a thesis is logged.\n"
    "Next Round: Do not add risk until the market view and levels are updated."
)


def _default_frontal_lobe() -> str:
    return DEFAULT_FRONTAL_LOBE_TEMPLATE


def _render_frontal_lobe_note(sections: Dict[str, str]) -> str:
    lines = [f"{field}: {sections.get(field, '')}" for field in FRONTAL_LOBE_FIELDS]
    if sections.get("Context Note"):
        lines.append(f"Context Note: {sections['Context Note']}")
    return "\n".join(lines)
```

```python
def _default_state() -> Dict[str, Any]:
    return {
        "frontalLobe": _default_frontal_lobe(),
        "emotion": "neutral",
        "marketRegime": _default_market_regime(),
        "heartbeat": _default_heartbeat()
    }
```

```python
class Brain:
    def _normalized_current_frontal_lobe(self) -> str:
        current_note = self.state.get("frontalLobe") or _default_frontal_lobe()
        return _render_frontal_lobe_note(_coerce_frontal_lobe_sections(current_note))

    def _frontal_lobe_write_is_unchanged(self, normalized_note: str) -> bool:
        return self._normalized_current_frontal_lobe() == normalized_note

    def update_lobe_section(self, section_name: str, new_content: str, source: str = "system") -> Dict[str, Any]:
        if section_name not in FRONTAL_LOBE_FIELDS:
            return {"success": False, "message": f"Invalid section: {section_name}"}

        current_note = self.state.get("frontalLobe") or _default_frontal_lobe()
        sections = _coerce_frontal_lobe_sections(current_note)
        sections[section_name] = new_content.strip()
        normalized_note = _render_frontal_lobe_note(sections)

        if self._frontal_lobe_write_is_unchanged(normalized_note):
            return {
                "success": True,
                "unchanged": True,
                "message": f"Frontal lobe section '{section_name}' unchanged."
            }

        self.state["frontalLobe"] = normalized_note
        summary = f"🧠 {section_name.upper()} AUTO-UPDATE: {_shorten(new_content, 80)}"
        delta_key = section_name.lower().replace(" ", "_")
        self._create_commit(
            "frontal_lobe_patch",
            summary,
            delta={delta_key: new_content.strip()},
            key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
            frontal_lobe_ref=self._build_frontal_lobe_ref(normalized_note),
            source=source
        )
        return {"success": True, "unchanged": False, "message": f"Frontal lobe section '{section_name}' updated."}

    def update_frontal_lobe(self, content: str) -> Dict[str, Any]:
        normalized_note = normalize_frontal_lobe_note(content)
        if self._is_placeholder_content(normalized_note):
            logger.warning("[Brain] Rejected placeholder-quality frontal lobe write.")
            return {"success": False, "message": "Rejected: content is too vague to persist."}

        if self._frontal_lobe_write_is_unchanged(normalized_note):
            return {"success": True, "unchanged": True, "message": "Frontal lobe unchanged; skipped commit."}

        snapshot_head = self.head
        self.state["frontalLobe"] = normalized_note
        sections = parse_frontal_lobe_note(normalized_note)
        summary = self._build_frontal_lobe_commit_summary(sections)
        committed = self._create_commit(
            "frontal_lobe",
            summary,
            delta={
                "market_view": sections.get("Market View", ""),
                "core_levels": sections.get("Core Levels", ""),
                "portfolio_health": sections.get("Portfolio Health", ""),
                "next_round": sections.get("Next Round", ""),
            },
            key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
            frontal_lobe_ref=self._build_frontal_lobe_ref(normalized_note),
            source="frontal_lobe_write",
            expected_head=snapshot_head
        )
        if not committed:
            self._load()
            return {"success": False, "message": "Rejected: concurrent frontal lobe update detected."}
        return {"success": True, "unchanged": False, "message": "Frontal lobe updated successfully"}
```

- [ ] **Step 4: Run the same focused tests and verify they pass**

Run:

```bash
cd /home/margincaller/MarginCall_2X && \
./venv/bin/python -m unittest \
  test_brain_memory.BrainMemoryTests.test_default_state_starts_with_structured_frontal_lobe_template \
  test_brain_memory.BrainMemoryTests.test_update_lobe_section_skips_identical_normalized_content \
  test_brain_memory.BrainMemoryTests.test_update_frontal_lobe_skips_identical_normalized_content \
  test_brain_memory.BrainMemoryTests.test_update_frontal_lobe_rejects_placeholder_quality_note
```

Expected: `Ran 4 tests` and `OK`

- [ ] **Step 5: Commit the structured-default/no-op change**

```bash
cd /home/margincaller/MarginCall_2X && \
git add test_brain_memory.py engine_memory.py && \
git commit -m "fix: harden frontal lobe no-op writes" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Market heartbeat no-change gating

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/test_brain_memory.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_memory.py`
- Test: `/home/margincaller/MarginCall_2X/test_brain_memory.py`

- [ ] **Step 1: Write the failing regression test for signal-only heartbeat refreshes**

```python
class BrainMemoryTests(unittest.TestCase):
    def test_update_market_regime_signal_only_refresh_updates_state_without_commit(self):
        brain = memory.Brain()
        first = brain.update_market_regime(
            summary="Macro backdrop unchanged; wait for confirmation.",
            regime="🟡 整理",
            risk_score=33,
            watchpoints=["SPX 20MA"],
            reasons=["Macro steady"],
            signals={"spx": 5200.4},
            source="unit_test",
            updated_at="2026-01-02T03:04:05Z",
        )
        first_change_at = brain.state["heartbeat"]["lastMacroChangeAt"]
        commit_count = len(brain.commits)

        second = brain.update_market_regime(
            summary="Macro backdrop unchanged; wait for confirmation.",
            regime="🟡 整理",
            risk_score=33,
            watchpoints=["SPX 20MA"],
            reasons=["Macro steady"],
            signals={"spx": 5198.8, "yieldCurve10Y2Y": -0.21},
            source="unit_test",
            updated_at="2026-01-02T03:09:05Z",
        )

        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        self.assertEqual(len(brain.commits), commit_count)
        self.assertEqual(brain.state["marketRegime"]["signals"]["spx"], 5198.8)
        self.assertEqual(brain.state["marketRegime"]["signals"]["yieldCurve10Y2Y"], -0.21)
        self.assertEqual(brain.state["heartbeat"]["lastSyncStatus"], "no_change")
        self.assertEqual(brain.state["heartbeat"]["lastMacroChangeAt"], first_change_at)
        self.assertEqual(brain.state["heartbeat"]["lastSyncMessage"], "Macro backdrop unchanged; wait for confirmation.")
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd /home/margincaller/MarginCall_2X && \
./venv/bin/python -m unittest \
  test_brain_memory.BrainMemoryTests.test_update_market_regime_signal_only_refresh_updates_state_without_commit
```

Expected: FAIL because `_market_regime_changed()` still treats `signals` as a commit-driving field, so `second["changed"]` is currently `True`

- [ ] **Step 3: Narrow market-regime change detection to material fields only**

```python
# near the other top-level constants in engine_memory.py
MARKET_REGIME_COMMIT_KEYS = ("summary", "state", "riskScore", "watchpoints", "reasons")
```

```python
class Brain:
    def _market_regime_changed(self, new_market: Dict[str, Any]) -> bool:
        current = self.state.get("marketRegime", _default_market_regime())
        return any(
            _to_comparable(current.get(key)) != _to_comparable(new_market.get(key))
            for key in MARKET_REGIME_COMMIT_KEYS
        )
```

```python
    def update_market_regime(
        self,
        summary: str,
        regime: Optional[str] = None,
        risk_score: Optional[int] = None,
        watchpoints: Optional[List[str]] = None,
        reasons: Optional[List[str]] = None,
        signals: Optional[Dict[str, Any]] = None,
        source: str = "manual",
        updated_at: Optional[str] = None
    ) -> Dict[str, Any]:
        previous_market = copy.deepcopy(self.state["marketRegime"])
        normalized_watchpoints = [item.strip() for item in _coerce_text_items(watchpoints, default="") if item and item.strip()]
        normalized_reasons = [item.strip() for item in _coerce_text_items(reasons, default="") if item and item.strip()]
        normalized_signals = _to_comparable(dict(signals)) if isinstance(signals, dict) else {}
        cleaned_market = {
            "summary": summary.strip(),
            "state": regime or self.state["marketRegime"].get("state") or "未初始化",
            "riskScore": risk_score if risk_score is not None else self.state["marketRegime"].get("riskScore"),
            "updatedAt": updated_at or _utc_now_iso(),
            "source": source,
            "watchpoints": normalized_watchpoints,
            "reasons": normalized_reasons,
            "signals": normalized_signals,
        }
        if not cleaned_market["watchpoints"]:
            cleaned_market["watchpoints"] = self.state["marketRegime"].get("watchpoints", [])
        if not cleaned_market["reasons"]:
            cleaned_market["reasons"] = self.state["marketRegime"].get("reasons", [])
        if not cleaned_market["signals"]:
            cleaned_market["signals"] = self.state["marketRegime"].get("signals", {})

        changed = self._market_regime_changed(cleaned_market)
        self.state["marketRegime"] = cleaned_market
        heartbeat = self.state["heartbeat"]
        heartbeat["lastMacroSyncAt"] = _utc_now_iso()
        heartbeat["lastSyncMessage"] = cleaned_market["summary"]

        if changed:
            heartbeat["lastMacroChangeAt"] = heartbeat["lastMacroSyncAt"]
            heartbeat["lastSyncStatus"] = "updated"
            self._create_commit(
                "market_regime",
                self._build_market_regime_commit_summary(previous_market, cleaned_market),
                delta={
                    "risk_score_from": previous_market.get("riskScore"),
                    "risk_score_to": cleaned_market.get("riskScore"),
                    "state_from": previous_market.get("state"),
                    "state_to": cleaned_market.get("state"),
                    "summary": cleaned_market.get("summary", ""),
                },
                key_signals=format_key_signals(cleaned_market.get("signals", {})),
                frontal_lobe_ref=self._build_frontal_lobe_ref(),
                source=source
            )
        else:
            heartbeat["lastSyncStatus"] = "no_change"
            self._save()

        return {
            "success": True,
            "changed": changed,
            "message": "Market regime updated successfully" if changed else "Macro heartbeat completed with no material change"
        }
```

- [ ] **Step 4: Run the regression test and verify it passes**

Run:

```bash
cd /home/margincaller/MarginCall_2X && \
./venv/bin/python -m unittest \
  test_brain_memory.BrainMemoryTests.test_update_market_regime_signal_only_refresh_updates_state_without_commit
```

Expected: `Ran 1 test` and `OK`

- [ ] **Step 5: Commit the market-regime gating change**

```bash
cd /home/margincaller/MarginCall_2X && \
git add test_brain_memory.py engine_memory.py && \
git commit -m "fix: ignore signal-only market heartbeat churn" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Commit-history retention and final focused verification

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/test_brain_memory.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_memory.py`
- Test: `/home/margincaller/MarginCall_2X/test_brain_memory.py`
- Verify: `/home/margincaller/MarginCall_2X/test_agent_runtime.py`

- [ ] **Step 1: Write the failing regression test for the 200-commit cap**

```python
class BrainMemoryTests(unittest.TestCase):
    def test_commit_history_is_capped_at_200_entries(self):
        brain = memory.Brain()

        for idx in range(205):
            result = brain.update_emotion(
                "cautious" if idx % 2 else "neutral",
                f"reason {idx}"
            )
            self.assertTrue(result["success"])

        self.assertEqual(len(brain.commits), 200)
        self.assertEqual(brain.head, brain.commits[-1]["hash"])
        self.assertEqual(brain.commits[0]["delta"]["reason"], "reason 5")

        reloaded = memory.Brain()
        self.assertEqual(len(reloaded.commits), 200)
        self.assertEqual(reloaded.head, reloaded.commits[-1]["hash"])
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd /home/margincaller/MarginCall_2X && \
./venv/bin/python -m unittest \
  test_brain_memory.BrainMemoryTests.test_commit_history_is_capped_at_200_entries
```

Expected: FAIL because `len(brain.commits)` is currently `205`

- [ ] **Step 3: Add centralized commit-history trimming**

```python
class Brain:
    def _trim_commit_history(self):
        if len(self.commits) <= MAX_COMMITS:
            return
        self.commits = self.commits[-MAX_COMMITS:]
        self.head = self.commits[-1]["hash"] if self.commits else None

    def _save(self):
        """將狀態持久化至本地端"""
        try:
            BRAIN_DIR.mkdir(exist_ok=True)
            self._trim_commit_history()
            with open(BRAIN_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "state": self.state,
                    "commits": self.commits,
                    "head": self.head
                }, f, ensure_ascii=False, indent=2)
            self._persist_views()
        except Exception as e:
            logger.error(f"Failed to save brain state: {e}")
```

- [ ] **Step 4: Run the cap regression and the full documented memory checks**

Run:

```bash
cd /home/margincaller/MarginCall_2X && \
./venv/bin/python -m unittest \
  test_brain_memory.BrainMemoryTests.test_commit_history_is_capped_at_200_entries && \
./venv/bin/python -m unittest test_brain_memory test_agent_runtime && \
./venv/bin/python -m py_compile engine_memory.py main.py src/agent.py
```

Expected:

- the cap regression passes
- the full focused memory regression suite passes
- `py_compile` exits successfully with no syntax errors

- [ ] **Step 5: Commit the retention change and final verification result**

```bash
cd /home/margincaller/MarginCall_2X && \
git add test_brain_memory.py engine_memory.py && \
git commit -m "fix: cap persisted brain commit history" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Self-review checklist

### Spec coverage

- **Structured default frontal-lobe content** -> Task 1
- **No-op frontal-lobe writes** -> Task 1
- **Signal-only heartbeat updates do not create commits** -> Task 2
- **200-commit retention cap** -> Task 3
- **Regression coverage** -> Tasks 1-3
- **Legacy compatibility** -> protected by existing `test_legacy_state_loads_with_new_market_fields` in `test_brain_memory.py` and the full-suite run in Task 3

### Placeholder scan

- No `TODO` / `TBD` placeholders remain.
- Every code-changing step includes concrete code snippets.
- Every test step names the exact command to run and the expected outcome.

### Type consistency

- `unchanged` is the return key used consistently for both frontal-lobe no-op write paths.
- `MAX_COMMITS` is the single retention constant used by `_trim_commit_history()`.
- `MARKET_REGIME_COMMIT_KEYS` is the single material-change field list used by `_market_regime_changed()`.
