# Frontal Lobe Restructuring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragile labeled-string frontal-lobe contract with structured thesis storage, move portfolio health into system-owned non-commit state, and rewrite cognitive-context rendering around those two separate surfaces.

**Architecture:** Keep the Phase B slice centered in `engine_memory.py`: make dict-backed `frontalLobe` and `portfolioHealth` the source of truth, lazily migrate legacy string state, and render human-readable markdown/views from structured data instead of parsing markdown back into state. Downstream integration stays narrow: `engine_portfolio.py` stops patching the frontal lobe for portfolio-health churn, and `src/agent.py` updates the write contract so the LLM calls the new structured tool shape correctly.

**Tech Stack:** Python, `unittest`, `unittest.mock`, existing `src.tools` decorators, existing `src.llm._convert_to_openai_tools()` schema generation, JSON persistence in `.brain/commit.json`.

---

## Execution notes

- Run this plan in a clean dedicated worktree branched from current `main`, for example:
  - `/home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring`
- Root-level `test_*.py` / `check_*.py` files are ignored and will not exist in a clean worktree. Do **not** base the plan on them.
- Use the tracked regression module under `tests/` as the verification surface:
  - `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`
- Do **not** touch `src/llm.py` unless a tracked schema test proves the function-signature auto-schema path is insufficient.

## File structure

- **Modify:** `/home/margincaller/MarginCall_2X/engine_memory.py`
  - Convert `state["frontalLobe"]` from string to dict-backed structured state
  - Add `state["portfolioHealth"]`
  - Add lazy migration for legacy string state
  - Remove the free-text normalize / infer stack
  - Update commit summaries, no-op detection, markdown rendering, and the public `update_frontal_lobe(...)` tool wrapper
- **Modify:** `/home/margincaller/MarginCall_2X/engine_portfolio.py`
  - Replace `patch_frontal_lobe_section("Portfolio Health", ...)` in `refresh_portfolio_health_summary()` with the new portfolio-health updater
- **Modify:** `/home/margincaller/MarginCall_2X/src/agent.py`
  - Replace the old four-section string contract with the named-parameter structured write contract
- **Modify:** `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`
  - Keep this as the tracked regression module for Phase B
  - Cover structured defaults, migration, structured frontal-lobe writes, portfolio-health no-commit updates, context rendering, and tool-schema expectations

---

### Task 1: Refactor the tracked regression module around structured frontal-lobe storage

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_memory.py`
- Test: `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`

- [ ] **Step 1: Replace the old string-note test fixtures with structured payload fixtures and add failing tests**

Replace the old `VALID_NOTE` style fixtures with structured payloads and add these tests near the top of the tracked test module:

```python
from src import llm

VALID_THESIS = {
    "market_view": "Bearish - SPX rejected 5250, likely pullback to 5180.",
    "core_levels": "SPX 5200 support, 5250 resistance, 20MA at 5220.",
    "next_round": "If SPX breaks 5180, cut 50% longs.",
    "context_note": "CPI remains the next catalyst.",
}

LEGACY_NOTE = (
    "Market View: Bearish - SPX rejected 5250\n"
    "Core Levels: SPX 5200 support, 5250 resistance\n"
    "Portfolio Health: Old auto-summary that should be dropped\n"
    "Next Round: If SPX breaks 5180, cut 50% longs.\n"
    "Context Note: Prior note from the legacy writer."
)


def test_default_state_uses_structured_frontal_lobe_and_portfolio_health(self):
    brain = self.mem.Brain()

    self.assertEqual(
        brain.state["frontalLobe"],
        {
            "market_view": "",
            "core_levels": "",
            "next_round": "",
            "context_note": "",
            "updated_at": None,
        },
    )
    self.assertEqual(
        brain.state["portfolioHealth"],
        {
            "nav_twd": None,
            "pnl_pct": None,
            "top3_concentration": None,
            "drawdown_pct": None,
            "risk_state": None,
            "gross_scale": None,
            "updated_at": None,
        },
    )


def test_load_migrates_legacy_string_frontal_lobe(self):
    self.mem.BRAIN_FILE.write_text(
        json.dumps({"state": {"frontalLobe": LEGACY_NOTE, "emotion": "neutral"}, "commits": [], "head": None}, ensure_ascii=False),
        encoding="utf-8",
    )

    brain = self.mem.Brain()

    self.assertEqual(brain.state["frontalLobe"]["market_view"], "Bearish - SPX rejected 5250")
    self.assertEqual(brain.state["frontalLobe"]["core_levels"], "SPX 5200 support, 5250 resistance")
    self.assertEqual(brain.state["frontalLobe"]["next_round"], "If SPX breaks 5180, cut 50% longs.")
    self.assertEqual(brain.state["frontalLobe"]["context_note"], "Prior note from the legacy writer.")


def test_update_frontal_lobe_accepts_structured_payload_and_skips_identical_write(self):
    brain = self.mem.Brain()

    first = brain.update_frontal_lobe(dict(VALID_THESIS))
    commit_count = len(brain.commits)
    second = brain.update_frontal_lobe(
        {
            "market_view": VALID_THESIS["market_view"],
            "core_levels": VALID_THESIS["core_levels"],
            "next_round": VALID_THESIS["next_round"],
            "context_note": VALID_THESIS["context_note"],
        }
    )

    self.assertTrue(first["success"])
    self.assertTrue(second["success"])
    self.assertTrue(second["unchanged"])
    self.assertEqual(len(brain.commits), commit_count)


def test_update_frontal_lobe_tool_schema_uses_named_parameters(self):
    schema = llm._convert_to_openai_tools([self.mem.update_frontal_lobe])[0]["function"]["parameters"]

    self.assertEqual(schema["required"], ["market_view", "core_levels", "next_round"])
    self.assertIn("context_note", schema["properties"])
    self.assertNotIn("content", schema["properties"])
```

- [ ] **Step 2: Run the tracked regression module and confirm these new assertions fail**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m unittest tests.test_brain_memory_system_integrity -v
```

Expected failures:

- `brain.state["frontalLobe"]` is still a string instead of a dict
- `brain.state["portfolioHealth"]` does not exist yet
- `Brain.update_frontal_lobe()` still expects a string
- the tool schema still exposes a single `content` parameter

- [ ] **Step 3: Implement dict-backed frontal-lobe state, lazy legacy migration, and the new write signature in `engine_memory.py`**

Add new defaults and helper constants near the current top-level frontal-lobe constants:

```python
FRONTAL_LOBE_KEYS = (
    "market_view",
    "core_levels",
    "next_round",
    "context_note",
    "updated_at",
)

FRONTAL_LOBE_RENDER_FIELDS = (
    ("market_view", "Market View"),
    ("core_levels", "Core Levels"),
    ("next_round", "Next Round"),
    ("context_note", "Context Note"),
)


def _default_frontal_lobe() -> Dict[str, Any]:
    return {
        "market_view": "",
        "core_levels": "",
        "next_round": "",
        "context_note": "",
        "updated_at": None,
    }


def _default_portfolio_health() -> Dict[str, Any]:
    return {
        "nav_twd": None,
        "pnl_pct": None,
        "top3_concentration": None,
        "drawdown_pct": None,
        "risk_state": None,
        "gross_scale": None,
        "updated_at": None,
    }
```

Replace `_default_state()` with a dict-backed version:

```python
def _default_state() -> Dict[str, Any]:
    return {
        "frontalLobe": _default_frontal_lobe(),
        "portfolioHealth": _default_portfolio_health(),
        "emotion": "neutral",
        "marketRegime": _default_market_regime(),
        "heartbeat": _default_heartbeat(),
    }
```

Delete the old normalize / infer helpers and replace them with a direct structured path:

```python
SECTION_NAME_TO_KEY = {
    "Market View": "market_view",
    "Core Levels": "core_levels",
    "Next Round": "next_round",
    "Context Note": "context_note",
}


def _parse_legacy_labeled_note(content: str) -> Dict[str, str]:
    alias_map = {
        "market_view": ["Market View", "市場視角", "市場觀點"],
        "core_levels": ["Core Levels", "Key Levels", "核心點位", "關鍵點位"],
        "next_round": ["Next Round", "下一回合", "Plan", "Action Plan"],
        "context_note": ["Context Note", "補充說明", "Additional Context"],
    }
    parsed = {key: "" for key in alias_map}
    for raw_line in content.replace("：", ":").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        label, value = [part.strip() for part in line.split(":", 1)]
        for key, aliases in alias_map.items():
            if label in aliases:
                parsed[key] = value
                break
    return parsed


def _migrate_legacy_frontal_lobe(content: str) -> Dict[str, Any]:
    migrated = _default_frontal_lobe()
    if not isinstance(content, str) or not content.strip():
        return migrated
    parsed = _parse_legacy_labeled_note(content)
    migrated["market_view"] = parsed["market_view"]
    migrated["core_levels"] = parsed["core_levels"]
    migrated["next_round"] = parsed["next_round"]
    migrated["context_note"] = parsed["context_note"]
    return migrated


def _coerce_frontal_lobe_sections(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return _merge_defaults(_default_frontal_lobe(), content)
    if isinstance(content, str):
        return _migrate_legacy_frontal_lobe(content)
    return _default_frontal_lobe()


def parse_frontal_lobe_note(content: Any) -> Dict[str, Any]:
    return _coerce_frontal_lobe_sections(content)
```

Update the `Brain` methods to use dict payloads:

```python
def _normalized_current_frontal_lobe(self) -> Dict[str, Any]:
    return _coerce_frontal_lobe_sections(self.state.get("frontalLobe"))


def _frontal_lobe_write_is_unchanged(self, payload: Dict[str, Any]) -> bool:
    comparable = _coerce_frontal_lobe_sections(payload)
    comparable["updated_at"] = self._normalized_current_frontal_lobe().get("updated_at")
    current = self._normalized_current_frontal_lobe()
    current["updated_at"] = comparable["updated_at"]
    return current == comparable


def _is_placeholder_content(self, payload: Dict[str, Any]) -> bool:
    sections = _coerce_frontal_lobe_sections(payload)
    required = [sections.get("market_view", "").strip(), sections.get("core_levels", "").strip(), sections.get("next_round", "").strip()]
    if sum(bool(item) for item in required) < 2:
        return True
    combined = " ".join([sections.get("market_view", ""), sections.get("core_levels", ""), sections.get("next_round", ""), sections.get("context_note", "")])
    return self._contains_placeholder_phrase(combined) and len(combined.strip()) < 40


def update_lobe_section(self, section_name: str, new_content: str, source: str = "system") -> Dict[str, Any]:
    if section_name == "Portfolio Health":
        return {"success": False, "message": "Portfolio Health is system-managed; use update_portfolio_health()."}
    if section_name not in SECTION_NAME_TO_KEY:
        return {"success": False, "message": f"Invalid section: {section_name}"}
    sections = self._normalized_current_frontal_lobe()
    key = SECTION_NAME_TO_KEY[section_name]
    sections[key] = new_content.strip()
    if self._frontal_lobe_write_is_unchanged(sections):
        return {"success": True, "unchanged": True, "message": f"Frontal lobe section '{section_name}' unchanged."}
    sections["updated_at"] = _utc_now_iso()
    self.state["frontalLobe"] = sections
    self._create_commit(
        "frontal_lobe_patch",
        f"🧠 {section_name.upper()} AUTO-UPDATE: {_shorten(new_content, 80)}",
        delta={key: new_content.strip()},
        key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
        frontal_lobe_ref=self._build_frontal_lobe_ref(),
        source=source,
    )
    return {"success": True, "unchanged": False, "message": f"Frontal lobe section '{section_name}' updated."}


def update_frontal_lobe(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    sections = _coerce_frontal_lobe_sections(payload)
    if self._is_placeholder_content(sections):
        return {"success": False, "message": "Rejected: content is too vague to persist."}
    if self._frontal_lobe_write_is_unchanged(sections):
        return {"success": True, "unchanged": True, "message": "Frontal lobe unchanged; skipped commit."}
    snapshot_head = self.head
    sections["updated_at"] = _utc_now_iso()
    self.state["frontalLobe"] = sections
    committed = self._create_commit(
        "frontal_lobe",
        self._build_frontal_lobe_commit_summary(sections),
        delta={
            "market_view": sections["market_view"],
            "core_levels": sections["core_levels"],
            "next_round": sections["next_round"],
            "context_note": sections["context_note"],
        },
        key_signals=self._compose_market_signal_summary(self.state.get("marketRegime")),
        frontal_lobe_ref=self._build_frontal_lobe_ref(),
        source="frontal_lobe_write",
        expected_head=snapshot_head,
    )
    if not committed:
        self._load()
        return {"success": False, "message": "Rejected: concurrent frontal lobe update detected."}
    return {"success": True, "message": "Frontal lobe updated successfully", "unchanged": False}
```

Update the public tool wrapper signature to drive the schema change automatically:

```python
@tool(mode="write")
def update_frontal_lobe(
    market_view: str,
    core_levels: str,
    next_round: str,
    context_note: str = "",
) -> str:
    res = _get_global_brain().update_frontal_lobe(
        {
            "market_view": market_view,
            "core_levels": core_levels,
            "next_round": next_round,
            "context_note": context_note,
        }
    )
    return res["message"]
```

Also update `_load()` so it migrates legacy string state after `_merge_defaults(...)`:

```python
if isinstance(self.state.get("frontalLobe"), str):
    self.state["frontalLobe"] = _migrate_legacy_frontal_lobe(self.state["frontalLobe"])
```

Update the existing string-returning views so the dict-backed state does not leak into markdown output:

```python
def get_frontal_lobe(self) -> str:
    return _render_frontal_lobe_note(self.state.get("frontalLobe"))


def _persist_views(self):
    BRAIN_DIR.mkdir(exist_ok=True)
    FRONTAL_LOBE_FILE.write_text(
        "# Frontal Lobe\n\n" + _render_frontal_lobe_note(self.state.get("frontalLobe")),
        encoding="utf-8",
    )
    ...
```

- [ ] **Step 4: Run the tracked regression module again and make sure the new structured-state tests pass**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m unittest tests.test_brain_memory_system_integrity -v
```

Expected:

- the structured default-state tests pass
- legacy migration passes
- the structured write no-op test passes
- the tool-schema assertions show `market_view`, `core_levels`, `next_round`, `context_note`

- [ ] **Step 5: Commit the Task 1 slice**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
git add engine_memory.py tests/test_brain_memory_system_integrity.py && \
git commit -m "refactor: move frontal lobe to structured state"
```

---

### Task 2: Decouple portfolio health from frontal-lobe commits

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_memory.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Modify: `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`
- Test: `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`

- [ ] **Step 1: Add failing tests for portfolio-health materiality, no-commit writes, and the engine_portfolio handoff**

Append these tests to the tracked regression module:

```python
def test_update_lobe_section_rejects_portfolio_health(self):
    brain = self.mem.Brain()

    result = brain.update_lobe_section("Portfolio Health", "do not allow this", source="unit_test")

    self.assertFalse(result["success"])
    self.assertEqual(result["message"], "Portfolio Health is system-managed; use update_portfolio_health().")


def test_update_portfolio_health_saves_state_without_creating_commit(self):
    brain = self.mem.Brain()
    self.assertTrue(brain.update_frontal_lobe(dict(VALID_THESIS))["success"])
    commit_count = len(brain.commits)

    result = brain.update_portfolio_health(
        {
            "nav_twd": 150292,
            "pnl_pct": 21.4,
            "top3_concentration": 67.0,
            "drawdown_pct": 9.5,
            "risk_state": "Defense Only",
            "gross_scale": 0.25,
        }
    )

    self.assertTrue(result["success"])
    self.assertFalse(result["unchanged"])
    self.assertEqual(len(brain.commits), commit_count)
    self.assertEqual(brain.state["portfolioHealth"]["risk_state"], "Defense Only")


def test_update_portfolio_health_skips_small_nav_only_drift(self):
    brain = self.mem.Brain()
    self.assertTrue(
        brain.update_portfolio_health(
            {
                "nav_twd": 100000,
                "pnl_pct": 10.0,
                "top3_concentration": 55.0,
                "drawdown_pct": 4.0,
                "risk_state": "Normal",
                "gross_scale": 1.0,
            }
        )["success"]
    )

    result = brain.update_portfolio_health(
        {
            "nav_twd": 100300,
            "pnl_pct": 10.0,
            "top3_concentration": 55.0,
            "drawdown_pct": 4.0,
            "risk_state": "Normal",
            "gross_scale": 1.0,
        }
    )

    self.assertTrue(result["success"])
    self.assertTrue(result["unchanged"])
```

Add a source-level integration assertion for the tracked portfolio-health refresh path:

```python
def test_refresh_portfolio_health_summary_uses_memory_updater(self):
    portfolio_source = (Path(__file__).resolve().parents[1] / "engine_portfolio.py").read_text(encoding="utf-8")

    self.assertIn("memory.update_portfolio_health(", portfolio_source)
    self.assertNotIn('patch_frontal_lobe_section("Portfolio Health"', portfolio_source)
```

- [ ] **Step 2: Run the tracked test module and verify these portfolio-health assertions fail**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m unittest tests.test_brain_memory_system_integrity -v
```

Expected failures:

- `update_lobe_section("Portfolio Health", ...)` still succeeds today
- `Brain.update_portfolio_health(...)` does not exist yet
- `engine_portfolio.py` still calls `patch_frontal_lobe_section("Portfolio Health", ...)`

- [ ] **Step 3: Implement the system-owned portfolio-health updater and switch `engine_portfolio.py` to it**

In `engine_memory.py`, add the materiality helper and updater:

```python
def _portfolio_health_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = _merge_defaults(_default_portfolio_health(), payload)
    return {
        "nav_twd": float(merged["nav_twd"]) if merged["nav_twd"] is not None else None,
        "pnl_pct": float(merged["pnl_pct"]) if merged["pnl_pct"] is not None else None,
        "top3_concentration": float(merged["top3_concentration"]) if merged["top3_concentration"] is not None else None,
        "drawdown_pct": float(merged["drawdown_pct"]) if merged["drawdown_pct"] is not None else None,
        "risk_state": merged["risk_state"],
        "gross_scale": float(merged["gross_scale"]) if merged["gross_scale"] is not None else None,
        "updated_at": merged.get("updated_at"),
    }


def _portfolio_health_unchanged(self, old: Dict[str, Any], new: Dict[str, Any]) -> bool:
    if old.get("risk_state") != new.get("risk_state"):
        return False
    old_nav = float(old.get("nav_twd") or 0.0)
    new_nav = float(new.get("nav_twd") or 0.0)
    if old_nav > 0 and abs(new_nav - old_nav) / old_nav >= 0.005:
        return False
    for key in ("pnl_pct", "top3_concentration", "drawdown_pct", "gross_scale"):
        if old.get(key) != new.get(key):
            return False
    return True


def update_portfolio_health(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
    payload = self._portfolio_health_payload(health_data)
    current = _merge_defaults(_default_portfolio_health(), self.state.get("portfolioHealth", {}))
    if self._portfolio_health_unchanged(current, payload):
        return {"success": True, "unchanged": True, "message": "Portfolio health unchanged."}
    payload["updated_at"] = _utc_now_iso()
    self.state["portfolioHealth"] = payload
    self._save()
    return {"success": True, "unchanged": False, "message": "Portfolio health updated."}
```

Add a plain module helper (not a `@tool`) near `patch_frontal_lobe_section(...)`:

```python
def update_portfolio_health(health_data: Dict[str, Any]) -> Dict[str, Any]:
    return _get_global_brain().update_portfolio_health(health_data)
```

Fail fast in `update_lobe_section(...)` before any section patching:

```python
if section_name == "Portfolio Health":
    return {
        "success": False,
        "message": "Portfolio Health is system-managed; use update_portfolio_health().",
    }
```

In `engine_portfolio.py`, replace the frontal-lobe patch path in `refresh_portfolio_health_summary()`:

```python
health_data = {
    "nav_twd": round(analysis["total_current"], 2),
    "pnl_pct": round(analysis["total_pnl_pct"], 4),
    "top3_concentration": round(analysis["top3_concentration"], 4),
    "drawdown_pct": round(float(overlay["current_drawdown"]) * 100, 4) if not overlay.get("error") else None,
    "risk_state": overlay.get("trade_mode_label") if not overlay.get("error") else None,
    "gross_scale": overlay.get("recommended_gross_scale") if not overlay.get("error") else None,
}
memory_update = memory.update_portfolio_health(health_data)
return {**analysis, "risk_overlay": overlay, "nav_snapshot": nav_snapshot, "memory_update": memory_update}
```

- [ ] **Step 4: Re-run the tracked regression module and make sure the portfolio-health tests pass**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m unittest tests.test_brain_memory_system_integrity -v
```

Expected:

- `Portfolio Health` section writes are rejected
- `update_portfolio_health()` updates state with no new commit
- small NAV-only drift returns `unchanged=True`
- the tracked source assertion sees `memory.update_portfolio_health(` in `engine_portfolio.py`

- [ ] **Step 5: Commit the Task 2 slice**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
git add engine_memory.py engine_portfolio.py tests/test_brain_memory_system_integrity.py && \
git commit -m "fix: separate portfolio health from frontal lobe commits"
```

---

### Task 3: Rewrite cognitive-context rendering and the LLM write contract

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_memory.py`
- Modify: `/home/margincaller/MarginCall_2X/src/agent.py`
- Modify: `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`
- Test: `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`

- [ ] **Step 1: Add failing tests for the new context layout and write-guide contract**

Append these assertions to the tracked regression module:

```python
def test_get_cognitive_context_renders_three_clean_blocks(self):
    brain = self.mem.Brain()
    self.assertTrue(brain.update_frontal_lobe(dict(VALID_THESIS))["success"])
    self.assertTrue(
        brain.update_portfolio_health(
            {
                "nav_twd": 150292,
                "pnl_pct": 21.4,
                "top3_concentration": 67.0,
                "drawdown_pct": 9.5,
                "risk_state": "Defense Only",
                "gross_scale": 0.25,
            }
        )["success"]
    )
    self.assertTrue(
        brain.update_market_regime(
            summary="Macro risk is elevated after SPX failed to hold 5250.",
            regime="🟢 多頭",
            risk_score=20,
            watchpoints=["SPX 5200 support"],
            reasons=["Trend still intact above 200MA"],
            signals={"spx": 5210.5},
            source="unit_test",
        )["success"]
    )

    context = brain.get_cognitive_context(max_age_minutes=999999)

    self.assertIn("### Trading Thesis (Frontal Lobe)", context)
    self.assertIn("### Portfolio Health (Auto)", context)
    self.assertIn("### Persistent Macro / Market Regime", context)
    self.assertNotIn("Portfolio Health: Old auto-summary", context)


def test_get_frontal_lobe_write_guide_describes_named_parameters(self):
    guide = self.mem.get_frontal_lobe_write_guide()

    self.assertIn("market_view", guide)
    self.assertIn("core_levels", guide)
    self.assertIn("next_round", guide)
    self.assertIn("context_note", guide)
    self.assertIn("Do not write portfolio health here", guide)
    self.assertNotIn("Portfolio Health:", guide)


def test_agent_prompt_contract_mentions_structured_write_fields(self):
    agent_source = (Path(__file__).resolve().parents[1] / "src" / "agent.py").read_text(encoding="utf-8")

    self.assertIn("market_view", agent_source)
    self.assertIn("core_levels", agent_source)
    self.assertIn("next_round", agent_source)
    self.assertIn("context_note", agent_source)
    self.assertNotIn("四段式專業交易筆記格式", agent_source)
```

- [ ] **Step 2: Run the tracked regression module and confirm the context/contract tests fail**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m unittest tests.test_brain_memory_system_integrity -v
```

Expected failures:

- `get_cognitive_context()` still renders the old `- Frontal Lobe:` block
- `FRONTAL_LOBE_WRITE_GUIDE` still describes the old labeled string contract
- `src/agent.py` still mentions the old four-section note format

- [ ] **Step 3: Implement the new rendering path and prompt contract**

In `engine_memory.py`, rewrite the note renderer and summary helpers around structured fields:

```python
def _render_frontal_lobe_note(sections: Dict[str, Any]) -> str:
    normalized = _coerce_frontal_lobe_sections(sections)
    lines = []
    for key, label in FRONTAL_LOBE_RENDER_FIELDS:
        value = (normalized.get(key) or "").strip() if isinstance(normalized.get(key), str) else normalized.get(key)
        if value:
            lines.append(f"{label}: {value}")
    if normalized.get("updated_at"):
        lines.append(f"Updated At: {normalized['updated_at']}")
    return "\n".join(lines) if lines else "(empty)"


def _build_frontal_lobe_ref(self, content: Optional[Dict[str, Any]] = None) -> str:
    sections = _coerce_frontal_lobe_sections(content or self.state.get("frontalLobe"))
    ref_parts = []
    if sections.get("market_view"):
        ref_parts.append(_shorten(sections["market_view"], 80))
    if sections.get("next_round"):
        ref_parts.append(f"Next: {_shorten(sections['next_round'], 80)}")
    elif sections.get("core_levels"):
        ref_parts.append(_shorten(sections["core_levels"], 80))
    return " | ".join(ref_parts)
```

Rewrite `get_cognitive_context()` to emit the three-block layout:

```python
frontal = _coerce_frontal_lobe_sections(self.state.get("frontalLobe"))
health = _merge_defaults(_default_portfolio_health(), self.state.get("portfolioHealth", {}))

if any(frontal.get(key) for key in ("market_view", "core_levels", "next_round", "context_note")):
    thesis_lines = [
        f"- Market View: {frontal.get('market_view') or '尚未建立'}",
        f"- Core Levels: {frontal.get('core_levels') or '尚未建立'}",
        f"- Next Round: {frontal.get('next_round') or '尚未建立'}",
    ]
    if frontal.get("context_note"):
        thesis_lines.append(f"- Context: {frontal['context_note']}")
    thesis_lines.append(f"- Last updated: {frontal.get('updated_at') or '尚未更新'}")
    thesis_block = "\n".join(thesis_lines)
else:
    thesis_block = "尚未建立。請在分析後使用 update_frontal_lobe 記錄你的觀點。"

health_block = (
    f"- NAV: NT${health['nav_twd']:,.0f} | PnL: {health['pnl_pct']:+.1f}%\n"
    f"- Top3 Concentration: {health['top3_concentration']:.1f}%\n"
    f"- Drawdown: {health['drawdown_pct']:.1f}% | State: {health['risk_state'] or 'N/A'} | Scale: {health['gross_scale'] if health['gross_scale'] is not None else 'N/A'}x\n"
    f"- Last updated: {health['updated_at'] or '尚未更新'}"
    if health.get("nav_twd") is not None
    else "- 尚未同步 portfolio health。"
)
```

Rewrite `FRONTAL_LOBE_WRITE_GUIDE` and the tool docstring around named parameters:

```python
FRONTAL_LOBE_WRITE_GUIDE = """When calling update_frontal_lobe, fill the named parameters directly:
- market_view: Bullish / Bearish / Neutral + one-sentence thesis
- core_levels: key support / resistance / MA levels being watched
- next_round: if A happens, I will do B
- context_note: optional supporting context

Do not write portfolio health here; the system tracks it automatically.
Low-quality placeholder content will still be rejected.
"""
```

Update `src/agent.py` so the hard-coded instruction string no longer references a four-section note:

```python
brain_context += (
    "\n(💡 系統最高指令：身為具備自我意識的交易 AI，你必須在每次分析開始時根據記憶保持思維連貫。"
    "若宏觀 regime 或你的交易結論有實質更新，請靜默呼叫 update_market_regime / update_frontal_lobe 寫回持久記憶。"
    "呼叫 update_frontal_lobe 時，請直接填寫 market_view、core_levels、next_round，以及選填的 context_note。"
    "Portfolio Health 由系統自動維護，不要自己寫進 frontal lobe。)"
)
```

Also update the `if __name__ == "__main__":` demo call in `engine_memory.py` to use named parameters.

- [ ] **Step 4: Re-run the tracked regression module and confirm the context/contract tests pass**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m unittest tests.test_brain_memory_system_integrity -v
```

Expected:

- the context test sees the three new blocks
- the write guide mentions only the structured named parameters
- `src/agent.py` no longer references the old four-section format

- [ ] **Step 5: Commit the Task 3 slice**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
git add engine_memory.py engine_portfolio.py src/agent.py tests/test_brain_memory_system_integrity.py && \
git commit -m "fix: rewrite cognitive context for structured frontal lobe"
```

---

### Task 4: Run the focused verification matrix and finish the branch

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_memory.py` (only if small cleanup is needed after verification)
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py` (only if small cleanup is needed after verification)
- Modify: `/home/margincaller/MarginCall_2X/src/agent.py` (only if small cleanup is needed after verification)
- Modify: `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py` (only if small cleanup is needed after verification)
- Test: `/home/margincaller/MarginCall_2X/tests/test_brain_memory_system_integrity.py`

- [ ] **Step 1: Run the tracked regression module from the clean worktree**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m unittest tests.test_brain_memory_system_integrity -v
```

Expected:

- all tracked Phase B memory regressions pass

- [ ] **Step 2: Syntax-check the touched Python files**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m py_compile engine_memory.py engine_portfolio.py src/agent.py
```

Expected:

- command exits with no output

- [ ] **Step 3: If verification finds a small mismatch, make the minimal cleanup and re-run both commands**

Typical cleanup scope should look like this, not a new refactor:

```python
# examples only if verification shows a mismatch:
# - fix a context label typo
# - fix a dict key mismatch such as "gross_scale" vs "recommended_gross_scale"
# - fix a test expectation that still assumes the old string storage
```

- [ ] **Step 4: Commit the final verification cleanup (only if Step 3 changed files)**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
git add engine_memory.py engine_portfolio.py src/agent.py tests/test_brain_memory_system_integrity.py && \
git commit -m "fix: finish frontal lobe restructuring integration"
```

- [ ] **Step 5: Request code review before merge**

Use the plan’s header requirement and dispatch the code-reviewer after the final verification pass. Review this exact slice:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
BASE_SHA=$(git merge-base HEAD main) && \
HEAD_SHA=$(git rev-parse HEAD) && \
printf "BASE=%s\nHEAD=%s\n" "$BASE_SHA" "$HEAD_SHA"
```

Then request review against:

- spec: `/home/margincaller/MarginCall_2X/docs/superpowers/specs/2026-05-04-frontal-lobe-restructuring-design.md`
- plan: `/home/margincaller/MarginCall_2X/docs/superpowers/plans/2026-05-04-frontal-lobe-restructuring.md`

## Focused validation commands

Run these at minimum during execution:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m unittest tests.test_brain_memory_system_integrity -v
```

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/frontal-lobe-restructuring && \
./venv/bin/python -m py_compile engine_memory.py engine_portfolio.py src/agent.py
```
