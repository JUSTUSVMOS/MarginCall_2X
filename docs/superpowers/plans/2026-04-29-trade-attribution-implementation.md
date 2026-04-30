# Trade Attribution & Journal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tracked, worktree-safe trade journal that captures decision-time attribution inputs, settles `T+5` / `T+20` checkpoints, and publishes a weekly Beta / Sector / Timing report.

**Architecture:** Extend `engine_portfolio.py` so new risk-increasing trades persist the attribution inputs and enqueue post-commit journal work. Implement the queue, settlement, and reporting logic in a new `engine_journal.py` module, then wire scheduler/runtime entry points around that module. Keep tests under `tests/` so the clean worktree no longer depends on ignored top-level `test_*.py` / `check_*.py` files.

**Tech Stack:** Python, sqlite3 via `src.database`, `unittest`, APScheduler, pandas, yfinance, existing portfolio/risk/router helpers.

---

## Execution notes

- Work only in `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics`.
- Do not call `git fetch origin` in this environment; `git-remote-https` is unavailable here.
- Use the shared interpreter at `/home/margincaller/MarginCall_2X/venv/bin/python`.
- Do not nest `db_lock`: enqueue journal rows only **after** the portfolio transaction commits.
- Do not place new tracked tests at repo root with `test_*.py` / `check_*.py` names because `.gitignore` ignores those patterns. Use `tests/`.

## File structure

- **Create:** `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/engine_journal.py`
  - Own checkpoint enqueueing, settlement, weekly report aggregation, and journal tool exports.
- **Modify:** `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/engine_portfolio.py`
  - Extend decision snapshots, create the `trade_outcome_checkpoints` schema, and enqueue journal work after committed `buy` / `sync_buy` trades.
- **Modify:** `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/src/bot.py`
  - Import `engine_journal` into `_TOOL_MODULES` so journal tools register with the bot runtime.
- **Modify:** `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/src/scheduler.py`
  - Add daily checkpoint settlement and Sunday weekly report jobs.
- **Create:** `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/__init__.py`
  - Make `tests` importable for `python -m unittest`.
- **Create:** `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal.py`
  - Cover schema creation, enqueue behavior, snapshot enrichment, checkpoint settlement, and weekly report math.
- **Create:** `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal_scheduler.py`
  - Cover scheduler wiring and runtime journal-tool registration.

---

### Task 1: Add a tracked journal test harness and schema scaffold

**Files:**
- Create: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/__init__.py`
- Create: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal.py`
- Create: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/engine_journal.py`
- Modify: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/engine_portfolio.py`
- Test: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal.py`

- [ ] **Step 1: Write the failing tracked tests for schema creation and checkpoint enqueueing.**

```python
# tests/__init__.py

# Intentionally empty; allows `python -m unittest tests.test_trade_journal`.
```

```python
# tests/test_trade_journal.py
import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import engine_portfolio
import engine_journal


def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


class TradeJournalSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "portfolio.db")
        self.addCleanup(self.tmpdir.cleanup)

    def _patch_db(self):
        return patch.multiple(
            engine_portfolio,
            get_connection=lambda check_same_thread=False: _connect(self.db_path),
        )

    def test_init_db_creates_trade_outcome_checkpoints_table(self):
        with self._patch_db():
            engine_portfolio.init_db()
        with sqlite3.connect(self.db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(trade_outcome_checkpoints)")}
        self.assertIn("trade_log_id", columns)
        self.assertIn("horizon_label", columns)
        self.assertIn("benchmark_entry_price", columns)
        self.assertIn("beta_component_pct", columns)
        self.assertIn("timing_component_twd", columns)

    def test_enqueue_trade_outcome_checkpoints_creates_t5_and_t20_rows(self):
        with self._patch_db(), patch.object(
            engine_journal, "get_connection", side_effect=lambda check_same_thread=False: _connect(self.db_path)
        ), patch.object(engine_journal, "_load_price_on_or_after", side_effect=[101.0, 102.0]):
            engine_portfolio.init_db()
            with engine_portfolio.get_connection() as conn:
                cursor = conn.cursor()
                trade_log_id = engine_portfolio._record_trade_log(
                    cursor,
                    symbol="NVDA",
                    action="buy",
                    price=100.0,
                    shares=2.0,
                    settle_currency="CASH_USD",
                    settle_amount=200.0,
                    fx_rate=32.0,
                    note="unit test buy",
                    decision_snapshot={
                        "captured_at": "2026-04-29T01:00:00Z",
                        "symbol": "NVDA",
                        "benchmark_symbol": "SPY",
                        "sector_proxy_symbol": "SOXX",
                        "beta_proxy_at_entry": 1.2,
                        "entry_notional_twd": 6400.0,
                    },
                )
                conn.commit()
            result = engine_journal.enqueue_trade_outcome_checkpoints([trade_log_id])
            self.assertEqual(result["created"], 2)
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT horizon_label, benchmark_symbol, benchmark_entry_price, sector_proxy_symbol, sector_entry_price "
                    "FROM trade_outcome_checkpoints "
                    "WHERE trade_log_id = ? ORDER BY horizon_label",
                    (trade_log_id,),
                ).fetchall()
        self.assertEqual(rows, [("T+20", "SPY", 101.0, "SOXX", 102.0), ("T+5", "SPY", 101.0, "SOXX", 102.0)])
```

- [ ] **Step 2: Run the test file and verify it fails.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest tests.test_trade_journal
```

Expected: FAIL because `engine_journal.py` does not exist yet and `trade_outcome_checkpoints` is not part of `init_db()`.

- [ ] **Step 3: Add the minimal schema and enqueue scaffold to make the tests pass.**

```python
# engine_journal.py
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd

from src.database import db_lock, get_connection
from yf_session import get_ticker

CHECKPOINT_HORIZONS = (("T+5", 5), ("T+20", 20))
ELIGIBLE_TRADE_ACTIONS = {"buy", "sync_buy"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _add_business_days(timestamp: str, days: int) -> str:
    base = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    cursor = base
    remaining = days
    while remaining > 0:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            remaining -= 1
    return cursor.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_price_on_or_after(symbol: str, as_of: str) -> float | None:
    start = datetime.fromisoformat(as_of.replace("Z", "+00:00")).date().isoformat()
    hist = get_ticker(symbol, cache_level="daily").history(start=start, period="10d")
    closes = hist.get("Close") if isinstance(hist, pd.DataFrame) else None
    if closes is None:
        return None
    closes = pd.Series(closes).dropna()
    if closes.empty:
        return None
    return round(float(closes.iloc[0]), 6)


def enqueue_trade_outcome_checkpoints(trade_log_ids: Iterable[int]) -> dict[str, int]:
    created = 0
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            for trade_log_id in trade_log_ids:
                row = cursor.execute(
                    "SELECT id, timestamp, symbol, action, price, shares, decision_snapshot FROM trade_log WHERE id = ?",
                    (int(trade_log_id),),
                ).fetchone()
                if not row or row[3] not in ELIGIBLE_TRADE_ACTIONS:
                    continue
                snapshot = json.loads(row[6]) if row[6] else {}
                entry_notional_twd = float(snapshot.get("entry_notional_twd") or row[4] * row[5])
                benchmark_symbol = snapshot.get("benchmark_symbol", "SPY")
                sector_proxy_symbol = snapshot.get("sector_proxy_symbol")
                benchmark_entry_price = _load_price_on_or_after(benchmark_symbol, row[1]) if benchmark_symbol else None
                sector_entry_price = _load_price_on_or_after(sector_proxy_symbol, row[1]) if sector_proxy_symbol else None
                for horizon_label, day_count in CHECKPOINT_HORIZONS:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO trade_outcome_checkpoints (
                            trade_log_id, symbol, horizon_label, due_at, status, entry_timestamp,
                            entry_price, entry_notional_twd, benchmark_symbol, benchmark_entry_price,
                            sector_proxy_symbol, sector_entry_price,
                            beta_proxy_at_entry, beta_coverage, sector_coverage, retry_count
                        ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                        """,
                        (
                            row[0],
                            row[2],
                            horizon_label,
                            _add_business_days(row[1], day_count),
                            row[1],
                            row[4],
                            entry_notional_twd,
                            benchmark_symbol,
                            benchmark_entry_price,
                            sector_proxy_symbol,
                            sector_entry_price,
                            snapshot.get("beta_proxy_at_entry"),
                            1 if snapshot.get("beta_proxy_at_entry") is not None else 0,
                            1 if sector_proxy_symbol and sector_entry_price is not None else 0,
                        ),
                    )
                    created += cursor.rowcount
            conn.commit()
        finally:
            conn.close()
    return {"created": created}
```

```python
# engine_portfolio.py -> inside init_db()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS trade_outcome_checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_log_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        horizon_label TEXT NOT NULL,
        due_at TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        entry_timestamp TEXT NOT NULL,
        entry_price REAL NOT NULL,
        entry_notional_twd REAL NOT NULL,
        benchmark_symbol TEXT,
        benchmark_entry_price REAL,
        sector_proxy_symbol TEXT,
        sector_entry_price REAL,
        beta_proxy_at_entry REAL,
        beta_coverage INTEGER NOT NULL DEFAULT 0,
        sector_coverage INTEGER NOT NULL DEFAULT 0,
        resolved_price REAL,
        benchmark_return_pct REAL,
        sector_return_pct REAL,
        actual_return_pct REAL,
        beta_component_pct REAL,
        sector_component_pct REAL,
        timing_component_pct REAL,
        beta_component_twd REAL,
        sector_component_twd REAL,
        timing_component_twd REAL,
        last_error TEXT,
        retry_count INTEGER NOT NULL DEFAULT 0,
        resolved_at TEXT,
        FOREIGN KEY(trade_log_id) REFERENCES trade_log(id) ON DELETE CASCADE
    )
    """
)
cursor.execute(
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_outcome_checkpoints_trade_horizon
    ON trade_outcome_checkpoints(trade_log_id, horizon_label)
    """
)
```

- [ ] **Step 4: Re-run the journal test file and verify it passes.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest tests.test_trade_journal
```

Expected: PASS for the schema and enqueue tests.

- [ ] **Step 5: Commit the scaffold layer.**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add engine_portfolio.py engine_journal.py tests/__init__.py tests/test_trade_journal.py && \
git commit -m "feat: add trade journal schema scaffold" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Enrich decision snapshots and enqueue journal work only after committed entry trades

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/engine_portfolio.py`
- Modify: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal.py`
- Test: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal.py`

- [ ] **Step 1: Add failing tests for snapshot enrichment and post-commit enqueue behavior.**

```python
# tests/test_trade_journal.py
class TradeJournalEntryFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "portfolio.db")
        self.addCleanup(self.tmpdir.cleanup)

    def _patch_db(self):
        return patch.object(engine_portfolio, "get_connection", side_effect=lambda check_same_thread=False: _connect(self.db_path))

    @patch.object(engine_portfolio, "_estimate_entry_beta_proxy", return_value=1.18)
    @patch.object(
        engine_portfolio,
        "_fetch_sync_nlp_payload",
        return_value={"nlp_alpha": 0.21, "alpha_macro": 0.03, "alpha_retail": -0.05, "alpha_sec": 0.11},
    )
    @patch.object(engine_portfolio.market, "get_asset_profile", return_value={"sector": "Technology", "industry": "Semiconductors"})
    def test_build_trade_decision_snapshot_includes_attribution_inputs(self, *_mocks):
        snapshot = engine_portfolio._build_trade_decision_snapshot("NVDA", entry_notional_twd=6400.0)
        self.assertEqual(snapshot["benchmark_symbol"], "SPY")
        self.assertEqual(snapshot["sector_proxy_symbol"], "SOXX")
        self.assertEqual(snapshot["beta_proxy_at_entry"], 1.18)
        self.assertEqual(snapshot["alpha_sec"], 0.11)
        self.assertEqual(snapshot["entry_notional_twd"], 6400.0)

    def test_execute_position_update_buy_enqueues_after_commit(self):
        with self._patch_db(), patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0), patch.object(
            engine_portfolio, "refresh_portfolio_health_summary", return_value={"summary": "ok", "memory_update": {"message": "ok"}}
        ), patch.object(engine_portfolio, "_build_trade_decision_snapshot", return_value={
            "captured_at": "2026-04-29T01:00:00Z",
            "symbol": "NVDA",
            "benchmark_symbol": "SPY",
            "sector_proxy_symbol": "SOXX",
            "beta_proxy_at_entry": 1.2,
            "entry_notional_twd": 6400.0,
        }), patch("engine_journal.enqueue_trade_outcome_checkpoints") as enqueue_mock:
            engine_portfolio.init_db()
            with engine_portfolio.get_connection() as conn:
                conn.execute("INSERT INTO portfolio(symbol, cost, shares, twd_cost, locked) VALUES ('CASH_USD', 32.0, 1000.0, 32000.0, 0)")
                conn.commit()
            message = engine_portfolio.execute_position_update("NVDA", 100.0, 2.0, action="buy", total_amount_twd=6400.0, sync_memory=False)
        self.assertIn("買進成功", message)
        enqueue_mock.assert_called_once()
```

- [ ] **Step 2: Run the entry-flow tests and verify they fail.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest tests.test_trade_journal.TradeJournalEntryFlowTests
```

Expected: FAIL because `_build_trade_decision_snapshot()`, `_fetch_sync_nlp_payload()`, `_estimate_entry_beta_proxy()`, and post-commit enqueue logic do not exist yet.

- [ ] **Step 3: Implement the shared snapshot builder and move journal enqueueing outside the database lock.**

```python
# engine_portfolio.py
def _fetch_sync_nlp_payload(symbol: str, lookup_symbol: str) -> Dict[str, float | None]:
    try:
        import engine_router as router
    except Exception as exc:
        logger.debug(f"NLP router import failed during trade snapshot: {exc}")
        return {"nlp_alpha": None, "alpha_macro": None, "alpha_retail": None, "alpha_sec": None}

    candidates = [normalize_ticker(symbol)]
    if lookup_symbol and lookup_symbol not in candidates:
        candidates.append(lookup_symbol)
    for candidate in candidates:
        payload = router.fetch_nlp_alpha(candidate)
        if payload and not payload.get("error"):
            return {
                "nlp_alpha": _safe_round(payload.get("nlp_alpha"), 4),
                "alpha_macro": _safe_round(payload.get("alpha_macro"), 4),
                "alpha_retail": _safe_round(payload.get("alpha_retail"), 4),
                "alpha_sec": _safe_round(payload.get("alpha_official"), 4),
            }
    return {"nlp_alpha": None, "alpha_macro": None, "alpha_retail": None, "alpha_sec": None}


def _estimate_entry_beta_proxy(symbol: str, benchmark: str = "SPY", period: str = "6mo") -> float | None:
    attribution = compute_portfolio_beta_attribution({normalize_ticker(symbol): 1.0}, benchmark=benchmark, period=period)
    if attribution.get("error"):
        return None
    positions = attribution.get("positions") or {}
    if not positions:
        return None
    only_symbol = next(iter(positions.values()))
    return _safe_round(only_symbol.get("beta"), 4)


def _build_trade_decision_snapshot(symbol: str, *, entry_notional_twd: float | None = None) -> Dict[str, Any]:
    lookup_symbol = _resolve_sync_lookup_symbol(symbol)
    profile = market.get_asset_profile(lookup_symbol or symbol)
    sector_proxy_symbol = _select_sector_proxy(profile)
    nlp_payload = _fetch_sync_nlp_payload(symbol, lookup_symbol or symbol)
    snapshot = {
        "captured_at": _utc_now_iso(),
        "symbol": normalize_ticker(symbol),
        "lookup_symbol": lookup_symbol,
        "sector": profile.get("sector", "Unknown"),
        "industry": profile.get("industry", "Unknown"),
        "benchmark_symbol": "SPY",
        "sector_proxy_symbol": sector_proxy_symbol,
        "beta_proxy_period": "6mo",
        "beta_proxy_at_entry": _estimate_entry_beta_proxy(lookup_symbol or symbol),
        "entry_notional_twd": _safe_round(entry_notional_twd, 2) if entry_notional_twd is not None else None,
        **nlp_payload,
    }
    try:
        import engine_risk as risk_engine
        risk_snapshot = risk_engine.get_global_risk_snapshot()
        snapshot["risk_state"] = risk_snapshot.get("state")
        snapshot["risk_score"] = risk_snapshot.get("riskScore")
    except Exception as exc:
        snapshot["risk_snapshot_error"] = str(exc)
    return snapshot


def _build_sync_decision_snapshot(symbol: str) -> Dict[str, Any]:
    return _build_trade_decision_snapshot(symbol)
```

```python
# engine_portfolio.py -> inside sync_fubon_portfolio_state() and execute_position_update()
pending_trade_journal_ids: List[int] = []

# when recording eligible entries:
decision_snapshot = _build_trade_decision_snapshot(symbol, entry_notional_twd=actual_twd_total)
trade_log_id = _record_trade_log(..., decision_snapshot=decision_snapshot)
pending_trade_journal_ids.append(trade_log_id)

# after conn.commit() and after conn.close():
if pending_trade_journal_ids:
    try:
        import engine_journal as journal
        journal.enqueue_trade_outcome_checkpoints(pending_trade_journal_ids)
    except Exception as exc:
        logger.warning(f"Trade journal enqueue failed: {exc}")
```

- [ ] **Step 4: Re-run the entry-flow tests and verify they pass.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest tests.test_trade_journal.TradeJournalEntryFlowTests
```

Expected: PASS, with `buy` trades producing enriched snapshots and enqueueing journal work after the transaction commits.

- [ ] **Step 5: Commit the entry-capture layer.**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add engine_portfolio.py tests/test_trade_journal.py && \
git commit -m "feat: capture attribution inputs on entry trades" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Implement checkpoint settlement, additive attribution math, and the journal tool surface

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/engine_journal.py`
- Modify: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/src/bot.py`
- Modify: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal.py`
- Test: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal.py`

- [ ] **Step 1: Add failing tests for checkpoint settlement and weekly report output.**

```python
# tests/test_trade_journal.py
class TradeJournalSettlementTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "portfolio.db")
        self.addCleanup(self.tmpdir.cleanup)
        self.portfolio_conn = patch.object(engine_portfolio, "get_connection", side_effect=lambda check_same_thread=False: _connect(self.db_path))
        self.journal_conn = patch.object(engine_journal, "get_connection", side_effect=lambda check_same_thread=False: _connect(self.db_path))
        self.portfolio_conn.start()
        self.journal_conn.start()
        self.addCleanup(self.portfolio_conn.stop)
        self.addCleanup(self.journal_conn.stop)
        engine_portfolio.init_db()

    def _seed_checkpoint(self):
        with engine_portfolio.get_connection() as conn:
            cursor = conn.cursor()
            trade_log_id = engine_portfolio._record_trade_log(
                cursor,
                symbol="NVDA",
                action="buy",
                price=100.0,
                shares=2.0,
                settle_currency="CASH_USD",
                settle_amount=200.0,
                fx_rate=32.0,
                decision_snapshot={
                    "captured_at": "2026-04-29T01:00:00Z",
                    "symbol": "NVDA",
                    "benchmark_symbol": "SPY",
                    "sector_proxy_symbol": "SOXX",
                    "beta_proxy_at_entry": 1.2,
                    "entry_notional_twd": 6400.0,
                },
            )
            conn.commit()
        with patch.object(engine_journal, "_load_price_on_or_after", side_effect=[100.0, 105.0]):
            engine_journal.enqueue_trade_outcome_checkpoints([trade_log_id])
        return trade_log_id

    @patch.object(engine_journal, "_load_price_on_or_after", side_effect=[110.0, 104.0, 108.0])
    def test_settle_due_trade_outcomes_computes_additive_components(self, _price_mock):
        trade_log_id = self._seed_checkpoint()
        result = engine_journal.settle_due_trade_outcomes(as_of="2026-05-10T00:00:00Z")
        self.assertEqual(result["resolved"], 2)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT actual_return_pct, beta_component_pct, sector_component_pct, timing_component_pct "
                "FROM trade_outcome_checkpoints WHERE trade_log_id = ? AND horizon_label = 'T+5'",
                (trade_log_id,),
            ).fetchone()
        self.assertAlmostEqual(row[0], 0.10, places=6)
        self.assertAlmostEqual(row[1], 0.048, places=6)
        self.assertAlmostEqual(row[2] + row[1] + row[3], row[0], places=6)

    @patch.object(engine_journal, "_load_price_on_or_after", side_effect=[110.0, 104.0, 108.0, 115.0, 106.0, 109.0])
    def test_build_weekly_attribution_report_includes_coverage_and_totals(self, _price_mock):
        self._seed_checkpoint()
        engine_journal.settle_due_trade_outcomes(as_of="2026-05-10T00:00:00Z")
        report = engine_journal.build_weekly_attribution_report(as_of="2026-05-10T00:00:00Z")
        self.assertIn("Beta", report)
        self.assertIn("Sector", report)
        self.assertIn("Timing", report)
        self.assertIn("Coverage", report)
        self.assertIn("resolved checkpoints", report)
```

- [ ] **Step 2: Run the settlement tests and verify they fail.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest tests.test_trade_journal.TradeJournalSettlementTests
```

Expected: FAIL because `settle_due_trade_outcomes()`, `_load_price_on_or_after()`, and `build_weekly_attribution_report()` do not exist yet.

- [ ] **Step 3: Implement settlement, report generation, and journal tool registration.**

```python
# engine_journal.py
import json
from datetime import datetime, timezone

import pandas as pd

from src.tools import tool
from yf_session import get_ticker


def _load_price_on_or_after(symbol: str, as_of: str) -> float | None:
    start = datetime.fromisoformat(as_of.replace("Z", "+00:00")).date().isoformat()
    hist = get_ticker(symbol, cache_level="daily").history(start=start, period="10d")
    closes = hist.get("Close") if isinstance(hist, pd.DataFrame) else None
    if closes is None:
        return None
    closes = pd.Series(closes).dropna()
    if closes.empty:
        return None
    return round(float(closes.iloc[0]), 6)


def settle_due_trade_outcomes(as_of: str | None = None) -> dict[str, int]:
    cutoff = as_of or _utc_now_iso()
    resolved = 0
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            rows = cursor.execute(
                "SELECT id, symbol, due_at, entry_price, entry_notional_twd, benchmark_symbol, sector_proxy_symbol, "
                "beta_proxy_at_entry, beta_coverage, sector_coverage, benchmark_entry_price, sector_entry_price "
                "FROM trade_outcome_checkpoints WHERE status = 'pending' AND due_at <= ? ORDER BY id",
                (cutoff,),
            ).fetchall()
            for row in rows:
                actual_price = _load_price_on_or_after(row[1], row[2])
                benchmark_price = _load_price_on_or_after(row[5], row[2]) if row[5] else None
                sector_price = _load_price_on_or_after(row[6], row[2]) if row[6] else None
                benchmark_entry_price = row[10]
                sector_entry_price = row[11]
                if actual_price is None or benchmark_price is None:
                    cursor.execute(
                        "UPDATE trade_outcome_checkpoints SET status = 'error', retry_count = retry_count + 1, last_error = ? WHERE id = ?",
                        ("missing market price", row[0]),
                    )
                    continue
                actual_return = actual_price / row[3] - 1
                benchmark_return = benchmark_price / benchmark_entry_price - 1 if benchmark_entry_price else None
                beta_component = (row[7] or 0.0) * benchmark_return if row[8] else None
                if row[9] and sector_price is not None and beta_component is not None:
                    sector_return = sector_price / sector_entry_price - 1 if sector_entry_price else None
                    sector_component = sector_return - beta_component
                    timing_component = actual_return - sector_return
                elif beta_component is not None:
                    sector_return = None
                    sector_component = 0.0
                    timing_component = actual_return - beta_component
                else:
                    sector_return = None
                    sector_component = None
                    timing_component = None
                cursor.execute(
                    """
                    UPDATE trade_outcome_checkpoints
                    SET status = 'resolved',
                        resolved_price = ?,
                        benchmark_return_pct = ?,
                        sector_return_pct = ?,
                        actual_return_pct = ?,
                        benchmark_entry_price = ?,
                        sector_entry_price = ?,
                        beta_component_pct = ?,
                        sector_component_pct = ?,
                        timing_component_pct = ?,
                        beta_component_twd = ?,
                        sector_component_twd = ?,
                        timing_component_twd = ?,
                        resolved_at = ?
                    WHERE id = ?
                    """,
                    (
                        actual_price,
                        benchmark_return,
                        sector_return,
                        actual_return,
                        benchmark_entry_price,
                        sector_entry_price,
                        beta_component,
                        sector_component,
                        timing_component,
                        row[4] * beta_component if beta_component is not None else None,
                        row[4] * sector_component if sector_component is not None else None,
                        row[4] * timing_component if timing_component is not None else None,
                        cutoff,
                        row[0],
                    ),
                )
                resolved += 1
            conn.commit()
        finally:
            conn.close()
    return {"resolved": resolved}


def build_weekly_attribution_report(as_of: str | None = None) -> str:
    cutoff = datetime.fromisoformat((as_of or _utc_now_iso()).replace("Z", "+00:00"))
    window_start = cutoff - pd.Timedelta(days=7)
    with db_lock:
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM trade_outcome_checkpoints WHERE status = 'resolved' AND resolved_at >= ? AND resolved_at <= ?",
                conn,
                params=(window_start.isoformat().replace("+00:00", "Z"), cutoff.isoformat().replace("+00:00", "Z")),
            )
        finally:
            conn.close()
    if df.empty:
        return "Trade Journal Weekly Attribution\n- resolved checkpoints: 0"
    beta_cov = float(df["beta_coverage"].fillna(0).mean())
    sector_cov = float(df["sector_coverage"].fillna(0).mean())
    beta_twd = float(df["beta_component_twd"].fillna(0).sum())
    sector_twd = float(df["sector_component_twd"].fillna(0).sum())
    timing_twd = float(df["timing_component_twd"].fillna(0).sum())
    return (
        "Trade Journal Weekly Attribution\n"
        f"- resolved checkpoints: {len(df)}\n"
        f"- Beta: NT${beta_twd:+.0f}\n"
        f"- Sector: NT${sector_twd:+.0f}\n"
        f"- Timing: NT${timing_twd:+.0f}\n"
        f"- Coverage: beta {beta_cov:.0%}, sector {sector_cov:.0%}"
    )


@tool()
def get_trade_journal_weekly_report() -> str:
    return build_weekly_attribution_report()
```

```python
# src/bot.py
import engine_journal as journal

_TOOL_MODULES = tuple(
    module for module in (portfolio, risk, fundamentals, technical, market, fubon, journal) if module is not None
)
```

- [ ] **Step 4: Re-run the journal tests and verify they pass.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest tests.test_trade_journal
```

Expected: PASS, with the additive decomposition summing back to `actual_return_pct` and the weekly report exposing coverage counts.

- [ ] **Step 5: Commit the journal math layer.**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add engine_journal.py src/bot.py tests/test_trade_journal.py && \
git commit -m "feat: add trade journal attribution reports" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Wire scheduler jobs and run the final tracked verification suite

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/src/scheduler.py`
- Create: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal_scheduler.py`
- Test: `/home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics/tests/test_trade_journal_scheduler.py`

- [ ] **Step 1: Write failing tests for scheduler registration and job behavior.**

```python
# tests/test_trade_journal_scheduler.py
import unittest
from unittest.mock import MagicMock, patch

import src.scheduler as scheduler


class TradeJournalSchedulerTests(unittest.TestCase):
    @patch("src.scheduler.BackgroundScheduler")
    def test_start_scheduler_registers_trade_journal_jobs(self, scheduler_cls):
        scheduler_obj = MagicMock()
        scheduler_obj.running = False
        scheduler_cls.return_value = scheduler_obj
        with patch("src.scheduler.macro_brain_heartbeat"), patch("src.scheduler.daily_portfolio_review"):
            scheduler.start_scheduler()
        added_ids = [call.kwargs["id"] for call in scheduler_obj.add_job.call_args_list]
        self.assertIn("trade-journal-settlement", added_ids)
        self.assertIn("weekly-trade-journal", added_ids)

    @patch("engine_journal.settle_due_trade_outcomes", return_value={"resolved": 2})
    def test_trade_journal_checkpoint_job_calls_settlement(self, settle_mock):
        result = scheduler.trade_journal_checkpoint_job()
        self.assertEqual(result["resolved"], 2)
        settle_mock.assert_called_once()
```

- [ ] **Step 2: Run the scheduler tests and verify they fail.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest tests.test_trade_journal_scheduler
```

Expected: FAIL because the journal scheduler jobs do not exist yet.

- [ ] **Step 3: Add the scheduler jobs with explicit safe timings.**

```python
# src/scheduler.py
def trade_journal_checkpoint_job():
    try:
        import engine_journal as journal

        result = journal.settle_due_trade_outcomes()
        logger.info("🧾 [TradeJournal] resolved=%s", result.get("resolved"))
        return result
    except Exception as exc:
        logger.error(f"Trade journal checkpoint job failed: {exc}")
        return None


def weekly_trade_journal_job():
    try:
        import engine_journal as journal
        from src import bot as bot_runtime

        report = journal.build_weekly_attribution_report()
        bot_runtime._send_or_edit(bot_runtime.AUTHORIZED_USER_ID, report)
        logger.info("🧾 [TradeJournalWeekly] weekly attribution sent")
        return report
    except Exception as exc:
        logger.error(f"Weekly trade journal job failed: {exc}")
        return None


# inside start_scheduler()
scheduler.add_job(
    trade_journal_checkpoint_job,
    "cron",
    day_of_week="tue-sat",
    hour=7,
    minute=15,
    id="trade-journal-settlement",
    replace_existing=True,
)
scheduler.add_job(
    weekly_trade_journal_job,
    "cron",
    day_of_week="sun",
    hour=18,
    minute=0,
    id="weekly-trade-journal",
    replace_existing=True,
)
```

- [ ] **Step 4: Re-run the scheduler tests and then the full tracked verification suite.**

Run:

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
/home/margincaller/MarginCall_2X/venv/bin/python -m unittest tests.test_trade_journal tests.test_trade_journal_scheduler && \
/home/margincaller/MarginCall_2X/venv/bin/python -m py_compile engine_portfolio.py engine_journal.py src/bot.py src/scheduler.py main.py src/database.py
```

Expected: all tracked tests PASS and all touched runtime files compile cleanly.

- [ ] **Step 5: Commit the scheduler/runtime layer.**

```bash
cd /home/margincaller/MarginCall_2X/.worktrees/cognitive-risk-analytics && \
git add src/scheduler.py tests/test_trade_journal_scheduler.py engine_journal.py src/bot.py engine_portfolio.py && \
git commit -m "feat: wire trade journal scheduler" -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## Self-review checklist

Before executing this plan, confirm:

1. Every spec requirement maps to a task in this file.
2. All new tracked tests live under `tests/` instead of ignored root-level patterns.
3. `engine_portfolio.py` never calls `engine_journal.enqueue_trade_outcome_checkpoints()` while still inside `with db_lock:`.
4. The additive formulas still satisfy:
   - `beta_component_pct + sector_component_pct + timing_component_pct = actual_return_pct`
5. The weekly report prints coverage, not just totals.
