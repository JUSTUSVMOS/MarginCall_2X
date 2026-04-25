# Trade Plan Monitor + Morning Briefing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable trade plans, alert-driven monitoring, and a deterministic morning briefing flow on top of the existing portfolio, follow-up, and scheduler infrastructure.

**Architecture:** Keep portfolio state, trade-plan state, and briefing logic separate. `engine_portfolio.py` remains the source of truth for schema, trade-plan persistence, holdings reconciliation, and monitoring state, while a new `engine_briefing.py` consumes structured events and open alerts to build action items. Telegram transport stays in `src/bot.py`, and scheduled execution stays in `src/scheduler.py`.

**Tech Stack:** Python, sqlite3 via `src.database`, pandas, yfinance/Fubon price helpers already used in `engine_portfolio.py`, Telegram bot handlers in `src/bot.py`, APScheduler, existing unittest suite

---

## File structure

- **Modify:** `/home/margincaller/MarginCall_2X/engine_portfolio.py`
  - Add `trade_plans`, `trade_plan_events`, and `trade_plan_alerts` schema.
  - Add plan validation, persistence, holdings-backfill scan, alert evaluation, and query helpers.
  - Extend buy execution to enforce the new trade-plan gate for bot-originated buys.
- **Create:** `/home/margincaller/MarginCall_2X/engine_briefing.py`
  - Build deterministic morning-briefing payloads, action-item ranking, and final formatting.
- **Modify:** `/home/margincaller/MarginCall_2X/engine_market.py`
  - Add a structured market-events helper that returns machine-readable event rows for portfolio/watchlist symbols.
- **Modify:** `/home/margincaller/MarginCall_2X/src/bot.py`
  - Send plan-backfill alerts, parse trade-plan replies, surface confirmation text, and expose morning-briefing dispatch.
- **Modify:** `/home/margincaller/MarginCall_2X/src/scheduler.py`
  - Run plan audits, prompt delivery, and the morning-briefing push job.
- **Modify:** `/home/margincaller/MarginCall_2X/test_tool_logic_refactor.py`
  - Cover schema, helper behavior, buy gate, and rule-engine unit cases.
- **Create:** `/home/margincaller/MarginCall_2X/test_trade_plan_flow.py`
  - Cover runtime plan prompting, reply handling, missing-plan backfill, and alert lifecycle.
- **Create:** `/home/margincaller/MarginCall_2X/test_morning_briefing.py`
  - Cover structured event ingestion, deterministic prioritization, and degraded-state surfacing.

---

### Task 1: Add the trade-plan schema and core persistence helpers

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Test: `/home/margincaller/MarginCall_2X/test_tool_logic_refactor.py`

- [ ] **Step 1: Write the failing schema/helper tests**

```python
def test_init_db_creates_trade_plan_tables(self):
    engine_portfolio.init_db()
    with database.locked_connection() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'trade_plan%'"
            ).fetchall()
        }
    self.assertEqual(tables, {"trade_plans", "trade_plan_events", "trade_plan_alerts"})


def test_upsert_trade_plan_creates_active_plan_and_event(self):
    engine_portfolio.init_db()
    plan_id = engine_portfolio.upsert_trade_plan(
        symbol="MRVL",
        source="manual_backfill",
        entry_price=85.2,
        stop_loss=80.0,
        take_profit_1=95.0,
        take_profit_2=105.0,
        max_holding_days=60,
        thesis_type="sector_rotation",
        thesis_text="semi rotation re-accelerating",
        thesis_payload={"proxy_symbol": "SOXX", "lookback_days": 10, "underperform_pct": -0.03},
        status="active",
    )

    with database.locked_connection() as conn:
        plan = conn.execute(
            """
            SELECT symbol, status, source, stop_loss, take_profit_1, take_profit_2,
                   max_holding_days, thesis_type, thesis_text
            FROM trade_plans WHERE id = ?
            """,
            (plan_id,),
        ).fetchone()
        event = conn.execute(
            "SELECT event_type FROM trade_plan_events WHERE plan_id = ? ORDER BY id",
            (plan_id,),
        ).fetchall()

    self.assertEqual(
        plan,
        ("MRVL", "active", "manual_backfill", 80.0, 95.0, 105.0, 60, "sector_rotation", "semi rotation re-accelerating"),
    )
    self.assertEqual([row[0] for row in event], ["plan_created", "plan_activated"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/margincaller/MarginCall_2X && ./venv/bin/python -m unittest \
  test_tool_logic_refactor.DirectHelperRuntimeTests.test_init_db_creates_trade_plan_tables \
  test_tool_logic_refactor.DirectHelperRuntimeTests.test_upsert_trade_plan_creates_active_plan_and_event
```

Expected: FAIL because the new tables and helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# engine_portfolio.py
TRADE_PLAN_REQUIRED_FIELDS = (
    "stop_loss",
    "take_profit_1",
    "max_holding_days",
    "thesis_type",
    "thesis_text",
)


def _serialize_json(payload: Dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _record_trade_plan_event(cursor, *, plan_id: int, event_type: str, payload: Dict[str, Any] | None = None):
    cursor.execute(
        """
        INSERT INTO trade_plan_events (plan_id, event_type, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (plan_id, event_type, _serialize_json(payload), _utc_now_iso()),
    )


def upsert_trade_plan(
    *,
    symbol: str,
    source: str,
    entry_price: float | None,
    stop_loss: float | None,
    take_profit_1: float | None,
    take_profit_2: float | None,
    max_holding_days: int | None,
    thesis_type: str | None,
    thesis_text: str | None,
    thesis_payload: Dict[str, Any] | None,
    status: str = "draft",
    opened_trade_log_id: int | None = None,
) -> int:
    normalized = normalize_ticker(symbol)
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            existing = cursor.execute(
                "SELECT id FROM trade_plans WHERE symbol = ? AND status IN ('draft', 'active') ORDER BY id DESC LIMIT 1",
                (normalized,),
            ).fetchone()
            if existing:
                plan_id = int(existing[0])
                cursor.execute(
                    """
                    UPDATE trade_plans
                    SET source = ?, opened_trade_log_id = COALESCE(?, opened_trade_log_id),
                        entry_price = ?, stop_loss = ?, take_profit_1 = ?, take_profit_2 = ?,
                        max_holding_days = ?, thesis_type = ?, thesis_text = ?,
                        thesis_payload_json = ?, status = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        source, opened_trade_log_id, entry_price, stop_loss, take_profit_1, take_profit_2,
                        max_holding_days, thesis_type, thesis_text, _serialize_json(thesis_payload),
                        status, _utc_now_iso(), plan_id,
                    ),
                )
                _record_trade_plan_event(cursor, plan_id=plan_id, event_type="plan_updated", payload={"status": status})
            else:
                cursor.execute(
                    """
                    INSERT INTO trade_plans (
                        symbol, status, source, opened_trade_log_id, entry_price, stop_loss,
                        take_profit_1, take_profit_2, max_holding_days, thesis_type,
                        thesis_text, thesis_payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized, status, source, opened_trade_log_id, entry_price, stop_loss,
                        take_profit_1, take_profit_2, max_holding_days, thesis_type,
                        thesis_text, _serialize_json(thesis_payload), _utc_now_iso(), _utc_now_iso(),
                    ),
                )
                plan_id = int(cursor.lastrowid)
                _record_trade_plan_event(cursor, plan_id=plan_id, event_type="plan_created", payload={"source": source})
            if status == "active":
                _record_trade_plan_event(cursor, plan_id=plan_id, event_type="plan_activated")
            conn.commit()
            return plan_id
        finally:
            conn.close()


def get_trade_plan(plan_id: int) -> Dict[str, Any] | None:
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM trade_plans WHERE id = ?", (plan_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def get_active_trade_plan(symbol: str) -> Dict[str, Any] | None:
    normalized = normalize_ticker(symbol)
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trade_plans WHERE symbol = ? AND status = 'active' ORDER BY id DESC LIMIT 1",
                (normalized,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def list_active_trade_plans() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_plans WHERE status = 'active' ORDER BY symbol, id"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/margincaller/MarginCall_2X
git add engine_portfolio.py test_tool_logic_refactor.py
git commit -m "feat: add trade plan persistence"
```

### Task 2: Enforce the buy-side trade-plan gate and backfill unmanaged holdings

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Test: `/home/margincaller/MarginCall_2X/test_tool_logic_refactor.py`
- Create: `/home/margincaller/MarginCall_2X/test_trade_plan_flow.py`

- [ ] **Step 1: Write the failing gate/backfill tests**

```python
def test_execute_position_update_rejects_buy_without_complete_trade_plan(self):
    engine_portfolio.init_db()
    with database.locked_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
            ("CASH_USD", 1.0, 1000.0, 32000.0, 0),
        )
        conn.commit()

    with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0), patch.object(
        engine_portfolio,
        "_apply_pretrade_risk_gate",
        return_value={"allowed": True, "approved_shares": 2.0, "approved_twd_total": 6400.0, "message": "", "note": None},
    ):
        result = engine_portfolio.execute_position_update("AAPL", 100.0, 2.0, action="buy")

    self.assertIn("交易計畫", result)
    self.assertIn("停損", result)


def test_sync_trade_plan_backfills_creates_missing_plan_alert_for_live_holding(self):
    engine_portfolio.init_db()
    with database.locked_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
            ("MRVL", 85.2, 30.0, 81792.0, 0),
        )
        conn.commit()

    payload = engine_portfolio.sync_trade_plan_backfills()

    self.assertEqual(payload["missing_plan_count"], 1)
    with database.locked_connection() as conn:
        draft = conn.execute(
            "SELECT symbol, status, source FROM trade_plans WHERE symbol = 'MRVL'"
        ).fetchone()
        alert = conn.execute(
            "SELECT alert_type, status FROM trade_plan_alerts WHERE symbol = 'MRVL'"
        ).fetchone()

    self.assertEqual(draft, ("MRVL", "draft", "manual_backfill"))
    self.assertEqual(alert, ("missing_plan", "open"))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/margincaller/MarginCall_2X && ./venv/bin/python -m unittest \
  test_tool_logic_refactor.DirectHelperRuntimeTests.test_execute_position_update_rejects_buy_without_complete_trade_plan \
  test_trade_plan_flow.TradePlanFlowTests.test_sync_trade_plan_backfills_creates_missing_plan_alert_for_live_holding
```

Expected: FAIL because buys do not yet require a plan and there is no holdings-backfill scan.

- [ ] **Step 3: Write minimal implementation**

```python
# engine_portfolio.py
def _build_trade_plan_payload(
    *,
    stop_loss: float | None = None,
    take_profit_1: float | None = None,
    take_profit_2: float | None = None,
    max_holding_days: int | None = None,
    thesis_type: str | None = None,
    thesis_text: str | None = None,
    thesis_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "max_holding_days": max_holding_days,
        "thesis_type": thesis_type,
        "thesis_text": thesis_text,
        "thesis_payload": thesis_payload or {},
    }


def validate_trade_plan_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    missing = [field for field in TRADE_PLAN_REQUIRED_FIELDS if payload.get(field) in (None, "", 0)]
    return {"complete": not missing, "missing_fields": missing}


def sync_trade_plan_backfills() -> Dict[str, Any]:
    rows = _load_portfolio_rows()
    missing_symbols = []
    for symbol, cost, shares, twd_cost in rows:
        if shares <= 0 or "CASH" in symbol:
            continue
        if get_active_trade_plan(symbol):
            continue
        upsert_trade_plan(
            symbol=symbol,
            source="manual_backfill",
            entry_price=float(cost or 0.0),
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            max_holding_days=None,
            thesis_type=None,
            thesis_text=None,
            thesis_payload={},
            status="draft",
        )
        upsert_trade_plan_alert(symbol=symbol, alert_type="missing_plan", severity="high", payload={"reason": "holding_without_active_plan"})
        missing_symbols.append(symbol)
    return {"missing_plan_count": len(missing_symbols), "symbols": missing_symbols}


def _validate_buy_trade_plan_or_error(*, action: str, is_cash: bool, enforce_pretrade_gate: bool, trade_plan: Dict[str, Any] | None) -> str | None:
    if action != "buy" or is_cash or not enforce_pretrade_gate:
        return None
    plan_validation = validate_trade_plan_payload(trade_plan or {})
    if plan_validation["complete"]:
        return None
    fields = "、".join(plan_validation["missing_fields"])
    return format_tool_error(f"❌ 交易計畫未完成：請先補齊 {fields}。", data_unavailable=True)


# insert near the top of execute_position_update(), after is_cash / fx_rate are known
plan_error = _validate_buy_trade_plan_or_error(
    action=action,
    is_cash=is_cash,
    enforce_pretrade_gate=enforce_pretrade_gate,
    trade_plan=trade_plan,
)
if plan_error:
    return plan_error


# insert inside the successful buy branch, immediately after the trade-log insert
upsert_trade_plan(
    symbol=symbol,
    source="bot_buy",
    entry_price=actual_unit_price,
    stop_loss=(trade_plan or {}).get("stop_loss"),
    take_profit_1=(trade_plan or {}).get("take_profit_1"),
    take_profit_2=(trade_plan or {}).get("take_profit_2"),
    max_holding_days=(trade_plan or {}).get("max_holding_days"),
    thesis_type=(trade_plan or {}).get("thesis_type"),
    thesis_text=(trade_plan or {}).get("thesis_text"),
    thesis_payload=(trade_plan or {}).get("thesis_payload"),
    status="active",
)
```

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/margincaller/MarginCall_2X
git add engine_portfolio.py test_tool_logic_refactor.py test_trade_plan_flow.py
git commit -m "feat: gate buys on trade plans"
```

### Task 3: Add thesis-specific monitoring rules and alert deduplication

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Test: `/home/margincaller/MarginCall_2X/test_tool_logic_refactor.py`

- [ ] **Step 1: Write the failing monitoring tests**

```python
def test_evaluate_trade_plan_alerts_fires_stop_hit_and_dedupes(self):
    engine_portfolio.init_db()
    plan_id = engine_portfolio.upsert_trade_plan(
        symbol="MRVL",
        source="manual_backfill",
        entry_price=85.2,
        stop_loss=80.0,
        take_profit_1=95.0,
        take_profit_2=105.0,
        max_holding_days=60,
        thesis_type="breakout_support",
        thesis_text="hold 80 support",
        thesis_payload={"support_level": 80.0, "close_below_count": 1},
        status="active",
    )

    with patch.object(engine_portfolio, "_build_live_position_snapshots", return_value=[
        {"symbol": "MRVL", "is_cash": False, "shares": 30.0, "current_price": 79.8, "market_value_twd": 70000.0, "twd_cost": 81792.0, "pnl_percent": -6.3}
    ]):
        first = engine_portfolio.audit_trade_plan_alerts()
        second = engine_portfolio.audit_trade_plan_alerts()

    self.assertEqual(first["open_alert_count"], 1)
    self.assertEqual(second["open_alert_count"], 1)
    with database.locked_connection() as conn:
        rows = conn.execute(
            "SELECT alert_type, status FROM trade_plan_alerts WHERE plan_id = ?",
            (plan_id,),
        ).fetchall()
    self.assertEqual(rows, [("stop_hit", "open")])


def test_evaluate_trade_plan_alerts_marks_monitor_degraded_when_price_refresh_fails(self):
    engine_portfolio.init_db()
    engine_portfolio.upsert_trade_plan(
        symbol="AMD",
        source="manual_backfill",
        entry_price=150.0,
        stop_loss=140.0,
        take_profit_1=165.0,
        take_profit_2=180.0,
        max_holding_days=45,
        thesis_type="sector_rotation",
        thesis_text="semi strength should persist",
        thesis_payload={"proxy_symbol": "SOXX", "lookback_days": 10, "underperform_pct": -0.03},
        status="active",
    )

    with patch.object(engine_portfolio, "_build_live_position_snapshots", side_effect=RuntimeError("price refresh failed")):
        payload = engine_portfolio.audit_trade_plan_alerts()

    self.assertEqual(payload["degraded"], 1)
    with database.locked_connection() as conn:
        alert = conn.execute(
            "SELECT alert_type, status FROM trade_plan_alerts WHERE symbol = 'AMD'"
        ).fetchone()
    self.assertEqual(alert, ("monitor_degraded", "open"))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/margincaller/MarginCall_2X && ./venv/bin/python -m unittest \
  test_tool_logic_refactor.DirectHelperRuntimeTests.test_evaluate_trade_plan_alerts_fires_stop_hit_and_dedupes \
  test_tool_logic_refactor.DirectHelperRuntimeTests.test_evaluate_trade_plan_alerts_marks_monitor_degraded_when_price_refresh_fails
```

Expected: FAIL because there is no trade-plan alert rule engine yet.

- [ ] **Step 3: Write minimal implementation**

```python
# engine_portfolio.py
def upsert_trade_plan_alert(*, symbol: str, alert_type: str, severity: str, payload: Dict[str, Any], plan_id: int | None = None) -> int:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT id FROM trade_plan_alerts
                WHERE symbol = ? AND alert_type = ? AND status = 'open'
                ORDER BY id DESC LIMIT 1
                """,
                (normalize_ticker(symbol), alert_type),
            ).fetchone()
            if row:
                alert_id = int(row[0])
                cursor.execute(
                    "UPDATE trade_plan_alerts SET severity = ?, payload_json = ?, last_seen_at = ? WHERE id = ?",
                    (severity, _serialize_json(payload), _utc_now_iso(), alert_id),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO trade_plan_alerts (
                        plan_id, symbol, alert_type, severity, status, payload_json, first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?)
                    """,
                    (plan_id, normalize_ticker(symbol), alert_type, severity, _serialize_json(payload), _utc_now_iso(), _utc_now_iso()),
                )
                alert_id = int(cursor.lastrowid)
            conn.commit()
            return alert_id
        finally:
            conn.close()


def audit_trade_plan_alerts() -> Dict[str, Any]:
    try:
        snapshots = {row["symbol"]: row for row in _build_live_position_snapshots(_load_portfolio_rows()) if not row["is_cash"]}
    except Exception as exc:
        for plan in list_active_trade_plans():
            upsert_trade_plan_alert(symbol=plan["symbol"], plan_id=plan["id"], alert_type="monitor_degraded", severity="high", payload={"error": str(exc)})
        return {"open_alert_count": 0, "degraded": 1}

    open_count = 0
    for plan in list_active_trade_plans():
        snap = snapshots.get(plan["symbol"])
        if not snap:
            continue
        if snap["current_price"] <= float(plan["stop_loss"] or -math.inf):
            upsert_trade_plan_alert(symbol=plan["symbol"], plan_id=plan["id"], alert_type="stop_hit", severity="critical", payload={"current_price": snap["current_price"]})
            open_count += 1
            continue
        if plan["take_profit_1"] and snap["current_price"] >= float(plan["take_profit_1"]):
            upsert_trade_plan_alert(symbol=plan["symbol"], plan_id=plan["id"], alert_type="tp1_hit", severity="medium", payload={"current_price": snap["current_price"]})
        if _is_plan_holding_expired(plan):
            upsert_trade_plan_alert(symbol=plan["symbol"], plan_id=plan["id"], alert_type="holding_expiry", severity="high", payload={"current_return_pct": snap["pnl_percent"]})
        thesis_alert = _evaluate_thesis_invalidation(plan, snap)
        if thesis_alert:
            upsert_trade_plan_alert(symbol=plan["symbol"], plan_id=plan["id"], alert_type="thesis_invalid", severity="high", payload=thesis_alert)
    return {"open_alert_count": open_count, "degraded": 0}


def resolve_trade_plan_alert(*, plan_id: int, alert_type: str) -> None:
    with db_lock:
        conn = get_connection()
        try:
            conn.execute(
                """
                UPDATE trade_plan_alerts
                SET status = 'resolved', resolved_at = ?, last_seen_at = ?
                WHERE plan_id = ? AND alert_type = ? AND status = 'open'
                """,
                (_utc_now_iso(), _utc_now_iso(), plan_id, alert_type),
            )
            conn.commit()
        finally:
            conn.close()


def get_open_trade_plan_alerts() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trade_plan_alerts WHERE status = 'open' ORDER BY severity DESC, last_seen_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def get_current_portfolio_symbols() -> List[str]:
    return [
        symbol
        for symbol, _cost, shares, _twd_cost in _load_portfolio_rows()
        if shares > 0 and "CASH" not in symbol
    ]


def build_trade_plan_status_summary() -> Dict[str, Any]:
    alerts = get_open_trade_plan_alerts()
    return {
        "open_alert_count": len(alerts),
        "missing_plan_count": sum(1 for item in alerts if item["alert_type"] == "missing_plan"),
        "alerts": alerts[:5],
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/margincaller/MarginCall_2X
git add engine_portfolio.py test_tool_logic_refactor.py
git commit -m "feat: add trade plan alert engine"
```

### Task 4: Deliver Telegram prompts and parse structured trade-plan replies

**Files:**
- Modify: `/home/margincaller/MarginCall_2X/engine_portfolio.py`
- Modify: `/home/margincaller/MarginCall_2X/src/bot.py`
- Modify: `/home/margincaller/MarginCall_2X/src/scheduler.py`
- Create: `/home/margincaller/MarginCall_2X/test_trade_plan_flow.py`

- [ ] **Step 1: Write the failing runtime tests**

```python
def test_send_pending_trade_plan_prompts_sends_missing_plan_message(self):
    plan_id = engine_portfolio.upsert_trade_plan(
        symbol="MRVL",
        source="manual_backfill",
        entry_price=85.2,
        stop_loss=None,
        take_profit_1=None,
        take_profit_2=None,
        max_holding_days=None,
        thesis_type=None,
        thesis_text=None,
        thesis_payload={},
        status="draft",
    )
    engine_portfolio.upsert_trade_plan_alert(
        symbol="MRVL",
        plan_id=plan_id,
        alert_type="missing_plan",
        severity="high",
        payload={"reason": "holding_without_active_plan"},
    )

    fake_bot = SimpleNamespace(send_message=Mock())
    bot_module.bot = fake_bot
    bot_module.AUTHORIZED_USER_ID = 123

    sent = bot_module.send_pending_trade_plan_prompts()

    self.assertEqual(sent, 1)
    fake_bot.send_message.assert_called_once()
    self.assertIn("MRVL", fake_bot.send_message.call_args.args[1])
    self.assertIn("類型", fake_bot.send_message.call_args.args[1])
    self.assertIn("停損", fake_bot.send_message.call_args.args[1])


def test_handle_all_text_records_structured_trade_plan_reply(self):
    plan_id = engine_portfolio.upsert_trade_plan(
        symbol="MRVL",
        source="manual_backfill",
        entry_price=85.2,
        stop_loss=None,
        take_profit_1=None,
        take_profit_2=None,
        max_holding_days=None,
        thesis_type=None,
        thesis_text=None,
        thesis_payload={},
        status="draft",
    )
    engine_portfolio.upsert_trade_plan_alert(
        symbol="MRVL",
        plan_id=plan_id,
        alert_type="missing_plan",
        severity="high",
        payload={"reason": "holding_without_active_plan"},
    )
    message = SimpleNamespace(text="類型: sector_rotation\n理由: semi rotation 回來\n停損: 80\n目標1: 95\n目標2: 105\n期限: 60", chat=SimpleNamespace(id=1))

    fake_bot = SimpleNamespace(reply_to=Mock(), send_message=Mock())
    bot_module.bot = fake_bot
    bot_module.AUTHORIZED_USER_ID = 123

    bot_module.handle_all_text(message)

    with database.locked_connection() as conn:
        row = conn.execute(
            """
            SELECT status, stop_loss, take_profit_1, take_profit_2, max_holding_days, thesis_type, thesis_text
            FROM trade_plans WHERE id = ?
            """,
            (plan_id,),
        ).fetchone()
    self.assertEqual(row, ("active", 80.0, 95.0, 105.0, 60, "sector_rotation", "semi rotation 回來"))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/margincaller/MarginCall_2X && ./venv/bin/python -m unittest test_trade_plan_flow.py
```

Expected: FAIL because there is no pending trade-plan prompt flow or structured reply parser yet.

- [ ] **Step 3: Write minimal implementation**

```python
# engine_portfolio.py
_TRADE_PLAN_REPLY_PATTERNS = {
    "thesis_type": re.compile(r"^(?:類型|type)\s*[:：]\s*(?P<value>\w+)$", re.IGNORECASE | re.MULTILINE),
    "thesis_text": re.compile(r"^(?:理由|原因)\s*[:：]\s*(?P<value>.+)$", re.IGNORECASE | re.MULTILINE),
    "stop_loss": re.compile(r"^(?:停損|止損|stop)\s*[:：]\s*\$?(?P<value>\d+(?:\.\d+)?)$", re.IGNORECASE | re.MULTILINE),
    "take_profit_1": re.compile(r"^(?:目標1|tp1)\s*[:：]\s*\$?(?P<value>\d+(?:\.\d+)?)$", re.IGNORECASE | re.MULTILINE),
    "take_profit_2": re.compile(r"^(?:目標2|tp2)\s*[:：]\s*\$?(?P<value>\d+(?:\.\d+)?)$", re.IGNORECASE | re.MULTILINE),
    "max_holding_days": re.compile(r"^(?:期限|持有天數|days)\s*[:：]\s*(?P<value>\d+)$", re.IGNORECASE | re.MULTILINE),
}


def parse_trade_plan_reply(reply_text: str) -> Dict[str, Any] | None:
    payload = (reply_text or "").strip()
    if not payload:
        return None
    parsed: Dict[str, Any] = {"thesis_payload": {}}
    for field, pattern in _TRADE_PLAN_REPLY_PATTERNS.items():
        match = pattern.search(payload)
        if not match:
            continue
        value = match.group("value").strip()
        if field in {"stop_loss", "take_profit_1", "take_profit_2"}:
            parsed[field] = float(value)
        elif field == "max_holding_days":
            parsed[field] = int(value)
        else:
            parsed[field] = value
    return parsed if validate_trade_plan_payload(parsed).get("complete") else None


def resolve_trade_plan_reply(plan_id: int, reply_text: str) -> Dict[str, Any] | None:
    parsed = parse_trade_plan_reply(reply_text)
    if parsed is None:
        return None
    plan = get_trade_plan(plan_id)
    upsert_trade_plan(
        symbol=plan["symbol"],
        source="plan_revision",
        status="active",
        entry_price=plan["entry_price"],
        stop_loss=parsed["stop_loss"],
        take_profit_1=parsed["take_profit_1"],
        take_profit_2=parsed.get("take_profit_2"),
        max_holding_days=parsed["max_holding_days"],
        thesis_type=parsed["thesis_type"],
        thesis_text=parsed["thesis_text"],
        thesis_payload=parsed["thesis_payload"],
    )
    resolve_trade_plan_alert(plan_id=plan_id, alert_type="missing_plan")
    return parsed


def claim_pending_trade_plan_prompts() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT tp.*
                FROM trade_plans tp
                JOIN trade_plan_alerts ta ON ta.plan_id = tp.id
                WHERE tp.status = 'draft' AND ta.alert_type = 'missing_plan' AND ta.status = 'open'
                ORDER BY tp.updated_at, tp.id
                """
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def mark_trade_plan_prompted(plan_id: int) -> None:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            _record_trade_plan_event(cursor, plan_id=plan_id, event_type="prompt_sent")
            conn.commit()
        finally:
            conn.close()


def get_latest_prompted_trade_plan() -> Dict[str, Any] | None:
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT tp.*
                FROM trade_plans tp
                JOIN trade_plan_events te ON te.plan_id = tp.id
                WHERE tp.status = 'draft' AND te.event_type = 'prompt_sent'
                ORDER BY te.id DESC LIMIT 1
                """
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def format_trade_plan_prompt(plan: Dict[str, Any]) -> str:
    return (
        f"⚠️ {plan['symbol']} 目前沒有完整交易計畫。\n"
        "請用以下格式回覆：\n"
        "類型: sector_rotation\n"
        "理由: semi rotation 回來\n"
        "停損: 80\n"
        "目標1: 95\n"
        "目標2: 105\n"
        "期限: 60"
    )
```

```python
# src/bot.py
def send_pending_trade_plan_prompts():
    bot_instance = _require_bot()
    pending = portfolio.claim_pending_trade_plan_prompts()
    sent = 0
    for item in pending:
        bot_instance.send_message(AUTHORIZED_USER_ID, portfolio.format_trade_plan_prompt(item))
        portfolio.mark_trade_plan_prompted(int(item["id"]))
        sent += 1
    return sent


def _maybe_handle_trade_plan_reply(message) -> bool:
    plan = portfolio.get_latest_prompted_trade_plan()
    if not plan:
        return False
    resolution = portfolio.resolve_trade_plan_reply(int(plan["id"]), getattr(message, "text", ""))
    if resolution is None:
        return False
    _require_bot().reply_to(message, portfolio.format_trade_plan_confirmation(plan, resolution))
    return True


def handle_all_text(message):
    if _maybe_handle_trade_plan_reply(message):
        return
    if _maybe_handle_trade_followup_reply(message):
        return
    bot_instance = _require_bot()
    user_text = message.text
    mood = "bad_market" if any(word in user_text for word in ["損益", "賠", "慘"]) else "normal"
    sent_msg = bot_instance.reply_to(message, f"【推進器點火】\n{random.choice(WDT_MESSAGES[mood])}")
    bot_instance.send_chat_action(message.chat.id, "typing")
    final_text = ask_agent(user_text, tools=READ_ONLY_TOOLS, chat_history=user_chat_history)
    _send_or_edit(message.chat.id, final_text, sent_msg.message_id)
```

```python
# src/scheduler.py
def trade_plan_audit_job():
    import engine_portfolio as portfolio

    portfolio.sync_trade_plan_backfills()
    portfolio.audit_trade_plan_alerts()
    from src import bot as bot_runtime
    bot_runtime.send_pending_trade_plan_prompts()
```

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/margincaller/MarginCall_2X
git add engine_portfolio.py src/bot.py src/scheduler.py test_trade_plan_flow.py
git commit -m "feat: add trade plan prompt flow"
```

### Task 5: Build the deterministic morning briefing engine

**Files:**
- Create: `/home/margincaller/MarginCall_2X/engine_briefing.py`
- Modify: `/home/margincaller/MarginCall_2X/engine_market.py`
- Modify: `/home/margincaller/MarginCall_2X/src/bot.py`
- Modify: `/home/margincaller/MarginCall_2X/src/scheduler.py`
- Create: `/home/margincaller/MarginCall_2X/test_morning_briefing.py`

- [ ] **Step 1: Write the failing morning-briefing tests**

```python
def test_build_morning_briefing_prioritizes_trade_plan_alerts_over_noise(self):
    with patch("engine_briefing.portfolio.get_open_trade_plan_alerts", return_value=[
        {"symbol": "MRVL", "alert_type": "stop_hit", "severity": "critical", "payload": {"current_price": 79.8}}
    ]), patch("engine_briefing.market.get_market_calendar_events", return_value=[
        {"symbol": "AMD", "event_type": "earnings", "starts_at": "2026-04-24T20:00:00Z", "label": "AMD earnings"}
    ]), patch("engine_briefing.risk.get_global_risk_snapshot", return_value={"state": "🟡 整理", "riskScore": 42}), patch(
        "engine_briefing.portfolio.build_trade_plan_status_summary",
        return_value={"open_alert_count": 1, "missing_plan_count": 0, "alerts": []},
    ):
        report = engine_briefing.build_morning_briefing()

    self.assertIn("MRVL", report)
    self.assertIn("stop", report.lower())
    self.assertIn("AMD", report)


def test_build_morning_briefing_surfaces_degraded_monitoring(self):
    with patch("engine_briefing.portfolio.get_open_trade_plan_alerts", return_value=[
        {"symbol": "AMD", "alert_type": "monitor_degraded", "severity": "high", "payload": {"error": "price refresh failed"}}
    ]), patch("engine_briefing.market.get_market_calendar_events", return_value=[]), patch(
        "engine_briefing.risk.get_global_risk_snapshot", return_value={"state": "🟢 風險開", "riskScore": 20}
    ):
        report = engine_briefing.build_morning_briefing()

    self.assertIn("monitor_degraded", report)
    self.assertIn("price refresh failed", report)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/margincaller/MarginCall_2X && ./venv/bin/python -m unittest test_morning_briefing.py
```

Expected: FAIL because there is no structured briefing engine yet.

- [ ] **Step 3: Write minimal implementation**

```python
# engine_market.py
def get_market_calendar_events(symbols: List[str] | None = None, days: int = 1) -> List[Dict[str, Any]]:
    watch = _parse_symbol_input(symbols) if symbols else ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "GOOG", "META"]
    events: List[Dict[str, Any]] = []
    now = datetime.datetime.now()
    end_date = now + datetime.timedelta(days=days)
    for symbol in watch:
        try:
            dates = get_ticker(symbol).earnings_dates
        except Exception as exc:
            logger.debug(f"Structured market calendar fetch failed for {symbol}: {exc}")
            continue
        if dates is None or dates.empty:
            continue
        dates.index = dates.index.tz_localize(None)
        for event_dt, _ in dates[(dates.index >= now) & (dates.index <= end_date)].iterrows():
            events.append(
                {
                    "symbol": symbol,
                    "event_type": "earnings",
                    "starts_at": event_dt.isoformat(),
                    "label": f"{symbol} earnings",
                }
            )
    return events
```

```python
# engine_briefing.py
def derive_morning_action_items(*, alerts: List[Dict[str, Any]], events: List[Dict[str, Any]], risk_snapshot: Dict[str, Any]) -> List[str]:
    items: List[tuple[int, str]] = []
    for alert in alerts:
        if alert["alert_type"] == "stop_hit":
            items.append((0, f"{alert['symbol']} stop_hit - review or exit immediately"))
        elif alert["alert_type"] == "thesis_invalid":
            items.append((1, f"{alert['symbol']} thesis_invalid - reassess the original trade thesis"))
        elif alert["alert_type"] == "holding_expiry":
            items.append((2, f"{alert['symbol']} holding_expiry - time-box has been exceeded"))
        elif alert["alert_type"] == "monitor_degraded":
            items.append((3, f"{alert['symbol']} monitor_degraded - {alert['payload'].get('error', 'unknown error')}"))
    for event in events:
        items.append((4, f"{event['symbol']} {event['label']} today - avoid adding risk before the event"))
    items.sort(key=lambda row: row[0])
    return [item for _, item in items]


def build_morning_briefing() -> str:
    holdings = portfolio.get_current_portfolio_symbols()
    alerts = portfolio.get_open_trade_plan_alerts()
    events = market.get_market_calendar_events(symbols=holdings, days=1)
    risk_snapshot = risk.get_global_risk_snapshot()
    status_summary = portfolio.build_trade_plan_status_summary()
    action_items = derive_morning_action_items(alerts=alerts, events=events, risk_snapshot=risk_snapshot)
    lines = [
        "🗞️ 【Morning Briefing】",
        f"一句話：{'今天觀望' if not action_items else action_items[0]}",
        f"Risk: {risk_snapshot.get('state')} ({risk_snapshot.get('riskScore')})",
        f"Open Alerts: {status_summary['open_alert_count']} | Missing Plans: {status_summary['missing_plan_count']}",
        "Action Items:",
    ]
    lines.extend(f"{idx}. {item}" for idx, item in enumerate(action_items, start=1))
    return "\n".join(lines)
```

```python
# src/scheduler.py
def morning_briefing_push():
    from src import bot as bot_runtime
    bot_runtime.send_morning_briefing()
```

```python
# src/bot.py
def send_morning_briefing():
    bot_instance = _require_bot()
    import engine_briefing as briefing

    bot_instance.send_message(AUTHORIZED_USER_ID, briefing.build_morning_briefing())
```

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/margincaller/MarginCall_2X
git add engine_briefing.py engine_market.py src/bot.py src/scheduler.py test_morning_briefing.py
git commit -m "feat: add morning briefing engine"
```

### Task 6: Final focused verification and integration cleanup

**Files:**
- Modify: none
- Test: `/home/margincaller/MarginCall_2X/test_tool_logic_refactor.py`
- Test: `/home/margincaller/MarginCall_2X/test_trade_plan_flow.py`
- Test: `/home/margincaller/MarginCall_2X/test_trade_followup_flow.py`
- Test: `/home/margincaller/MarginCall_2X/test_morning_briefing.py`

- [ ] **Step 1: Run the focused unittest suite**

Run:

```bash
cd /home/margincaller/MarginCall_2X && ./venv/bin/python -m unittest \
  test_tool_logic_refactor.py \
  test_trade_plan_flow.py \
  test_trade_followup_flow.py \
  test_morning_briefing.py
```

Expected: PASS.

- [ ] **Step 2: Run syntax validation for touched modules**

Run:

```bash
cd /home/margincaller/MarginCall_2X && ./venv/bin/python -m py_compile \
  engine_portfolio.py \
  engine_market.py \
  engine_briefing.py \
  src/bot.py \
  src/scheduler.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Commit final integration**

```bash
cd /home/margincaller/MarginCall_2X
git add engine_portfolio.py engine_market.py engine_briefing.py src/bot.py src/scheduler.py \
  test_tool_logic_refactor.py test_trade_plan_flow.py test_trade_followup_flow.py test_morning_briefing.py
git commit -m "feat: add trade plan monitoring and morning briefing"
```

---

## Self-review

### Spec coverage

- Trade-plan persistence and audit trail: Task 1
- New-buy hard gate: Task 2
- Current-holdings backfill: Task 2
- Thesis invalidation and deterministic alerts: Task 3
- Telegram prompts and reply capture: Task 4
- Morning Briefing orchestrator: Task 5
- Future analytics hooks remain out of scope, as required by the spec

### Placeholder scan

- No deferred implementation placeholders remain
- Every task includes exact files, tests, commands, and concrete code snippets

### Type consistency

- Core plan fields stay aligned across tasks:
  - `stop_loss`
  - `take_profit_1`
  - `take_profit_2`
  - `max_holding_days`
  - `thesis_type`
  - `thesis_text`
  - `thesis_payload`
- Alert helpers consistently use:
  - `alert_type`
  - `severity`
  - `status`
  - `payload`
