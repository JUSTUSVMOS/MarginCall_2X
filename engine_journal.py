"""
engine_journal.py — Trade outcome checkpoint scheduling and settlement.

Enqueues T+5 / T+20 business-day checkpoints for eligible buy trades so that
post-trade performance can be measured against entry, benchmark, and sector prices.
Provides additive attribution settlement and weekly report generation.
"""

import logging
import sqlite3
from datetime import date, datetime, timedelta, timezone

from src.database import db_lock, get_connection
from src.tools import tool

logger = logging.getLogger(__name__)

# Horizons to track: (label, business days to horizon)
CHECKPOINT_HORIZONS = (("T+5", 5), ("T+20", 20))

# Only these trade actions trigger outcome checkpoints
ELIGIBLE_TRADE_ACTIONS = {"buy", "sync_buy"}

# Default benchmark used for all checkpoints
_DEFAULT_BENCHMARK = "SPY"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _add_business_days(from_date: date, n: int) -> date:
    """Return the date that is *n* business days (Mon–Fri) after *from_date*."""
    current = from_date
    remaining = n
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon=0 … Fri=4
            remaining -= 1
    return current


def _load_price_on_or_after(symbol: str, target_date: date) -> float | None:
    """Return the closing price of *symbol* on *target_date* (or the earliest available date
    on or after it).  Returns None when no data is available (e.g. future date).

    *target_date* must be a ``datetime.date`` object, not a string — callers are responsible
    for converting ISO-format strings before passing them here (Task 2/3 code should do the
    same; do not pass raw timestamp strings directly).
    """
    try:
        # Deferred import: yf_session is optional at module load time.
        from yf_session import get_ticker

        ticker = get_ticker(symbol)
        # 7-day window absorbs weekends and market holidays near the target date.
        end_date = target_date + timedelta(days=7)
        hist = ticker.history(
            start=target_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=True,
        )
        if hist.empty:
            return None
        return float(hist["Close"].iloc[0])
    except Exception as exc:
        logger.debug("_load_price_on_or_after(%s, %s) failed: %s", symbol, target_date, exc)
        return None


def _resolve_sector_symbol(symbol: str) -> str:
    """Return the sector ETF proxy for *symbol*, falling back to SPY."""
    try:
        # Deferred import: yf_session is optional at module load time.
        from yf_session import get_ticker

        info = get_ticker(symbol).info or {}
        sector = str(info.get("sector") or "")
        industry = str(info.get("industry") or "").lower()

        if "semiconductor" in industry:
            return "SOXX"

        _sector_map = {
            "Technology": "XLK",
            "Communication Services": "XLC",
            "Energy": "XLE",
            "Financial Services": "XLF",
            "Healthcare": "XLV",
            "Industrials": "XLI",
            "Utilities": "XLU",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Real Estate": "XLRE",
            "Materials": "XLB",
        }
        return _sector_map.get(sector, _DEFAULT_BENCHMARK)
    except Exception:
        return _DEFAULT_BENCHMARK


def enqueue_trade_outcome_checkpoints(trade_log_ids: list[int]) -> dict:
    """Insert pending checkpoint rows for each eligible trade ID.

    Skips IDs whose action is not in ELIGIBLE_TRADE_ACTIONS and silently ignores
    duplicate (trade_log_id, horizon_label) pairs via INSERT OR IGNORE.

    Returns {"created": <number of rows actually inserted>}.
    """
    if not trade_log_ids:
        return {"created": 0}

    import json

    placeholders = ",".join("?" * len(trade_log_ids))
    inserted = 0

    # Phase 1: fetch trade rows — release lock before any network I/O.
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, symbol, action, price, shares, timestamp, decision_snapshot"
                f" FROM trade_log WHERE id IN ({placeholders})",
                trade_log_ids,
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

    # Phase 2: resolve sector and prices outside the lock (network I/O).
    work_items = []
    for row in rows:
        trade_id, symbol, action, entry_price = row[0], row[1], row[2], row[3]
        shares = row[4] or 0
        trade_ts = row[5]
        decision_snapshot_raw = row[6] if len(row) > 6 else None

        if action not in ELIGIBLE_TRADE_ACTIONS:
            continue

        # Parse the decision snapshot once for all derived fields.
        snap: dict = {}
        if decision_snapshot_raw:
            try:
                snap = json.loads(decision_snapshot_raw) or {}
            except Exception:
                pass

        # entry_notional_twd: prefer snapshot value (already in TWD), fall back to price × shares.
        if snap.get("entry_notional_twd") is not None:
            entry_notional_twd = float(snap["entry_notional_twd"])
        else:
            entry_notional_twd = float(entry_price * shares)

        # benchmark_symbol: prefer snapshot, fall back to module-level default.
        benchmark_sym = snap.get("benchmark_symbol") or _DEFAULT_BENCHMARK

        # Sector proxy: prefer snapshot, fall back to live lookup.
        sector_sym = snap.get("sector_proxy_symbol") or _resolve_sector_symbol(symbol)

        # Beta coverage metadata from snapshot.
        beta_proxy_at_entry = snap.get("beta_proxy_at_entry")

        try:
            trade_date = (
                datetime.fromisoformat(trade_ts.replace("Z", "+00:00"))
                .astimezone(timezone.utc)
                .date()
            )
        except Exception:
            logger.warning(
                "enqueue: could not parse trade timestamp %r for trade_id=%s; using today",
                trade_ts,
                trade_id,
            )
            trade_date = datetime.now(timezone.utc).date()

        bmark_entry = _load_price_on_or_after(benchmark_sym, trade_date)
        sector_entry = _load_price_on_or_after(sector_sym, trade_date)

        work_items.append(
            (trade_id, symbol, action, entry_price, entry_notional_twd, trade_ts, trade_date,
             benchmark_sym, bmark_entry, sector_sym, sector_entry, beta_proxy_at_entry)
        )

    # Phase 3: write checkpoint rows under lock.
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            for (trade_id, symbol, action, entry_price, entry_notional_twd, trade_ts, trade_date,
                 benchmark_sym, bmark_entry, sector_sym, sector_entry,
                 beta_proxy_at_entry) in work_items:
                beta_coverage = 1 if beta_proxy_at_entry is not None else 0
                sector_coverage = 1 if sector_entry is not None else 0
                for label, bdays in CHECKPOINT_HORIZONS:
                    due_date = _add_business_days(trade_date, bdays)
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO trade_outcome_checkpoints
                            (trade_log_id, horizon_label, symbol, action, entry_price,
                             entry_notional_twd, entry_timestamp, due_at,
                             benchmark_symbol, benchmark_entry_price,
                             sector_proxy_symbol, sector_entry_price,
                             beta_proxy_at_entry, beta_coverage, sector_coverage)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trade_id,
                            label,
                            symbol,
                            action,
                            entry_price,
                            entry_notional_twd,
                            trade_ts or _utc_now_iso(),
                            due_date.isoformat(),
                            benchmark_sym,
                            bmark_entry,
                            sector_sym,
                            sector_entry,
                            beta_proxy_at_entry,
                            beta_coverage,
                            sector_coverage,
                        ),
                    )
                    inserted += cursor.rowcount
            conn.commit()
        finally:
            conn.close()

    return {"created": inserted}


def settle_due_trade_outcomes(as_of: date | None = None) -> dict:
    """Settle all pending checkpoints whose due_at <= as_of.

    For each due checkpoint:
    - Fetches the resolved price for the stock, benchmark, and (if covered) sector.
    - Computes additive attribution: beta + sector + timing = actual_return_pct.
    - Updates the row to status='resolved' with all computed fields.
    - Rows whose resolved price is unavailable are left pending and retry_count is incremented.

    Returns {"settled": N, "errors": M}.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc).date()

    # Phase 1: load pending due rows under lock.
    with db_lock:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM trade_outcome_checkpoints"
                " WHERE status='pending' AND due_at <= ?",
                (as_of.isoformat(),),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

    # Phase 2: resolve prices and compute attribution outside the lock (network I/O).
    settlements = []
    errors = []

    for row in rows:
        row_id = row["id"]
        symbol = row["symbol"]
        entry_price = float(row["entry_price"])
        entry_notional_twd = float(row["entry_notional_twd"] or 0)
        due_date = date.fromisoformat(row["due_at"])
        benchmark_sym = row["benchmark_symbol"]
        benchmark_entry = row["benchmark_entry_price"]
        sector_sym = row["sector_proxy_symbol"]
        sector_entry = row["sector_entry_price"]
        beta_proxy = row["beta_proxy_at_entry"]
        sector_coverage = int(row["sector_coverage"] or 0)

        try:
            resolved_price = _load_price_on_or_after(symbol, due_date)
            if resolved_price is None:
                errors.append((row_id, "resolved price unavailable"))
                continue

            actual_return_pct = (resolved_price - entry_price) / entry_price * 100

            # Benchmark return
            benchmark_return_pct = None
            if benchmark_sym and benchmark_entry:
                bmark_resolved = _load_price_on_or_after(benchmark_sym, due_date)
                if bmark_resolved is not None:
                    benchmark_return_pct = (
                        (bmark_resolved - float(benchmark_entry)) / float(benchmark_entry) * 100
                    )

            # Sector return (only when sector coverage was recorded)
            sector_return_pct = None
            if sector_coverage and sector_sym and sector_entry:
                sec_resolved = _load_price_on_or_after(sector_sym, due_date)
                if sec_resolved is not None:
                    sector_return_pct = (
                        (sec_resolved - float(sector_entry)) / float(sector_entry) * 100
                    )

            # Additive attribution: beta + sector + timing = actual
            # Only use the no-sector path when sector coverage was never recorded (sector_coverage=0).
            # When sector_coverage=1 but the resolved price is missing, leave sector/timing as None
            # rather than silently inventing zero sector contribution.
            beta_component_pct = None
            sector_component_pct = None
            timing_component_pct = None

            if beta_proxy is not None and benchmark_return_pct is not None:
                beta_component_pct = float(beta_proxy) * benchmark_return_pct
                if sector_return_pct is not None:
                    # Full three-way split.
                    sector_component_pct = sector_return_pct - beta_component_pct
                    timing_component_pct = actual_return_pct - sector_return_pct
                elif not sector_coverage:
                    # No sector proxy was ever tracked — collapse to beta + timing.
                    sector_component_pct = 0.0
                    timing_component_pct = actual_return_pct - beta_component_pct
                # else: sector_coverage=1 but resolved price missing — leave None to avoid fake numbers.

            # TWD P&L components
            def _to_twd(pct):
                return pct / 100 * entry_notional_twd if pct is not None else None

            settlements.append({
                "id": row_id,
                "resolved_price": resolved_price,
                "actual_return_pct": actual_return_pct,
                "benchmark_return_pct": benchmark_return_pct,
                "sector_return_pct": sector_return_pct,
                "beta_component_pct": beta_component_pct,
                "sector_component_pct": sector_component_pct,
                "timing_component_pct": timing_component_pct,
                "beta_component_twd": _to_twd(beta_component_pct),
                "sector_component_twd": _to_twd(sector_component_pct),
                "timing_component_twd": _to_twd(timing_component_pct),
            })
        except Exception as exc:
            logger.warning("settle error for checkpoint id=%s: %s", row_id, exc)
            errors.append((row_id, str(exc)))

    # Phase 3: write back under lock.
    now_iso = _utc_now_iso()
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            for s in settlements:
                cursor.execute(
                    """
                    UPDATE trade_outcome_checkpoints SET
                        status='resolved',
                        resolved_price=?,
                        actual_return_pct=?,
                        benchmark_return_pct=?,
                        sector_return_pct=?,
                        beta_component_pct=?,
                        sector_component_pct=?,
                        timing_component_pct=?,
                        beta_component_twd=?,
                        sector_component_twd=?,
                        timing_component_twd=?,
                        resolved_at=?
                    WHERE id=?
                    """,
                    (
                        s["resolved_price"],
                        s["actual_return_pct"],
                        s["benchmark_return_pct"],
                        s["sector_return_pct"],
                        s["beta_component_pct"],
                        s["sector_component_pct"],
                        s["timing_component_pct"],
                        s["beta_component_twd"],
                        s["sector_component_twd"],
                        s["timing_component_twd"],
                        now_iso,
                        s["id"],
                    ),
                )
            for row_id, err_msg in errors:
                cursor.execute(
                    """
                    UPDATE trade_outcome_checkpoints SET
                        last_error=?,
                        retry_count=COALESCE(retry_count, 0) + 1
                    WHERE id=?
                    """,
                    (err_msg, row_id),
                )
            conn.commit()
        finally:
            conn.close()

    return {"settled": len(settlements), "errors": len(errors)}


def build_weekly_attribution_report(as_of: date | None = None) -> dict:
    """Return a dict summarising resolved checkpoints from the past 7 days up to as_of.

    Keys: as_of, resolved_checkpoints, avg_actual_return_pct,
          avg_beta_component_pct, avg_sector_component_pct, avg_timing_component_pct,
          beta_coverage_count, sector_coverage_count.
    """
    if as_of is None:
        as_of = datetime.now(timezone.utc).date()

    week_start = as_of - timedelta(days=7)

    with db_lock:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM trade_outcome_checkpoints
                WHERE status='resolved'
                  AND date(resolved_at) >= ?
                  AND date(resolved_at) <= ?
                """,
                (week_start.isoformat(), as_of.isoformat()),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

    total = len(rows)

    def _avg(field):
        vals = [row[field] for row in rows if row[field] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "as_of": as_of.isoformat(),
        "resolved_checkpoints": total,
        "avg_actual_return_pct": _avg("actual_return_pct"),
        "avg_beta_component_pct": _avg("beta_component_pct"),
        "avg_sector_component_pct": _avg("sector_component_pct"),
        "avg_timing_component_pct": _avg("timing_component_pct"),
        "beta_coverage_count": sum(1 for r in rows if r["beta_coverage"]),
        "sector_coverage_count": sum(1 for r in rows if r["sector_coverage"]),
    }


@tool()
def get_trade_journal_weekly_report() -> str:
    """Return the trade journal weekly attribution report (past 7 days)."""
    r = build_weekly_attribution_report()
    n = r["resolved_checkpoints"]
    lines = [
        f"📊 Weekly Attribution Report (as of {r['as_of']})",
        f"Resolved checkpoints: {n}",
        f"Avg actual return:   {r['avg_actual_return_pct']:.2f}%",
        f"Beta component avg:   {r['avg_beta_component_pct']:.2f}%"
        f"  (Beta coverage: {r['beta_coverage_count']}/{n})",
        f"Sector component avg: {r['avg_sector_component_pct']:.2f}%"
        f"  (Sector coverage: {r['sector_coverage_count']}/{n})",
        f"Timing component avg: {r['avg_timing_component_pct']:.2f}%",
        "Coverage note: rows without beta_proxy_at_entry have no beta/sector/timing split.",
    ]
    return "\n".join(lines)
