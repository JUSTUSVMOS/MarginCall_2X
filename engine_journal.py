"""
engine_journal.py — Trade outcome checkpoint scheduling.

Enqueues T+5 / T+20 business-day checkpoints for eligible buy trades so that
post-trade performance can be measured against entry, benchmark, and sector prices.
"""

import logging
from datetime import date, datetime, timedelta, timezone

from src.database import db_lock, get_connection

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
    on or after it).  Returns None when no data is available (e.g. future date)."""
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
                            trade_ts or "",
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
