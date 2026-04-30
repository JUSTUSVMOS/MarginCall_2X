"""
engine_journal.py — Trade outcome checkpoint scheduling.

Enqueues T+5 / T+20 business-day checkpoints for eligible buy trades so that
post-trade performance can be measured against entry, benchmark, and sector prices.
"""

import logging
from datetime import date, datetime, timedelta, timezone

from src.database import db_lock, get_connection

logger = logging.getLogger(__name__)

# Horizons to track: (label, calendar days that approximate N business days)
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
        from yf_session import get_ticker

        ticker = get_ticker(symbol)
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


def enqueue_trade_outcome_checkpoints(trade_log_ids: list[int]) -> int:
    """Insert pending checkpoint rows for each eligible trade ID.

    Skips IDs whose action is not in ELIGIBLE_TRADE_ACTIONS and silently ignores
    duplicate (trade_log_id, horizon_label) pairs via INSERT OR IGNORE.

    Returns the number of rows actually inserted.
    """
    if not trade_log_ids:
        return 0

    placeholders = ",".join("?" * len(trade_log_ids))
    today = datetime.now(timezone.utc).date()
    inserted = 0

    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id, symbol, action, price FROM trade_log WHERE id IN ({placeholders})",
                trade_log_ids,
            )
            rows = cursor.fetchall()

            for row in rows:
                trade_id, symbol, action, entry_price = row[0], row[1], row[2], row[3]
                if action not in ELIGIBLE_TRADE_ACTIONS:
                    continue

                benchmark_sym = _DEFAULT_BENCHMARK
                sector_sym = _resolve_sector_symbol(symbol)

                for label, bdays in CHECKPOINT_HORIZONS:
                    due_date = _add_business_days(today, bdays)
                    bmark_price = _load_price_on_or_after(benchmark_sym, due_date)
                    sector_price = _load_price_on_or_after(sector_sym, due_date)

                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO trade_outcome_checkpoints
                            (trade_log_id, horizon_label, symbol, action, entry_price,
                             due_date, benchmark_symbol, benchmark_entry_price,
                             sector_symbol, sector_entry_price)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trade_id,
                            label,
                            symbol,
                            action,
                            entry_price,
                            due_date.isoformat(),
                            benchmark_sym,
                            bmark_price,
                            sector_sym,
                            sector_price,
                        ),
                    )
                    inserted += cursor.rowcount

            conn.commit()
        finally:
            conn.close()

    return inserted


def _resolve_sector_symbol(symbol: str) -> str:
    """Return the sector ETF proxy for *symbol*, falling back to SPY."""
    try:
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
