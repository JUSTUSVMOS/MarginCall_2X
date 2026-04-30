import json
import math
import time
import os
import csv
import logging
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import fubon
from typing import Any, Dict, List
from config import WATCH_LIST
import engine_market as market
from yf_session import get_ticker
from src.database import db_lock, get_connection
from src.symbols import normalize_ticker
from src.tools import format_tool_error, tool

logger = logging.getLogger(__name__)

CSV_BACKUP = "my_portfolio.csv"

# --- 匯率快取 ---
_fx_cache = {"rate": 32.0, "timestamp": 0}
_fx_cache_lock = threading.Lock()

TRADING_DAYS_PER_YEAR = 252
MIN_BETA_OBSERVATIONS = 20
DEFAULT_RISK_FREE_RATE = 0.04
DRAWDOWN_SOFT_THRESHOLD = 0.03
DRAWDOWN_HARD_THRESHOLD = 0.05
DRAWDOWN_DEFENSIVE_THRESHOLD = 0.08
DRAWDOWN_KILL_SWITCH_THRESHOLD = 0.10
PORTFOLIO_VOL_SOFT_SYMBOL_CAP = 5
PORTFOLIO_VOL_HARD_SYMBOL_CAP = 8
PORTFOLIO_VOL_MIN_WEIGHT_COVERAGE = 0.85
TRADE_SIZE_DECIMALS = 4
FUBON_SYNC_SHARE_TOL = 1e-6
FUBON_SYNC_COST_TOL = 1e-4
_TRADE_FOLLOWUP_UNSET = object()
TRADE_PLAN_REQUIRED_FIELDS = (
    "stop_loss",
    "take_profit_1",
    "max_holding_days",
    "thesis_type",
    "thesis_text",
)
TRADE_PLAN_FIELD_LABELS = {
    "stop_loss": "停損",
    "take_profit_1": "第一止盈",
    "max_holding_days": "最長持有天數",
    "thesis_type": "交易分類",
    "thesis_text": "交易理由",
}
TRADE_PLAN_COLUMNS = (
    "id",
    "symbol",
    "status",
    "source",
    "opened_trade_log_id",
    "entry_price",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "max_holding_days",
    "thesis_type",
    "thesis_text",
    "thesis_payload_json",
    "created_at",
    "updated_at",
)
TRADE_PLAN_SELECT = ", ".join(TRADE_PLAN_COLUMNS)
TRADE_PLAN_ALERT_COLUMNS = (
    "id",
    "plan_id",
    "symbol",
    "alert_type",
    "severity",
    "status",
    "payload_json",
    "first_seen_at",
    "last_seen_at",
    "resolved_at",
)
TRADE_PLAN_ALERT_SELECT = ", ".join(TRADE_PLAN_ALERT_COLUMNS)
TRADE_PLAN_MONITOR_ALERT_TYPES = (
    "stop_hit",
    "tp1_hit",
    "tp2_hit",
    "holding_expiry",
    "thesis_invalid",
)

TRADE_GOVERNOR_LIMITS = {
    "normal": {"single_name_cap": 0.15, "sector_cap": 0.35},
    "soft_throttle": {"single_name_cap": 0.12, "sector_cap": 0.30},
    "risk_off": {"single_name_cap": 0.10, "sector_cap": 0.25},
    "defensive": {"single_name_cap": 0.08, "sector_cap": 0.20},
    "kill_switch": {"single_name_cap": 0.05, "sector_cap": 0.15},
}

RISK_OVERLAY_TARGETS = {
    "🟢": {"label": "Risk-On", "beta_band": (0.80, 1.10), "target_vol": 0.18},
    "🟡": {"label": "Balanced", "beta_band": (0.40, 0.70), "target_vol": 0.12},
    "🔴": {"label": "Defensive", "beta_band": (0.15, 0.40), "target_vol": 0.08},
    "💀": {"label": "Capital Preservation", "beta_band": (0.00, 0.20), "target_vol": 0.05},
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def fetch_exchange_rate() -> float:
    """Pure FX-rate logic for direct callers and tests."""
    global _fx_cache
    current_time = time.time()
    with _fx_cache_lock:
        if current_time - _fx_cache["timestamp"] < 600:
            return _fx_cache["rate"]
    try:
        ticker = get_ticker("TWD=X")
        fast_info = getattr(ticker, "fast_info", {}) or {}
        rate = fast_info.get("last_price")
        if rate is None:
            hist = ticker.history(period="1d")
            if hist.empty:
                raise ValueError("TWD=X history is empty")
            rate = hist["Close"].iloc[-1]
        fresh_rate = round(float(rate), 2)
    except requests.RequestException as e:
        logger.warning(f"Exchange rate network refresh failed, using cache: {e}")
        with _fx_cache_lock:
            return _fx_cache["rate"]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.error(f"Exchange rate payload parsing failed, using cache: {e}")
        with _fx_cache_lock:
            return _fx_cache["rate"]
    except Exception:
        logger.exception("Exchange rate refresh failed unexpectedly, using cache")
        with _fx_cache_lock:
            return _fx_cache["rate"]

    with _fx_cache_lock:
        if current_time >= _fx_cache["timestamp"]:
            _fx_cache["rate"] = fresh_rate
            _fx_cache["timestamp"] = time.time()
        return _fx_cache["rate"]


@tool()
def get_exchange_rate() -> float:
    return fetch_exchange_rate()


def _upsert_portfolio_row(cursor, symbol: str, cost: float, shares: float, twd_cost: float, locked: int = 0):
    cursor.execute(
        "INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?, ?, ?)",
        (symbol, cost, shares, twd_cost, locked),
    )


def _record_trade_log(
    cursor,
    *,
    symbol: str,
    action: str,
    price: float,
    shares: float,
    settle_currency: str | None = None,
    settle_amount: float | None = None,
    fx_rate: float | None = None,
    realized_pnl: float | None = None,
    cash_before: float | None = None,
    cash_after: float | None = None,
    note: str | None = None,
    decision_snapshot: Dict[str, Any] | str | None = None,
):
    serialized_snapshot = decision_snapshot
    if isinstance(decision_snapshot, dict):
        serialized_snapshot = json.dumps(decision_snapshot, ensure_ascii=False, sort_keys=True)
    cursor.execute(
        """
        INSERT INTO trade_log (
            symbol, action, price, shares, settle_currency, settle_amount, fx_rate,
            realized_pnl, cash_before, cash_after, note, decision_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            symbol,
            action,
            price,
            shares,
            settle_currency,
            settle_amount,
            fx_rate,
            realized_pnl,
            cash_before,
            cash_after,
            note,
            serialized_snapshot,
        ),
    )
    return cursor.lastrowid


def _record_trade_followup(
    cursor,
    *,
    trade_log_id: int,
    symbol: str,
    action: str,
    prompt_text: str,
    prompt_state: str = "pending",
    status: str = "pending",
    user_reason: str | None = None,
    target_price: float | None = None,
    stop_price: float | None = None,
    skipped: int = 0,
):
    cursor.execute(
        """
        INSERT INTO trade_followups (
            trade_log_id, symbol, action, status, prompt_state, prompt_text,
            user_reason, target_price, stop_price, skipped, created_at, responded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade_log_id,
            symbol,
            action,
            status,
            prompt_state,
            prompt_text,
            user_reason,
            target_price,
            stop_price,
            int(skipped),
            _utc_now_iso(),
            None,
        ),
    )
    return cursor.lastrowid


def _serialize_json(payload: Dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _list_trade_followups(cursor, *, status: str | None = "pending"):
    if status is None:
        cursor.execute(
            "SELECT id, trade_log_id, symbol, action, status, prompt_state, prompt_text, user_reason, target_price, stop_price, skipped, created_at, responded_at FROM trade_followups ORDER BY id"
        )
    else:
        cursor.execute(
            "SELECT id, trade_log_id, symbol, action, status, prompt_state, prompt_text, user_reason, target_price, stop_price, skipped, created_at, responded_at FROM trade_followups WHERE status = ? ORDER BY id",
            (status,),
        )
    return cursor.fetchall()


def _update_trade_followup_status(
    cursor,
    followup_id: int,
    *,
    status: str,
    prompt_state: str | None = None,
    user_reason: Any = _TRADE_FOLLOWUP_UNSET,
    target_price: Any = _TRADE_FOLLOWUP_UNSET,
    stop_price: Any = _TRADE_FOLLOWUP_UNSET,
    skipped: Any = _TRADE_FOLLOWUP_UNSET,
    responded_at: Any = _TRADE_FOLLOWUP_UNSET,
):
    fields = ["status = ?"]
    params: List[Any] = [status]
    if prompt_state is not None:
        fields.append("prompt_state = ?")
        params.append(prompt_state)
    if user_reason is not _TRADE_FOLLOWUP_UNSET:
        fields.append("user_reason = ?")
        params.append(user_reason)
    if target_price is not _TRADE_FOLLOWUP_UNSET:
        fields.append("target_price = ?")
        params.append(target_price)
    if stop_price is not _TRADE_FOLLOWUP_UNSET:
        fields.append("stop_price = ?")
        params.append(stop_price)
    if skipped is not _TRADE_FOLLOWUP_UNSET:
        fields.append("skipped = ?")
        params.append(int(skipped))
    if responded_at is not _TRADE_FOLLOWUP_UNSET:
        fields.append("responded_at = ?")
        params.append(responded_at)
    params.append(followup_id)
    cursor.execute(f"UPDATE trade_followups SET {', '.join(fields)} WHERE id = ?", params)
    if cursor.rowcount == 0:
        raise ValueError(f"trade_followup {followup_id} not found")


def list_pending_trade_followups() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, trade_log_id, symbol, action, status, prompt_state, prompt_text,
                       user_reason, target_price, stop_price, skipped, created_at, responded_at
                FROM trade_followups
                WHERE status = 'pending' AND prompt_state = 'pending'
                ORDER BY id
                """
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

    columns = [
        "id",
        "trade_log_id",
        "symbol",
        "action",
        "status",
        "prompt_state",
        "prompt_text",
        "user_reason",
        "target_price",
        "stop_price",
        "skipped",
        "created_at",
        "responded_at",
    ]
    return [dict(zip(columns, row)) for row in rows]


def claim_pending_trade_followups() -> List[Dict[str, Any]]:
    columns = [
        "id",
        "trade_log_id",
        "symbol",
        "action",
        "status",
        "prompt_state",
        "prompt_text",
        "user_reason",
        "target_price",
        "stop_price",
        "skipped",
        "created_at",
        "responded_at",
    ]

    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, trade_log_id, symbol, action, status, prompt_state, prompt_text,
                       user_reason, target_price, stop_price, skipped, created_at, responded_at
                FROM trade_followups
                WHERE status = 'pending' AND prompt_state = 'pending'
                ORDER BY id
                """
            )
            rows = cursor.fetchall()
            if not rows:
                return []

            claimed_followups = []
            for row in rows:
                followup_id = int(row[0])
                _update_trade_followup_status(cursor, followup_id, status="pending", prompt_state="sending")
                followup = dict(zip(columns, row))
                followup["prompt_state"] = "sending"
                claimed_followups.append(followup)

            conn.commit()
            return claimed_followups
        finally:
            conn.close()


def get_latest_prompted_trade_followup() -> Dict[str, Any] | None:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, trade_log_id, symbol, action, status, prompt_state, prompt_text,
                       user_reason, target_price, stop_price, skipped, created_at, responded_at
                FROM trade_followups
                WHERE status = 'pending' AND prompt_state = 'prompted'
                ORDER BY id DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()
        finally:
            conn.close()

    if row is None:
        return None

    columns = [
        "id",
        "trade_log_id",
        "symbol",
        "action",
        "status",
        "prompt_state",
        "prompt_text",
        "user_reason",
        "target_price",
        "stop_price",
        "skipped",
        "created_at",
        "responded_at",
    ]
    return dict(zip(columns, row))


def get_latest_prompted_pending_trade_followup() -> Dict[str, Any] | None:
    return get_latest_prompted_trade_followup()


_TRADE_FOLLOWUP_TARGET_PATTERN = re.compile(
    r"(?:目標價?|target(?:\s+price)?|tp)\s*[:：=]?\s*\$?(?P<target>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_TRADE_FOLLOWUP_STOP_PATTERN = re.compile(
    r"(?:停損|止損|stop(?:\s+loss)?|sl)\s*[:：=]?\s*\$?(?P<stop>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_TRADE_FOLLOWUP_REASON_PREFIX_PATTERN = re.compile(r"^(?:原因|理由|原因是|理由是)[:：\s]*")
_TRADE_FOLLOWUP_REASON_HINT_PATTERN = re.compile(r"^(?:原因|理由|原因是|理由是|因為|因为)[:：\s]*")


def parse_trade_followup_reply(reply_text: str) -> Dict[str, Any] | None:
    payload = (reply_text or "").strip()
    if not payload:
        return None

    normalized = re.sub(r"\s+", " ", payload)
    if "跳過" in normalized or "跳过" in normalized or re.search(r"\bskip\b", normalized, re.IGNORECASE):
        return {"skipped": 1, "user_reason": None, "target_price": None, "stop_price": None}

    target_price = None
    stop_price = None
    reason_text = normalized
    reason_hint_match = _TRADE_FOLLOWUP_REASON_HINT_PATTERN.match(reason_text)

    target_match = _TRADE_FOLLOWUP_TARGET_PATTERN.search(reason_text)
    if target_match:
        target_price = float(target_match.group("target"))
        reason_text = (reason_text[: target_match.start()] + " " + reason_text[target_match.end() :]).strip()

    stop_match = _TRADE_FOLLOWUP_STOP_PATTERN.search(reason_text)
    if stop_match:
        stop_price = float(stop_match.group("stop"))
        reason_text = (reason_text[: stop_match.start()] + " " + reason_text[stop_match.end() :]).strip()

    if _TRADE_FOLLOWUP_REASON_PREFIX_PATTERN.match(reason_text):
        reason_text = _TRADE_FOLLOWUP_REASON_PREFIX_PATTERN.sub("", reason_text, count=1)

    user_reason = reason_text
    user_reason = re.sub(r"[，,。.;；:：\-_/]+", " ", user_reason)
    user_reason = re.sub(r"\s+", " ", user_reason).strip()

    has_structured_reply = (
        target_price is not None or stop_price is not None or reason_hint_match is not None
    )
    if not has_structured_reply:
        return None

    if not user_reason and target_price is None and stop_price is None:
        return None

    return {
        "skipped": 0,
        "user_reason": user_reason or None,
        "target_price": target_price,
        "stop_price": stop_price,
    }


def parse_trade_followup_reply_text(reply_text: str) -> Dict[str, Any] | None:
    return parse_trade_followup_reply(reply_text)


def resolve_trade_followup_reply(followup_id: int, reply_text: str) -> Dict[str, Any] | None:
    parsed = parse_trade_followup_reply(reply_text)
    if parsed is None:
        return None

    responded_at = _utc_now_iso()
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            _update_trade_followup_status(
                cursor,
                followup_id,
                status="resolved",
                prompt_state="resolved",
                user_reason=parsed["user_reason"],
                target_price=parsed["target_price"],
                stop_price=parsed["stop_price"],
                skipped=parsed["skipped"],
                responded_at=responded_at,
            )
            conn.commit()
        finally:
            conn.close()

    return {**parsed, "responded_at": responded_at}


def format_trade_followup_confirmation(followup: Dict[str, Any], resolution: Dict[str, Any]) -> str:
    symbol = str(followup.get("symbol") or "").strip() or "UNKNOWN"
    if int(resolution.get("skipped") or 0):
        return f"✅ 已略過 {symbol} 的追問。"

    parts = [f"✅ 已記錄 {symbol} 的追問回覆"]
    user_reason = str(resolution.get("user_reason") or "").strip()
    if user_reason:
        parts.append(f"原因：{user_reason}")
    target_price = resolution.get("target_price")
    if target_price is not None:
        parts.append(f"目標價：{target_price:g}")
    stop_price = resolution.get("stop_price")
    if stop_price is not None:
        parts.append(f"停損：{stop_price:g}")
    return "\n".join(parts)


def format_trade_followup_reply_confirmation(followup: Dict[str, Any], resolution: Dict[str, Any]) -> str:
    return format_trade_followup_confirmation(followup, resolution)


def mark_trade_followup_prompted(followup_id: int) -> None:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            _update_trade_followup_status(cursor, followup_id, status="pending", prompt_state="prompted")
            conn.commit()
        finally:
            conn.close()


def mark_trade_followup_pending(followup_id: int) -> None:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            _update_trade_followup_status(cursor, followup_id, status="pending", prompt_state="pending")
            conn.commit()
        finally:
            conn.close()


def build_trade_followup_prompt(followup: Dict[str, Any]) -> str:
    symbol = str(followup.get("symbol") or "").strip() or "UNKNOWN"
    action = str(followup.get("action") or "").strip()
    action_labels = {
        "sync_buy": "同步買進",
        "sync_sell": "同步賣出",
        "sync_adjust": "同步調整",
    }
    action_label = action_labels.get(action, action or "交易")
    prompt_text = str(followup.get("prompt_text") or "").strip()
    parts = [f"📣 交易追問：{symbol}（{action_label}）"]
    if prompt_text:
        parts.append(prompt_text)
    parts.append("請回覆原因＋目標價/停損；若不處理可直接回「跳過」。")
    return "\n".join(parts)


# --- 資料庫初始化與遷移 ---
def init_db():
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                symbol TEXT PRIMARY KEY,
                cost REAL,
                shares REAL,
                twd_cost REAL,
                locked INTEGER DEFAULT 0
            )
        """)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                shares REAL NOT NULL,
                settle_currency TEXT,
                settle_amount REAL,
                fx_rate REAL,
                realized_pnl REAL,
                cash_before REAL,
                cash_after REAL,
                note TEXT,
                decision_snapshot TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_followups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_log_id INTEGER NOT NULL UNIQUE,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                prompt_state TEXT NOT NULL DEFAULT 'pending',
                prompt_text TEXT,
                user_reason TEXT,
                target_price REAL,
                stop_price REAL,
                skipped INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                responded_at TEXT,
                FOREIGN KEY(trade_log_id) REFERENCES trade_log(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft',
                source TEXT NOT NULL,
                opened_trade_log_id INTEGER,
                entry_price REAL,
                stop_loss REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                max_holding_days INTEGER,
                thesis_type TEXT,
                thesis_text TEXT,
                thesis_payload_json TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                FOREIGN KEY(opened_trade_log_id) REFERENCES trade_log(id) ON DELETE SET NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_plan_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                FOREIGN KEY(plan_id) REFERENCES trade_plans(id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_plan_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id INTEGER,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                payload_json TEXT,
                first_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                last_seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                resolved_at TEXT,
                FOREIGN KEY(plan_id) REFERENCES trade_plans(id) ON DELETE SET NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_nav_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                nav_twd REAL NOT NULL,
                total_cost_twd REAL,
                gross_exposure_twd REAL,
                cash_twd REAL,
                pnl_pct REAL,
                source TEXT
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_nav_history_timestamp ON portfolio_nav_history(timestamp)"
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS trade_outcome_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_log_id INTEGER NOT NULL,
                horizon_label TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                entry_notional_twd REAL NOT NULL,
                due_at TEXT NOT NULL,
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
                status TEXT NOT NULL DEFAULT 'pending',
                resolved_at TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
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
        # 執行遷移：如果舊資料庫沒有 locked 欄位，手動補上
        try:
            cursor.execute("ALTER TABLE portfolio ADD COLUMN locked INTEGER DEFAULT 0")
        except Exception as e:
            logger.debug(f"Portfolio migration skipped: {e}")
        try:
            cursor.execute("ALTER TABLE trade_log ADD COLUMN decision_snapshot TEXT")
        except Exception as e:
            logger.debug(f"Trade log migration skipped: {e}")
        try:
            cursor.execute(
                "ALTER TABLE trade_outcome_checkpoints ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'"
            )
        except Exception as e:
            logger.debug(f"Trade outcome checkpoints status migration skipped: {e}")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_followups_status ON trade_followups(status)")
        except Exception as e:
            logger.debug(f"Trade followup index skipped: {e}")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_plans_symbol_status ON trade_plans(symbol, status)")
        except Exception as e:
            logger.debug(f"Trade plans index skipped: {e}")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_plan_events_plan_id ON trade_plan_events(plan_id)")
        except Exception as e:
            logger.debug(f"Trade plan events index skipped: {e}")
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_plan_alerts_symbol_status ON trade_plan_alerts(symbol, status)")
        except Exception as e:
            logger.debug(f"Trade plan alerts index skipped: {e}")
        try:
            _dedupe_open_trade_plan_alerts_locked(cursor)
        except Exception as e:
            logger.debug(f"Trade plan alert dedupe migration skipped: {e}")
        try:
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_plan_alerts_open_plan_type
                ON trade_plan_alerts(plan_id, alert_type)
                WHERE status = 'open' AND plan_id IS NOT NULL
                """
            )
        except Exception as e:
            logger.debug(f"Trade plan alert plan dedupe index skipped: {e}")
        try:
            cursor.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_plan_alerts_open_symbol_type_null_plan
                ON trade_plan_alerts(symbol, alert_type)
                WHERE status = 'open' AND plan_id IS NULL
                """
            )
        except Exception as e:
            logger.debug(f"Trade plan alert symbol dedupe index skipped: {e}")
        conn.commit()

        # 檢查是否需要從 CSV 遷移
        if os.path.exists(CSV_BACKUP):
            logger.info(f"📦 偵測到舊帳本 {CSV_BACKUP}，正在執行自動遷移...")
            try:
                with open(CSV_BACKUP, mode='r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader, None) # 跳過標頭
                    for row in reader:
                        if len(row) >= 3:
                            sym = row[0].upper()
                            cost = float(row[1])
                            shares = float(row[2])
                            twd_c = float(row[3]) if len(row) >= 4 else (cost * shares * (fetch_exchange_rate() if ".TW" not in sym and "CASH" not in sym else 1.0))
                            locked = int(row[4]) if len(row) >= 5 else 0
                            cursor.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?, ?, ?)", (sym, cost, shares, twd_c, locked))
                conn.commit()
                # 遷移完成後將舊檔改名備份
                os.rename(CSV_BACKUP, f"{CSV_BACKUP}.migrated_{int(time.time())}")
                logger.info("✅ 遷移完成，舊檔已備份。")
            except Exception as e:
                logger.warning(f"⚠️ 遷移失敗: {e}")
        conn.close()


def _record_trade_plan_event(
    cursor,
    *,
    plan_id: int,
    event_type: str,
    payload: Dict[str, Any] | None = None,
):
    cursor.execute(
        """
        INSERT INTO trade_plan_events (plan_id, event_type, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (plan_id, event_type, _serialize_json(payload), _utc_now_iso()),
    )


def _dedupe_open_trade_plan_alerts_locked(cursor) -> None:
    cursor.execute(
        """
        DELETE FROM trade_plan_alerts
        WHERE id IN (
            SELECT older.id
            FROM trade_plan_alerts AS older
            JOIN trade_plan_alerts AS newer
              ON older.id < newer.id
             AND older.status = 'open'
             AND newer.status = 'open'
             AND older.plan_id IS NOT NULL
             AND newer.plan_id = older.plan_id
             AND newer.alert_type = older.alert_type
        )
        """
    )
    cursor.execute(
        """
        DELETE FROM trade_plan_alerts
        WHERE id IN (
            SELECT older.id
            FROM trade_plan_alerts AS older
            JOIN trade_plan_alerts AS newer
              ON older.id < newer.id
             AND older.status = 'open'
             AND newer.status = 'open'
             AND older.plan_id IS NULL
             AND newer.plan_id IS NULL
             AND newer.symbol = older.symbol
             AND newer.alert_type = older.alert_type
        )
        """
    )


def _merge_trade_plan_values(
    existing: sqlite3.Row | None,
    *,
    source: str,
    opened_trade_log_id: int | None,
    entry_price: float | None,
    stop_loss: float | None,
    take_profit_1: float | None,
    take_profit_2: float | None,
    max_holding_days: int | None,
    thesis_type: str | None,
    thesis_text: str | None,
    thesis_payload: Dict[str, Any] | None,
    status: str,
) -> Dict[str, Any]:
    return {
        "source": source,
        "opened_trade_log_id": opened_trade_log_id if opened_trade_log_id is not None else (existing["opened_trade_log_id"] if existing else None),
        "entry_price": entry_price if entry_price is not None else (existing["entry_price"] if existing else None),
        "stop_loss": stop_loss if stop_loss is not None else (existing["stop_loss"] if existing else None),
        "take_profit_1": take_profit_1 if take_profit_1 is not None else (existing["take_profit_1"] if existing else None),
        "take_profit_2": take_profit_2 if take_profit_2 is not None else (existing["take_profit_2"] if existing else None),
        "max_holding_days": max_holding_days if max_holding_days is not None else (existing["max_holding_days"] if existing else None),
        "thesis_type": thesis_type if thesis_type is not None else (existing["thesis_type"] if existing else None),
        "thesis_text": thesis_text if thesis_text is not None else (existing["thesis_text"] if existing else None),
        "thesis_payload_json": _serialize_json(thesis_payload) if thesis_payload is not None else (existing["thesis_payload_json"] if existing else None),
        "status": status,
    }


def validate_trade_plan_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return completeness metadata for a candidate trade-plan payload."""
    payload = payload or {}
    missing_fields = [
        field
        for field in TRADE_PLAN_REQUIRED_FIELDS
        if payload.get(field) is None or (isinstance(payload.get(field), str) and not payload.get(field).strip())
    ]
    return {"complete": not missing_fields, "missing_fields": missing_fields}


def _validate_trade_plan_payload(plan_values: Dict[str, Any]) -> None:
    if plan_values["status"] != "active":
        return
    missing_fields = validate_trade_plan_payload(plan_values)["missing_fields"]
    if missing_fields:
        raise ValueError(f"active trade plan requires fields: {', '.join(missing_fields)}")


def _query_trade_plan_rows(query: str, params: tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def _upsert_trade_plan_locked(
    cursor,
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
    existing = cursor.execute(
        f"""
        SELECT {TRADE_PLAN_SELECT}
        FROM trade_plans
        WHERE symbol = ? AND status IN ('draft', 'active')
        ORDER BY id DESC
        LIMIT 1
        """,
        (normalized,),
    ).fetchone()
    now = _utc_now_iso()
    plan_values = _merge_trade_plan_values(
        existing,
        source=source,
        opened_trade_log_id=opened_trade_log_id,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        max_holding_days=max_holding_days,
        thesis_type=thesis_type,
        thesis_text=thesis_text,
        thesis_payload=thesis_payload,
        status=status,
    )
    _validate_trade_plan_payload(plan_values)
    newly_active = status == "active" and (existing is None or existing["status"] != "active")
    if existing:
        plan_id = int(existing["id"])
        cursor.execute(
            """
            UPDATE trade_plans
            SET source = ?, opened_trade_log_id = ?,
                entry_price = ?, stop_loss = ?, take_profit_1 = ?, take_profit_2 = ?,
                max_holding_days = ?, thesis_type = ?, thesis_text = ?,
                thesis_payload_json = ?, status = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                plan_values["source"],
                plan_values["opened_trade_log_id"],
                plan_values["entry_price"],
                plan_values["stop_loss"],
                plan_values["take_profit_1"],
                plan_values["take_profit_2"],
                plan_values["max_holding_days"],
                plan_values["thesis_type"],
                plan_values["thesis_text"],
                plan_values["thesis_payload_json"],
                plan_values["status"],
                now,
                plan_id,
            ),
        )
        _record_trade_plan_event(
            cursor,
            plan_id=plan_id,
            event_type="plan_updated",
            payload={"status": plan_values["status"]},
        )
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
                normalized,
                plan_values["status"],
                plan_values["source"],
                plan_values["opened_trade_log_id"],
                plan_values["entry_price"],
                plan_values["stop_loss"],
                plan_values["take_profit_1"],
                plan_values["take_profit_2"],
                plan_values["max_holding_days"],
                plan_values["thesis_type"],
                plan_values["thesis_text"],
                plan_values["thesis_payload_json"],
                now,
                now,
            ),
        )
        plan_id = int(cursor.lastrowid)
        _record_trade_plan_event(cursor, plan_id=plan_id, event_type="plan_created", payload={"source": source})
    if newly_active:
        _record_trade_plan_event(cursor, plan_id=plan_id, event_type="plan_activated")
    return plan_id


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
    """Create or update the latest draft/active trade plan for ``symbol``.

    ``source`` always records the latest writer/provenance for the persisted row.
    On updates, ``None`` means "keep the current stored value" for optional fields,
    including ``thesis_payload``; this API has no sentinel-based clear behavior.
    """
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            plan_id = _upsert_trade_plan_locked(
                cursor,
                symbol=symbol,
                source=source,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                max_holding_days=max_holding_days,
                thesis_type=thesis_type,
                thesis_text=thesis_text,
                thesis_payload=thesis_payload,
                status=status,
                opened_trade_log_id=opened_trade_log_id,
            )
            conn.commit()
            return plan_id
        finally:
            conn.close()


def get_trade_plan(plan_id: int) -> Dict[str, Any] | None:
    rows = _query_trade_plan_rows(f"SELECT {TRADE_PLAN_SELECT} FROM trade_plans WHERE id = ?", (plan_id,))
    return rows[0] if rows else None


def get_active_trade_plan(symbol: str) -> Dict[str, Any] | None:
    normalized = normalize_ticker(symbol)
    rows = _query_trade_plan_rows(
        f"SELECT {TRADE_PLAN_SELECT} FROM trade_plans WHERE symbol = ? AND status = 'active' ORDER BY id DESC LIMIT 1",
        (normalized,),
    )
    return rows[0] if rows else None


def list_active_trade_plans() -> List[Dict[str, Any]]:
    return _query_trade_plan_rows(
        f"SELECT {TRADE_PLAN_SELECT} FROM trade_plans WHERE status = 'active' ORDER BY symbol, id DESC"
    )


def _build_trade_plan_payload(
    *,
    symbol: str,
    entry_price: float | None,
    trade_plan: Dict[str, Any] | None = None,
    source: str,
    status: str,
    opened_trade_log_id: int | None = None,
) -> Dict[str, Any]:
    """Normalize runtime trade-plan input into the persisted payload shape."""
    trade_plan = trade_plan or {}
    thesis_payload = trade_plan.get("thesis_payload")
    return {
        "symbol": normalize_ticker(symbol),
        "source": source,
        "opened_trade_log_id": opened_trade_log_id,
        "entry_price": entry_price,
        "stop_loss": trade_plan.get("stop_loss"),
        "take_profit_1": trade_plan.get("take_profit_1"),
        "take_profit_2": trade_plan.get("take_profit_2"),
        "max_holding_days": trade_plan.get("max_holding_days"),
        "thesis_type": trade_plan.get("thesis_type"),
        "thesis_text": trade_plan.get("thesis_text"),
        "thesis_payload": thesis_payload if isinstance(thesis_payload, dict) or thesis_payload is None else None,
        "status": status,
    }


def _format_trade_plan_validation_error(validation: Dict[str, Any]) -> str | None:
    if validation["complete"]:
        return None
    missing_labels = ", ".join(
        TRADE_PLAN_FIELD_LABELS.get(field, field) for field in validation["missing_fields"]
    )
    return format_tool_error(f"❌ 買進前需提供完整交易計畫，缺少：{missing_labels}。")


def _upsert_trade_plan_alert_locked(
    cursor,
    *,
    symbol: str,
    alert_type: str,
    severity: str,
    status: str = "open",
    plan_id: int | None = None,
    payload: Dict[str, Any] | None = None,
) -> int:
    normalized = normalize_ticker(symbol)
    if plan_id is None:
        existing = cursor.execute(
            """
            SELECT id, plan_id
            FROM trade_plan_alerts
            WHERE symbol = ? AND plan_id IS NULL AND alert_type = ? AND status = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (normalized, alert_type, status),
        ).fetchone()
    else:
        existing = cursor.execute(
            """
            SELECT id, plan_id
            FROM trade_plan_alerts
            WHERE plan_id = ? AND alert_type = ? AND status = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (plan_id, alert_type, status),
        ).fetchone()
    now = _utc_now_iso()
    serialized_payload = _serialize_json(payload)
    if existing:
        if isinstance(existing, sqlite3.Row):
            alert_id = int(existing["id"])
            existing_plan_id = existing["plan_id"]
        else:
            alert_id = int(existing[0])
            existing_plan_id = existing[1]
        cursor.execute(
            """
            UPDATE trade_plan_alerts
            SET plan_id = ?, severity = ?, payload_json = ?, last_seen_at = ?
            WHERE id = ?
            """,
            (
                plan_id if plan_id is not None else existing_plan_id,
                severity,
                serialized_payload,
                now,
                alert_id,
            ),
        )
        return alert_id

    cursor.execute(
        """
        INSERT INTO trade_plan_alerts (
            plan_id, symbol, alert_type, severity, status, payload_json, first_seen_at, last_seen_at, resolved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (plan_id, normalized, alert_type, severity, status, serialized_payload, now, now, None),
    )
    return int(cursor.lastrowid)


def _deserialize_json(payload_json: str | None) -> Dict[str, Any] | None:
    if not payload_json:
        return None
    try:
        payload = json.loads(payload_json)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _hydrate_trade_plan_alert(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    payload_json = row["payload_json"]
    return {
        "id": int(row["id"]),
        "plan_id": int(row["plan_id"]) if row["plan_id"] is not None else None,
        "symbol": normalize_ticker(row["symbol"]),
        "alert_type": row["alert_type"],
        "severity": row["severity"],
        "status": row["status"],
        "payload_json": payload_json,
        "payload": _deserialize_json(payload_json),
        "first_seen_at": row["first_seen_at"],
        "last_seen_at": row["last_seen_at"],
        "resolved_at": row["resolved_at"],
    }


def upsert_trade_plan_alert(
    *,
    symbol: str,
    alert_type: str,
    severity: str,
    status: str = "open",
    plan_id: int | None = None,
    payload: Dict[str, Any] | None = None,
) -> int:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            alert_id = _upsert_trade_plan_alert_locked(
                cursor,
                symbol=symbol,
                alert_type=alert_type,
                severity=severity,
                status=status,
                plan_id=plan_id,
                payload=payload,
            )
            conn.commit()
            return alert_id
        finally:
            conn.close()


def resolve_trade_plan_alert(*, symbol: str, alert_type: str, plan_id: int | None = None) -> int:
    normalized = normalize_ticker(symbol)
    now = _utc_now_iso()
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            if plan_id is None:
                cursor.execute(
                    """
                    UPDATE trade_plan_alerts
                    SET status = 'resolved', resolved_at = ?
                    WHERE symbol = ? AND alert_type = ? AND plan_id IS NULL AND status = 'open'
                    """,
                    (now, normalized, alert_type),
                )
            else:
                cursor.execute(
                    """
                    UPDATE trade_plan_alerts
                    SET status = 'resolved', resolved_at = ?
                    WHERE symbol = ? AND alert_type = ? AND plan_id = ? AND status = 'open'
                    """,
                    (now, normalized, alert_type, plan_id),
                )
            resolved_count = int(cursor.rowcount)
            conn.commit()
            return resolved_count
        finally:
            conn.close()


def get_open_trade_plan_alerts(
    *,
    symbol: str | None = None,
    alert_type: str | None = None,
) -> List[Dict[str, Any]]:
    query = [f"SELECT {TRADE_PLAN_ALERT_SELECT} FROM trade_plan_alerts WHERE status = 'open'"]
    params: List[Any] = []
    if symbol is not None:
        query.append("AND symbol = ?")
        params.append(normalize_ticker(symbol))
    if alert_type is not None:
        query.append("AND alert_type = ?")
        params.append(alert_type)
    query.append("ORDER BY symbol, alert_type, id DESC")

    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(" ".join(query), tuple(params)).fetchall()
            return [_hydrate_trade_plan_alert(row) for row in rows]
        finally:
            conn.close()


def get_current_portfolio_symbols() -> List[str]:
    symbols = {
        normalize_ticker(symbol)
        for symbol, _cost, shares, _twd_cost in _load_portfolio_rows()
        if float(shares or 0) > 0 and not normalize_ticker(symbol).startswith("CASH")
    }
    return sorted(symbols)


def _build_trade_plan_snapshot_lookup(snapshots: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for snapshot in snapshots:
        symbol = snapshot.get("symbol")
        if not symbol:
            continue
        lookup[normalize_ticker(str(symbol))] = snapshot
    return lookup


def _parse_trade_plan_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _snapshot_current_return_pct(plan: Dict[str, Any], snapshot: Dict[str, Any]) -> float | None:
    pnl_percent = snapshot.get("pnl_percent")
    if isinstance(pnl_percent, (int, float)):
        return round(float(pnl_percent), 4)
    current_price = snapshot.get("current_price")
    entry_price = plan.get("entry_price")
    if not isinstance(current_price, (int, float)) or not isinstance(entry_price, (int, float)) or float(entry_price) == 0:
        return None
    return round((float(current_price) / float(entry_price)) - 1.0, 4)


def _trade_plan_held_days(plan: Dict[str, Any], now_iso: str) -> int | None:
    now_dt = _parse_trade_plan_timestamp(now_iso)
    opened_dt = _parse_trade_plan_timestamp(plan.get("created_at"))
    if now_dt is None or opened_dt is None or now_dt < opened_dt:
        return None
    return int((now_dt - opened_dt).days)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_trade_plan_price_alert(
    *,
    alert_type: str,
    severity: str,
    symbol: str,
    current_price: float,
    threshold_key: str,
    threshold_value: float,
    thesis_type: str | None,
) -> Dict[str, Any]:
    return {
        "alert_type": alert_type,
        "severity": severity,
        "payload": {
            "symbol": symbol,
            "current_price": round(float(current_price), 4),
            threshold_key: float(threshold_value),
            "thesis_type": thesis_type,
        },
    }


def _evaluate_trade_plan_thesis_invalid_spec(
    plan: Dict[str, Any],
    snapshot: Dict[str, Any],
    *,
    now_iso: str,
) -> Dict[str, Any] | None:
    thesis_type = str(plan.get("thesis_type") or "").strip()
    thesis_payload = _deserialize_json(plan.get("thesis_payload_json"))
    current_price = _coerce_float(snapshot.get("current_price"))
    symbol = normalize_ticker(plan["symbol"])
    if not thesis_type or not isinstance(thesis_payload, dict) or current_price is None:
        return None

    if thesis_type == "breakout_support":
        support_level = _coerce_float(
            thesis_payload.get("support_level", thesis_payload.get("breakout_level"))
        )
        close_below_count = _coerce_int(thesis_payload.get("close_below_count"))
        confirmation_rule = str(
            thesis_payload.get("grace_rule")
            or thesis_payload.get("breach_basis")
            or thesis_payload.get("invalidation_basis")
            or ""
        ).strip().lower()
        uses_close_confirmation = (
            thesis_payload.get("close_based") is True
            or thesis_payload.get("use_close_confirmation") is True
            or confirmation_rule in {"close", "close_below", "close_based", "daily_close"}
            or (close_below_count is not None and close_below_count > 0)
        )
        if support_level is None:
            return None
        if uses_close_confirmation:
            if close_below_count is None or close_below_count <= 0:
                return None
            recent_closes = _fetch_recent_closes(symbol, close_below_count)
            if recent_closes is None or len(recent_closes) < close_below_count:
                return None
            if any(close >= support_level for close in recent_closes[-close_below_count:]):
                return None
            return {
                "alert_type": "thesis_invalid",
                "severity": "warning",
                "payload": {
                    "symbol": symbol,
                    "thesis_type": thesis_type,
                    "current_price": round(current_price, 4),
                    "support_level": support_level,
                    "close_below_count": close_below_count,
                },
            }
        if current_price >= support_level:
            return None
        return {
            "alert_type": "thesis_invalid",
            "severity": "warning",
            "payload": {
                "symbol": symbol,
                "thesis_type": thesis_type,
                "current_price": round(current_price, 4),
                "support_level": support_level,
            },
        }

    if thesis_type == "sector_rotation":
        proxy_symbol = thesis_payload.get("proxy_symbol")
        if not proxy_symbol:
            return None
        lookback_days = _coerce_int(thesis_payload.get("lookback_days")) or 1
        symbol_change = _fetch_recent_change(symbol, lookback_days)
        proxy_change = _fetch_recent_change(str(proxy_symbol), lookback_days)
        threshold = _coerce_float(thesis_payload.get("underperform_pct"))
        if symbol_change is None or proxy_change is None:
            return None
        relative_performance = round(float(symbol_change) - float(proxy_change), 4)
        threshold_pct = 0.0 if threshold is None else float(threshold)
        if relative_performance > threshold_pct:
            return None
        return {
            "alert_type": "thesis_invalid",
            "severity": "warning",
            "payload": {
                "symbol": symbol,
                "thesis_type": thesis_type,
                "proxy_symbol": normalize_ticker(str(proxy_symbol)),
                "symbol_change_pct": round(float(symbol_change), 4),
                "proxy_change_pct": round(float(proxy_change), 4),
                "relative_performance_pct": relative_performance,
                "threshold_pct": threshold_pct,
                "lookback_days": lookback_days,
            },
        }

    if thesis_type == "mean_reversion":
        recovery_window_days = _coerce_int(thesis_payload.get("recovery_window_days"))
        held_days = _trade_plan_held_days(plan, now_iso)
        recovery_price = _coerce_float(
            thesis_payload.get("recovery_price", thesis_payload.get("reference_price", plan.get("entry_price")))
        )
        if (
            recovery_window_days is None
            or recovery_window_days < 0
            or held_days is None
            or held_days < recovery_window_days
        ):
            return None
        if recovery_price is not None and current_price < recovery_price:
            return {
                "alert_type": "thesis_invalid",
                "severity": "warning",
                "payload": {
                    "symbol": symbol,
                    "thesis_type": thesis_type,
                    "current_price": round(current_price, 4),
                    "reference_price": recovery_price,
                    "held_days": held_days,
                    "recovery_window_days": recovery_window_days,
                },
            }
        proxy_symbol = thesis_payload.get("proxy_symbol")
        lookback_days = _coerce_int(thesis_payload.get("lookback_days"))
        threshold = _coerce_float(thesis_payload.get("underperform_pct"))
        if not proxy_symbol or lookback_days is None or lookback_days <= 0 or threshold is None:
            return None
        symbol_change = _fetch_recent_change(symbol, lookback_days)
        proxy_change = _fetch_recent_change(str(proxy_symbol), lookback_days)
        if symbol_change is None or proxy_change is None:
            return None
        relative_performance = round(float(symbol_change) - float(proxy_change), 4)
        threshold_pct = float(threshold)
        if relative_performance > threshold_pct:
            return None
        return {
            "alert_type": "thesis_invalid",
            "severity": "warning",
            "payload": {
                "symbol": symbol,
                "thesis_type": thesis_type,
                "proxy_symbol": normalize_ticker(str(proxy_symbol)),
                "symbol_change_pct": round(float(symbol_change), 4),
                "proxy_change_pct": round(float(proxy_change), 4),
                "relative_performance_pct": relative_performance,
                "threshold_pct": threshold_pct,
                "lookback_days": lookback_days,
                "held_days": held_days,
                "recovery_window_days": recovery_window_days,
            },
        }

    if thesis_type == "earnings":
        review_window_days = _coerce_int(thesis_payload.get("review_window_days"))
        now_dt = _parse_trade_plan_timestamp(now_iso)
        event_dt = _parse_trade_plan_timestamp(
            thesis_payload.get("earnings_date", thesis_payload.get("event_date"))
        )
        expected_direction = str(thesis_payload.get("expected_direction") or "").strip().lower()
        reference_price = _coerce_float(thesis_payload.get("reference_price", plan.get("entry_price")))
        if (
            review_window_days is None
            or review_window_days < 0
            or now_dt is None
            or event_dt is None
            or now_dt < (event_dt + timedelta(days=review_window_days))
            or expected_direction not in {"up", "down"}
            or reference_price is None
            or reference_price == 0
        ):
            return None
        move_pct = round((current_price / reference_price) - 1.0, 4)
        moved_opposite = (expected_direction == "up" and move_pct < 0) or (
            expected_direction == "down" and move_pct > 0
        )
        if not moved_opposite:
            return None
        return {
            "alert_type": "thesis_invalid",
            "severity": "warning",
            "payload": {
                "symbol": symbol,
                "thesis_type": thesis_type,
                "expected_direction": expected_direction,
                "move_pct": move_pct,
                "event_date": event_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "review_window_days": review_window_days,
            },
        }

    if thesis_type == "event_driven":
        deadline_raw = thesis_payload.get(
            "invalidation_deadline",
            thesis_payload.get("catalyst_deadline", thesis_payload.get("catalyst_date")),
        )
        deadline_dt = _parse_trade_plan_timestamp(deadline_raw)
        now_dt = _parse_trade_plan_timestamp(now_iso)
        if deadline_dt is None or now_dt is None or now_dt <= deadline_dt:
            return None
        if thesis_payload.get("confirmed") is True or thesis_payload.get("catalyst_confirmed") is True:
            return None
        return {
            "alert_type": "thesis_invalid",
            "severity": "warning",
            "payload": {
                "symbol": symbol,
                "thesis_type": thesis_type,
                "deadline": deadline_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "confirmed": False,
            },
        }

    return None


def _resolve_stale_trade_plan_alerts(symbol: str, plan_id: int, active_alert_types: set[str]) -> None:
    for alert in get_open_trade_plan_alerts(symbol=symbol):
        if alert["plan_id"] != plan_id:
            continue
        if alert["alert_type"] not in TRADE_PLAN_MONITOR_ALERT_TYPES:
            continue
        if alert["alert_type"] in active_alert_types:
            continue
        resolve_trade_plan_alert(symbol=symbol, alert_type=alert["alert_type"], plan_id=plan_id)


def _evaluate_trade_plan_alert_specs(
    plan: Dict[str, Any],
    snapshot: Dict[str, Any],
    *,
    now_iso: str,
) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    current_price = _coerce_float(snapshot.get("current_price"))
    symbol = normalize_ticker(plan["symbol"])
    thesis_type = plan.get("thesis_type")

    stop_loss = _coerce_float(plan.get("stop_loss"))
    if current_price is not None and stop_loss is not None and current_price <= stop_loss:
        alerts.append(
            _build_trade_plan_price_alert(
                alert_type="stop_hit",
                severity="critical",
                symbol=symbol,
                current_price=current_price,
                threshold_key="stop_loss",
                threshold_value=stop_loss,
                thesis_type=thesis_type,
            )
        )

    take_profit_1 = _coerce_float(plan.get("take_profit_1"))
    if current_price is not None and take_profit_1 is not None and current_price >= take_profit_1:
        alerts.append(
            _build_trade_plan_price_alert(
                alert_type="tp1_hit",
                severity="info",
                symbol=symbol,
                current_price=current_price,
                threshold_key="target_price",
                threshold_value=take_profit_1,
                thesis_type=thesis_type,
            )
        )

    take_profit_2 = _coerce_float(plan.get("take_profit_2"))
    if current_price is not None and take_profit_2 is not None and current_price >= take_profit_2:
        alerts.append(
            _build_trade_plan_price_alert(
                alert_type="tp2_hit",
                severity="info",
                symbol=symbol,
                current_price=current_price,
                threshold_key="target_price",
                threshold_value=take_profit_2,
                thesis_type=thesis_type,
            )
        )

    max_holding_days = _coerce_int(plan.get("max_holding_days"))
    held_days = _trade_plan_held_days(plan, now_iso)
    if max_holding_days is not None and max_holding_days >= 0 and held_days is not None and held_days >= max_holding_days:
        holding_payload = {
            "symbol": symbol,
            "held_days": held_days,
            "max_days": max_holding_days,
            "current_return_pct": _snapshot_current_return_pct(plan, snapshot),
        }
        alerts.append(
            {
                "alert_type": "holding_expiry",
                "severity": "warning",
                "payload": holding_payload,
            }
        )

    thesis_invalid_spec = _evaluate_trade_plan_thesis_invalid_spec(plan, snapshot, now_iso=now_iso)
    if thesis_invalid_spec is not None:
        alerts.append(thesis_invalid_spec)
    return alerts


def audit_trade_plan_alerts() -> Dict[str, Any]:
    active_plans = list_active_trade_plans()
    result: Dict[str, Any] = {
        "audited": len(active_plans),
        "triggered": 0,
        "degraded": 0,
        "symbols": [],
    }
    if not active_plans:
        return result

    try:
        snapshot_lookup = _build_trade_plan_snapshot_lookup(
            _build_live_position_snapshots(_load_portfolio_rows())
        )
    except Exception as exc:
        for plan in active_plans:
            upsert_trade_plan_alert(
                symbol=plan["symbol"],
                alert_type="monitor_degraded",
                severity="warning",
                plan_id=int(plan["id"]),
                payload={"error": str(exc), "reason": "live_snapshot_build_failed"},
            )
        result["degraded"] = len(active_plans)
        result["symbols"] = [normalize_ticker(plan["symbol"]) for plan in active_plans]
        return result

    run_now_iso = _utc_now_iso()
    for plan in active_plans:
        symbol = normalize_ticker(plan["symbol"])
        plan_id = int(plan["id"])
        resolve_trade_plan_alert(symbol=symbol, alert_type="monitor_degraded", plan_id=plan_id)
        snapshot = snapshot_lookup.get(symbol)
        if snapshot is None:
            _resolve_stale_trade_plan_alerts(symbol, plan_id, set())
            continue

        alert_specs = _evaluate_trade_plan_alert_specs(plan, snapshot, now_iso=run_now_iso)
        for spec in alert_specs:
            upsert_trade_plan_alert(
                symbol=symbol,
                alert_type=spec["alert_type"],
                severity=spec["severity"],
                plan_id=plan_id,
                payload=spec["payload"],
            )
        _resolve_stale_trade_plan_alerts(symbol, plan_id, {spec["alert_type"] for spec in alert_specs})
        if alert_specs:
            result["symbols"].append(symbol)
            result["triggered"] += len(alert_specs)

    return result


def build_trade_plan_status_summary() -> Dict[str, Any]:
    active_plans = list_active_trade_plans()
    open_alerts = get_open_trade_plan_alerts()
    missing_plan_count = sum(1 for alert in open_alerts if alert["alert_type"] == "missing_plan")
    return {
        "generated_at": _utc_now_iso(),
        "portfolio_symbols": get_current_portfolio_symbols(),
        "active_plan_count": len(active_plans),
        "active_plans": active_plans,
        "open_alert_count": len(open_alerts),
        "missing_plan_count": missing_plan_count,
        "alerts": open_alerts,
        "open_alerts": open_alerts,
    }


def sync_trade_plan_backfills() -> Dict[str, Any]:
    """Backfill draft trade plans for live holdings missing tracked plans.

    This helper is best-effort and idempotent, not fully atomic: it snapshots
    current portfolio rows before re-entering ``db_lock`` to upsert missing
    plans and alerts. Callers should treat the result as a best-effort backfill
    against a potentially changing portfolio view rather than a single locked
    transaction over both reads and writes.
    """
    created_symbols: List[str] = []
    failed_symbols: List[Dict[str, str]] = []
    rows = _load_portfolio_rows()
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            for symbol, cost, shares, _ in rows:
                normalized = normalize_ticker(symbol)
                if normalized.startswith("CASH") or float(shares or 0) <= 0:
                    continue

                cursor.execute("SAVEPOINT trade_plan_backfill_symbol")
                try:
                    existing = cursor.execute(
                        f"""
                        SELECT {TRADE_PLAN_SELECT}
                        FROM trade_plans
                        WHERE symbol = ? AND status IN ('draft', 'active')
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (normalized,),
                    ).fetchone()
                    if existing:
                        cursor.execute("RELEASE SAVEPOINT trade_plan_backfill_symbol")
                        continue

                    plan_id = _upsert_trade_plan_locked(
                        cursor,
                        symbol=symbol,
                        source="manual_backfill",
                        entry_price=float(cost or 0.0),
                        stop_loss=None,
                        take_profit_1=None,
                        take_profit_2=None,
                        max_holding_days=None,
                        thesis_type=None,
                        thesis_text=None,
                        thesis_payload=None,
                        status="draft",
                    )
                    _upsert_trade_plan_alert_locked(
                        cursor,
                        symbol=normalized,
                        alert_type="missing_plan",
                        severity="warning",
                        plan_id=plan_id,
                        payload={"reason": "live_holding_missing_trade_plan"},
                    )
                    created_symbols.append(normalized)
                    cursor.execute("RELEASE SAVEPOINT trade_plan_backfill_symbol")
                except Exception as exc:
                    cursor.execute("ROLLBACK TO SAVEPOINT trade_plan_backfill_symbol")
                    cursor.execute("RELEASE SAVEPOINT trade_plan_backfill_symbol")
                    failed_symbols.append({"symbol": normalized, "error": str(exc)})

            conn.commit()
        finally:
            conn.close()

    return {
        "missing_plan_count": len(created_symbols),
        "symbols": created_symbols,
        "failed_count": len(failed_symbols),
        "failed_symbols": failed_symbols,
    }


def _floor_trade_quantity(quantity: float, decimals: int = TRADE_SIZE_DECIMALS) -> float:
    if quantity <= 0:
        return 0.0
    factor = 10 ** max(int(decimals), 0)
    return math.floor(float(quantity) * factor) / factor


def _is_fubon_sync_candidate(symbol: str, locked: int = 0) -> bool:
    if symbol.startswith("CASH"):
        return False
    clean_symbol = _clean_fubon_sync_symbol(symbol)
    is_taiwan = clean_symbol.isdigit() or (any(char.isdigit() for char in clean_symbol[:4]) and len(clean_symbol) <= 6)
    is_trust = '_TRUST' in symbol or '_ESOP' in symbol
    return is_taiwan and not is_trust and int(locked or 0) != 1


def _clean_fubon_sync_symbol(symbol: str) -> str:
    return normalize_ticker(symbol).replace('.TW', '').replace('.TWO', '').replace('_TRUST', '').replace('_ESOP', '')


def _resolve_sync_lookup_symbol(symbol: str) -> str:
    clean_symbol = normalize_ticker(symbol).replace('_TRUST', '').replace('_ESOP', '')
    return market._normalize_lookup_symbol(clean_symbol)


def _fetch_recent_change(symbol: str, days: int = 1) -> float | None:
    try:
        lookback_days = max(int(days), 1)
        hist = get_ticker(symbol, cache_level="daily").history(
            period=f"{max(lookback_days * 2, lookback_days + 4)}d"
        )
        closes = hist.get("Close") if isinstance(hist, pd.DataFrame) else None
        if closes is None:
            return None
        closes = pd.Series(closes).dropna()
        if len(closes) <= lookback_days:
            return None
        prev_close = float(closes.iloc[-(lookback_days + 1)])
        last_close = float(closes.iloc[-1])
        if prev_close == 0:
            return None
        return round((last_close / prev_close) - 1.0, 4)
    except Exception as exc:
        logger.debug(f"Recent change lookup failed for {symbol}: {exc}")
        return None


def _fetch_recent_closes(symbol: str, days: int = 1) -> List[float] | None:
    try:
        lookback_days = max(int(days), 1)
        hist = get_ticker(symbol, cache_level="daily").history(
            period=f"{max(lookback_days * 2, lookback_days + 4)}d"
        )
        closes = hist.get("Close") if isinstance(hist, pd.DataFrame) else None
        if closes is None:
            return None
        closes = pd.Series(closes).dropna()
        if len(closes) < lookback_days:
            return None
        return [round(float(close), 4) for close in closes.iloc[-lookback_days:]]
    except Exception as exc:
        logger.debug(f"Recent close lookup failed for {symbol}: {exc}")
        return None


def _fetch_recent_change_1d(symbol: str) -> float | None:
    return _fetch_recent_change(symbol, 1)


def _fetch_latest_price_value(symbol: str) -> float | None:
    try:
        ticker = get_ticker(symbol, cache_level="daily")
        fast_info = getattr(ticker, "fast_info", {}) or {}
        price = fast_info.get("last_price")
        if price is not None:
            return round(float(price), 4)
        hist = ticker.history(period="5d")
        closes = hist.get("Close") if isinstance(hist, pd.DataFrame) else None
        if closes is None:
            return None
        closes = pd.Series(closes).dropna()
        if closes.empty:
            return None
        return round(float(closes.iloc[-1]), 4)
    except Exception as exc:
        logger.debug(f"Latest price lookup failed for {symbol}: {exc}")
        return None


def _select_sector_proxy(profile: Dict[str, Any]) -> str:
    sector = str(profile.get("sector") or "Unknown")
    industry = str(profile.get("industry") or "")
    industry_l = industry.lower()
    if "semiconductor" in industry_l:
        return "SOXX"
    mapping = {
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
        "Macro Hedge": "GLD",
    }
    return mapping.get(sector, "SPY")


def _fetch_symbol_rsi_14(symbol: str) -> float | None:
    try:
        import engine_technical

        hist = get_ticker(symbol, cache_level="daily").history(period="3mo")
        if not isinstance(hist, pd.DataFrame) or hist.empty or "Close" not in hist:
            return None
        closes = pd.Series(hist["Close"]).dropna().astype(float)
        if len(closes) < 15:
            return None
        calc = engine_technical.IndicatorCalculator()
        rsi_series = pd.Series(calc.RSI(closes.to_numpy(dtype=float))).dropna()
        if rsi_series.empty:
            return None
        return round(float(rsi_series.iloc[-1]), 2)
    except Exception as exc:
        logger.debug(f"RSI lookup failed for {symbol}: {exc}")
        return None


def _fetch_sync_nlp_payload(symbol: str, lookup_symbol: str) -> Dict[str, Any]:
    """Fetch full NLP alpha payload and remap alpha_official → alpha_sec."""
    empty: Dict[str, Any] = {
        "nlp_alpha": None,
        "alpha_macro": None,
        "alpha_retail": None,
        "alpha_sec": None,
    }
    try:
        import engine_router as router
    except Exception as exc:
        logger.debug(f"NLP router import failed during trade snapshot: {exc}")
        return empty

    candidates = [normalize_ticker(symbol)]
    if lookup_symbol and lookup_symbol not in candidates:
        candidates.append(lookup_symbol)
    base_lookup = (lookup_symbol or "").replace(".TW", "").replace(".TWO", "")
    if base_lookup and base_lookup not in candidates:
        candidates.append(base_lookup)

    for candidate in candidates:
        payload = router.fetch_nlp_alpha(candidate)
        if payload.get("error") or not isinstance(payload.get("nlp_alpha"), (int, float)):
            continue
        return {
            "nlp_alpha": round(float(payload["nlp_alpha"]), 4),
            "alpha_macro": round(float(payload["alpha_macro"]), 4) if isinstance(payload.get("alpha_macro"), (int, float)) else None,
            "alpha_retail": round(float(payload["alpha_retail"]), 4) if isinstance(payload.get("alpha_retail"), (int, float)) else None,
            "alpha_sec": round(float(payload["alpha_official"]), 4) if isinstance(payload.get("alpha_official"), (int, float)) else None,
        }
    return empty


def _estimate_entry_beta_proxy(
    symbol: str,
    benchmark: str = "SPY",
    period: str = "6mo",
) -> float | None:
    """Return the OLS beta for a single-symbol portfolio vs. the benchmark."""
    try:
        result = compute_portfolio_beta_attribution(
            {symbol: 1.0}, benchmark=benchmark, period=period
        )
        if result.get("error"):
            return None
        positions = result.get("positions") or {}
        for pos_data in positions.values():
            beta = pos_data.get("beta")
            if isinstance(beta, (int, float)):
                return round(float(beta), 4)
        return None
    except Exception as exc:
        logger.debug(f"Beta proxy estimation failed for {symbol}: {exc}")
        return None


def _build_trade_decision_snapshot(
    symbol: str,
    *,
    entry_notional_twd: float | None = None,
) -> Dict[str, Any]:
    """Build a rich attribution snapshot for an entry trade (buy or sync_buy).

    Prioritises attribution inputs used downstream by engine_journal:
    benchmark_symbol, sector_proxy_symbol, and the beta/NLP alpha dimensions.
    The older ad hoc market-context fields are intentionally excluded here.
    """
    lookup_symbol = _resolve_sync_lookup_symbol(symbol)
    effective_lookup = lookup_symbol or normalize_ticker(symbol)
    profile = market.get_asset_profile(effective_lookup)
    sector_proxy = _select_sector_proxy(profile)
    nlp_payload = _fetch_sync_nlp_payload(symbol, effective_lookup)
    beta_proxy = _estimate_entry_beta_proxy(normalize_ticker(symbol))

    snapshot: Dict[str, Any] = {
        "captured_at": _utc_now_iso(),
        "symbol": normalize_ticker(symbol),
        "lookup_symbol": lookup_symbol,
        "sector": profile.get("sector", "Unknown"),
        "industry": profile.get("industry", "Unknown"),
        "benchmark_symbol": "SPY",
        "sector_proxy_symbol": sector_proxy,
        "beta_proxy_period": "6mo",
        "beta_proxy_at_entry": beta_proxy,
        "entry_notional_twd": round(entry_notional_twd, 2) if entry_notional_twd is not None else None,
        "nlp_alpha": nlp_payload.get("nlp_alpha"),
        "alpha_macro": nlp_payload.get("alpha_macro"),
        "alpha_retail": nlp_payload.get("alpha_retail"),
        "alpha_sec": nlp_payload.get("alpha_sec"),
    }
    snapshot["risk_state"] = None
    snapshot["risk_score"] = None
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


def sync_fubon_portfolio_state(source: str = "scheduler", sync_memory: bool = False) -> Dict[str, Any]:
    result: Dict[str, Any] = {"synced": False, "event_count": 0, "followup_count": 0, "events": [], "message": "富邦未啟動。"}
    if not fubon.fubon_ready:
        return result

    account_snapshot = fubon.get_fubon_account_snapshot()
    if account_snapshot.get("success") is not True:
        error_message = str(account_snapshot.get("error") or "broker snapshot unavailable")
        result["message"] = f"Fubon sync skipped: {error_message}"
        return result

    fubon_inv = account_snapshot.get("inventories") or {}
    fubon_cash = account_snapshot.get("cash_twd")
    events: List[Dict[str, Any]] = []
    followup_count = 0
    skipped_aliases: set[str] = set()

    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, cost, shares, twd_cost, locked FROM portfolio")
            db_rows = cursor.fetchall()
        finally:
            conn.close()

    alias_groups: Dict[str, List[str]] = {}
    existing_sync_symbols = []
    for row in db_rows:
        symbol = str(row[0])
        if not _is_fubon_sync_candidate(symbol, int(row[4] or 0)):
            continue
        existing_sync_symbols.append(symbol)
        alias_groups.setdefault(_clean_fubon_sync_symbol(symbol), []).append(symbol)
    ambiguous_aliases = {alias for alias, symbols in alias_groups.items() if len(symbols) > 1}
    snapshot_symbols = {normalize_ticker(symbol) for symbol in fubon_inv.keys()}
    snapshot_symbols.update(existing_sync_symbols)
    decision_cache = {
        symbol: _build_sync_decision_snapshot(symbol)
        for symbol in sorted(snapshot_symbols)
    }

    pending_trade_journal_ids: list[int] = []
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            db_dict = {row[0]: list(row) for row in db_rows}
            db_aliases = {
                _clean_fubon_sync_symbol(row[0]): row[0]
                for row in db_rows
                if _is_fubon_sync_candidate(str(row[0]), int(row[4] or 0))
                and _clean_fubon_sync_symbol(row[0]) not in ambiguous_aliases
            }

            for symbol, data in sorted(fubon_inv.items()):
                normalized = normalize_ticker(symbol)
                clean_alias = _clean_fubon_sync_symbol(normalized)
                if clean_alias in ambiguous_aliases:
                    skipped_aliases.add(clean_alias)
                    continue
                db_symbol = db_aliases.get(clean_alias, normalized)
                fb_shares = float(data.get("shares") or 0.0)
                fb_cost = float(data.get("cost") or 0.0)
                old_pos = db_dict.get(db_symbol)

                if old_pos is None:
                    if fb_shares > 0:
                        _raw_snap = decision_cache.get(normalize_ticker(normalized))
                        _snap = {**_raw_snap, "entry_notional_twd": round(fb_cost * fb_shares, 2)} if _raw_snap else _raw_snap
                        trade_log_id = _record_trade_log(
                            cursor,
                            symbol=normalized,
                            action="sync_buy",
                            price=fb_cost,
                            shares=fb_shares,
                            note=f"[auto sync:{source}] new broker position detected.",
                            decision_snapshot=_snap,
                        )
                        pending_trade_journal_ids.append(trade_log_id)
                        if source != "portfolio_query":
                            _record_trade_followup(
                                cursor,
                                trade_log_id=trade_log_id,
                                symbol=normalized,
                                action="sync_buy",
                                prompt_text=f"Broker sync detected a new position in {normalized}. Please outline the plan.",
                            )
                            followup_count += 1
                        events.append(
                            {"symbol": normalized, "action": "sync_buy", "shares_delta": fb_shares, "price": fb_cost}
                        )
                    _upsert_portfolio_row(cursor, normalized, fb_cost, fb_shares, fb_cost * fb_shares, 0)
                    db_dict[normalized] = [normalized, fb_cost, fb_shares, fb_cost * fb_shares, 0]
                    db_aliases[_clean_fubon_sync_symbol(normalized)] = normalized
                    continue

                old_cost = float(old_pos[1] or 0.0)
                old_shares = float(old_pos[2] or 0.0)
                old_locked = int(old_pos[4] or 0)

                shares_changed = not math.isclose(fb_shares, old_shares, rel_tol=0.0, abs_tol=FUBON_SYNC_SHARE_TOL)
                cost_changed = not math.isclose(fb_cost, old_cost, rel_tol=0.0, abs_tol=FUBON_SYNC_COST_TOL)

                if fb_shares > old_shares + FUBON_SYNC_SHARE_TOL:
                    added = fb_shares - old_shares
                    old_total = old_shares * old_cost
                    new_total = fb_shares * fb_cost
                    inferred_price = (new_total - old_total) / added if added > 0 else fb_cost
                    if inferred_price <= 0:
                        inferred_price = fb_cost
                    _raw_snap = decision_cache.get(normalize_ticker(db_symbol))
                    _snap = {**_raw_snap, "entry_notional_twd": round(inferred_price * added, 2)} if _raw_snap else _raw_snap
                    trade_log_id = _record_trade_log(
                        cursor,
                        symbol=db_symbol,
                        action="sync_buy",
                        price=inferred_price,
                        shares=added,
                        note=(
                            f"[auto sync:{source}] broker sync inferred average add "
                            f"(old_avg={old_cost:.4f} -> new_avg={fb_cost:.4f})."
                        ),
                        decision_snapshot=_snap,
                    )
                    pending_trade_journal_ids.append(trade_log_id)
                    if source != "portfolio_query":
                        _record_trade_followup(
                            cursor,
                            trade_log_id=trade_log_id,
                            symbol=db_symbol,
                            action="sync_buy",
                            prompt_text=f"Broker sync increased {db_symbol}. Please outline the plan.",
                        )
                        followup_count += 1
                    events.append(
                        {"symbol": db_symbol, "action": "sync_buy", "shares_delta": round(added, 4), "price": round(inferred_price, 4)}
                    )
                elif fb_shares < old_shares - FUBON_SYNC_SHARE_TOL:
                    reduced = old_shares - fb_shares
                    placeholder_price = old_cost if old_cost > 0 else fb_cost
                    _record_trade_log(
                        cursor,
                        symbol=db_symbol,
                        action="sync_sell",
                        price=placeholder_price,
                        shares=reduced,
                        note=(
                            f"[auto sync:{source}] broker sync detected share reduction; "
                            "execution price unavailable, stored previous average cost placeholder."
                        ),
                        decision_snapshot=decision_cache.get(normalize_ticker(db_symbol)),
                    )
                    events.append(
                        {"symbol": db_symbol, "action": "sync_sell", "shares_delta": round(-reduced, 4), "price": round(placeholder_price, 4)}
                    )
                elif cost_changed:
                    _record_trade_log(
                        cursor,
                        symbol=db_symbol,
                        action="sync_adjust",
                        price=fb_cost,
                        shares=fb_shares,
                        note=f"[auto sync:{source}] average cost adjusted ({old_cost:.4f} -> {fb_cost:.4f}).",
                        decision_snapshot=decision_cache.get(normalize_ticker(db_symbol)),
                    )
                    events.append(
                        {"symbol": db_symbol, "action": "sync_adjust", "shares_delta": 0.0, "price": round(fb_cost, 4)}
                    )

                new_twd_cost = fb_cost * fb_shares
                if (
                    shares_changed
                    or cost_changed
                    or not math.isclose(float(old_pos[3] or 0.0), new_twd_cost, rel_tol=0.0, abs_tol=FUBON_SYNC_COST_TOL)
                ):
                    _upsert_portfolio_row(cursor, db_symbol, fb_cost, fb_shares, new_twd_cost, old_locked)
                    db_dict[db_symbol] = [db_symbol, fb_cost, fb_shares, new_twd_cost, old_locked]

            fb_symbols = {_clean_fubon_sync_symbol(symbol) for symbol in fubon_inv.keys()}
            for symbol, row in list(db_dict.items()):
                clean_alias = _clean_fubon_sync_symbol(symbol)
                if clean_alias in ambiguous_aliases:
                    skipped_aliases.add(clean_alias)
                    continue
                if not _is_fubon_sync_candidate(symbol, row[4]) or clean_alias in fb_symbols:
                    continue
                old_cost = float(row[1] or 0.0)
                old_shares = float(row[2] or 0.0)
                placeholder_price = old_cost if old_cost > 0 else 0.0
                _record_trade_log(
                    cursor,
                    symbol=symbol,
                    action="sync_sell",
                    price=placeholder_price,
                    shares=old_shares,
                    note=(
                        f"[auto sync:{source}] broker sync detected position close-out; "
                        "execution price unavailable, stored previous average cost placeholder."
                    ),
                    decision_snapshot=decision_cache.get(normalize_ticker(symbol)),
                )
                cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
                del db_dict[symbol]
                events.append(
                    {"symbol": symbol, "action": "sync_sell", "shares_delta": round(-old_shares, 4), "price": round(placeholder_price, 4)}
                )

            if fubon_cash is not None:
                cash_value = float(fubon_cash)
                _upsert_portfolio_row(cursor, "CASH_TWD", 1.0, cash_value, cash_value, 0)

            conn.commit()
        finally:
            conn.close()

    # Enqueue journal checkpoints after the lock is fully released (entry trades only).
    if pending_trade_journal_ids:
        try:
            import engine_journal
            engine_journal.enqueue_trade_outcome_checkpoints(pending_trade_journal_ids)
        except Exception as exc:
            logger.warning(f"Journal enqueue failed after Fubon sync: {exc}")

    if sync_memory and events:
        try:
            refresh_portfolio_health_summary(source="fubon_sync")
        except Exception as exc:
            logger.warning(f"Portfolio health refresh failed after Fubon sync: {exc}")

    result["synced"] = True
    result["event_count"] = len(events)
    result["events"] = events
    result["followup_count"] = followup_count
    result["skipped_aliases"] = sorted(skipped_aliases)
    result["message"] = f"Fubon sync complete ({len(events)} inferred events)."
    return result


def _find_snapshot_by_symbol(snapshots: List[Dict[str, Any]], symbol: str) -> Dict[str, Any] | None:
    normalized = normalize_ticker(symbol)
    for snapshot in snapshots:
        if normalize_ticker(str(snapshot.get("symbol") or "")) == normalized:
            return snapshot
    return None


def _get_trade_governor_limits(trade_mode: str, asset_type: str | None = None) -> Dict[str, float]:
    limits = dict(TRADE_GOVERNOR_LIMITS.get(trade_mode or "normal", TRADE_GOVERNOR_LIMITS["normal"]))
    if asset_type == "Tech_Momentum":
        limits["single_name_cap"] = max(0.06, limits["single_name_cap"] - 0.02)
        limits["sector_cap"] = max(0.18, limits["sector_cap"] - 0.05)
    return limits


def _estimate_symbol_beta(
    symbol: str,
    benchmark: str = "SPY",
    period: str = "6mo",
    series_cache: Dict[tuple[str, str], pd.Series] | None = None,
) -> Dict[str, Any]:
    resolved_symbol, stock_returns, stock_error = _load_daily_return_series(
        symbol,
        period=period,
        series_cache=series_cache,
    )
    if stock_error:
        return {"symbol": resolved_symbol or symbol, "error": stock_error}

    benchmark_symbol, bench_returns, bench_error = _load_daily_return_series(
        benchmark,
        period=period,
        series_cache=series_cache,
    )
    if bench_error:
        return {"symbol": resolved_symbol or symbol, "benchmark": benchmark_symbol or benchmark, "error": bench_error}

    aligned = pd.concat([stock_returns, bench_returns], axis=1, join="inner").dropna()
    if len(aligned) < MIN_BETA_OBSERVATIONS:
        return {
            "symbol": resolved_symbol or symbol,
            "benchmark": benchmark_symbol,
            "error": f"有效樣本不足 ({len(aligned)})",
        }

    stock_vals = aligned.iloc[:, 0].to_numpy(dtype=float)
    bench_vals = aligned.iloc[:, 1].to_numpy(dtype=float)
    bench_var = float(np.var(bench_vals))
    if bench_var <= 0:
        return {"symbol": resolved_symbol or symbol, "benchmark": benchmark_symbol, "error": "基準波動為 0"}

    beta = float(np.cov(stock_vals, bench_vals, ddof=0)[0, 1] / bench_var)
    return {
        "symbol": resolved_symbol or symbol,
        "benchmark": benchmark_symbol,
        "beta": beta,
        "observations": int(len(aligned)),
    }


def _estimate_sector_exposure_twd(
    snapshots: List[Dict[str, Any]],
    target_sector: str,
) -> float:
    exposure_twd = 0.0
    for snapshot in snapshots:
        if snapshot.get("is_cash") or snapshot.get("market_value_twd", 0) <= 0:
            continue
        lookup_symbol = _resolve_lookup_symbol(str(snapshot.get("symbol") or ""))
        profile = market.get_asset_profile(lookup_symbol or str(snapshot.get("symbol") or ""))
        if _get_concentration_bucket(profile) == target_sector:
            exposure_twd += float(snapshot.get("market_value_twd") or 0.0)
    return exposure_twd


def _get_concentration_bucket(profile: Dict[str, Any]) -> str:
    bucket = str(profile.get("concentration_bucket") or "").strip()
    if bucket and bucket != "Unknown":
        return bucket
    sector = str(profile.get("sector") or "").strip()
    if sector and sector not in {"Unknown", "ETF"}:
        return sector
    asset_type = str(profile.get("asset_type") or "Unknown")
    if asset_type == "Tech_Momentum":
        return "Technology"
    if asset_type == "Macro_Hedge":
        return "Macro Hedge"
    if asset_type == "Value_Holding" and bool(profile.get("is_etf")):
        return "Diversified Equity"
    return "Unknown"


def _is_concentration_cap_applicable(bucket: str) -> bool:
    return bucket not in {"", "Unknown", "Diversified Equity"}


def _format_trade_limit_source(
    overlay: Dict[str, Any],
    *,
    limit_key: str,
    limits: Dict[str, float],
    asset_type: str,
) -> str:
    base_limits = TRADE_GOVERNOR_LIMITS.get(str(overlay.get("trade_mode") or "normal"), TRADE_GOVERNOR_LIMITS["normal"])
    source = f"{overlay.get('trade_mode_label', '🟢 Normal')} 檔位"
    if abs(float(limits.get(limit_key, 0.0)) - float(base_limits.get(limit_key, 0.0))) > 1e-9 and asset_type == "Tech_Momentum":
        source += " + Tech_Momentum tighten"
    return source


def _apply_pretrade_risk_gate(
    symbol: str,
    action: str,
    shares: float,
    actual_twd_total: float,
    benchmark: str = "SPY",
) -> Dict[str, Any]:
    gate_result = {
        "allowed": True,
        "approved_shares": float(shares),
        "approved_twd_total": float(actual_twd_total),
        "message": "",
        "note": None,
    }
    if action != "buy" or "CASH" in symbol:
        return gate_result

    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    if not snapshots:
        return gate_result

    overlay = compute_portfolio_risk_overlay(benchmark=benchmark, snapshots=snapshots)
    if overlay.get("error"):
        return {
            **gate_result,
            "allowed": False,
            "message": format_tool_error(
                f"❌ 風控拒單：組合風險 overlay 不可用（{overlay['error']}），暫停新增風險單。",
                data_unavailable=True,
            ),
        }

    current_nav_twd = float(overlay.get("current_nav_twd") or 0.0)
    if current_nav_twd <= 0:
        return {
            **gate_result,
            "allowed": False,
            "message": format_tool_error("❌ 風控拒單：無法取得有效 NAV，暫停新增風險單。", data_unavailable=True),
        }

    if not overlay.get("allow_new_longs", True):
        return {
            **gate_result,
            "allowed": False,
            "message": (
                f"❌ 風控拒單：{overlay.get('trade_mode_label', '組合節流')} "
                f"禁止新增多單。{overlay.get('governor_message', '')}"
            ).strip(),
        }

    existing_snapshot = _find_snapshot_by_symbol(snapshots, symbol)
    if (
        existing_snapshot
        and float(existing_snapshot.get("market_value_twd") or 0.0) > 0
        and float(existing_snapshot.get("pnl_value_twd") or 0.0) < 0
        and not overlay.get("allow_average_down", True)
    ):
        return {
            **gate_result,
            "allowed": False,
            "message": (
                f"❌ 風控拒單：{overlay.get('trade_mode_label', '組合節流')} "
                f"禁止攤平虧損部位，{symbol} 目前浮虧 NT${float(existing_snapshot.get('pnl_value_twd') or 0.0):,.0f}。"
            ),
        }

    lookup_symbol = _resolve_lookup_symbol(symbol)
    profile = market.get_asset_profile(lookup_symbol or symbol)
    asset_type = str(profile.get("asset_type") or "Unknown")
    sector = _get_concentration_bucket(profile)
    tracking_index = str(profile.get("tracking_index") or "").strip()
    is_etf = bool(profile.get("is_etf"))
    limits = _get_trade_governor_limits(str(overlay.get("trade_mode") or "normal"), asset_type)
    sector_limit_source = _format_trade_limit_source(
        overlay,
        limit_key="sector_cap",
        limits=limits,
        asset_type=asset_type,
    )
    single_name_limit_source = _format_trade_limit_source(
        overlay,
        limit_key="single_name_cap",
        limits=limits,
        asset_type=asset_type,
    )

    requested_twd_total = float(actual_twd_total)
    current_position_mv = float(existing_snapshot.get("market_value_twd") or 0.0) if existing_snapshot else 0.0
    current_gross_twd = float(overlay.get("gross_exposure_twd") or 0.0)
    gross_budget_twd = current_nav_twd * float(overlay.get("recommended_gross_scale") or 0.0)
    gross_headroom_twd = max(0.0, gross_budget_twd - current_gross_twd)
    if gross_headroom_twd <= 0:
        return {
            **gate_result,
            "allowed": False,
            "message": (
                f"❌ 風控拒單：目前 Gross Scale 僅允許 {float(overlay.get('recommended_gross_scale') or 0.0):.2f}x，"
                "組合已無新增風險空間。"
            ),
        }

    single_name_cap_twd = current_nav_twd * limits["single_name_cap"]
    position_headroom_twd = max(0.0, single_name_cap_twd - current_position_mv)

    # ETF Look-through Pre-trade Check
    ticker = get_ticker(symbol)
    holdings = ticker.get_holdings() if hasattr(ticker, 'get_holdings') else []
    if holdings:
        # Calculate current total exposure for all assets
        exposure_map = {}
        for s in snapshots:
            sym = s['symbol']
            mv = s['market_value_twd']
            t_ticker = get_ticker(sym)
            t_holdings = t_ticker.get_holdings() if hasattr(t_ticker, 'get_holdings') else []
            if t_holdings:
                for th in t_holdings:
                    exposure_map[th['Symbol']] = exposure_map.get(th['Symbol'], 0) + (mv * th['Percent'])
            else:
                exposure_map[sym] = exposure_map.get(sym, 0) + mv
        
        # Check against single_name_cap for each component
        for h in holdings:
            h_sym = h['Symbol']
            h_weight = h['Percent']
            h_current_exp = exposure_map.get(h_sym, 0.0)
            h_requested_add = requested_twd_total * h_weight
            h_headroom = max(0.0, single_name_cap_twd - h_current_exp)
            if h_headroom < h_requested_add:
                return {
                    **gate_result,
                    "allowed": False,
                    "message": (
                        f"❌ 風控拒單：ETF 穿透風險檢查未通過。買入 {symbol} 將使底層成分股 {h_sym} "
                        f"的總曝險達到 {(h_current_exp + h_requested_add) / current_nav_twd * 100:.1f}%，"
                        f"超過單一持股上限 {limits['single_name_cap'] * 100:.1f}%。"
                    ),
                }
    if position_headroom_twd <= 0:
        return {
            **gate_result,
            "allowed": False,
            "message": (
                f"❌ 風控拒單：{symbol} 已達單一持股上限 "
                f"{limits['single_name_cap'] * 100:.1f}% of NAV。"
                f" 目前 {current_position_mv / current_nav_twd * 100:.1f}% | "
                f"滿單後 {(current_position_mv + requested_twd_total) / current_nav_twd * 100:.1f}% | "
                f"來源: {single_name_limit_source}。"
            ),
        }

    headrooms: List[tuple[str, float]] = [("gross_scale", gross_headroom_twd), ("single_name_cap", position_headroom_twd)]

    if _is_concentration_cap_applicable(sector):
        sector_cap_twd = current_nav_twd * limits["sector_cap"]
        current_sector_mv = _estimate_sector_exposure_twd(snapshots, sector)
        sector_headroom_twd = max(0.0, sector_cap_twd - current_sector_mv)
        if sector_headroom_twd <= 0:
            current_sector_ratio = current_sector_mv / current_nav_twd if current_nav_twd > 0 else 0.0
            requested_sector_ratio = (current_sector_mv + requested_twd_total) / current_nav_twd if current_nav_twd > 0 else 0.0
            etf_context = f" | ETF 指數: {tracking_index}" if is_etf and tracking_index else ""
            return {
                **gate_result,
                "allowed": False,
                "message": (
                    f"❌ 風控拒單：{sector} 曝險已達集中上限 "
                    f"{limits['sector_cap'] * 100:.1f}% of NAV。"
                    f" 目前 {current_sector_ratio * 100:.1f}% | 滿單後 {requested_sector_ratio * 100:.1f}% | "
                    f"來源: {sector_limit_source}{etf_context}。"
                ),
            }
        headrooms.append((f"sector_cap:{sector}", sector_headroom_twd))

    target_beta_band = overlay.get("target_beta_band") or [None, None]
    current_beta_to_nav = overlay.get("current_beta_to_nav")
    target_beta_high = target_beta_band[1] if len(target_beta_band) >= 2 else None
    beta_estimate = _estimate_symbol_beta(lookup_symbol or symbol, benchmark=benchmark, series_cache={})
    if (
        not beta_estimate.get("error")
        and isinstance(current_beta_to_nav, (int, float))
        and isinstance(target_beta_high, (int, float))
        and float(beta_estimate.get("beta") or 0.0) > 0
    ):
        beta_headroom_twd = max(
            0.0,
            ((float(target_beta_high) - float(current_beta_to_nav)) * current_nav_twd) / float(beta_estimate["beta"]),
        )
        if beta_headroom_twd <= 0:
            return {
                **gate_result,
                "allowed": False,
                "message": (
                    f"❌ 風控拒單：下單前組合 β 預算已滿 "
                    f"(目前 {float(current_beta_to_nav):.2f} / 上限 {float(target_beta_high):.2f})。"
                ),
            }
        headrooms.append(("beta_budget", beta_headroom_twd))

    binding_constraint, approved_twd_total = min(headrooms, key=lambda item: item[1])
    approved_twd_total = min(requested_twd_total, float(approved_twd_total))
    if approved_twd_total + 1e-9 >= requested_twd_total:
        return gate_result

    approved_shares = _floor_trade_quantity(float(shares) * (approved_twd_total / requested_twd_total))
    if approved_shares <= 0:
        return {
            **gate_result,
            "allowed": False,
            "message": f"❌ 風控拒單：{binding_constraint} 限制後可下股數趨近於 0。",
        }

    approved_twd_total = float(actual_twd_total) * (approved_shares / float(shares))
    cap_labels = {
        "gross_scale": f"Gross Scale {float(overlay.get('recommended_gross_scale') or 0.0):.2f}x",
        "single_name_cap": f"單一持股上限 {limits['single_name_cap'] * 100:.1f}% NAV",
        "beta_budget": f"組合 β 上限 {float(target_beta_high):.2f}",
    }
    if binding_constraint.startswith("sector_cap:"):
        sector_name = binding_constraint.split(":", 1)[1]
        cap_labels[binding_constraint] = f"{sector_name} 集中上限 {limits['sector_cap'] * 100:.1f}% NAV"
    binding_label = cap_labels.get(binding_constraint, binding_constraint)
    context_suffix = ""
    if binding_constraint == "single_name_cap":
        context_suffix = (
            f" 目前 {current_position_mv / current_nav_twd * 100:.1f}% -> "
            f"滿單後 {(current_position_mv + requested_twd_total) / current_nav_twd * 100:.1f}% | "
            f"來源: {single_name_limit_source}。"
        )
    elif binding_constraint.startswith("sector_cap:") and current_nav_twd > 0:
        current_sector_mv = _estimate_sector_exposure_twd(snapshots, sector)
        approved_sector_ratio = (current_sector_mv + approved_twd_total) / current_nav_twd
        requested_sector_ratio = (current_sector_mv + requested_twd_total) / current_nav_twd
        etf_context = f" | ETF 指數: {tracking_index}" if is_etf and tracking_index else ""
        context_suffix = (
            f" 目前 {current_sector_mv / current_nav_twd * 100:.1f}% -> "
            f"滿單後 {requested_sector_ratio * 100:.1f}% -> "
            f"核准後 {approved_sector_ratio * 100:.1f}% | "
            f"來源: {sector_limit_source}{etf_context}。"
        )
    return {
        **gate_result,
        "approved_shares": approved_shares,
        "approved_twd_total": approved_twd_total,
        "message": (
            f"⚠️ 風控縮倉：{symbol} 由 {float(shares):.4f} 股縮至 {approved_shares:.4f} 股 "
            f"({binding_label})。{context_suffix}".strip()
        ),
        "note": (
            f"risk_gate:{binding_label}; requested_shares={float(shares):.4f}; "
            f"approved_shares={approved_shares:.4f}; source={sector_limit_source if binding_constraint.startswith('sector_cap:') else single_name_limit_source if binding_constraint == 'single_name_cap' else binding_label}"
        ),
    }

def _format_confirmed_trade_risk_feedback(gate: Dict[str, Any], requested_shares: float) -> Dict[str, Any]:
    gate_message = str(gate.get("message") or "").strip()
    gate_note = str(gate.get("note") or "").strip()
    approved_shares = float(gate.get("approved_shares") or requested_shares)

    if gate.get("allowed", True) and approved_shares + 1e-9 >= float(requested_shares) and not gate_message and not gate_note:
        return {"message": "", "note": None}

    detail = gate_message
    for prefix in ("❌ 風控拒單：", "⚠️ 風控縮倉："):
        if detail.startswith(prefix):
            detail = detail[len(prefix):].strip()
            break
    if not detail:
        detail = "此筆交易觸發組合風控限制。"

    warning_message = (
        f"⚠️ 成交後風控警告：{detail} "
        f"本次因 /trade 視為已成交，仍已照原始 {float(requested_shares):.4f} 股入帳。"
    ).strip()
    warning_note = f"post_trade_warning:{gate_note or detail}"
    return {"message": warning_message, "note": warning_note}


def execute_position_update(
    symbol: str,
    price: float,
    shares: float,
    action: str = 'set',
    total_amount_twd: float = None,
    locked: int = None,
    sync_memory: bool = False,
    enforce_pretrade_gate: bool = True,
    trade_plan: Dict[str, Any] | None = None,
) -> str:
    """Pure portfolio-write logic for direct callers and tests."""
    symbol = normalize_ticker(symbol)
    try:
        shares = float(shares)
        price = float(price)
    except (TypeError, ValueError):
        return format_tool_error("❌ 價格與股數必須是數字。", data_unavailable=True)
    if action in {"buy", "sell"} and shares <= 0:
        return format_tool_error("❌ 買賣股數必須大於 0。", data_unavailable=True)
    if action == "set" and shares < 0:
        return format_tool_error("❌ 校正股數不能是負數。", data_unavailable=True)
    if price < 0:
        return format_tool_error("❌ 價格不能是負數。", data_unavailable=True)
    clean_sym_for_market = symbol.replace('.TW', '').replace('.TWO', '').replace('_ESOP', '').replace('_TRUST', '')
    is_taiwan = (any(char.isdigit() for char in clean_sym_for_market) and len(clean_sym_for_market) <= 6) or symbol.endswith('.TW') or symbol.endswith('.TWO')
    is_cash = 'CASH' in symbol
    fx_rate = fetch_exchange_rate() if (not is_taiwan and not is_cash) else 1.0
    
    # 核心邏輯：計算該次異動的台幣價值
    if total_amount_twd:
        actual_twd_total = total_amount_twd
        actual_unit_price = total_amount_twd / shares / fx_rate if (shares > 0 and fx_rate > 0) else price
    else:
        actual_unit_price = price
        actual_twd_total = price * shares * fx_rate

    settle_currency = 'CASH_TWD' if is_taiwan else 'CASH_USD'
    # 美股扣款原幣，台股扣款台幣
    settle_amount = actual_unit_price * shares if not is_taiwan else actual_twd_total
    gate_message = ""
    trade_note = None
    candidate_trade_plan: Dict[str, Any] | None = None
    trade_plan_validation: Dict[str, Any] | None = None
    persisted_trade_plan: Dict[str, Any] | None = None
    skipped_trade_plan_warning = ""

    if action == "buy" and not is_cash:
        candidate_trade_plan = _build_trade_plan_payload(
            symbol=symbol,
            entry_price=actual_unit_price,
            trade_plan=trade_plan,
            source="bot_trade",
            status="active",
        )
        trade_plan_validation = validate_trade_plan_payload(candidate_trade_plan)
        if enforce_pretrade_gate:
            trade_plan_error = _format_trade_plan_validation_error(trade_plan_validation)
            if trade_plan_error:
                return trade_plan_error
        gate = _apply_pretrade_risk_gate(symbol, action, shares, actual_twd_total)
        if enforce_pretrade_gate:
            if not gate.get("allowed", True):
                return gate.get("message") or "❌ 風控拒單。"
            approved_shares = float(gate.get("approved_shares", shares))
            approved_twd_total = float(gate.get("approved_twd_total", actual_twd_total))
            if approved_shares < shares:
                shares = approved_shares
                actual_twd_total = approved_twd_total
                settle_amount = actual_unit_price * shares if not is_taiwan else actual_twd_total
                gate_message = gate.get("message", "")
                trade_note = gate.get("note")
        else:
            confirmed_feedback = _format_confirmed_trade_risk_feedback(gate, shares)
            gate_message = confirmed_feedback.get("message", "")
            trade_note = confirmed_feedback.get("note")

        if trade_plan_validation["complete"]:
            persisted_trade_plan = candidate_trade_plan
        elif trade_plan and not enforce_pretrade_gate:
            skipped_trade_plan_warning = "⚠️ 交易計畫未儲存（欄位不完整）"

    result_message = ""
    should_refresh_memory = False

    # Build enriched snapshot before acquiring the lock (avoids network I/O inside lock).
    entry_decision_snapshot: Dict[str, Any] | None = None
    if action == "buy" and not is_cash:
        try:
            entry_decision_snapshot = _build_trade_decision_snapshot(
                symbol, entry_notional_twd=actual_twd_total
            )
        except Exception as exc:
            logger.warning(f"Trade decision snapshot build failed for {symbol}: {exc}")

    pending_trade_journal_ids: list[int] = []
    with db_lock:
        conn = get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # 取得標的與現金池現況
            cursor.execute("SELECT cost, shares, twd_cost, locked FROM portfolio WHERE symbol = ?", (symbol,))
            old_pos = cursor.fetchone() or (0.0, 0.0, 0.0, 0)
            
            # 覆寫鎖定狀態
            current_locked = locked if locked is not None else old_pos[3]

            cursor.execute("SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = ?", (settle_currency,))
            cash_pos = cursor.fetchone() or (1.0 if 'TWD' in settle_currency else fx_rate, 0.0, 0.0)

            if action == 'buy':
                if cash_pos[1] < settle_amount:
                    result_message = f"❌ 買進失敗：{settle_currency} 餘額不足！(剩 {cash_pos[1]:.2f})"
                else:
                    new_shares = old_pos[1] + shares
                    new_twd_cost = old_pos[2] + actual_twd_total
                    new_cost = (old_pos[0] * old_pos[1] + actual_unit_price * shares) / new_shares
                    cash_before = cash_pos[1]
                    cash_after = cash_before - settle_amount
                    _upsert_portfolio_row(cursor, symbol, new_cost, new_shares, new_twd_cost, current_locked)
                    _upsert_portfolio_row(cursor, settle_currency, cash_pos[0], cash_after, cash_pos[2] - actual_twd_total, 0)
                    trade_log_id = _record_trade_log(
                        cursor,
                        symbol=symbol,
                        action='buy',
                        price=actual_unit_price,
                        shares=shares,
                        settle_currency=settle_currency,
                        settle_amount=settle_amount,
                        fx_rate=fx_rate,
                        cash_before=cash_before,
                        cash_after=cash_after,
                        note=trade_note,
                        decision_snapshot=entry_decision_snapshot,
                    )
                    if not is_cash:
                        pending_trade_journal_ids.append(trade_log_id)
                    if persisted_trade_plan:
                        _upsert_trade_plan_locked(
                            cursor,
                            **(persisted_trade_plan | {"opened_trade_log_id": trade_log_id}),
                        )
                    result_message = f"✅ 買進成功！從 {settle_currency} 扣款 {settle_amount:.2f}"
                    if gate_message:
                        result_message = f"{gate_message} {result_message}".strip()
                    if skipped_trade_plan_warning:
                        result_message = f"{result_message} {skipped_trade_plan_warning}".strip()
                    should_refresh_memory = True
            
            elif action == 'sell':
                if old_pos[3] == 1:
                    result_message = f"❌ 賣出失敗：標的 {symbol} 被鎖定 (福利信託/長期持有)，禁止機器人操作。請手動解除鎖定後再試。"
                elif old_pos[1] < shares:
                    result_message = f"❌ 賣出失敗：持股不足 (只有 {old_pos[1]})"
                else:
                    cost_ratio = shares / old_pos[1]
                    realized_twd_cost = old_pos[2] * cost_ratio
                    realized_pnl = actual_twd_total - realized_twd_cost
                    new_shares = old_pos[1] - shares
                    cash_before = cash_pos[1]
                    cash_after = cash_before + settle_amount
                    if new_shares > 0:
                        cursor.execute("UPDATE portfolio SET shares = ?, twd_cost = twd_cost - ? WHERE symbol = ?", (new_shares, realized_twd_cost, symbol))
                    else:
                        cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
                    _upsert_portfolio_row(cursor, settle_currency, cash_pos[0], cash_after, cash_pos[2] + actual_twd_total, 0)
                    _record_trade_log(
                        cursor,
                        symbol=symbol,
                        action='sell',
                        price=actual_unit_price,
                        shares=shares,
                        settle_currency=settle_currency,
                        settle_amount=settle_amount,
                        fx_rate=fx_rate,
                        realized_pnl=realized_pnl,
                        cash_before=cash_before,
                        cash_after=cash_after,
                    )
                    result_message = f"✅ 賣出成功！實現損益: NT${realized_pnl:+.0f}"
                    should_refresh_memory = True
            
            elif action == 'set':
                _upsert_portfolio_row(cursor, symbol, actual_unit_price, shares, actual_twd_total, current_locked)
                _record_trade_log(
                    cursor,
                    symbol=symbol,
                    action='set',
                    price=actual_unit_price,
                    shares=shares,
                    settle_currency=symbol if is_cash else None,
                    settle_amount=(shares - old_pos[1]) if is_cash else None,
                    fx_rate=fx_rate,
                    cash_before=old_pos[1] if is_cash else None,
                    cash_after=shares if is_cash else None,
                    note=f"manual set; locked={current_locked}",
                )
                result_message = f"✅ 校正成功！{symbol} 已更新 (Locked: {current_locked})。"
                should_refresh_memory = True
            else:
                result_message = f"❌ 未知操作: {action}"

            conn.commit()
        except Exception as e:
            logger.error(f"Position update failed for {symbol}: {e}")
            return format_tool_error(f"❌ 記帳異常: {e}", transient=True)
        finally:
            conn.close()

    # Enqueue journal checkpoints after the lock is fully released.
    if pending_trade_journal_ids:
        try:
            import engine_journal
            engine_journal.enqueue_trade_outcome_checkpoints(pending_trade_journal_ids)
        except Exception as exc:
            logger.warning(f"Journal enqueue failed after buy for {symbol}: {exc}")

    if sync_memory and should_refresh_memory:
        try:
            refresh_portfolio_health_summary(source="portfolio_trade")
        except Exception as e:
            logger.warning(f"Portfolio health refresh failed after updating {symbol}: {e}")

    return result_message

@tool(mode="write")
def update_position(symbol: str, price: float, shares: float, action: str = 'set', total_amount_twd: float = None, locked: int = None) -> str:
    """
    Updates a portfolio position or cash balance.
    action: 'buy', 'sell', or 'set' (manual adjustment).
    price: unit price in original currency.
    shares: quantity to change.
    locked: 1 to lock position from AI trading, 0 to unlock.
    """
    return execute_position_update(symbol, price, shares, action, total_amount_twd, locked, sync_memory=True)

# --- 標的名對應表 (手動維護優先，其餘自動偵測) ---
SYMBOL_NAME_MAP = {
    "CASH_TWD": "台幣現金池",
    "CASH_USD": "美金現金池",
}

_AUTO_NAME_CACHE = {}

def get_symbol_name(symbol: str) -> str:
    symbol = normalize_ticker(symbol)
    if symbol in SYMBOL_NAME_MAP:
        return SYMBOL_NAME_MAP[symbol]
    
    if symbol in _AUTO_NAME_CACHE:
        return _AUTO_NAME_CACHE[symbol]

    # 自動偵測邏輯
    clean_sym = symbol.replace('.TW', '').replace('.TWO', '').replace('_ESOP', '').replace('_TRUST', '')
    is_taiwan = (any(char.isdigit() for char in clean_sym) and len(clean_sym) <= 6)
    
    name = symbol
    try:
        if is_taiwan and fubon.fubon_ready:
            # 嘗試從 Fubon 抓取名稱
            reststock = fubon.fubon_sdk.marketdata.rest_client.stock
            # 先試 intraday quote
            quote = reststock.intraday.quote(symbol=clean_sym)
            if isinstance(quote, dict) and quote.get('name'):
                name = f"{quote['name']}"
            else:
                # 再試 historical stats
                stats = reststock.historical.stats(symbol=clean_sym)
                if isinstance(stats, dict) and stats.get('name'):
                    name = f"{stats['name']}"
        else:
            # 嘗試 yfinance
            import yfinance as yf
            query_sym = market._normalize_lookup_symbol(clean_sym)
            ticker = get_ticker(query_sym, cache_level="daily")
            name = ticker.info.get('shortName') or ticker.info.get('longName') or clean_sym
    except Exception as e:
        logger.debug(f"Symbol name lookup failed for {symbol}: {e}")

    # 特殊字尾裝飾
    if '_ESOP' in symbol or '_TRUST' in symbol:
        name = f"{name} (員工福利信託)"

    _AUTO_NAME_CACHE[symbol] = name
    return name

def build_portfolio_raw_data() -> str:
    """Pure portfolio snapshot logic for direct callers and tests."""
    if fubon.fubon_ready:
        sync_fubon_portfolio_state(source="portfolio_query", sync_memory=False)

    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()

        try:
            # 1. 取得資料庫目前的倉位
            cursor.execute("SELECT symbol, cost, shares, twd_cost, locked FROM portfolio")
            db_rows = cursor.fetchall()
            db_dict = {r[0]: list(r) for r in db_rows}

            # 2. 組裝回傳資料
            records = []
            for sym, data in db_dict.items():
                # 【V5.4 強化】精準市場分類邏輯
                if sym.startswith('CASH'):
                    market_type = "CASH"
                elif sym.endswith('.L') or sym.endswith('.IL'):
                    market_type = "UK"
                elif (sym.replace('.TW','').replace('.TWO','').replace('_TRUST','').replace('_ESOP','').isdigit()) or \
                     (any(c.isdigit() for c in sym[:4]) and len(sym.split('.')[0]) <= 6):
                    # 規則：純數字、或前四碼含數字且長度<=6 (涵蓋 00981A, 2330.TW 等)
                    market_type = "TW"
                else:
                    market_type = "US"

                records.append({
                    "symbol": sym,
                    "name": get_symbol_name(sym),
                    "cost": data[1],
                    "shares": data[2],
                    "twd_cost": data[3],
                    "locked": bool(data[4]),
                    "market": market_type
                })
            lines = [f"{r['symbol']}|{r['shares']}sh|cost={r['cost']}|{r['market']}" for r in records]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Portfolio snapshot failed: {e}")
            return format_tool_error("[]", transient=True)
        finally:
            conn.close()


def build_portfolio_detailed_raw_data() -> str:
    """Pure portfolio snapshot logic with live prices and PnL."""
    # 1. 觸發與 Fubon 同步，並清除幽靈庫存
    build_portfolio_raw_data()
    
    # 2. 取得包含即時報價與 TWD PnL 的快照
    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    if not snapshots:
        return "[]"
    
    lines = []
    for s in snapshots:
        name = get_symbol_name(s['symbol'])
        price = s.get('current_price', 0.0)
        cost = s.get('cost', 0.0)
        mv_twd = s.get('market_value_twd', 0.0)
        pnl_twd = s.get('pnl_value_twd', 0.0)
        pnl_pct = s.get('pnl_percent', 0.0)
        lines.append(
            f"{s['symbol']}|{name}|{s['shares']}sh|cost={cost:.4f}|price={price:.4f}|market_value_twd={mv_twd:.0f}|pnl_twd={pnl_twd:+.0f}|pnl_pct={pnl_pct:+.1f}%|{s['market']}"
        )
    return "\n".join(lines)


@tool()
def get_portfolio_raw_data() -> str:
    """Retrieves current portfolio positions, prices, and TWD balances."""
    return build_portfolio_detailed_raw_data()


def _classify_portfolio_market(symbol: str) -> str:
    if symbol.startswith("CASH"):
        return "CASH"
    if symbol.endswith(".L") or symbol.endswith(".IL"):
        return "UK"
    clean_symbol = symbol.replace(".TW", "").replace(".TWO", "").replace("_TRUST", "").replace("_ESOP", "")
    if clean_symbol.isdigit() or (any(c.isdigit() for c in clean_symbol[:4]) and len(clean_symbol) <= 6):
        return "TW"
    return "US"


def _load_portfolio_rows() -> List[tuple]:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, cost, shares, twd_cost FROM portfolio")
            return cursor.fetchall()
        finally:
            conn.close()


def _build_live_position_snapshots(rows: List[tuple]) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    for sym, cost, shares, twd_cost in rows:
        if shares <= 0:
            continue

        market = _classify_portfolio_market(sym)
        is_cash = market == "CASH"
        is_us_stock = market == "US"
        current_price = float(cost or 0.0)

        try:
            if is_cash:
                current_price = 1.0 if sym == "CASH_TWD" else fetch_exchange_rate()
            elif market == "TW":
                refreshed = False
                if fubon.fubon_ready:
                    try:
                        quote = fubon.fubon_sdk.marketdata.rest_client.stock.intraday.quote(
                            symbol=sym.replace(".TW", "").replace(".TWO", "")
                        )
                        refreshed_price = quote.get("closePrice") or quote.get("lastPrice")
                        if refreshed_price is not None and not pd.isna(refreshed_price):
                            current_price = float(refreshed_price)
                            refreshed = True
                    except Exception as exc:
                        logger.debug(f"Fubon price refresh failed for {sym}: {exc}")
                
                if not refreshed:
                    # Fallback to Yahoo Finance with normalized symbol
                    clean_sym_for_yf = sym.replace('_TRUST', '').replace('_ESOP', '')
                    lookup_sym = _resolve_lookup_symbol(clean_sym_for_yf)
                    if lookup_sym:
                        ticker = get_ticker(lookup_sym, cache_level="daily")
                        fast_info = getattr(ticker, "fast_info", {}) or {}
                        refreshed_price = fast_info.get("last_price")
                        if refreshed_price is None:
                            hist = ticker.history(period="5d")
                            if not hist.empty and "Close" in hist.columns:
                                close_series = pd.to_numeric(hist["Close"], errors="coerce").dropna()
                                if not close_series.empty:
                                    refreshed_price = close_series.iloc[-1]
                        
                        if refreshed_price is not None and not pd.isna(refreshed_price):
                            current_price = float(refreshed_price)
            else:
                ticker = get_ticker(sym, cache_level="daily")
                fast_info = getattr(ticker, "fast_info", {}) or {}
                refreshed_price = fast_info.get("last_price")
                if refreshed_price is None:
                    hist = ticker.history(period="5d")
                    if not hist.empty and "Close" in hist.columns:
                        close_series = pd.to_numeric(hist["Close"], errors="coerce").dropna()
                        if not close_series.empty:
                            refreshed_price = close_series.iloc[-1]
                if refreshed_price is not None and not pd.isna(refreshed_price):
                    current_price = float(refreshed_price)
        except Exception as e:
            logger.warning(f"Portfolio analysis price refresh failed for {sym}: {e}")

        pnl = calculate_position_pnl(sym, current_price, shares, twd_cost, is_us_stock)
        snapshots.append(
            {
                "symbol": sym,
                "market": market,
                "is_cash": is_cash,
                "is_us_stock": is_us_stock,
                "shares": float(shares),
                "cost": float(cost or 0.0),
                "twd_cost": float(twd_cost or 0.0),
                "current_price": float(current_price),
                "market_value_twd": float(pnl["market_value_twd"]),
                "pnl_value_twd": float(pnl["pnl_value_twd"]),
                "pnl_percent": float(pnl["pnl_percent"]),
            }
        )
    return snapshots


def _build_current_holdings_weights() -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    holdings = [pos for pos in snapshots if not pos["is_cash"] and pos["market_value_twd"] > 0]
    total_mv = sum(pos["market_value_twd"] for pos in holdings)
    if total_mv <= 0:
        return {}, holdings
    return ({pos["symbol"]: pos["market_value_twd"] / total_mv for pos in holdings}, holdings)


def _resolve_lookup_symbol(symbol: str) -> str:
    normalized = normalize_ticker(str(symbol or "")).upper()
    normalized = normalized.replace("_TRUST", "").replace("_ESOP", "")
    if not normalized:
        return ""
    try:
        return market._normalize_lookup_symbol(normalized)
    except Exception as exc:
        logger.debug(f"Portfolio symbol normalization fallback for {normalized}: {exc}")
        return normalized


def _load_daily_return_series(
    symbol: str,
    period: str = "6mo",
    series_cache: Dict[tuple[str, str], pd.Series] | None = None,
) -> tuple[str, pd.Series, str | None]:
    resolved = _resolve_lookup_symbol(symbol)
    if not resolved:
        return "", pd.Series(dtype=float), "無效代碼"

    cache_key = (resolved, period)
    if series_cache is not None and cache_key in series_cache:
        return resolved, series_cache[cache_key].copy(), None

    try:
        history = get_ticker(resolved, cache_level="daily").history(period=period, interval="1d")
    except Exception as exc:
        return resolved, pd.Series(dtype=float), f"價格抓取失敗: {exc}"
    if history.empty or "Close" not in history.columns:
        return resolved, pd.Series(dtype=float), "缺少 Close 歷史資料"

    close = pd.to_numeric(history["Close"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if series_cache is not None:
        series_cache[cache_key] = returns.copy()
    return resolved, returns, None


def _parse_symbol_input(symbols: str | List[str] | None) -> List[str]:
    if symbols is None:
        return []
    raw_items = symbols if isinstance(symbols, list) else str(symbols).replace(",", " ").split()
    normalized = []
    for item in raw_items:
        symbol = normalize_ticker(str(item))
        if not symbol:
            continue
        s = _resolve_lookup_symbol(symbol)
        if s not in normalized:
            normalized.append(s)
    return normalized


def _summarize_live_snapshots(snapshots: List[Dict[str, Any]]) -> Dict[str, float]:
    total_cost_twd = sum(pos["twd_cost"] for pos in snapshots)
    total_nav_twd = sum(pos["market_value_twd"] for pos in snapshots)
    gross_exposure_twd = sum(pos["market_value_twd"] for pos in snapshots if not pos["is_cash"])
    cash_twd = sum(pos["market_value_twd"] for pos in snapshots if pos["is_cash"])
    total_pnl_pct = ((total_nav_twd - total_cost_twd) / total_cost_twd * 100) if total_cost_twd > 0 else 0.0
    return {
        "total_cost_twd": float(total_cost_twd),
        "total_nav_twd": float(total_nav_twd),
        "gross_exposure_twd": float(gross_exposure_twd),
        "cash_twd": float(cash_twd),
        "gross_exposure_ratio": float(gross_exposure_twd / total_nav_twd) if total_nav_twd > 0 else 0.0,
        "cash_ratio": float(cash_twd / total_nav_twd) if total_nav_twd > 0 else 0.0,
        "total_pnl_pct": float(total_pnl_pct),
    }


def _build_holdings_weights_from_snapshots(snapshots: List[Dict[str, Any]]) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
    holdings = [pos for pos in snapshots if not pos["is_cash"] and pos["market_value_twd"] > 0]
    total_mv = sum(pos["market_value_twd"] for pos in holdings)
    if total_mv <= 0:
        return {}, holdings
    return ({pos["symbol"]: pos["market_value_twd"] / total_mv for pos in holdings}, holdings)


def build_portfolio_analysis(snapshots: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """生成持倉健檢摘要，用於系統自動更新額葉。"""
    snapshots = snapshots if snapshots is not None else _build_live_position_snapshots(_load_portfolio_rows())
    if not snapshots:
        return {
            "total_current": 0,
            "total_pnl_pct": 0,
            "top3_concentration": 0,
            "position_count": 0,
            "summary": "無有效持倉數據；請確認帳本或券商同步狀態。",
        }

    totals = _summarize_live_snapshots(snapshots)
    total_cost_twd = totals["total_cost_twd"]
    total_market_value_twd = totals["total_nav_twd"]
    assets = [{"symbol": pos["symbol"], "mv": pos["market_value_twd"]} for pos in snapshots if not pos["is_cash"]]

    # 計算集中度
    assets.sort(key=lambda x: x['mv'], reverse=True)
    top3_mv = sum(a['mv'] for a in assets[:3])
    top3_pct = (top3_mv / total_market_value_twd * 100) if total_market_value_twd > 0 else 0

    summary = (
        f"NAV: NT${total_market_value_twd:,.0f} | "
        f"PnL: {totals['total_pnl_pct']:+.1f}% | "
        f"Top3 集中度: {top3_pct:.0f}%"
    )
    
    # 如果有大幅變動，增加警語
    if abs(totals["total_pnl_pct"]) > 5:
        summary += f" (⚠️ 總體損益波動劇烈)"

    return {
        "total_current": total_market_value_twd,
        "total_pnl_pct": totals["total_pnl_pct"],
        "top3_concentration": top3_pct,
        "position_count": len(snapshots),
        "summary": summary
    }

def refresh_portfolio_health_summary(source: str = "portfolio_review") -> Dict[str, Any]:
    """Builds a portfolio-health snapshot and patches the frontal lobe section."""
    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    nav_snapshot = record_portfolio_nav_snapshot(source=source, snapshots=snapshots) if snapshots else {"error": "無有效持倉"}
    analysis = build_portfolio_analysis(snapshots=snapshots)
    overlay = compute_portfolio_risk_overlay(snapshots=snapshots)
    summary = analysis["summary"]
    if not overlay.get("error"):
        summary += (
            f" | DD: {overlay['current_drawdown'] * 100:.1f}%"
            f" | {overlay['trade_mode_label']}"
            f" | Gross Scale {overlay['recommended_gross_scale']:.2f}x"
        )
        analysis["summary"] = summary
    import engine_memory as memory
 
    memory_update = memory.patch_frontal_lobe_section("Portfolio Health", analysis["summary"], source=source)
    return {**analysis, "risk_overlay": overlay, "nav_snapshot": nav_snapshot, "memory_update": memory_update}

@tool()
def get_portfolio_analysis() -> str:
    """Returns a high-level summary of portfolio health, NAV, and concentration."""
    res = build_portfolio_analysis()
    return res['summary']


def compute_portfolio_analytics(risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> Dict[str, Any]:
    with db_lock:
        conn = get_connection()
        try:
            trades = pd.read_sql(
                "SELECT timestamp, action, settle_amount, fx_rate, realized_pnl FROM trade_log WHERE action = 'sell'",
                conn,
            )
        finally:
            conn.close()

    if trades.empty:
        return {"error": "尚無已實現賣出交易，無法計算量化績效。"}

    sells = trades.copy()
    for column in ("settle_amount", "fx_rate", "realized_pnl"):
        sells[column] = pd.to_numeric(sells[column], errors="coerce")
    sells["timestamp"] = pd.to_datetime(sells["timestamp"], utc=True, errors="coerce")
    sells["fx_rate"] = sells["fx_rate"].replace(0, np.nan).fillna(1.0)
    sells = sells.dropna(subset=["timestamp", "settle_amount", "realized_pnl"]).copy()
    if sells.empty:
        return {"error": "trade_log 缺少可用的賣出審計資料。"}

    sells["proceeds_twd"] = sells["settle_amount"] * sells["fx_rate"]
    sells["cost_basis_twd"] = sells["proceeds_twd"] - sells["realized_pnl"]
    sells = sells[sells["cost_basis_twd"] > 0].copy()
    if sells.empty:
        return {"error": "賣出審計資料無法還原成本基礎，無法計算績效。"}

    sells["trade_return"] = sells["realized_pnl"] / sells["cost_basis_twd"]
    sells["trade_day"] = sells["timestamp"].dt.tz_convert(None).dt.normalize()

    daily = sells.groupby("trade_day", as_index=True).agg(
        realized_pnl=("realized_pnl", "sum"),
        cost_basis_twd=("cost_basis_twd", "sum"),
    )
    daily["closed_return"] = daily["realized_pnl"] / daily["cost_basis_twd"]
    daily_index = pd.bdate_range(start=daily.index.min(), end=daily.index.max())
    daily_returns = daily["closed_return"].reindex(daily_index, fill_value=0.0)

    rf_daily = risk_free_rate / TRADING_DAYS_PER_YEAR
    mean_return = float(daily_returns.mean()) if not daily_returns.empty else 0.0
    daily_std = float(daily_returns.std(ddof=0)) if len(daily_returns) > 1 else 0.0
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = float(downside_returns.std(ddof=0)) if len(downside_returns) > 0 else 0.0

    sharpe_ratio = ((mean_return - rf_daily) / daily_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if daily_std > 0 else None
    sortino_ratio = ((mean_return - rf_daily) / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else None
    sortino_unbounded = downside_std == 0 and mean_return > rf_daily

    equity_curve = (1.0 + daily_returns).cumprod()
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    max_drawdown = abs(float(drawdown.min())) if not drawdown.empty else 0.0
    annual_return = (
        float(equity_curve.iloc[-1]) ** (TRADING_DAYS_PER_YEAR / len(daily_returns)) - 1.0
        if len(daily_returns) > 0 and float(equity_curve.iloc[-1]) > 0
        else None
    )
    calmar_ratio = (annual_return / max_drawdown) if annual_return is not None and max_drawdown > 0 else None
    calmar_unbounded = max_drawdown == 0 and annual_return is not None and annual_return > 0

    wins = int((sells["realized_pnl"] > 0).sum())
    losses = int((sells["realized_pnl"] < 0).sum())
    total_trades = wins + losses
    gross_profit = float(sells.loc[sells["realized_pnl"] > 0, "realized_pnl"].sum())
    gross_loss = abs(float(sells.loc[sells["realized_pnl"] < 0, "realized_pnl"].sum()))
    avg_win = float(sells.loc[sells["realized_pnl"] > 0, "realized_pnl"].mean()) if wins else None
    avg_loss = abs(float(sells.loc[sells["realized_pnl"] < 0, "realized_pnl"].mean())) if losses else None

    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
    profit_factor_unbounded = gross_loss == 0 and gross_profit > 0
    avg_win_loss_ratio = (avg_win / avg_loss) if avg_win is not None and avg_loss not in (None, 0) else None
    avg_win_loss_unbounded = avg_loss in (None, 0) and avg_win is not None and avg_win > 0
    expectancy_twd = float(sells["realized_pnl"].mean())
    win_rate = (wins / total_trades) if total_trades > 0 else 0.0
    net_realized_pnl_twd = float(sells["realized_pnl"].sum())

    methodology = "以已實現 sell 審計資料重建 closed-book 日報酬；Sharpe/Sortino/Drawdown 不含未實現部位 MTM。"
    summary = (
        f"Closed-book Sharpe {sharpe_ratio:.2f}" if sharpe_ratio is not None else "Closed-book Sharpe N/A"
    )
    summary += f" | Win Rate {win_rate:.1%} | Profit Factor "
    summary += f"{profit_factor:.2f}" if profit_factor is not None else ("∞" if profit_factor_unbounded else "N/A")

    return {
        "basis": "closed_book_daily",
        "methodology": methodology,
        "closed_trade_count": total_trades,
        "daily_observations": int(len(daily_returns)),
        "net_realized_pnl_twd": round(net_realized_pnl_twd, 2),
        "expectancy_twd": round(expectancy_twd, 2),
        "win_rate": round(win_rate, 4),
        "gross_profit_twd": round(gross_profit, 2),
        "gross_loss_twd": round(gross_loss, 2),
        "sharpe_ratio": round(float(sharpe_ratio), 2) if sharpe_ratio is not None else None,
        "sortino_ratio": round(float(sortino_ratio), 2) if sortino_ratio is not None else None,
        "sortino_unbounded": sortino_unbounded,
        "max_drawdown": round(max_drawdown, 4),
        "annual_return": round(float(annual_return), 4) if annual_return is not None else None,
        "calmar_ratio": round(float(calmar_ratio), 2) if calmar_ratio is not None else None,
        "calmar_unbounded": calmar_unbounded,
        "profit_factor": round(float(profit_factor), 2) if profit_factor is not None else None,
        "profit_factor_unbounded": profit_factor_unbounded,
        "avg_win_loss_ratio": round(float(avg_win_loss_ratio), 2) if avg_win_loss_ratio is not None else None,
        "avg_win_loss_unbounded": avg_win_loss_unbounded,
        "summary": summary,
    }


def _format_metric(value: float | None, *, digits: int = 2, pct: bool = False, unbounded: bool = False) -> str:
    if unbounded:
        return "∞"
    if value is None:
        return "N/A"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.{digits}f}"


def build_portfolio_analytics_report() -> str:
    analytics = compute_portfolio_analytics()
    if analytics.get("error"):
        return format_tool_error(f"❌ {analytics['error']}", data_unavailable=True)

    report = "📊 === Portfolio Quant Analytics ===\n"
    report += "● 基礎: 已實現平倉日序列 (未含未實現 MTM)\n"
    report += (
        f"● Sharpe: {_format_metric(analytics.get('sharpe_ratio'))} | "
        f"Sortino: {_format_metric(analytics.get('sortino_ratio'), unbounded=analytics.get('sortino_unbounded', False))} | "
        f"Max DD: {_format_metric(analytics.get('max_drawdown'), pct=True)} | "
        f"Calmar: {_format_metric(analytics.get('calmar_ratio'), unbounded=analytics.get('calmar_unbounded', False))}\n"
    )
    report += (
        f"● Win Rate: {_format_metric(analytics.get('win_rate'), pct=True)} | "
        f"Profit Factor: {_format_metric(analytics.get('profit_factor'), unbounded=analytics.get('profit_factor_unbounded', False))} | "
        f"Avg Win/Loss: {_format_metric(analytics.get('avg_win_loss_ratio'), unbounded=analytics.get('avg_win_loss_unbounded', False))}\n"
    )
    report += (
        f"● Closed Trades: {analytics['closed_trade_count']} | "
        f"Net Realized PnL: NT${analytics['net_realized_pnl_twd']:,.0f} | "
        f"Expectancy: NT${analytics['expectancy_twd']:,.0f}/筆\n"
    )
    if analytics.get("annual_return") is not None:
        report += f"● Closed-book Annual Return*: {_format_metric(analytics.get('annual_return'), pct=True)}\n"
    report += f"● 註記: {analytics['methodology']}"
    return report


def build_trade_followup_weekly_report(days: int = 7) -> str:
    lookback_days = max(int(days), 1)
    cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=lookback_days)
    cutoff_iso = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    with db_lock:
        conn = get_connection()
        try:
            followups = pd.read_sql(
                """
                SELECT
                    tf.id AS followup_id,
                    tf.symbol,
                    tf.status,
                    tf.prompt_state,
                    tf.user_reason,
                    tf.target_price,
                    tf.stop_price,
                    tf.skipped,
                    tf.responded_at,
                    tf.created_at,
                    tl.timestamp AS followup_timestamp,
                    tl.decision_snapshot
                FROM trade_followups tf
                JOIN trade_log tl ON tl.id = tf.trade_log_id
                WHERE tl.timestamp >= ?
                ORDER BY tf.symbol, tl.timestamp, tf.id
                """,
                conn,
                params=(cutoff_iso,),
            )
            sells = pd.read_sql(
                """
                SELECT
                    id AS sell_trade_log_id,
                    timestamp,
                    symbol,
                    settle_amount,
                    fx_rate,
                    realized_pnl
                FROM trade_log
                WHERE action = 'sell' AND timestamp >= ?
                ORDER BY symbol, timestamp, id
                """,
                conn,
                params=(cutoff_iso,),
            )
        finally:
            conn.close()

    if followups.empty:
        return format_tool_error(f"❌ 過去 {lookback_days} 天沒有 broker trade follow-up。", data_unavailable=True)

    followups["followup_timestamp"] = pd.to_datetime(followups["followup_timestamp"], utc=True, errors="coerce")
    followups = followups.dropna(subset=["followup_timestamp"]).copy()
    if followups.empty:
        return format_tool_error("❌ trade_followups 缺少可用的時間戳，無法建立週報。", data_unavailable=True)

    followups["next_followup_timestamp"] = followups.groupby("symbol")["followup_timestamp"].shift(-1)

    if sells.empty:
        sells = pd.DataFrame(columns=["sell_trade_log_id", "timestamp", "symbol", "settle_amount", "fx_rate", "realized_pnl"])
    else:
        sells["timestamp"] = pd.to_datetime(sells["timestamp"], utc=True, errors="coerce")
        for column in ("settle_amount", "fx_rate", "realized_pnl"):
            sells[column] = pd.to_numeric(sells[column], errors="coerce")
        sells["fx_rate"] = sells["fx_rate"].replace(0, np.nan).fillna(1.0)
        sells["cost_basis_twd"] = (sells["settle_amount"] * sells["fx_rate"]) - sells["realized_pnl"]
        sells = sells.dropna(subset=["timestamp", "realized_pnl", "cost_basis_twd"]).copy()
        sells = sells[sells["cost_basis_twd"] > 0].copy()

    bucket_stats: Dict[str, Dict[str, Any]] = {
        "planned": {"count": 0, "pending_count": 0, "closed_count": 0, "win_count": 0, "return_sum": 0.0, "pnl_sum": 0.0, "alpha_sum": 0.0, "alpha_count": 0},
        "unplanned": {"count": 0, "pending_count": 0, "closed_count": 0, "win_count": 0, "return_sum": 0.0, "pnl_sum": 0.0, "alpha_sum": 0.0, "alpha_count": 0},
    }

    for row in followups.itertuples(index=False):
        has_plan = (
            str(row.status or "").strip() == "resolved"
            and int(row.skipped or 0) == 0
            and any(value not in (None, "") for value in (row.user_reason, row.target_price, row.stop_price))
        )
        bucket_name = "planned" if has_plan else "unplanned"
        bucket = bucket_stats[bucket_name]
        bucket["count"] += 1
        if str(row.status or "").strip() != "resolved":
            bucket["pending_count"] += 1

        snapshot = {}
        if row.decision_snapshot:
            try:
                snapshot = json.loads(row.decision_snapshot)
            except (TypeError, json.JSONDecodeError):
                snapshot = {}
        nlp_alpha = snapshot.get("nlp_alpha")
        if isinstance(nlp_alpha, (int, float)):
            bucket["alpha_sum"] += float(nlp_alpha)
            bucket["alpha_count"] += 1

        if sells.empty:
            continue

        matched_sells = sells[
            (sells["symbol"] == row.symbol)
            & (sells["timestamp"] >= row.followup_timestamp)
            & (
                sells["timestamp"] < row.next_followup_timestamp
                if pd.notna(row.next_followup_timestamp)
                else True
            )
        ]
        if matched_sells.empty:
            continue

        realized_pnl = float(matched_sells["realized_pnl"].sum())
        cost_basis_twd = float(matched_sells["cost_basis_twd"].sum())
        if cost_basis_twd <= 0:
            continue

        bucket["closed_count"] += 1
        bucket["pnl_sum"] += realized_pnl
        bucket["return_sum"] += realized_pnl / cost_basis_twd
        if realized_pnl > 0:
            bucket["win_count"] += 1

    report = f"📝 === Trade Follow-up Weekly Review ({lookback_days}d) ===\n"
    report += f"● Follow-ups: {len(followups)} 筆\n"
    for bucket_name, label in (("planned", "有計畫"), ("unplanned", "無計畫")):
        bucket = bucket_stats[bucket_name]
        line = f"● {label}: {bucket['count']} 筆"
        if bucket["pending_count"] > 0:
            line += f" | 待補 {bucket['pending_count']} 筆"
        line += f" | 已平倉 {bucket['closed_count']} 筆"
        if bucket["closed_count"] > 0:
            win_rate = bucket["win_count"] / bucket["closed_count"]
            avg_return = bucket["return_sum"] / bucket["closed_count"]
            avg_pnl = bucket["pnl_sum"] / bucket["closed_count"]
            line += f" | 勝率 {win_rate:.1%} | 平均報酬 {avg_return:.1%} | 平均已實現 NT${avg_pnl:,.0f}"
        else:
            line += " | 勝率 N/A | 平均報酬 N/A"
        if bucket["alpha_count"] > 0:
            avg_alpha = bucket["alpha_sum"] / bucket["alpha_count"]
            line += f" | 平均偵測 Alpha {avg_alpha:+.2f}"
        report += line + "\n"

    report += "● 註記: 以 trade_followups 分 bucket，並將同 symbol 後續 sell audit 歸到最近一筆 follow-up；未平倉不納入勝率/平均報酬。"
    return report


@tool()
def get_portfolio_analytics() -> str:
    """Returns realized closed-book performance analytics built from trade_log sells."""
    return build_portfolio_analytics_report()


@tool()
def get_trade_followup_weekly_report(days: int = 7) -> str:
    """Returns the weekly planned-vs-unplanned report for broker-detected trade followups."""
    return build_trade_followup_weekly_report(days)


def compute_portfolio_beta_attribution(
    holdings: Dict[str, float],
    benchmark: str = "SPY",
    period: str = "6mo",
    series_cache: Dict[tuple[str, str], pd.Series] | None = None,
) -> Dict[str, Any]:
    clean_holdings: Dict[str, float] = {}
    for symbol, weight in holdings.items():
        if not isinstance(weight, (int, float)) or weight <= 0:
            continue
        resolved = _resolve_lookup_symbol(symbol)
        if not resolved:
            continue
        clean_holdings[resolved] = clean_holdings.get(resolved, 0.0) + float(weight)
    total_weight = sum(clean_holdings.values())
    if total_weight <= 0:
        return {"error": "無有效持倉權重可做 beta 分解。"}
    normalized_holdings = {symbol: weight / total_weight for symbol, weight in clean_holdings.items()}

    benchmark_symbol, bench_returns, bench_error = _load_daily_return_series(
        benchmark,
        period=period,
        series_cache=series_cache,
    )
    if bench_error:
        return {"error": f"{benchmark_symbol or benchmark} 無法取得基準歷史價格：{bench_error}"}
    if len(bench_returns) < MIN_BETA_OBSERVATIONS:
        return {"error": f"{benchmark_symbol} 歷史資料不足，無法穩健估 beta。"}

    positions: Dict[str, Dict[str, Any]] = {}
    skipped_positions: Dict[str, str] = {}
    portfolio_beta = 0.0
    portfolio_alpha_daily = 0.0
    coverage_weight = 0.0

    for symbol, weight in normalized_holdings.items():
        resolved_symbol, stock_returns, fetch_error = _load_daily_return_series(
            symbol,
            period=period,
            series_cache=series_cache,
        )
        symbol = resolved_symbol or symbol
        if fetch_error:
            skipped_positions[symbol] = fetch_error
            continue

        aligned = pd.concat([stock_returns, bench_returns], axis=1, join="inner").dropna()
        if len(aligned) < MIN_BETA_OBSERVATIONS:
            skipped_positions[symbol] = f"有效重疊樣本不足 ({len(aligned)})"
            continue

        stock_vals = aligned.iloc[:, 0].to_numpy(dtype=float)
        bench_vals = aligned.iloc[:, 1].to_numpy(dtype=float)
        bench_var = float(np.var(bench_vals))
        if bench_var <= 0:
            skipped_positions[symbol] = "基準波動為 0，無法回歸"
            continue

        beta = float(np.cov(stock_vals, bench_vals, ddof=0)[0, 1] / bench_var)
        alpha_daily = float(np.mean(stock_vals) - beta * np.mean(bench_vals))
        residual = stock_vals - (alpha_daily + beta * bench_vals)
        idio_vol = float(np.std(residual, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))

        positions[symbol] = {
            "weight": weight,
            "beta": beta,
            "alpha_annual": alpha_daily * TRADING_DAYS_PER_YEAR,
            "idio_vol": idio_vol,
            "risk_contribution": weight * beta,
            "observations": int(len(aligned)),
        }
        portfolio_beta += weight * beta
        portfolio_alpha_daily += weight * alpha_daily
        coverage_weight += weight

    if not positions:
        return {"error": "所有持倉都缺少足夠歷史資料，無法估 beta。"}

    return {
        "benchmark": benchmark_symbol,
        "period": period,
        "portfolio_beta": round(portfolio_beta, 4),
        "portfolio_alpha_annual": round(portfolio_alpha_daily * TRADING_DAYS_PER_YEAR, 4),
        "coverage_weight": round(coverage_weight, 4),
        "positions": positions,
        "skipped_positions": skipped_positions,
        "methodology": "用目前持倉權重對基準做單因子日報酬回歸；alpha 為年化截距，idio_vol 為殘差波動。",
    }


def build_portfolio_beta_report(benchmark: str = "SPY", period: str = "6mo") -> str:
    holdings, _ = _build_current_holdings_weights()
    if not holdings:
        return format_tool_error("❌ 無有效股票持倉可做 beta 分解。", data_unavailable=True)

    attribution = compute_portfolio_beta_attribution(holdings, benchmark=benchmark, period=period)
    if attribution.get("error"):
        return format_tool_error(f"❌ {attribution['error']}", data_unavailable=True)

    report = f"🧮 === Portfolio Beta Attribution vs {attribution['benchmark']} ===\n"
    report += (
        f"● Portfolio Beta: {attribution['portfolio_beta']:.2f} | "
        f"Annualized Alpha: {attribution['portfolio_alpha_annual']:+.1%} | "
        f"Coverage: {attribution['coverage_weight']:.1%}\n"
    )

    ranked = sorted(
        attribution["positions"].items(),
        key=lambda item: abs(item[1]["risk_contribution"]),
        reverse=True,
    )
    for symbol, payload in ranked:
        report += (
            f"● {symbol}: 權重 {payload['weight']:.1%} | β {payload['beta']:.2f} | "
            f"α_ann {payload['alpha_annual']:+.1%} | idio {payload['idio_vol']:.1%} | "
            f"風險貢獻 {payload['risk_contribution']:.3f}\n"
        )

    if attribution["skipped_positions"]:
        skipped = "; ".join(
            f"{symbol}({reason})" for symbol, reason in sorted(attribution["skipped_positions"].items())
        )
        report += f"● 跳過: {skipped}\n"

    report += f"● 註記: {attribution['methodology']}"
    return report


@tool()
def get_portfolio_beta_attribution(benchmark: str = "SPY", period: str = "6mo") -> str:
    """Decomposes current holdings into benchmark beta and residual alpha."""
    return build_portfolio_beta_report(benchmark, period)


def compute_inverse_vol_weights(symbols: List[str], lookback: int = 120, period: str = "1y") -> Dict[str, Any]:
    cleaned = _parse_symbol_input(symbols)
    if not cleaned:
        return {"error": "沒有可用標的可做 inverse-vol 配置。"}

    effective_lookback = max(int(lookback), 20)
    annualized_vols: Dict[str, float] = {}
    skipped: Dict[str, str] = {}

    for symbol in cleaned:
        resolved_symbol, returns, fetch_error = _load_daily_return_series(symbol, period=period)
        symbol = resolved_symbol or symbol
        if fetch_error:
            skipped[symbol] = fetch_error
            continue

        returns = returns.tail(effective_lookback)
        if len(returns) < min(effective_lookback, 20):
            skipped[symbol] = f"有效樣本不足 ({len(returns)})"
            continue

        vol = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if not np.isfinite(vol) or vol <= 0:
            skipped[symbol] = "波動率無法計算"
            continue
        annualized_vols[symbol] = vol

    if not annualized_vols:
        return {"error": "所有標的都缺少足夠資料，無法計算 inverse-vol 權重。"}

    inverse_vols = {symbol: 1 / vol for symbol, vol in annualized_vols.items()}
    total_inverse = sum(inverse_vols.values())
    weights = {symbol: inverse_vols[symbol] / total_inverse for symbol in annualized_vols}
    risk_budget_proxy = {symbol: weights[symbol] * annualized_vols[symbol] for symbol in annualized_vols}

    return {
        "lookback": effective_lookback,
        "weights": {symbol: round(weight, 4) for symbol, weight in weights.items()},
        "annualized_vols": {symbol: round(vol, 4) for symbol, vol in annualized_vols.items()},
        "risk_budget_proxy": {symbol: round(risk_budget_proxy[symbol], 4) for symbol in annualized_vols},
        "skipped": skipped,
        "methodology": "以近 N 日年化波動率做 inverse-vol weighting；這是簡化版 risk parity proxy，未納入協方差矩陣最佳化。",
    }


def build_risk_parity_report(symbols: str = "", lookback: int = 120, period: str = "1y") -> str:
    source_symbols = _parse_symbol_input(symbols)
    source_label = "指定標的"
    if not source_symbols:
        holdings, _ = _build_current_holdings_weights()
        source_symbols = list(holdings.keys())
        source_label = "目前持倉"
    if not source_symbols:
        return format_tool_error("❌ 無有效持倉或輸入標的可做配置。", data_unavailable=True)

    payload = compute_inverse_vol_weights(source_symbols, lookback=lookback, period=period)
    if payload.get("error"):
        return format_tool_error(f"❌ {payload['error']}", data_unavailable=True)

    ranked = sorted(payload["weights"], key=lambda symbol: payload["weights"][symbol], reverse=True)
    report = "⚖️ === Inverse-Vol Risk Parity Proxy ===\n"
    report += f"● 來源: {source_label} | Lookback: {payload['lookback']} 日\n"
    for symbol in ranked:
        report += (
            f"● {symbol}: 權重 {payload['weights'][symbol] * 100:.1f}% | "
            f"Ann Vol {payload['annualized_vols'][symbol] * 100:.1f}% | "
            f"Risk Budget Proxy {payload['risk_budget_proxy'][symbol]:.4f}\n"
        )

    if payload["skipped"]:
        skipped = "; ".join(f"{symbol}({reason})" for symbol, reason in sorted(payload["skipped"].items()))
        report += f"● 跳過: {skipped}\n"

    report += f"● 註記: {payload['methodology']}"
    return report


@tool()
def get_risk_parity_weights(symbols: str = "", lookback: int = 120, period: str = "1y") -> str:
    """Suggests simplified inverse-vol weights for supplied symbols or the current holdings."""
    return build_risk_parity_report(symbols, lookback, period)


def record_portfolio_nav_snapshot(source: str = "system", snapshots: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    snapshots = snapshots if snapshots is not None else _build_live_position_snapshots(_load_portfolio_rows())
    if not snapshots:
        return {"error": "無有效持倉可記錄 NAV。"}

    totals = _summarize_live_snapshots(snapshots)
    timestamp = _utc_now_iso()
    with db_lock:
        conn = get_connection()
        try:
            conn.execute(
                """
                INSERT INTO portfolio_nav_history (
                    timestamp, nav_twd, total_cost_twd, gross_exposure_twd, cash_twd, pnl_pct, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    totals["total_nav_twd"],
                    totals["total_cost_twd"],
                    totals["gross_exposure_twd"],
                    totals["cash_twd"],
                    totals["total_pnl_pct"],
                    source,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    return {
        "timestamp": timestamp,
        "source": source,
        "nav_twd": round(totals["total_nav_twd"], 2),
        "gross_exposure_twd": round(totals["gross_exposure_twd"], 2),
        "cash_twd": round(totals["cash_twd"], 2),
        "pnl_pct": round(totals["total_pnl_pct"], 4),
    }


def _load_portfolio_nav_history(max_days: int = 400) -> pd.DataFrame:
    with db_lock:
        conn = get_connection()
        try:
            df = pd.read_sql(
                """
                SELECT timestamp, nav_twd, gross_exposure_twd, cash_twd
                FROM portfolio_nav_history
                ORDER BY timestamp
                """,
                conn,
            )
        finally:
            conn.close()

    if df.empty:
        return pd.DataFrame(columns=["nav_twd", "gross_exposure_twd", "cash_twd"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for column in ("nav_twd", "gross_exposure_twd", "cash_twd"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["timestamp", "nav_twd"]).sort_values("timestamp")
    if df.empty:
        return pd.DataFrame(columns=["nav_twd", "gross_exposure_twd", "cash_twd"])

    df["trade_day"] = df["timestamp"].dt.tz_convert(None).dt.normalize()
    daily = (
        df.groupby("trade_day", as_index=True)
        .agg(
            nav_twd=("nav_twd", "last"),
            gross_exposure_twd=("gross_exposure_twd", "last"),
            cash_twd=("cash_twd", "last"),
        )
        .sort_index()
    )
    if max_days > 0:
        daily = daily.tail(max_days)
    return daily


def _ensure_current_nav_point(nav_history: pd.DataFrame, totals: Dict[str, float]) -> pd.DataFrame:
    current_day = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
    current_row = pd.DataFrame(
        {
            "nav_twd": [totals["total_nav_twd"]],
            "gross_exposure_twd": [totals["gross_exposure_twd"]],
            "cash_twd": [totals["cash_twd"]],
        },
        index=[current_day],
    )
    if nav_history.empty:
        return current_row
    merged = nav_history.copy()
    merged.loc[current_day, ["nav_twd", "gross_exposure_twd", "cash_twd"]] = [
        totals["total_nav_twd"],
        totals["gross_exposure_twd"],
        totals["cash_twd"],
    ]
    return merged.sort_index()


def _compute_current_drawdown(nav_series: pd.Series) -> float | None:
    if nav_series.empty:
        return None
    running_peak = nav_series.cummax()
    latest_peak = float(running_peak.iloc[-1])
    latest_nav = float(nav_series.iloc[-1])
    if latest_peak <= 0:
        return None
    return max(0.0, 1.0 - (latest_nav / latest_peak))


def _compute_max_drawdown(nav_series: pd.Series) -> float | None:
    if nav_series.empty:
        return None
    running_peak = nav_series.cummax()
    drawdown = (nav_series / running_peak) - 1.0
    if drawdown.empty:
        return None
    return abs(float(drawdown.min()))


def _compute_window_drawdown(nav_series: pd.Series, window: int) -> float | None:
    return _compute_current_drawdown(nav_series.tail(max(int(window), 1)))


def _compute_month_to_date_drawdown(nav_series: pd.Series) -> float | None:
    if nav_series.empty:
        return None
    current_day = nav_series.index[-1]
    month_series = nav_series[(nav_series.index.year == current_day.year) & (nav_series.index.month == current_day.month)]
    return _compute_current_drawdown(month_series)


def _classify_drawdown_governor(current_drawdown: float | None, drawdown_20d: float | None, drawdown_mtd: float | None) -> Dict[str, Any]:
    candidates = [value for value in (current_drawdown, drawdown_20d, drawdown_mtd) if value is not None]
    active_drawdown = max(candidates) if candidates else 0.0

    if active_drawdown >= DRAWDOWN_KILL_SWITCH_THRESHOLD:
        return {
            "active_drawdown": active_drawdown,
            "trade_mode": "kill_switch",
            "trade_mode_label": "💀 Kill Switch",
            "size_multiplier": 0.0,
            "allow_new_longs": False,
            "allow_average_down": False,
            "message": "回撤超過 10%，停止新增淨多單，只允許減倉/對沖。",
        }
    if active_drawdown >= DRAWDOWN_DEFENSIVE_THRESHOLD:
        return {
            "active_drawdown": active_drawdown,
            "trade_mode": "defensive",
            "trade_mode_label": "🔴 Defense Only",
            "size_multiplier": 0.25,
            "allow_new_longs": False,
            "allow_average_down": False,
            "message": "回撤進入 8%+ 防守區，只保留防禦與風險下降操作。",
        }
    if active_drawdown >= DRAWDOWN_HARD_THRESHOLD:
        return {
            "active_drawdown": active_drawdown,
            "trade_mode": "risk_off",
            "trade_mode_label": "🟠 Risk-Off",
            "size_multiplier": 0.50,
            "allow_new_longs": True,
            "allow_average_down": False,
            "message": "回撤超過 5%，新單砍半且禁止攤平虧損部位。",
        }
    if active_drawdown >= DRAWDOWN_SOFT_THRESHOLD:
        return {
            "active_drawdown": active_drawdown,
            "trade_mode": "soft_throttle",
            "trade_mode_label": "🟡 Soft Throttle",
            "size_multiplier": 0.70,
            "allow_new_longs": True,
            "allow_average_down": False,
            "message": "回撤超過 3%，新倉位縮到 0.7x。",
        }
    return {
        "active_drawdown": active_drawdown,
        "trade_mode": "normal",
        "trade_mode_label": "🟢 Normal",
        "size_multiplier": 1.0,
        "allow_new_longs": True,
        "allow_average_down": True,
        "message": "回撤仍在可接受區間。",
    }


def _get_risk_overlay_target(state: str | None) -> Dict[str, Any]:
    state_text = state or "🟡"
    for key, payload in RISK_OVERLAY_TARGETS.items():
        if state_text.startswith(key):
            return payload
    return RISK_OVERLAY_TARGETS["🟡"]


def _select_volatility_proxy_holdings(holdings: Dict[str, float]) -> tuple[Dict[str, float], Dict[str, str]]:
    total_weight = sum(float(weight) for weight in holdings.values() if isinstance(weight, (int, float)) and weight > 0)
    if total_weight <= 0:
        return {}, {}

    ranked = sorted(
        ((symbol, float(weight) / total_weight) for symbol, weight in holdings.items() if isinstance(weight, (int, float)) and weight > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    selected: Dict[str, float] = {}
    skipped: Dict[str, str] = {}
    covered_weight = 0.0

    for index, (symbol, weight) in enumerate(ranked):
        within_soft_cap = index < PORTFOLIO_VOL_SOFT_SYMBOL_CAP
        needs_more_coverage = covered_weight < PORTFOLIO_VOL_MIN_WEIGHT_COVERAGE and index < PORTFOLIO_VOL_HARD_SYMBOL_CAP
        if within_soft_cap or needs_more_coverage:
            selected[symbol] = weight
            covered_weight += weight
            continue
        skipped[symbol] = "低權重，未納入快速波動估算"
    return selected, skipped


def _estimate_portfolio_volatility(
    holdings: Dict[str, float],
    invested_ratio: float,
    period: str = "6mo",
    lookback: int = 60,
    series_cache: Dict[tuple[str, str], pd.Series] | None = None,
) -> Dict[str, Any]:
    clean_holdings: Dict[str, float] = {}
    for symbol, weight in holdings.items():
        if not isinstance(weight, (int, float)) or weight <= 0:
            continue
        resolved = _resolve_lookup_symbol(symbol)
        if not resolved:
            continue
        clean_holdings[resolved] = clean_holdings.get(resolved, 0.0) + float(weight)
    if not clean_holdings:
        return {"error": "無有效持倉可估波動。"}

    series_map: Dict[str, pd.Series] = {}
    weight_map: Dict[str, float] = {}
    selected_holdings, skipped = _select_volatility_proxy_holdings(clean_holdings)
    if not selected_holdings:
        return {"error": "無有效持倉可估波動。"}

    coverage_weight = 0.0

    for symbol, weight in selected_holdings.items():
        resolved_symbol, returns, fetch_error = _load_daily_return_series(
            symbol,
            period=period,
            series_cache=series_cache,
        )
        symbol = resolved_symbol or symbol
        if fetch_error:
            skipped[symbol] = fetch_error
            continue
        if len(returns) < max(20, lookback // 2):
            skipped[symbol] = f"有效樣本不足 ({len(returns)})"
            continue
        series_map[symbol] = returns.tail(max(lookback, 20))
        weight_map[symbol] = weight
        coverage_weight += weight

    if not series_map:
        return {"error": "所有持倉都缺少足夠樣本，無法估波動。", "skipped": skipped}

    aligned = pd.concat(series_map, axis=1, join="inner").dropna()
    if aligned.empty or len(aligned) < max(20, lookback // 2):
        return {"error": "持倉報酬率重疊樣本不足，無法估波動。", "skipped": skipped}

    ordered_symbols = list(aligned.columns)
    weights = np.array([weight_map[symbol] for symbol in ordered_symbols], dtype=float)
    weights = weights / weights.sum()

    invested_returns = aligned.to_numpy(dtype=float) @ weights
    nav_returns = invested_returns * max(invested_ratio, 0.0)
    invested_vol = float(np.std(invested_returns, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    nav_vol = float(np.std(nav_returns, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))

    return {
        "invested_vol_annual": invested_vol,
        "nav_vol_annual": nav_vol,
        "observations": int(len(aligned)),
        "coverage_weight": round(float(coverage_weight), 4),
        "selected_symbol_count": len(selected_holdings),
        "requested_symbol_count": len(clean_holdings),
        "skipped": skipped,
    }


def compute_portfolio_risk_overlay(
    benchmark: str = "SPY",
    period: str = "6mo",
    snapshots: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    snapshots = snapshots if snapshots is not None else _build_live_position_snapshots(_load_portfolio_rows())
    if not snapshots:
        return {"error": "無有效持倉可計算風險節流。"}

    totals = _summarize_live_snapshots(snapshots)
    nav_history = _ensure_current_nav_point(_load_portfolio_nav_history(), totals)
    nav_series = pd.to_numeric(nav_history["nav_twd"], errors="coerce").dropna()
    if nav_series.empty:
        return {"error": "NAV 歷史資料不足。"}

    current_drawdown = _compute_current_drawdown(nav_series)
    max_drawdown = _compute_max_drawdown(nav_series)
    drawdown_20d = _compute_window_drawdown(nav_series, 20)
    drawdown_mtd = _compute_month_to_date_drawdown(nav_series)
    governor = _classify_drawdown_governor(current_drawdown, drawdown_20d, drawdown_mtd)
    current_drawdown = float(current_drawdown or 0.0)
    max_drawdown = float(max_drawdown or 0.0)
    drawdown_20d = float(drawdown_20d or 0.0)
    drawdown_mtd = float(drawdown_mtd or 0.0)

    holdings, _ = _build_holdings_weights_from_snapshots(snapshots)
    invested_ratio = totals["gross_exposure_ratio"]
    series_cache: Dict[tuple[str, str], pd.Series] = {}
    beta_payload = (
        compute_portfolio_beta_attribution(holdings, benchmark=benchmark, period=period, series_cache=series_cache)
        if holdings
        else {"error": "無股票持倉"}
    )
    book_beta = None if beta_payload.get("error") else float(beta_payload["portfolio_beta"])
    beta_to_nav = (book_beta * invested_ratio) if book_beta is not None else None

    vol_payload = (
        _estimate_portfolio_volatility(holdings, invested_ratio=invested_ratio, period=period, series_cache=series_cache)
        if holdings
        else {"error": "無股票持倉"}
    )
    nav_vol = None if vol_payload.get("error") else float(vol_payload["nav_vol_annual"])

    try:
        import engine_risk as risk_engine
        risk_snapshot = risk_engine.get_global_risk_snapshot()
        risk_state = risk_snapshot.get("state", "🟡 整理")
    except Exception as exc:
        logger.debug(f"Portfolio risk overlay could not load risk radar: {exc}")
        risk_state = "🟡 整理"

    target = _get_risk_overlay_target(risk_state)
    target_beta_low, target_beta_high = target["beta_band"]
    target_beta_mid = (target_beta_low + target_beta_high) / 2
    target_vol = target["target_vol"]
    active_limits = _get_trade_governor_limits(governor["trade_mode"])

    beta_scale = 1.0
    if beta_to_nav is not None and target_beta_high > 0 and beta_to_nav > target_beta_high:
        beta_scale = max(0.0, min(1.0, target_beta_high / beta_to_nav))

    vol_scale = 1.0
    if nav_vol is not None and target_vol > 0 and nav_vol > target_vol:
        vol_scale = max(0.0, min(1.0, target_vol / nav_vol))

    recommended_gross_scale = min(1.0, governor["size_multiplier"], beta_scale, vol_scale)
    trim_notional_twd = max(0.0, totals["gross_exposure_twd"] * (1.0 - recommended_gross_scale))
    hedge_notional_twd = (
        max(0.0, totals["total_nav_twd"] * (beta_to_nav - target_beta_mid))
        if beta_to_nav is not None and beta_to_nav > target_beta_mid
        else 0.0
    )

    constraints = []
    if governor["size_multiplier"] < 1.0:
        constraints.append(f"drawdown governor {governor['trade_mode_label']}")
    if beta_scale < 1.0:
        constraints.append(f"beta 高於目標上限 {target_beta_high:.2f}")
    if vol_scale < 1.0:
        constraints.append(f"波動高於目標 {target_vol * 100:.1f}%")
    primary_constraint = constraints[0] if constraints else "目前無強制降風險"

    summary = (
        f"{governor['trade_mode_label']} | DD {current_drawdown * 100:.1f}% | "
        f"Target β {target_beta_low:.2f}-{target_beta_high:.2f} | Gross Scale {recommended_gross_scale:.2f}x"
    )

    return {
        "generated_at": _utc_now_iso(),
        "summary": summary,
        "trade_mode": governor["trade_mode"],
        "trade_mode_label": governor["trade_mode_label"],
        "size_multiplier": round(float(governor["size_multiplier"]), 4),
        "allow_new_longs": governor["allow_new_longs"],
        "allow_average_down": governor["allow_average_down"],
        "governor_message": governor["message"],
        "current_drawdown": round(current_drawdown, 4),
        "drawdown_20d": round(drawdown_20d, 4),
        "drawdown_mtd": round(drawdown_mtd, 4),
        "max_drawdown_since_inception": round(max_drawdown, 4),
        "current_nav_twd": round(totals["total_nav_twd"], 2),
        "gross_exposure_twd": round(totals["gross_exposure_twd"], 2),
        "cash_twd": round(totals["cash_twd"], 2),
        "gross_exposure_ratio": round(totals["gross_exposure_ratio"], 4),
        "risk_state": risk_state,
        "risk_overlay_label": target["label"],
        "target_beta_band": [round(target_beta_low, 2), round(target_beta_high, 2)],
        "target_beta_mid": round(target_beta_mid, 2),
        "target_vol_annual": round(target_vol, 4),
        "current_book_beta": round(book_beta, 4) if book_beta is not None else None,
        "current_beta_to_nav": round(beta_to_nav, 4) if beta_to_nav is not None else None,
        "current_nav_vol_annual": round(nav_vol, 4) if nav_vol is not None else None,
        "beta_scale": round(float(beta_scale), 4),
        "vol_scale": round(float(vol_scale), 4),
        "recommended_gross_scale": round(float(recommended_gross_scale), 4),
        "trim_notional_twd": round(trim_notional_twd, 2),
        "hedge_notional_twd": round(hedge_notional_twd, 2),
        "primary_constraint": primary_constraint,
        "constraints": constraints,
        "base_single_name_cap": round(float(active_limits["single_name_cap"]), 4),
        "base_sector_cap": round(float(active_limits["sector_cap"]), 4),
        "concentration_methodology": "股票以 sector 計，ETF 以 tracking index / category / 名稱推斷 concentration bucket。",
        "beta_methodology": beta_payload.get("methodology") if not beta_payload.get("error") else beta_payload.get("error"),
        "vol_methodology": (
            "以代表性持倉權重重建近 60 日組合報酬；優先納入前 5 大權重，"
            "若覆蓋不足 85% 最多擴到前 8 檔，並重用同輪 beta 查詢的日報酬快取。"
        ),
    }


def build_portfolio_risk_overlay_report(benchmark: str = "SPY", period: str = "6mo") -> str:
    overlay = compute_portfolio_risk_overlay(benchmark=benchmark, period=period)
    if overlay.get("error"):
        return format_tool_error(f"❌ {overlay['error']}", data_unavailable=True)

    beta_to_nav = overlay.get("current_beta_to_nav")
    beta_text = f"{beta_to_nav:.2f}" if isinstance(beta_to_nav, (int, float)) else "N/A"
    nav_vol = overlay.get("current_nav_vol_annual")
    nav_vol_text = f"{nav_vol * 100:.1f}%" if isinstance(nav_vol, (int, float)) else "N/A"
    target_vol = overlay.get("target_vol_annual")
    target_vol_text = f"{target_vol * 100:.1f}%" if isinstance(target_vol, (int, float)) else "N/A"
    beta_band = overlay.get("target_beta_band", ["N/A", "N/A"])

    report = "🛡️ === Portfolio Risk Overlay ===\n"
    report += (
        f"● NAV: NT${overlay['current_nav_twd']:,.0f} | Gross Exposure: NT${overlay['gross_exposure_twd']:,.0f} "
        f"({overlay['gross_exposure_ratio'] * 100:.1f}%) | Cash: NT${overlay['cash_twd']:,.0f}\n"
    )
    report += (
        f"● Drawdown: 現在 {overlay['current_drawdown'] * 100:.1f}% | "
        f"20D {overlay['drawdown_20d'] * 100:.1f}% | MTD {overlay['drawdown_mtd'] * 100:.1f}% | "
        f"{overlay['trade_mode_label']}\n"
    )
    report += (
        f"● Beta/Vol Overlay: β(NAV) {beta_text} vs 目標 {beta_band[0]:.2f}-{beta_band[1]:.2f} | "
        f"Vol(NAV) {nav_vol_text} vs 目標 {target_vol_text}\n"
    )
    report += (
        f"● 交易上限: 單一持股 {overlay['base_single_name_cap'] * 100:.1f}% NAV | "
        f"集中桶 {overlay['base_sector_cap'] * 100:.1f}% NAV\n"
    )
    report += (
        f"● 建議: Gross Scale {overlay['recommended_gross_scale']:.2f}x | "
        f"Raise Cash ~NT${overlay['trim_notional_twd']:,.0f}"
    )
    if overlay.get("hedge_notional_twd", 0) > 0:
        report += f" | Benchmark Hedge ~NT${overlay['hedge_notional_twd']:,.0f}"
    report += "\n"
    report += f"● 約束: {overlay['primary_constraint']}\n"
    report += f"● Governor: {overlay['governor_message']}\n"
    report += f"● 集中度方法: {overlay['concentration_methodology']}"
    return report


@tool()
def get_portfolio_risk_overlay(benchmark: str = "SPY", period: str = "6mo") -> str:
    """Returns drawdown governor plus dynamic beta/vol overlay guidance for the current portfolio."""
    return build_portfolio_risk_overlay_report(benchmark, period)


def _build_current_nav_weight_map(snapshots: List[Dict[str, Any]]) -> Dict[str, float]:
    totals = _summarize_live_snapshots(snapshots)
    nav_twd = totals["total_nav_twd"]
    if nav_twd <= 0:
        return {}

    weights: Dict[str, float] = {}
    for pos in snapshots:
        if pos["is_cash"] or pos["market_value_twd"] <= 0:
            continue
        planning_symbol = _normalize_position_symbol_for_planning(str(pos.get("symbol") or ""))
        if not planning_symbol:
            continue
        weights[planning_symbol] = weights.get(planning_symbol, 0.0) + (float(pos["market_value_twd"]) / nav_twd)
    return weights


def _split_position_suffix(symbol: str) -> tuple[str, str]:
    normalized = normalize_ticker(str(symbol or ""))
    for suffix in ("_TRUST", "_ESOP"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)], suffix
    return normalized, ""


def _lookup_symbol_for_signal(symbol: str) -> str:
    base_symbol, _ = _split_position_suffix(symbol)
    if base_symbol.startswith("CASH"):
        return base_symbol
    return _resolve_lookup_symbol(base_symbol)


def _normalize_position_symbol_for_planning(symbol: str) -> str:
    base_symbol, suffix = _split_position_suffix(symbol)
    if base_symbol.startswith("CASH"):
        return base_symbol
    resolved = _resolve_lookup_symbol(base_symbol)
    return f"{resolved}{suffix}" if resolved else ""


def _build_rebalance_universe(symbols: str | List[str] | None, snapshots: List[Dict[str, Any]]) -> List[str]:
    universe = _parse_symbol_input(symbols)
    for symbol in WATCH_LIST:
        resolved = _resolve_lookup_symbol(symbol)
        if resolved and resolved not in universe:
            universe.append(resolved)
    for snapshot in snapshots:
        if snapshot.get("is_cash"):
            continue
        resolved = _lookup_symbol_for_signal(str(snapshot.get("symbol") or ""))
        if resolved and resolved not in universe:
            universe.append(resolved)
    return universe


def _is_accumulation_only_symbol(symbol: str) -> bool:
    normalized = normalize_ticker(str(symbol or ""))
    return normalized.endswith("_TRUST") or normalized.endswith("_ESOP")


def _normalize_rebalance_snapshots(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for snapshot in snapshots:
        market_value_twd = float(snapshot.get("market_value_twd") or 0.0)
        normalized.append(
            {
                **snapshot,
                "symbol": snapshot.get("symbol"),
                "is_cash": bool(snapshot.get("is_cash")),
                "market_value_twd": market_value_twd,
                "twd_cost": float(snapshot.get("twd_cost") or market_value_twd),
                "pnl_value_twd": float(snapshot.get("pnl_value_twd") or 0.0),
            }
        )
    return normalized


def _build_protected_position_targets(snapshots: List[Dict[str, Any]], nav_twd: float) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    protected_weights: Dict[str, float] = {}
    sector_allocations: Dict[str, float] = {}
    underlying_allocations: Dict[str, float] = {}
    if nav_twd <= 0:
        return protected_weights, sector_allocations, underlying_allocations

    for snapshot in snapshots:
        symbol = str(snapshot.get("symbol") or "")
        if snapshot.get("is_cash") or not _is_accumulation_only_symbol(symbol):
            continue
        planning_symbol = _normalize_position_symbol_for_planning(symbol)
        underlying_symbol = _lookup_symbol_for_signal(symbol)
        if not planning_symbol or not underlying_symbol:
            continue
        weight = float(snapshot.get("market_value_twd") or 0.0) / nav_twd
        protected_weights[planning_symbol] = protected_weights.get(planning_symbol, 0.0) + weight
        underlying_allocations[underlying_symbol] = underlying_allocations.get(underlying_symbol, 0.0) + weight
        try:
            sector = str(market.get_asset_profile(underlying_symbol).get("sector") or "Unknown")
        except Exception as exc:
            logger.debug(f"Protected position sector lookup failed for {underlying_symbol}: {exc}")
            sector = "Unknown"
        if sector != "Unknown":
            sector_allocations[sector] = sector_allocations.get(sector, 0.0) + weight
    return protected_weights, sector_allocations, underlying_allocations


def compute_portfolio_rebalance_plan(
    symbols: str | List[str] | None = None,
    benchmark: str = "SPY",
    period: str = "6mo",
    candidate_panel: Dict[str, Any] | None = None,
    snapshots: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    snapshots = snapshots if snapshots is not None else _build_live_position_snapshots(_load_portfolio_rows())
    if not snapshots:
        return {"error": "無有效持倉與現金，無法建立 rebalance plan。"}
    snapshots = _normalize_rebalance_snapshots(snapshots)

    overlay = compute_portfolio_risk_overlay(benchmark=benchmark, period=period, snapshots=snapshots)
    if overlay.get("error"):
        return {"error": overlay["error"]}

    totals = _summarize_live_snapshots(snapshots)
    nav_twd = totals["total_nav_twd"]
    if nav_twd <= 0:
        return {"error": "NAV 無效，無法建立 rebalance plan。"}

    current_weight_map = _build_current_nav_weight_map(snapshots)
    protected_weights, sector_allocations, underlying_allocations = _build_protected_position_targets(snapshots, nav_twd)
    universe = _build_rebalance_universe(symbols, snapshots)
    if not universe:
        return {"error": "沒有可用股票池可產生 rebalance plan。"}

    candidate_panel = candidate_panel or market.compute_candidate_alpha_panel(
        universe,
        benchmark=benchmark,
        period=period,
    )
    if candidate_panel.get("error"):
        return {"error": candidate_panel["error"]}
    panel_rows = candidate_panel.get("rows") or []
    if not panel_rows:
        return {"error": "candidate alpha panel 為空，無法配倉。"}

    trade_mode = str(overlay.get("trade_mode") or "normal")
    target_gross_ratio = float(overlay.get("recommended_gross_scale") or 0.0)
    current_gross_ratio = float(overlay.get("gross_exposure_ratio") or 0.0)

    ranked_rows = sorted(
        panel_rows,
        key=lambda row: (
            float(row.get("expected_return_bps") or 0.0) * float(row.get("forecast_confidence") or 0.0),
            float(row.get("final_alpha_score") or 0.0),
        ),
        reverse=True,
    )
    eligible_rows = [
        row for row in ranked_rows
        if float(row.get("expected_return_bps") or 0.0) > 0
        and float(row.get("forecast_confidence") or 0.0) >= 0.35
    ]
    total_positive_score = sum(
        float(row.get("expected_return_bps") or 0.0) * float(row.get("forecast_confidence") or 0.0)
        for row in eligible_rows
    )

    target_weights: Dict[str, float] = dict(protected_weights)
    blocked_by_risk: List[Dict[str, Any]] = []
    protected_gross_ratio = sum(protected_weights.values())
    remaining_gross = max(0.0, target_gross_ratio - protected_gross_ratio)
    if protected_gross_ratio > target_gross_ratio:
        blocked_by_risk.append(
            {
                "symbol": "ACCUMULATION_ONLY",
                "reason": (
                    f"non-sellable accumulation holdings already occupy {protected_gross_ratio:.2f} gross, "
                    f"above target gross {target_gross_ratio:.2f}"
                ),
            }
        )

    for row in eligible_rows:
        symbol = row["symbol"]
        asset_type = row.get("asset_type")
        limits = _get_trade_governor_limits(trade_mode, asset_type)
        single_name_cap = limits["single_name_cap"]
        sector_cap = limits["sector_cap"]
        sector = str(row.get("sector") or "Unknown")
        underlying_exposure = underlying_allocations.get(symbol, 0.0)
        score = float(row.get("expected_return_bps") or 0.0) * float(row.get("forecast_confidence") or 0.0)
        base_weight = (target_gross_ratio * score / total_positive_score) if total_positive_score > 0 else 0.0
        sector_headroom = remaining_gross if sector == "Unknown" else max(0.0, sector_cap - sector_allocations.get(sector, 0.0))
        single_name_headroom = max(0.0, single_name_cap - underlying_exposure)
        proposed_weight = min(base_weight, single_name_headroom, sector_headroom, remaining_gross)

        if proposed_weight <= 0:
            blocked_by_risk.append(
                {
                    "symbol": symbol,
                    "reason": (
                        f"sector cap {sector_cap:.2f}" if sector_headroom <= 0 and sector != "Unknown"
                        else f"single-name cap {single_name_cap:.2f}" if single_name_headroom <= 0
                        else "gross budget exhausted"
                    ),
                }
            )
            continue

        target_weights[symbol] = proposed_weight
        remaining_gross -= proposed_weight
        underlying_allocations[symbol] = underlying_allocations.get(symbol, 0.0) + proposed_weight
        if sector != "Unknown":
            sector_allocations[sector] = sector_allocations.get(sector, 0.0) + proposed_weight

    if remaining_gross > 0 and target_weights:
        for _ in range(2):
            expandable = []
            for row in eligible_rows:
                symbol = row["symbol"]
                if symbol not in target_weights:
                    continue
                asset_type = row.get("asset_type")
                limits = _get_trade_governor_limits(trade_mode, asset_type)
                sector = str(row.get("sector") or "Unknown")
                single_name_headroom = max(0.0, limits["single_name_cap"] - underlying_allocations.get(symbol, 0.0))
                sector_headroom = remaining_gross if sector == "Unknown" else max(0.0, limits["sector_cap"] - sector_allocations.get(sector, 0.0))
                headroom = min(single_name_headroom, sector_headroom, remaining_gross)
                if headroom <= 0:
                    continue
                score = float(row.get("expected_return_bps") or 0.0) * float(row.get("forecast_confidence") or 0.0)
                expandable.append((row, score, headroom))
            if not expandable:
                break
            score_sum = sum(item[1] for item in expandable)
            if score_sum <= 0:
                break
            for row, score, headroom in expandable:
                symbol = row["symbol"]
                sector = str(row.get("sector") or "Unknown")
                add_weight = min(remaining_gross * (score / score_sum), headroom)
                if add_weight <= 0:
                    continue
                target_weights[symbol] += add_weight
                remaining_gross -= add_weight
                underlying_allocations[symbol] = underlying_allocations.get(symbol, 0.0) + add_weight
                if sector != "Unknown":
                    sector_allocations[sector] = sector_allocations.get(sector, 0.0) + add_weight
                if remaining_gross <= 1e-4:
                    break
            if remaining_gross <= 1e-4:
                break

    recommendations = []
    all_symbols = sorted(set(current_weight_map) | set(target_weights))
    panel_lookup = {row["symbol"]: row for row in panel_rows}

    for symbol in all_symbols:
        current_weight = float(current_weight_map.get(symbol, 0.0))
        target_weight = float(target_weights.get(symbol, 0.0))

        delta_weight = target_weight - current_weight
        delta_notional_twd = delta_weight * nav_twd
        panel_row = panel_lookup.get(symbol, {})

        if delta_weight > 0.01:
            action = "buy"
        elif delta_weight < -0.01:
            action = "trim" if target_weight > 0.005 else "exit"
        else:
            action = "hold"

        recommendations.append(
            {
                "symbol": symbol,
                "sector": panel_row.get("sector", "Unknown"),
                "asset_type": panel_row.get("asset_type", "Unknown"),
                "current_weight": round(current_weight, 4),
                "target_weight": round(target_weight, 4),
                "delta_weight": round(delta_weight, 4),
                "delta_notional_twd": round(delta_notional_twd, 2),
                "expected_return_bps": panel_row.get("expected_return_bps"),
                "forecast_confidence": panel_row.get("forecast_confidence"),
                "final_alpha_score": panel_row.get("final_alpha_score"),
                "action": action,
            }
        )

    recommendations.sort(
        key=lambda row: (
            0 if row["action"] == "buy" else 1 if row["action"] == "trim" else 2,
            abs(float(row["delta_notional_twd"])),
        ),
        reverse=False,
    )

    summary = (
        f"{overlay.get('trade_mode_label', 'N/A')} | Current Gross {current_gross_ratio:.2f}x -> "
        f"Target Gross {target_gross_ratio:.2f}x | 候選池 {len(panel_rows)} 檔"
    )
    return {
        "generated_at": _utc_now_iso(),
        "summary": summary,
        "benchmark": benchmark,
        "trade_mode": overlay.get("trade_mode"),
        "trade_mode_label": overlay.get("trade_mode_label"),
        "target_gross_ratio": round(target_gross_ratio, 4),
        "current_gross_ratio": round(current_gross_ratio, 4),
        "protected_gross_ratio": round(protected_gross_ratio, 4),
        "allocated_gross_ratio": round(sum(target_weights.values()), 4),
        "cash_buffer_ratio": round(max(0.0, 1.0 - sum(target_weights.values())), 4),
        "current_nav_twd": round(nav_twd, 2),
        "overlay_constraint": overlay.get("primary_constraint"),
        "candidate_panel_generated_at": candidate_panel.get("generated_at"),
        "target_weights": {symbol: round(weight, 4) for symbol, weight in sorted(target_weights.items())},
        "sector_allocations": {sector: round(weight, 4) for sector, weight in sorted(sector_allocations.items())},
        "blocked_by_risk": blocked_by_risk,
        "recommendations": recommendations,
        "methodology": (
            "先用 candidate alpha panel 的 expected return × confidence 排序，再在現有 overlay 的 gross / single-name / sector 約束下做 explainable greedy sizing；"
            "輸出 target weights 與 rebalance proposal，不會自動下單。"
        ),
    }


def build_portfolio_rebalance_report(
    symbols: str = "",
    benchmark: str = "SPY",
    period: str = "6mo",
) -> str:
    payload = compute_portfolio_rebalance_plan(symbols=symbols, benchmark=benchmark, period=period)
    if payload.get("error"):
        return format_tool_error(f"❌ {payload['error']}", data_unavailable=True)

    report = "🧱 === Portfolio Rebalance Proposal ===\n"
    report += (
        f"● {payload['trade_mode_label']} | Current Gross {payload['current_gross_ratio']:.2f}x -> "
        f"Target {payload['target_gross_ratio']:.2f}x | Allocated {payload['allocated_gross_ratio']:.2f}x\n"
    )
    report += f"● 約束: {payload.get('overlay_constraint', 'N/A')}\n"

    for row in payload["recommendations"][:10]:
        conf = row.get("forecast_confidence")
        conf_text = f"{conf * 100:.0f}%" if isinstance(conf, (int, float)) else "N/A"
        report += (
            f"● {row['action'].upper()} {row['symbol']}: "
            f"{row['current_weight'] * 100:.1f}% -> {row['target_weight'] * 100:.1f}% "
            f"(Δ {row['delta_notional_twd']:+,.0f} TWD | ER {row.get('expected_return_bps', 'N/A')}bps | Conf {conf_text})\n"
        )

    if payload["blocked_by_risk"]:
        blocked = "; ".join(f"{item['symbol']}({item['reason']})" for item in payload["blocked_by_risk"][:5])
        report += f"● 受限: {blocked}\n"
    report += f"● 註記: {payload['methodology']}"
    return report


@tool()
def get_portfolio_rebalance_plan(symbols: str = "", benchmark: str = "SPY", period: str = "6mo") -> str:
    """Builds target weights and rebalance proposals from the candidate alpha panel without auto-executing trades."""
    return build_portfolio_rebalance_report(symbols, benchmark, period)


def calculate_position_pnl(symbol: str, current_price: float, shares: float, historical_twd_cost: float, is_us_stock: bool) -> dict:
    """Pure PnL logic for direct callers and tests."""
    # 識別市場
    is_uk_stock = symbol.endswith('.L') or symbol.endswith('.IL')
    is_foreign = is_us_stock or is_uk_stock or symbol == 'CASH_USD'
    
    # 取得匯率
    raw_fx = fetch_exchange_rate() if is_foreign else 1.0
    # 海外資產換回台幣需扣除約 0.2% 換匯手續費與價差
    settle_fx = raw_fx * 0.998 if is_foreign else 1.0

    # 特殊處理：現金池
    if symbol == 'CASH_TWD':
        return {"market_value_twd": round(shares, 2), "pnl_value_twd": 0, "pnl_percent": 0}
    if symbol == 'CASH_USD':
        cur_val = shares * settle_fx
        pnl = cur_val - historical_twd_cost
        return {"market_value_twd": round(cur_val, 2), "pnl_value_twd": round(pnl, 2), "pnl_percent": 0}

    # 【核心計算】
    current_market_value_twd = current_price * shares * settle_fx
    pnl_value_twd = current_market_value_twd - historical_twd_cost
    
    # 百分比防呆：若成本為 0，避免除以零或噴出天文數字
    if historical_twd_cost > 0:
        pnl_percent = (pnl_value_twd / historical_twd_cost) * 100
    else:
        pnl_percent = 0.0
    
    return {
        "market_value_twd": round(current_market_value_twd, 2),
        "pnl_value_twd": round(pnl_value_twd, 2),
        "pnl_percent": round(pnl_percent, 2),
        "market": "UK" if is_uk_stock else "US" if is_us_stock else "TW"
    }


def build_position_size_report(
    symbol: str,
    risk_pct: float = 2.0,
    total_capital_twd: float = None,
    stop_atr_multiple: float = 2.0,
) -> str:
    """以 ATR 估算風險預算下的建議倉位。"""
    try:
        if risk_pct <= 0 or stop_atr_multiple <= 0:
            return format_tool_error("❌ risk_pct 與 stop_atr_multiple 必須大於 0。", data_unavailable=True)

        symbol = normalize_ticker(symbol)
        from engine_technical import IndicatorCalculator

        calc = IndicatorCalculator()
        highs = calc.HIGH(symbol, '1d')
        lows = calc.LOW(symbol, '1d')
        closes = calc.CLOSE(symbol, '1d')
        atr_series = pd.Series(calc.ATR(highs, lows, closes)).dropna()
        if atr_series.empty:
            return format_tool_error(f"❌ {symbol} 無法計算 ATR。", data_unavailable=True)

        atr = float(atr_series.iloc[-1])
        price = float(closes[-1])
        clean_symbol = symbol.replace('.TW', '').replace('.TWO', '')
        is_taiwan = (any(char.isdigit() for char in clean_symbol) and len(clean_symbol) <= 6) or symbol.endswith('.TW') or symbol.endswith('.TWO')
        fx_rate = 1.0 if is_taiwan or 'CASH' in symbol else fetch_exchange_rate()

        if total_capital_twd is None:
            total_capital_twd = float(build_portfolio_analysis().get("total_current") or 0.0)
        else:
            total_capital_twd = float(total_capital_twd)

        if total_capital_twd <= 0:
            return format_tool_error("❌ 無法取得有效總資金，請先確認 portfolio。", data_unavailable=True)

        risk_budget_twd = total_capital_twd * (risk_pct / 100.0)
        stop_distance_local = atr * stop_atr_multiple
        stop_distance_twd = stop_distance_local * fx_rate
        if stop_distance_twd <= 0 or price <= 0:
            return format_tool_error(f"❌ {symbol} 的 ATR / 價格數據異常。", data_unavailable=True)

        risk_shares = int(risk_budget_twd / stop_distance_twd)
        affordable_shares = int(total_capital_twd / (price * fx_rate))
        recommended_shares = max(0, min(risk_shares, affordable_shares))
        capped_by_capital = risk_shares > affordable_shares
        position_value_local = recommended_shares * price
        position_value_twd = position_value_local * fx_rate

        report = f"📐 【ATR 倉位計算】 {symbol}\n"
        report += (
            f"● ATR(14): {atr:.2f} | 建議止損距離: {stop_distance_local:.2f} "
            f"({stop_atr_multiple:.1f}x ATR)\n"
        )
        report += f"● 風險預算: NT${risk_budget_twd:,.0f} ({risk_pct:.2f}% of NT${total_capital_twd:,.0f})\n"
        report += (
            f"● 建議股數: {recommended_shares} 股 | 部位市值: "
            f"{position_value_local:,.0f}{' 原幣' if fx_rate > 1 else ' TWD'}"
        )
        if fx_rate > 1:
            report += f" (~NT${position_value_twd:,.0f})"
        report += "\n"
        report += f"● 佔總資金: {(position_value_twd / total_capital_twd * 100) if total_capital_twd > 0 else 0:.1f}%"
        if capped_by_capital:
            report += " | ⚠️ 已受總資金上限限制"
        return report
    except Exception as e:
        logger.error(f"ATR position sizing failed for {symbol}: {e}")
        return format_tool_error(f"❌ 倉位計算失敗: {e}", data_unavailable=True)


@tool()
def calculate_position_size(
    symbol: str,
    risk_pct: float = 2.0,
    total_capital_twd: float = None,
    stop_atr_multiple: float = 2.0,
) -> str:
    """
    Calculates a suggested position size using ATR-based stop distance and portfolio risk budget.
    """
    return build_position_size_report(symbol, risk_pct, total_capital_twd, stop_atr_multiple)


@tool()
def calculate_pnl(symbol: str, current_price: float, shares: float, historical_twd_cost: float, is_us_stock: bool) -> dict:
    """
    Calculates profit and loss (PNL) for a specific position.
    Converts foreign values to TWD and accounts for estimated exchange fees.
    """
    return calculate_position_pnl(symbol, current_price, shares, historical_twd_cost, is_us_stock)

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
    if plan is None:
        return None
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
        thesis_payload=parsed.get("thesis_payload", {}),
    )
    resolve_trade_plan_alert(symbol=plan["symbol"], plan_id=plan_id, alert_type="missing_plan")
    return parsed

def claim_pending_trade_plan_prompts() -> List[Dict[str, Any]]:
    with db_lock:
        conn = get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            prompted_row = cursor.execute(
                """
                SELECT tp.*
                FROM trade_plans tp
                JOIN trade_plan_alerts ta ON ta.plan_id = tp.id
                WHERE tp.status = 'draft'
                  AND ta.alert_type = 'missing_plan'
                  AND ta.status = 'open'
                  AND EXISTS (
                      SELECT 1
                      FROM trade_plan_events te
                      WHERE te.plan_id = tp.id AND te.event_type = 'prompt_sent'
                  )
                ORDER BY tp.updated_at, tp.id
                LIMIT 1
                """
            ).fetchone()
            if prompted_row is not None:
                return []

            row = cursor.execute(
                """
                SELECT tp.*
                FROM trade_plans tp
                JOIN trade_plan_alerts ta ON ta.plan_id = tp.id
                WHERE tp.status = 'draft'
                  AND ta.alert_type = 'missing_plan'
                  AND ta.status = 'open'
                  AND NOT EXISTS (
                      SELECT 1
                      FROM trade_plan_events te
                      WHERE te.plan_id = tp.id AND te.event_type = 'prompt_sent'
                )
                ORDER BY tp.updated_at, tp.id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                return []
            _record_trade_plan_event(cursor, plan_id=int(row["id"]), event_type="prompt_sent")
            conn.commit()
            return [dict(row)]
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

def release_trade_plan_prompt(plan_id: int) -> None:
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM trade_plan_events
                WHERE id = (
                    SELECT id
                    FROM trade_plan_events
                    WHERE plan_id = ? AND event_type = 'prompt_sent'
                    ORDER BY id DESC
                    LIMIT 1
                )
                """,
                (plan_id,),
            )
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
                JOIN trade_plan_alerts ta ON ta.plan_id = tp.id
                WHERE tp.status = 'draft'
                  AND te.event_type = 'prompt_sent'
                  AND ta.alert_type = 'missing_plan'
                  AND ta.status = 'open'
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

def format_trade_plan_confirmation(plan: Dict[str, Any], resolution: Dict[str, Any]) -> str:
    return f"✅ 已記錄 {plan['symbol']} 的交易計畫"

@tool()
def get_portfolio_exposure_report() -> str:
    """
    Calculates true exposure by looking through ETF holdings.
    """
    rows = _load_portfolio_rows()
    snapshots = _build_live_position_snapshots(rows)
    
    exposure = {} # symbol -> market_value_twd
    total_mv_twd = 0
    
    for s in snapshots:
        symbol = s['symbol']
        mv_twd = s['market_value_twd']
        total_mv_twd += mv_twd
        
        ticker = get_ticker(symbol)
        holdings = ticker.get_holdings() if hasattr(ticker, 'get_holdings') else []
        
        if holdings:
            covered_weight = 0
            for h in holdings:
                h_sym = h['Symbol']
                h_weight = h['Percent']
                exposure[h_sym] = exposure.get(h_sym, 0) + (mv_twd * h_weight)
                covered_weight += h_weight
            
            others_weight = max(0, 1.0 - covered_weight)
            if others_weight > 0:
                other_name = f"OTHERS({symbol})"
                exposure[other_name] = exposure.get(other_name, 0) + (mv_twd * others_weight)
        else:
            exposure[symbol] = exposure.get(symbol, 0) + mv_twd
    
    if not exposure: return "Portfolio is empty."
    
    sorted_exp = sorted(exposure.items(), key=lambda x: x[1], reverse=True)
    
    report = "📊 === Portfolio True Exposure (Look-through) ===\n"
    for sym, val in sorted_exp[:15]:
        pct = (val / total_mv_twd) * 100 if total_mv_twd > 0 else 0
        report += f"- {sym}: {val:,.0f} TWD ({pct:.2f}%)\n"
    
    risks = [f"{sym}({ (val/total_mv_twd)*100 :.1f}%)" for sym, val in sorted_exp if total_mv_twd > 0 and (val/total_mv_twd) > 0.15 and not sym.startswith("OTHERS")]
    if risks:
        report += f"\n⚠️ Concentration Risk Alert: {', '.join(risks)}"
        
    return report
