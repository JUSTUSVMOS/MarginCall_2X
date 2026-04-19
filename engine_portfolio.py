import json
import math
import time
import os
import csv
import logging
import threading
from datetime import datetime, timezone
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
):
    cursor.execute(
        """
        INSERT INTO trade_log (
            symbol, action, price, shares, settle_currency, settle_amount, fx_rate,
            realized_pnl, cash_before, cash_after, note
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ),
    )

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
                note TEXT
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
        # 執行遷移：如果舊資料庫沒有 locked 欄位，手動補上
        try:
            cursor.execute("ALTER TABLE portfolio ADD COLUMN locked INTEGER DEFAULT 0")
        except Exception as e:
            logger.debug(f"Portfolio migration skipped: {e}")
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


def _floor_trade_quantity(quantity: float, decimals: int = TRADE_SIZE_DECIMALS) -> float:
    if quantity <= 0:
        return 0.0
    factor = 10 ** max(int(decimals), 0)
    return math.floor(float(quantity) * factor) / factor


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
        if str(profile.get("sector") or "Unknown") == target_sector:
            exposure_twd += float(snapshot.get("market_value_twd") or 0.0)
    return exposure_twd


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
    sector = str(profile.get("sector") or "Unknown")
    limits = _get_trade_governor_limits(str(overlay.get("trade_mode") or "normal"), asset_type)

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
    if position_headroom_twd <= 0:
        return {
            **gate_result,
            "allowed": False,
            "message": (
                f"❌ 風控拒單：{symbol} 已達單一持股上限 "
                f"{limits['single_name_cap'] * 100:.1f}% of NAV。"
            ),
        }

    headrooms: List[tuple[str, float]] = [("gross_scale", gross_headroom_twd), ("single_name_cap", position_headroom_twd)]

    if sector and sector != "Unknown":
        sector_cap_twd = current_nav_twd * limits["sector_cap"]
        current_sector_mv = _estimate_sector_exposure_twd(snapshots, sector)
        sector_headroom_twd = max(0.0, sector_cap_twd - current_sector_mv)
        if sector_headroom_twd <= 0:
            return {
                **gate_result,
                "allowed": False,
                "message": (
                    f"❌ 風控拒單：{sector} 曝險已達產業上限 "
                    f"{limits['sector_cap'] * 100:.1f}% of NAV。"
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
        cap_labels[binding_constraint] = f"{sector_name} 產業上限 {limits['sector_cap'] * 100:.1f}% NAV"
    binding_label = cap_labels.get(binding_constraint, binding_constraint)
    return {
        **gate_result,
        "approved_shares": approved_shares,
        "approved_twd_total": approved_twd_total,
        "message": (
            f"⚠️ 風控縮倉：{symbol} 由 {float(shares):.4f} 股縮至 {approved_shares:.4f} 股 "
            f"({binding_label})。"
        ),
        "note": (
            f"risk_gate:{binding_label}; requested_shares={float(shares):.4f}; "
            f"approved_shares={approved_shares:.4f}"
        ),
    }

def execute_position_update(symbol: str, price: float, shares: float, action: str = 'set', total_amount_twd: float = None, locked: int = None, sync_memory: bool = False) -> str:
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

    if action == "buy" and not is_cash:
        gate = _apply_pretrade_risk_gate(symbol, action, shares, actual_twd_total)
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

    result_message = ""
    should_refresh_memory = False
    with db_lock:
        conn = get_connection()
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
                    _record_trade_log(
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
                    )
                    result_message = f"✅ 買進成功！從 {settle_currency} 扣款 {settle_amount:.2f}"
                    if gate_message:
                        result_message = f"{gate_message} {result_message}".strip()
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
    with db_lock:
        conn = get_connection()
        cursor = conn.cursor()
        
        # 1. 取得富邦實體數據 (包含股數與買進成本)
        if fubon.fubon_ready:
            fubon_inv = fubon.get_fubon_inventories()
            fubon_cash = fubon.get_fubon_bank_remain()
        else:
            fubon_inv = {}
            fubon_cash = None
        
        try:
            # 2. 取得資料庫目前的倉位
            cursor.execute("SELECT symbol, cost, shares, twd_cost, locked FROM portfolio")
            db_rows = cursor.fetchall()
            db_dict = {r[0]: list(r) for r in db_rows}

            if fubon.fubon_ready:
                # 3. 智能合併與清洗
                # A. 遍歷富邦抓到的標的，更新或新增
                for symbol, data in fubon_inv.items():
                    fb_shares = data['shares']
                    fb_cost = data['cost']
                    
                    if symbol in db_dict:
                        # 已有紀錄，強制同步股數與平均成本 (Fubon 為準)
                        update_needed = False
                        if db_dict[symbol][2] != fb_shares or db_dict[symbol][1] != fb_cost:
                            db_dict[symbol][2] = fb_shares
                            db_dict[symbol][1] = fb_cost
                            db_dict[symbol][3] = fb_cost * fb_shares
                            update_needed = True
                        
                        if update_needed:
                            cursor.execute("UPDATE portfolio SET shares = ?, cost = ?, twd_cost = ? WHERE symbol = ?", (fb_shares, fb_cost, fb_cost * fb_shares, symbol))
                    else:
                        # 資料庫沒記錄，自動新增
                        cursor.execute("INSERT INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)", (symbol, fb_cost, fb_shares, fb_cost * fb_shares, 0))
                        db_dict[symbol] = [symbol, fb_cost, fb_shares, fb_cost * fb_shares, 0]

                # B. 【清洗邏輯】如果資料庫中的台股標的不在富邦清單內，且未被鎖定，則刪除
                fb_symbols = set(fubon_inv.keys())
                to_delete = []
                for sym in db_dict.keys():
                    # 判斷是否為台股 (非 CASH, 非海外股)
                    clean_sym = sym.replace('.TW', '').replace('.TWO', '').replace('_TRUST', '').replace('_ESOP', '')
                    is_taiwan = (any(char.isdigit() for char in clean_sym) and len(clean_sym) <= 6)
                    is_locked = db_dict[sym][4] == 1
                    is_trust = '_TRUST' in sym or '_ESOP' in sym
                    
                    # 只有一般台股才需要跟 Fubon 同步清理，鎖定單或信託/ESOP單不受券商清單影響
                    if is_taiwan and not is_trust and sym not in fb_symbols and not is_locked:
                        to_delete.append(sym)
                
                for sym in to_delete:
                    cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (sym,))
                    del db_dict[sym]
                    print(f"🧹 已自動清理幽靈庫存: {sym}")

                # C. 自動同步台幣現金
                if fubon_cash is not None:
                    cursor.execute("INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES ('CASH_TWD', 1.0, ?, ?, 0)", (float(fubon_cash), float(fubon_cash)))
                    # 更新 db_dict 讓回傳的 JSON 也有資料
                    db_dict['CASH_TWD'] = ['CASH_TWD', 1.0, float(fubon_cash), float(fubon_cash), 0]

            conn.commit()

            # 4. 組裝回傳資料
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


@tool()
def get_portfolio_raw_data() -> str:
    """Retrieves current portfolio positions, prices, and TWD balances."""
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


@tool()
def get_portfolio_analytics() -> str:
    """Returns realized closed-book performance analytics built from trade_log sells."""
    return build_portfolio_analytics_report()


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
        f"● 建議: Gross Scale {overlay['recommended_gross_scale']:.2f}x | "
        f"Raise Cash ~NT${overlay['trim_notional_twd']:,.0f}"
    )
    if overlay.get("hedge_notional_twd", 0) > 0:
        report += f" | Benchmark Hedge ~NT${overlay['hedge_notional_twd']:,.0f}"
    report += "\n"
    report += f"● 約束: {overlay['primary_constraint']}\n"
    report += f"● Governor: {overlay['governor_message']}"
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
