import json
import time
import os
import csv
import logging
import threading
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import fubon
from typing import Any, Dict, List
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

def execute_position_update(symbol: str, price: float, shares: float, action: str = 'set', total_amount_twd: float = None, locked: int = None, sync_memory: bool = False) -> str:
    """Pure portfolio-write logic for direct callers and tests."""
    symbol = normalize_ticker(symbol)
    is_taiwan = (any(char.isdigit() for char in symbol) and len(symbol) <= 6) or symbol.endswith('.TW') or symbol.endswith('.TWO')
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
                    )
                    result_message = f"✅ 買進成功！從 {settle_currency} 扣款 {settle_amount:.2f}"
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
            # 美股嘗試 yfinance
            import yfinance as yf
            ticker = get_ticker(clean_sym, cache_level="daily")
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
        # if fubon.fubon_ready:
        #     fubon_inv = fubon.get_fubon_inventories()
        #     fubon_cash = fubon.get_fubon_bank_remain()
        # else:
        #     fubon_inv = {}
        #     fubon_cash = None
        fubon_inv = {}
        fubon_cash = None
        
        try:
            # 2. 取得資料庫目前的倉位
            cursor.execute("SELECT symbol, cost, shares, twd_cost, locked FROM portfolio")
            db_rows = cursor.fetchall()
            db_dict = {r[0]: list(r) for r in db_rows}

        # if fubon.fubon_ready:
        #     # 3. 智能合併與清洗
        #     # A. 遍歷富邦抓到的標的，更新或新增
        #     for symbol, data in fubon_inv.items():
        #         fb_shares = data['shares']
        #         fb_cost = data['cost']
        #         
        #         if symbol in db_dict:
        #             # 已有紀錄，更新股數。只有當 db 成本為 0 時才更新成本。
        #             update_needed = False
        #             if db_dict[symbol][2] != fb_shares:
        #                 db_dict[symbol][2] = fb_shares
        #                 update_needed = True
        #             if db_dict[symbol][1] == 0.0 and fb_cost > 0:
        #                 db_dict[symbol][1] = fb_cost
        #                 update_needed = True
        #             
        #             if update_needed:
        #                 cursor.execute("UPDATE portfolio SET shares = ?, cost = ? WHERE symbol = ?", (fb_shares, db_dict[symbol][1], symbol))
        #         else:
        #             # 資料庫沒記錄，自動新增
        #             cursor.execute("INSERT INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)", (symbol, fb_cost, fb_shares, fb_cost * fb_shares, 0))
        #             db_dict[symbol] = [symbol, fb_cost, fb_shares, fb_cost * fb_shares, 0]
        #
        #     # B. 【清洗邏輯】如果資料庫中的台股標的不在富邦清單內，且未被鎖定，則刪除
        #     fb_symbols = set(fubon_inv.keys())
        #     to_delete = []
        #     for sym in db_dict.keys():
        #         # 判斷是否為台股 (非 CASH, 非海外股)
        #         is_taiwan = (any(char.isdigit() for char in sym) and len(sym) <= 6)
        #         is_locked = db_dict[sym][4] == 1
        #         
        #         if is_taiwan and sym not in fb_symbols and not is_locked:
        #             to_delete.append(sym)
        #     
        #     for sym in to_delete:
        #         cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (sym,))
        #         del db_dict[sym]
        #         print(f"🧹 已自動清理幽靈庫存: {sym}")
        #
        #     # C. 自動同步台幣現金
        #     if fubon_cash is not None:
        #         cursor.execute("INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES ('CASH_TWD', 1.0, ?, ?, 0)", (float(fubon_cash), float(fubon_cash)))
        #         # 更新 db_dict 讓回傳的 JSON 也有資料
        #         db_dict['CASH_TWD'] = ['CASH_TWD', 1.0, float(fubon_cash), float(fubon_cash), 0]
        
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
    """Retrieves current portfolio positions and balances."""
    return build_portfolio_raw_data()


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
            elif market == "TW" and fubon.fubon_ready:
                quote = fubon.fubon_sdk.marketdata.rest_client.stock.intraday.quote(
                    symbol=sym.replace(".TW", "").replace(".TWO", "")
                )
                refreshed_price = quote.get("closePrice") or quote.get("lastPrice")
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


def build_portfolio_analysis() -> Dict[str, Any]:
    """生成持倉健檢摘要，用於系統自動更新額葉。"""
    snapshots = _build_live_position_snapshots(_load_portfolio_rows())
    if not snapshots:
        return {
            "total_current": 0,
            "total_pnl_pct": 0,
            "top3_concentration": 0,
            "position_count": 0,
            "summary": "無有效持倉數據；請確認帳本或券商同步狀態。",
        }

    total_cost_twd = sum(pos["twd_cost"] for pos in snapshots)
    total_market_value_twd = sum(pos["market_value_twd"] for pos in snapshots)
    assets = [{"symbol": pos["symbol"], "mv": pos["market_value_twd"]} for pos in snapshots if not pos["is_cash"]]

    # 計算集中度
    assets.sort(key=lambda x: x['mv'], reverse=True)
    top3_mv = sum(a['mv'] for a in assets[:3])
    top3_pct = (top3_mv / total_market_value_twd * 100) if total_market_value_twd > 0 else 0
    
    total_pnl_pct = ((total_market_value_twd - total_cost_twd) / total_cost_twd * 100) if total_cost_twd > 0 else 0

    summary = (
        f"NAV: NT${total_market_value_twd:,.0f} | "
        f"PnL: {total_pnl_pct:+.1f}% | "
        f"Top3 集中度: {top3_pct:.0f}%"
    )
    
    # 如果有大幅變動，增加警語
    if abs(total_pnl_pct) > 5:
        summary += f" (⚠️ 總體損益波動劇烈)"

    return {
        "total_current": total_market_value_twd,
        "total_pnl_pct": total_pnl_pct,
        "top3_concentration": top3_pct,
        "position_count": len(snapshots),
        "summary": summary
    }

def refresh_portfolio_health_summary(source: str = "portfolio_review") -> Dict[str, Any]:
    """Builds a portfolio-health snapshot and patches the frontal lobe section."""
    analysis = build_portfolio_analysis()
    import engine_memory as memory

    memory_update = memory.patch_frontal_lobe_section("Portfolio Health", analysis["summary"], source=source)
    return {**analysis, "memory_update": memory_update}

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
) -> Dict[str, Any]:
    clean_holdings = {
        normalize_ticker(symbol): float(weight)
        for symbol, weight in holdings.items()
        if isinstance(weight, (int, float)) and weight > 0
    }
    total_weight = sum(clean_holdings.values())
    if total_weight <= 0:
        return {"error": "無有效持倉權重可做 beta 分解。"}
    normalized_holdings = {symbol: weight / total_weight for symbol, weight in clean_holdings.items()}

    benchmark_symbol = normalize_ticker(benchmark).upper()
    if benchmark_symbol.isdigit():
        benchmark_symbol += ".TW"

    bench_hist = get_ticker(benchmark_symbol, cache_level="daily").history(period=period, interval="1d")
    if bench_hist.empty or "Close" not in bench_hist.columns:
        return {"error": f"{benchmark_symbol} 無法取得基準歷史價格。"}
    bench_returns = pd.to_numeric(bench_hist["Close"], errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(bench_returns) < MIN_BETA_OBSERVATIONS:
        return {"error": f"{benchmark_symbol} 歷史資料不足，無法穩健估 beta。"}

    positions: Dict[str, Dict[str, Any]] = {}
    skipped_positions: Dict[str, str] = {}
    portfolio_beta = 0.0
    portfolio_alpha_daily = 0.0
    coverage_weight = 0.0

    for raw_symbol, weight in normalized_holdings.items():
        symbol = raw_symbol.upper()
        if symbol.isdigit():
            symbol += ".TW"
        try:
            stock_hist = get_ticker(symbol, cache_level="daily").history(period=period, interval="1d")
        except Exception as exc:
            skipped_positions[symbol] = f"價格抓取失敗: {exc}"
            continue
        if stock_hist.empty or "Close" not in stock_hist.columns:
            skipped_positions[symbol] = "缺少 Close 歷史資料"
            continue

        stock_returns = pd.to_numeric(stock_hist["Close"], errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).dropna()
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
