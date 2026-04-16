import json
import time
import os
import csv
import logging
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

def fetch_exchange_rate() -> float:
    """Pure FX-rate logic for direct callers and tests."""
    global _fx_cache
    current_time = time.time()
    if current_time - _fx_cache["timestamp"] < 600:
        return _fx_cache["rate"]
    try:
        ticker = get_ticker("TWD=X")
        rate = ticker.fast_info.get('last_price') or ticker.history(period="1d")['Close'].iloc[-1]
        _fx_cache["rate"] = round(float(rate), 2)
        _fx_cache["timestamp"] = current_time
        return _fx_cache["rate"]
    except Exception as e:
        logger.warning(f"Exchange rate refresh failed, using cache: {e}")
        return _fx_cache["rate"]


@tool()
def get_exchange_rate() -> float:
    return fetch_exchange_rate()

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
                    cursor.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?, ?, ?)", (symbol, new_cost, new_shares, new_twd_cost, current_locked))
                    cursor.execute("UPDATE portfolio SET shares = shares - ?, twd_cost = twd_cost - ? WHERE symbol = ?", (settle_amount, actual_twd_total, settle_currency))
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
                    if new_shares > 0:
                        cursor.execute("UPDATE portfolio SET shares = ?, twd_cost = twd_cost - ? WHERE symbol = ?", (new_shares, realized_twd_cost, symbol))
                    else:
                        cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
                    cursor.execute("UPDATE portfolio SET shares = shares + ?, twd_cost = twd_cost + ? WHERE symbol = ?", (settle_amount, actual_twd_total, settle_currency))
                    result_message = f"✅ 賣出成功！實現損益: NT${realized_pnl:+.0f}"
                    should_refresh_memory = True
            
            elif action == 'set':
                cursor.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?, ?, ?)", (symbol, actual_unit_price, shares, actual_twd_total, current_locked))
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


def build_portfolio_analysis() -> Dict[str, Any]:
    """生成持倉健檢摘要，用於系統自動更新額葉。"""
    with db_lock:
        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, cost, shares, twd_cost FROM portfolio")
            rows = cursor.fetchall()
        finally:
            conn.close()

    active_rows = [row for row in rows if row[2] > 0]
    if not active_rows:
        return {
            "total_current": 0,
            "total_pnl_pct": 0,
            "top3_concentration": 0,
            "position_count": 0,
            "summary": "無有效持倉數據；請確認帳本或券商同步狀態。",
        }

    total_cost_twd = 0.0
    total_market_value_twd = 0.0
    assets = []

    for sym, cost, shares, twd_cost in active_rows:
        current_price = cost
        is_us = False
        if 'CASH' in sym:
            current_price = 1.0 if 'TWD' in sym else fetch_exchange_rate()
        else:
            try:
                if (any(c.isdigit() for c in sym[:4]) and len(sym.split('.')[0]) <= 6) or '.TW' in sym or '.TWO' in sym:
                    if fubon.fubon_ready:
                        quote = fubon.fubon_sdk.marketdata.rest_client.stock.intraday.quote(
                            symbol=sym.replace('.TW', '').replace('.TWO', '')
                        )
                        current_price = quote.get('closePrice') or quote.get('lastPrice') or cost
                else:
                    is_us = True
                    ticker = get_ticker(sym, cache_level="daily")
                    current_price = ticker.fast_info.get('last_price') or cost
            except Exception as e:
                logger.warning(f"Portfolio analysis price refresh failed for {sym}: {e}")

        pnl = calculate_position_pnl(sym, current_price, shares, twd_cost, is_us)
        mv = pnl['market_value_twd']
        total_market_value_twd += mv
        total_cost_twd += twd_cost
        if 'CASH' not in sym:
            assets.append({"symbol": sym, "mv": mv})

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
        "position_count": len(active_rows),
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


@tool()
def calculate_pnl(symbol: str, current_price: float, shares: float, historical_twd_cost: float, is_us_stock: bool) -> dict:
    """
    Calculates profit and loss (PNL) for a specific position.
    Converts foreign values to TWD and accounts for estimated exchange fees.
    """
    return calculate_position_pnl(symbol, current_price, shares, historical_twd_cost, is_us_stock)
