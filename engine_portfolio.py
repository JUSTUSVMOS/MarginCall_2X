import sqlite3
import json
import time
import os
import csv
import yfinance as yf
from typing import Dict, List

DB_FILE = "portfolio.db"
CSV_BACKUP = "my_portfolio.csv"

# --- 匯率快取 ---
_fx_cache = {"rate": 32.0, "timestamp": 0}

def get_exchange_rate() -> float:
    global _fx_cache
    current_time = time.time()
    if current_time - _fx_cache["timestamp"] < 600:
        return _fx_cache["rate"]
    try:
        ticker = yf.Ticker("TWD=X")
        rate = ticker.fast_info.get('last_price') or ticker.history(period="1d")['Close'].iloc[-1]
        _fx_cache["rate"] = round(float(rate), 2)
        _fx_cache["timestamp"] = current_time
        return _fx_cache["rate"]
    except:
        return _fx_cache["rate"]

# --- 資料庫初始化與遷移 ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            symbol TEXT PRIMARY KEY,
            cost REAL,
            shares REAL,
            twd_cost REAL
        )
    """)
    conn.commit()

    # 檢查是否需要從 CSV 遷移
    if os.path.exists(CSV_BACKUP):
        print(f"📦 偵測到舊帳本 {CSV_BACKUP}，正在執行自動遷移...")
        try:
            with open(CSV_BACKUP, mode='r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                next(reader, None) # 跳過標頭
                for row in reader:
                    if len(row) >= 3:
                        sym = row[0].upper()
                        cost = float(row[1])
                        shares = float(row[2])
                        twd_c = float(row[3]) if len(row) >= 4 else (cost * shares * (get_exchange_rate() if ".TW" not in sym and "CASH" not in sym else 1.0))
                        cursor.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?, ?)", (sym, cost, shares, twd_c))
            conn.commit()
            # 遷移完成後將舊檔改名備份
            os.rename(CSV_BACKUP, f"{CSV_BACKUP}.migrated_{int(time.time())}")
            print("✅ 遷移完成，舊檔已備份。")
        except Exception as e:
            print(f"⚠️ 遷移失敗: {e}")
    conn.close()

# 啟動時自動初始化
init_db()

def update_position(symbol: str, price: float, shares: float, action: str = 'set', total_amount_twd: float = None) -> str:
    """
    更新持倉或現金。
    action: 'buy' (買入), 'sell' (賣出), 'set' (校正)
    price: 原幣單價
    shares: 股數 (action='sell' 時代表賣出股數)
    total_amount_twd: 選填，如果 AI 直接知道台幣總額可帶入
    """
    symbol = symbol.upper()
    is_taiwan = (any(char.isdigit() for char in symbol) and len(symbol) <= 6) or symbol.endswith('.TW') or symbol.endswith('.TWO')
    is_cash = 'CASH' in symbol
    fx_rate = get_exchange_rate() if (not is_taiwan and not is_cash) else 1.0
    
    # 核心邏輯：計算該次異動的台幣價值
    if total_amount_twd:
        actual_twd_total = total_amount_twd
        actual_unit_price = total_amount_twd / shares / fx_rate if shares > 0 else price
    else:
        actual_unit_price = price
        actual_twd_total = price * shares * fx_rate

    settle_currency = 'CASH_TWD' if is_taiwan else 'CASH_USD'
    # 美股扣款原幣，台股扣款台幣
    settle_amount = actual_unit_price * shares if not is_taiwan else actual_twd_total

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # 取得標的與現金池現況
        cursor.execute("SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = ?", (symbol,))
        old_pos = cursor.fetchone() or (0.0, 0.0, 0.0)
        
        cursor.execute("SELECT cost, shares, twd_cost FROM portfolio WHERE symbol = ?", (settle_currency,))
        cash_pos = cursor.fetchone() or (1.0 if 'TWD' in settle_currency else fx_rate, 0.0, 0.0)

        if action == 'buy':
            if cash_pos[1] < settle_amount:
                return f"❌ 買進失敗：{settle_currency} 餘額不足！(剩 {cash_pos[1]:.2f})"
            new_shares = old_pos[1] + shares
            new_twd_cost = old_pos[2] + actual_twd_total
            new_cost = (old_pos[0] * old_pos[1] + actual_unit_price * shares) / new_shares
            cursor.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?, ?)", (symbol, new_cost, new_shares, new_twd_cost))
            cursor.execute("UPDATE portfolio SET shares = shares - ?, twd_cost = twd_cost - ? WHERE symbol = ?", (settle_amount, actual_twd_total, settle_currency))
            msg = f"✅ 買進成功！從 {settle_currency} 扣款 {settle_amount:.2f}"
        
        elif action == 'sell':
            if old_pos[1] < shares:
                return f"❌ 賣出失敗：持股不足 (只有 {old_pos[1]})"
            cost_ratio = shares / old_pos[1]
            realized_twd_cost = old_pos[2] * cost_ratio
            realized_pnl = actual_twd_total - realized_twd_cost
            new_shares = old_pos[1] - shares
            if new_shares > 0:
                cursor.execute("UPDATE portfolio SET shares = ?, twd_cost = twd_cost - ? WHERE symbol = ?", (new_shares, realized_twd_cost, symbol))
            else:
                cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
            cursor.execute("UPDATE portfolio SET shares = shares + ?, twd_cost = twd_cost + ? WHERE symbol = ?", (settle_amount, actual_twd_total, settle_currency))
            msg = f"✅ 賣出成功！實現損益: NT${realized_pnl:+.0f}"
        
        elif action == 'set':
            cursor.execute("INSERT OR REPLACE INTO portfolio VALUES (?, ?, ?, ?)", (symbol, actual_unit_price, shares, actual_twd_total))
            msg = f"✅ 校正成功！{symbol} 已更新。"

        conn.commit()
        return msg
    except Exception as e:
        return f"❌ 記帳異常: {e}"
    finally:
        conn.close()

def get_portfolio_raw_data() -> str:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT symbol, cost, shares, twd_cost FROM portfolio")
        rows = cursor.fetchall()
        records = [{"symbol": r[0], "cost": r[1], "shares": r[2], "twd_cost": r[3]} for r in rows]
        return json.dumps(records)
    except:
        return "[]"
    finally:
        conn.close()

def calculate_pnl(symbol: str, current_price: float, shares: float, historical_twd_cost: float, is_us_stock: bool) -> dict:
    raw_fx = get_exchange_rate() if (is_us_stock or symbol == 'CASH_USD') else 1.0
    settle_fx = raw_fx * 0.998 if (is_us_stock or symbol == 'CASH_USD') else 1.0

    if symbol == 'CASH_TWD':
        return {"market_value_twd": round(shares, 2), "pnl_value_twd": 0, "pnl_percent": 0}
    
    current_market_value_twd = (shares * settle_fx) if 'CASH' in symbol else (current_price * shares * settle_fx)
    pnl_value_twd = current_market_value_twd - historical_twd_cost
    pnl_percent = (pnl_value_twd / historical_twd_cost * 100) if historical_twd_cost > 0 else 0
    
    return {
        "market_value_twd": round(current_market_value_twd, 2),
        "pnl_value_twd": round(pnl_value_twd, 2),
        "pnl_percent": round(pnl_percent, 2)
    }
