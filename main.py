import pytz
import os
import csv
import json
import urllib3
from dotenv import load_dotenv
from config import WDT_MESSAGES, system_prompt
from fubon import get_quote_and_orderbook
import telebot
from google import genai
from google.genai import types
import yfinance as yf
import random
import datetime
import requests
import time
from fubon_neo.sdk import FubonSDK  # 👈 新增引入富邦 SDK
# 關閉警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

FMP_KEY = os.getenv("FMP_API_KEY")
if not FMP_KEY:
    print("⚠️ 警告：沒讀到 FMP_API_KEY，將退回 Yahoo 備援模式。")
else:
    print("✅ FMP 引擎金鑰讀取成功！即將啟動機構級報價通道。")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN or not GEMINI_KEY:
    raise ValueError("兄弟，你的 .env 沒設定好 TOKEN 或 GEMINI API KEY 喔！")

print("啟動破產推進器：V8雙渦輪引擎 (含自動備用切換機制) 載入中...")
import fubon  # 👈 引入你剛剛寫好的 fubon.py

# 執行富邦初始化 (它會讀取 .env 並登入)
fubon.init_fubon()

# 為了方便後續調用，可以把狀態抓出來 (選配)
fubon_ready = fubon.fubon_ready
bot = telebot.TeleBot(BOT_TOKEN)

PORTFOLIO_FILE = "my_portfolio.csv"
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(["symbol", "cost", "shares"])

# ==========================================
# 🛠️ 數據與運算 API 層 (修改記帳工具的註解，防呆！)
# ==========================================
def update_position(symbol: str, price: float, shares: float, action: str = 'set') -> str:
    """
    【量級自動判定與歷史匯率鎖定系統 V7.0】
    支援 action: 'buy', 'sell', 'set'
    具備抽象化幣別分流邏輯，嚴禁二次換匯。
    """
    symbol = symbol.upper()
    # 1. 辨識屬性：台股、現金、還是美股
    is_taiwan = (any(char.isdigit() for char in symbol) and len(symbol) <= 6) or symbol.endswith('.TW') or symbol.endswith('.TWO')
    is_cash = 'CASH' in symbol
    
    # 取得最新市場匯率 (僅美股需要)
    fx_rate = get_exchange_rate() if (not is_taiwan and not is_cash) else 1.0
    
    # 2. 🛡️ 核心：幣別與量級自動判定 (Scale Heuristics)
    # 邏輯：美股若 price > 2000，判定為台幣總額；若 < 2000，判定為美金單價
    actual_twd_total = 0.0
    actual_unit_price = 0.0 # 存入 CSV 'cost' 欄位的數值 (美股為 USD, 台股為 TWD)

    if not is_taiwan and not is_cash:
        if price > 2000:
            # [總額模式] 用戶輸入的是台幣總支出
            actual_twd_total = price
            actual_unit_price = price / shares / fx_rate if shares > 0 else 0
        else:
            # [單價模式] 用戶輸入的是美金單價
            actual_unit_price = price
            actual_twd_total = price * shares * fx_rate
    else:
        # 台股或現金，單位統一為台幣
        actual_unit_price = price
        actual_twd_total = price * shares

    # 3. 讀取現有帳本
    records = {}
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    sym = row[0]
                    cost = float(row[1])
                    qty = float(row[2])
                    # 優先信任第 4 欄 (twd_cost)，若無則現場補算
                    twd_c = float(row[3]) if len(row) >= 4 else (cost * qty * (get_exchange_rate() if any(c.isalpha() for c in sym) and ".TW" not in sym and "CASH" not in sym else 1.0))
                    records[sym] = {"cost": cost, "shares": qty, "twd_cost": twd_c}

    # 4. 現金池初始化 (防呆：若無 CASH 則設為 0，不亂送錢)
    if 'CASH' not in records:
        records['CASH'] = {"cost": 1.0, "shares": 0.0, "twd_cost": 0.0}

    old_pos = records.get(symbol, {"cost": 0.0, "shares": 0.0, "twd_cost": 0.0})
    cash_pos = records['CASH']
    msg = ""

    try:
        # 5. 執行交易邏輯
        if action == 'buy':
            if cash_pos['shares'] < actual_twd_total:
                return f"❌ 買進失敗：現金不足！(需 NT${actual_twd_total:.0f}，剩 NT${cash_pos['shares']:.0f})"
            
            new_shares = old_pos['shares'] + shares
            new_twd_cost = old_pos['twd_cost'] + actual_twd_total
            # 重新加權平均單價 (美股存 USD, 台股存 TWD)
            new_cost = (old_pos['cost'] * old_pos['shares'] + actual_unit_price * shares) / new_shares
            
            records[symbol] = {"cost": new_cost, "shares": new_shares, "twd_cost": new_twd_cost}
            records['CASH']['shares'] -= actual_twd_total
            records['CASH']['twd_cost'] -= actual_twd_total
            msg = f"✅ 買進成功！已鎖定匯率扣款 NT${actual_twd_total:.0f}"

        elif action == 'sell':
            if old_pos['shares'] < shares:
                return f"❌ 賣出失敗：持股不足 (只有 {old_pos['shares']} 股)"
            
            # 按比例扣除歷史台幣成本
            cost_ratio = shares / old_pos['shares']
            realized_twd_cost = old_pos['twd_cost'] * cost_ratio
            realized_pnl = actual_twd_total - realized_twd_cost

            new_shares = old_pos['shares'] - shares
            new_twd_cost = old_pos['twd_cost'] - realized_twd_cost
            
            if new_shares > 0:
                records[symbol] = {"cost": old_pos['cost'], "shares": new_shares, "twd_cost": new_twd_cost}
            else:
                if symbol in records: del records[symbol]

            records['CASH']['shares'] += actual_twd_total
            records['CASH']['twd_cost'] += actual_twd_total
            msg = f"✅ 賣出成功！入帳 NT${actual_twd_total:.0f}，實現損益: {realized_pnl:+.0f}"

        elif action == 'set':
            # 直接校正模式
            records[symbol] = {"cost": actual_unit_price, "shares": shares, "twd_cost": actual_twd_total}
            msg = f"✅ 校正成功！{symbol} 已更新為 {shares} 股。"

        # 6. 寫回 CSV (確保 4 欄位完整)
        with open(PORTFOLIO_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "cost", "shares", "twd_cost"])
            for sym, data in records.items():
                writer.writerow([sym, data['cost'], data['shares'], data['twd_cost']])

        return msg

    except Exception as e:
        return f"❌ 記帳異常: {e}"
    
def get_portfolio_raw_data() -> str:
    """【防彈版】回傳持股 JSON，具備新舊格式自動轉換能力"""
    if not os.path.exists(PORTFOLIO_FILE): 
        return "[]"
    
    records = []
    try:
        with open(PORTFOLIO_FILE, mode='r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader, None) # 跳過標頭
            
            for row in reader:
                if not row or len(row) < 3: continue # 略過空行或爛掉的行
                
                symbol = row[0].upper()
                cost = float(row[1])
                shares = float(row[2])
                
                # 🛡️ 關鍵修正：如果沒有第四欄，就現場用目前匯率推算補齊
                if len(row) >= 4:
                    twd_cost = float(row[3])
                else:
                    is_taiwan = any(char.isdigit() for char in symbol) and (len(symbol) <= 6)
                    fx = get_exchange_rate() if not is_taiwan and symbol != 'CASH' else 1.0
                    twd_cost = cost * shares * fx
                
                records.append({
                    "symbol": symbol,
                    "cost": cost,
                    "shares": shares,
                    "twd_cost": twd_cost
                })
        
        return json.dumps(records)
    except Exception as e:
        print(f"❌ 讀取帳本失敗: {e}")
        return "[]"

def calculate_pnl(symbol: str, current_price: float, shares: float, historical_twd_cost: float, is_us_stock: bool) -> dict:
    # 1. 取得最新市場匯率
    raw_fx = get_exchange_rate() if (is_us_stock or symbol == 'CASH_USD') else 1.0
    
    # 2. 🛡️ 專業級「匯率摩擦力」校正
    # 當我們要算「目前價值」時，必須模擬「換回台幣」的價格（銀行買入價 Bid）
    # 通常買賣價差約 0.06 ~ 0.1 元，這裡我們保守打 0.998 折 (模擬 -0.2% 的換匯損失)
    settle_fx = raw_fx * 0.998 if (is_us_stock or symbol == 'CASH_USD') else 1.0

    # --- 分支 A：🇹🇼 台幣現金 ---
    if symbol == 'CASH_TWD':
        return {"market_value_twd": round(shares, 2), "pnl_value_twd": 0, "pnl_percent": 0}

    # --- 分支 B：🇺🇸 美金現金 (也是一種美金部位) ---
    if symbol == 'CASH_USD':
        current_market_value_twd = shares * settle_fx
        pnl_value_twd = current_market_value_twd - historical_twd_cost
        pnl_percent = (pnl_value_twd / historical_twd_cost * 100) if historical_twd_cost > 0 else 0
        return {
            "market_value_twd": round(current_market_value_twd, 2),
            "pnl_value_twd": round(pnl_value_twd, 2),
            "pnl_percent": round(pnl_percent, 2)
        }

    # --- 分支 C：📈 一般股票邏輯 (包含匯率摩擦) ---
    current_market_value_twd = current_price * shares * settle_fx
    pnl_value_twd = current_market_value_twd - historical_twd_cost
    pnl_percent = (pnl_value_twd / historical_twd_cost * 100) if historical_twd_cost > 0 else 0
    
    return {
        "market_value_twd": round(current_market_value_twd, 2),
        "pnl_value_twd": round(pnl_value_twd, 2),
        "pnl_percent": round(pnl_percent, 2)
    }
    
_fx_cache = {"rate": 32.0, "timestamp": 0}

def get_exchange_rate() -> float:
    global _fx_cache
    current_time = time.time()
    
    # 如果快取還沒超過 10 分鐘，直接回傳舊的，省下網路請求時間
    if current_time - _fx_cache["timestamp"] < 600:
        return _fx_cache["rate"]
        
    try:
        ticker = yf.Ticker("TWD=X")
        # 優先用 fast_info 拿價格，這比 info 快非常多
        rate = ticker.fast_info.get('last_price')
        
        if not rate:
            hist = ticker.history(period="1d")
            rate = hist['Close'].iloc[-1]
            
        _fx_cache["rate"] = round(float(rate), 2)
        _fx_cache["timestamp"] = current_time
        return _fx_cache["rate"]
    except Exception as e:
        print(f"⚠️ 匯率抓取失敗: {e}")
        return _fx_cache["rate"] # 失敗時回傳上一次成功的快取

def is_tw_market_open() -> bool:
    """
    判斷現在是否為台股正常交易時段。
    台灣時間：週一至週五 09:00 ~ 13:30。
    """
    now = datetime.datetime.now()
    weekday = now.weekday()
    
    # 週末直接睡死
    if weekday >= 5: 
        return False
        
    current_hour = now.hour
    current_minute = now.minute
    
    # 09:00 ~ 13:30 判定
    if 9 <= current_hour < 13:
        return True
    elif current_hour == 13 and current_minute <= 30:
        return True
        
    return False

def is_us_market_open() -> bool:
    """
    判斷現在是否為美股正常交易時段 (忽略夏令/冬令切換的細微差異，取最廣泛範圍)。
    台灣時間約為：週一至週五 21:00 ~ 隔日 05:00 (含盤前預熱與盤後緩衝)。
    """
    now = datetime.datetime.now()
    weekday = now.weekday()  # 0 是週一, 6 是週日
    
    # 週末不開盤 (週六早上 5 點後到週一晚上 9 點前)
    if weekday == 5 and now.hour >= 5: return False # 週六清晨後
    if weekday == 6: return False # 週日
    if weekday == 0 and now.hour < 21: return False # 週一晚上前
    
    # 簡單邏輯：晚上 9 點到凌晨 5 點視為「需要即時 FMP 數據」的時段
    current_hour = now.hour
    if current_hour >= 21 or current_hour < 5:
        return True
    
    return False

def get_dynamic_models():
    """根據台股與美股開盤狀態，動態切換引擎優先順序"""
    models = [
        'gemini-3.1-flash-lite-preview', # 衝鋒槍：3.1代輕量版，速度與額度的平衡點
        'gemini-2.5-pro',                # 備用大腦
        'gemini-2.5-flash',              # 主力部隊
        'gemini-2.0-flash-lite',         # 省油燈：額度快乾時的主力
        'gemini-flash-latest'            # 護城河：絕對能跑
    ]
    
    # 🚀 雙引擎點火：只要美股「或」台股開盤，直接拔出 3.1 Pro 狙擊槍！
    if is_us_market_open() or is_tw_market_open():
        models.insert(0, 'gemini-3.1-pro-preview')
    return models

def get_live_price(symbol: str) -> float:
    """
    【V9 雙引擎行情切換器】
    台股 ➡️ 優先走富邦 Fubon SDK (零延遲)
    美股/富邦故障 ➡️ 走 FMP/Yahoo Finance
    """
    global fubon_sdk, fubon_ready  # 確保抓得到全域的狀態
    symbol = symbol.upper()

    # 🛡️ MTK ESOP 轉接器
    if symbol == "2454_ESOP":
        symbol = "2454"

    # 判斷是否為台股 (數字開頭且長度 <= 6)
    is_taiwan_stock = any(char.isdigit() for char in symbol) and (len(symbol) <= 6)

    # --- 🚀 優先路徑：台股走富邦 SDK ---
    if is_taiwan_stock and fubon_ready:
        try:
            reststock = fubon_sdk.marketdata.rest_client.stock
            # 使用我們測試成功的 quote API
            quote_data = reststock.intraday.quote(symbol=symbol)
            
            # 解析現價 (依據我們剛才 debug 的格式)
            is_dict = isinstance(quote_data, dict)
            price = quote_data.get('closePrice') or quote_data.get('lastPrice') if is_dict else getattr(quote_data, 'closePrice', getattr(quote_data, 'lastPrice', None))
            
            if price and price > 0:
                print(f"🔥 [富邦 V8] 抓取 {symbol} 成功: {price}")
                return round(float(price), 2)
        except Exception as e:
            print(f"⚠️ 富邦通道異常 ({e})，準備切換備援模式...")
            # 這裡不 return，讓它往下走 Yahoo 備援

    # --- 🚀 條件 2：美股且 FMP 有效 ---
    if not is_taiwan_stock and FMP_KEY and is_us_market_open():
        try:
            url = f"https://financialmodelingprep.com/stable/quote?symbol={symbol}&apikey={FMP_KEY}"
            res = requests.get(url, timeout=5).json()
            if isinstance(res, list) and len(res) > 0:
                print(f"⚡ [FMP] 抓取 {symbol} 即時報價成功")
                return round(float(res[0]['price']), 2)
        except:
            pass 

    # --- 🛡️ 備援模式：Yahoo Finance (當富邦掛了或是非台股時) ---
    search_list = [symbol]
    if is_taiwan_stock and '.' not in symbol:
        search_list = [f"{symbol}.TW", f"{symbol}.TWO", symbol]

    for s in search_list:
        try:
            ticker = yf.Ticker(s)
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            
            if not price:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            if price and price > 0:
                source = "台股備援 (YF)" if is_taiwan_stock else "美股 YF"
                print(f"🛡️ [{source}] 抓取 {s} 成功")
                return round(float(price), 2)
        except:
            continue
            
    return None

def get_market_sentiment() -> str:
    """
    【🌐 全球資金流向雷達 V5.0 - Justus 修正版】
    使用測試成功的 YF history 5d 邏輯，徹底解決週末與收費牆問題。
    """
    # 這裡放的是你原本想要的所有 8 個宏觀指標，並對應到 YF 成功代碼
    indicators = {
        "^TNX": "美債10年期(估值重力)",
        "DX-Y.NYB": "美元指數(資金水龍頭)",
        "^VIX": "恐慌指數(波動絞肉機)",
        "^SOX": "費城半導體(科技基本面)",
        "HYG": "高收益債(企業違約)",
        "GC=F": "黃金期貨(避險)",
        "CL=F": "WTI原油(通膨指標)",
        "BZ=F": "布蘭特原油(地緣政治)"
    }
    
    now = datetime.datetime.now()
    # 週末判定
    if now.weekday() >= 5:
        report = "【⏸️ 週末休市模式：數據源已鎖定週五收盤價】\n"
    else:
        report = "【🌐 全球資金流向雷達與 Watchdog 狀態 (YF 穩定版)】\n"
        
    watchdog_alerts = []
    
    for symbol, name in indicators.items():
        try:
            ticker = yf.Ticker(symbol)
            # 🚀 使用你測試成功的 5d 歷史數據法，保證抓得到
            hist = ticker.history(period="5d")
            
            if not hist.empty and len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                direction = "📈" if change > 0 else "📉"
                
                # 抓取最後更新日期 (MM/DD)
                last_date = hist.index[-1].strftime('%m/%d')
                report += f"[{last_date}] {direction} {name}: {current:.2f} ({change:+.2f}%)\n"
                
                # --- Watchdog 攔截邏輯 (精準攔截你的策略風險) ---
                if "BZ=F" in symbol and current > 90:
                    watchdog_alerts.append("🚨 [通膨炸彈] 原油破 90！小心你的「賣鏟子」標的估值下殺！")
                
                if "HYG" in symbol and change < -1.5:
                    watchdog_alerts.append("🚨 [清償危機] 垃圾債暴跌！資金正在大撤退，小心 00631L 回測！")
                
                if "^VIX" in symbol and current > 25 and change > 5:
                    watchdog_alerts.append("🚨 [恐慌爆發] VIX 急升！現在絕對禁止追高任何槓桿 ETF！")
            else:
                report += f"⚠️ {name}: 數據讀取不足\n"
                
        except Exception as e:
            report += f"❌ {name}: 抓取失敗 ({str(e)[:15]}...)\n"
            
    if watchdog_alerts:
        report += "\n" + "\n".join(watchdog_alerts)
        
    return report
        
def get_market_history(symbol: str, days: int) -> str:
    """
    【強大歷史雷達】
    當用戶詢問「昨天」、「近5天」、「這禮拜」、「最近走勢」時，必須呼叫此工具。
    - 如果用戶問「昨天」，請傳入 days=2。
    - 如果用戶問「近5天」，請傳入 days=5。
    它會回傳過去 N 個交易日的「每日開高低收與成交量」，讓你進行跨日對比。
    """
    try:
        # 自動補台股後綴
        if symbol.isdigit() or (symbol.endswith('L') and symbol[:-1].isdigit()):
            if not symbol.endswith('.TW'):
                symbol += '.TW'
                
        ticker = yf.Ticker(symbol)
        
        # 故意抓長一點(1個月)的歷史資料，避免遇到假日沒有交易日
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            return f"抓不到 {symbol} 的歷史資料。"
            
        # 確保不會超過實際有的資料長度，然後取最後 days 天
        actual_days = min(days, len(hist))
        target_hist = hist.tail(actual_days)
        
        report = f"以下是 {symbol} 近 {actual_days} 個交易日的真實數據：\n"
        for date, row in target_hist.iterrows():
            date_str = date.strftime('%m/%d')
            report += f"[{date_str}] 開:{row['Open']:.2f} | 高:{row['High']:.2f} | 低:{row['Low']:.2f} | 收:{row['Close']:.2f} | 量:{int(row['Volume'])}\n"
            
        # 順便附上最新現價，讓 AI 可以拿昨天跟現在比
        current = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', '未知'))
        report += f"\n目前最新現價 (盤中): {current}"
        
        return report
    except Exception as e:
        return f"歷史報價系統異常: {e}"  
    
def get_fundamental_data(symbol: str) -> str:
    """
    獲取個股的基本面與估值數據 (本益比 P/E、EPS、市值等)。
    當用戶問「這檔現在算便宜還是貴」、「基本面如何」、「本益比多少」時必須呼叫。
    """
    try:
        search_symbol = symbol.upper()
        if search_symbol.isdigit() and len(search_symbol) <= 6:
            search_symbol += ".TW"
            
        ticker = yf.Ticker(search_symbol)
        info = ticker.info
        
        # 很多 ETF (像 VOO, 00631L) 沒有單一公司的 P/E，需要做防呆
        if 'trailingPE' not in info and 'navPrice' in info:
            return f"[{symbol}] 這是一檔 ETF/基金，不適用單一公司的本益比 (P/E) 估值模型，請直接分析其追蹤的底層指數或宏觀資金流向。"
            
        pe = info.get('trailingPE', '未知')
        fwd_pe = info.get('forwardPE', '未知')
        eps = info.get('trailingEps', '未知')
        pb = info.get('priceToBook', '未知') # 股價淨值比，金融股或破底股愛用
        
        report = f"【📊 {symbol} 基本面 X 光機】\n"
        report += f"● 近四季 EPS: {eps}\n"
        report += f"● 歷史本益比 (Trailing P/E): {pe}\n"
        report += f"● 預估本益比 (Forward P/E): {fwd_pe}\n"
        report += f"● 股價淨值比 (P/B): {pb}\n"
        
        return report
    except Exception as e:
        return f"基本面數據讀取失敗: {e}"

def get_stock_news(symbol: str) -> str:
    try:
        search_symbol = symbol.upper()
        if "2454_ESOP" in search_symbol: search_symbol = "2454.TW"
        elif search_symbol.isdigit() and len(search_symbol) <= 6: search_symbol += ".TW"
        
        ticker = yf.Ticker(search_symbol)
        news_list = ticker.news
        if not news_list: return f"【📰 {symbol}】Yahoo 端無數據。"

        report = f"【📰 {symbol} 全量情報解析 (10+10 飽和模式)】\n\n"
        count = 0
        
        for i, item in enumerate(news_list[:20]): 
            content = item.get('content', {})
            title = content.get('title') or item.get('title')
            if not title: continue

            # 🚀 數據清洗：去掉換行與前後空白，確保 Payload 最小化
            title = title.replace('\n', ' ').strip()
            publisher = (content.get('provider', {}).get('displayName') or item.get('publisher', '財經媒體')).strip()
            link = content.get('canonicalUrl', {}).get('url') or item.get('link')

            if i < 10:
                # 🛡️ 深度模式：連摘要也要清洗
                summary = content.get('summary') or ""
                # 洗掉換行，並把連續多個空白縮減為一個
                summary_clean = " ".join(summary.split())
                summary_text = (summary_clean[:150] + "...") if len(summary_clean) > 150 else summary_clean
                
                report += f"{i+1}. 🔥 *[{publisher}]* {title}\n"
                if summary_text:
                    report += f"   └ 📝 摘要：{summary_text}\n"
                report += f"   🔗 [Read More]({link})\n\n"
            else:
                # 🛡️ 廣度模式：僅標題
                report += f"{i+1}. ● *[{publisher}]* {title}\n"
            
            count += 1

        # 🚀 最終保護：如果總長度超過 Telegram 上限，截斷它
        if len(report) > 4000:
            report = report[:3950] + "\n\n...(訊息過長，已截斷剩餘部分)"

        return report

    except Exception as e:
        return f"❌ 新聞系統異常: {str(e)[:50]}"
# ==========================================
# 🧠 AI 大腦層與「自動降級機制」
# ==========================================
client = genai.Client(api_key=GEMINI_KEY)

# 引擎優先順序 (先燒最貴的，燒完自動換便宜的)
# 引擎優先順序：從最聰明的燒到最智障的，確保絕不斷線
AVAILABLE_MODELS = [
    'gemini-3.1-pro-preview',        # 狙擊槍：最精準，用來應付你刁鑽的技術分析
    'gemini-3.1-flash-lite-preview', # 衝鋒槍：3.1代輕量版，速度與額度的平衡點
    'gemini-2.5-pro',                # 備用大腦
    'gemini-2.5-flash',              # 主力部隊
    'gemini-2.0-flash-lite',         # 省油燈：額度快乾時的主力
    'gemini-flash-latest'            # 護城河：絕對能跑，這就是剛才 404 的正解！
]
current_model_idx = 0


def create_agent_chat(model_name, history=None):
    return client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            # 👈 加入 get_market_sentiment
        tools=[update_position, get_portfolio_raw_data, get_live_price, 
            get_market_history, calculate_pnl, get_exchange_rate, 
            get_market_sentiment, get_stock_news, get_fundamental_data,
            get_quote_and_orderbook, fubon.get_market_hot_stocks, fubon.get_intraday_trend],
        temperature=0.3, 
        ),
        history=history
    )

# 初始化第一順位引擎
chat = create_agent_chat(AVAILABLE_MODELS[current_model_idx])

# ==========================================
# 🗣️ Telegram 訊息接收、WDT 垃圾話與動態重試系統
# ==========================================

# 假設這些變數與函式已在外部定義
# AVAILABLE_MODELS, WDT_MESSAGES, bot, create_agent_chat, current_model_idx, chat

@bot.message_handler(func=lambda message: True)
def handle_all_text(message):
    global chat  # 🚀 必須宣告全域，不然會變成 local 變數
    user_text = message.text
    
    # --- 1. 決定心情並發送第一句垃圾話 ---
    mood = "normal"
    if any(word in user_text for word in ["損益", "倉位", "賠", "慘", "更改", "修改"]):
        mood = "bad_market"
    elif random.random() < 0.1:
        mood = "bad_market"
    
    wdt_text = random.choice(WDT_MESSAGES[mood])
    sent_msg = bot.reply_to(message, f"【推進器點火中...】\n{wdt_text}")
    bot.send_chat_action(message.chat.id, 'typing')
    
    # --- 2. 取得當下的動態引擎清單 ---
    current_models = get_dynamic_models()

    # --- 3. 進入 AI 思考迴圈 (記憶轉移與無縫降級) ---
    for model_idx, model_name in enumerate(current_models):
        try:
            # 🛡️ 修正 1：精準抓取新版 SDK 的隱藏模型屬性 _model
            current_chat_model = getattr(chat, '_model', getattr(chat, 'model', None))
            
            if current_chat_model != model_name:
                print(f"🔄 模型切換: {current_chat_model} -> {model_name}，正在轉移 Context...")
                
                # 🛡️ 修正 2：改用 .get_history() 提取記憶陣列
                old_history = chat.get_history() if hasattr(chat, 'get_history') else getattr(chat, 'history', None)
                chat = create_agent_chat(model_name, history=old_history)
            
            # 呼叫 Gemini
            response = chat.send_message(user_text)
            
            # --- 【關鍵修正：防斷片安全網】 ---
            final_text = response.text if (response and response.text) else "兄弟，我剛才算到一半突然靈魂出竅，沒吐出東西來。可能是這標的太妖，連我都無語了。你再問一次試試？"
            
            # --- 🎯 處理補刀邏輯 ---
            if mood == "bad_market" and random.random() < 0.3:
                insults = [
                    "\n\n(補刀：我看你這損益，還是先把 Telegram 關掉去寫 C 語言吧。)",
                    "\n\n(提醒：新竹公園的風大，記得帶件厚外套。)",
                    "\n\n(戰友碎念：這操作... 真是讓我大開眼界。)"
                ]
                final_text += random.choice(insults)
            
            # --- 🎯 送出修改訊息 ---
            try:
                bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=sent_msg.message_id,
                    text=final_text,
                    parse_mode='Markdown'
                )
            except Exception as parse_error:
                print(f"⚠️ Markdown 解析失敗，已切換至純文字模式。原因: {parse_error}")
                bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=sent_msg.message_id,
                    text=final_text
                )
            
            return  # 成功回覆，直接跳出

        except Exception as e:
            error_str = str(e).upper()
            
            # 遇到額度或伺服器問題時降級
            if any(key in error_str for key in ['429', 'RESOURCE_EXHAUSTED', 'QUOTA', '404', 'NOT FOUND', '403', '400', 'INVALID', '503', 'UNAVAILABLE']):
                if model_idx + 1 < len(current_models):
                    next_model_name = current_models[model_idx + 1]
                    
                    # 🛡️ 修正 3：降級時的記憶提取也要同步改成 get_history()
                    old_history = chat.get_history() if hasattr(chat, 'get_history') else getattr(chat, 'history', None)
                    chat = create_agent_chat(next_model_name, history=old_history)
                    
                    # 判定錯誤類型
                    if any(k in error_str for k in ['429', 'RESOURCE_EXHAUSTED', 'QUOTA']):
                        reason = "燃料耗盡 (429)"
                    elif any(k in error_str for k in ['503', 'UNAVAILABLE']):
                        reason = "伺服器超載 (503)"
                    else:
                        reason = f"引擎異常 ({error_str[:30]})"

                    bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=sent_msg.message_id,
                        text=f"⚠️ {model_name} {reason}！\n正在轉移記憶並切換至：{next_model_name} ..."
                    )
                    continue  # 帶著新大腦進入下一輪迴圈
                else:
                    bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=sent_msg.message_id,
                        text="兄弟，所有引擎都燒光了！Google 把我們趕出交易室了! 等幾分鐘後再來吧。"
                    )
                    return
            else:
                # 非 API 錯誤 (可能是 Code 寫錯)，直接噴錯
                bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=sent_msg.message_id,
                    text=f"兄弟，我思考迴圈卡死了：\n`{str(e)}`"
                )
                return
        
@bot.message_handler(commands=['reset'])
def reset_memory(message):
    global chat
    # 重新點火，換一個乾淨的大腦
    current_model = get_dynamic_models()[0] 
    chat = create_agent_chat(current_model)
    bot.reply_to(message, "🧹 推進器記憶體已排空！目前大腦已重新裝填，又是新的一天。")           

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    print("🚀 MarginCall Express 終極防護網模式上線！去 Telegram 測試吧。")
    
    # 幫輪詢機制加上超時與重試設定，避免被掛電話後崩潰
    while True:
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e:
            print(f"⚠️ 網路瞬斷或 Telegram 伺服器掛電話 ({e})，3秒後自動重連...")
            time.sleep(3)