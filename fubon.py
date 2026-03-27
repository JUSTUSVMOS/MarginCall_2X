# fubon.py
import os
import json
import pandas as pd
from fubon_neo.sdk import FubonSDK
from fubon_neo.fugle_marketdata.rest.base_rest import FugleAPIError

# 把實體化延後，避免 import 時就連線失敗崩潰
fubon_sdk = None
fubon_ready = False

from datetime import datetime, timedelta

def get_fubon_technical(symbol: str) -> str:
    if not fubon_ready: return "❌ 富邦 SDK 未啟動"
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        today = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        # 1. 抓取 52 週高低與基本報價
        stats = reststock.historical.stats(symbol=symbol)
        h52, l52 = stats.get('week52High', 0), stats.get('week52Low', 0)
        curr = stats.get('closePrice', 0)
        
        # 2. 抓取 RSI (週期改為國際標準 14)
        rsi_data = reststock.technical.rsi(symbol=symbol, timeframe='D', period=14, from_=start_date, to=today)
        rsi = rsi_data.get('data', [])[-1].get('rsi', 0) if rsi_data.get('data') else 0
        
        # 3. 抓取 MACD
        macd_data = reststock.technical.macd(symbol=symbol, timeframe='D', fast=12, slow=26, signal=9, from_=start_date, to=today)
        macd_last = macd_data.get('data', [])[-1] if macd_data.get('data') else {}
        dif, dea = macd_last.get('macdLine', 0), macd_last.get('signalLine', 0)
        macd_hist = (dif - dea) * 2
        
        # 4. 抓取布林通道
        bb_data = reststock.technical.bb(symbol=symbol, timeframe='D', period=20, from_=start_date, to=today)
        bb_last = bb_data.get('data', [])[-1] if bb_data.get('data') else {}
        upper, lower = bb_last.get('upper', 0), bb_last.get('lower', 0)
        
        report = f"🇹🇼 === {symbol} 台股全武裝分析 ===\n"
        report += f"● 現價: {curr} | 52週高: {h52} | 52週低: {l52}\n"
        report += f"● RSI(6): {rsi:.2f} ({'🔥超買' if rsi>75 else '❄️超跌' if rsi<25 else '⚖️中性'})\n"
        report += f"● MACD: DIF:{dif:.2f} | 柱狀體:{macd_hist:.2f} ({'📈多頭增強' if macd_hist>0 else '📉空頭衰退'})\n"
        report += f"● 布林通道: 上軌:{upper:.2f} | 下軌:{lower:.2f}\n"
        
        # 戰術建議
        if curr >= upper: report += "⚠️ 戰略：股價觸及布林上軌，短線噴發過頭，不建議追高。\n"
        elif curr <= lower: report += "🎯 戰略：股價觸及布林下軌，且 RSI 偏低，具備反彈潛力！\n"
        elif rsi < 30: report += "🔥 戰略：RSI 極度超跌，隨時可能暴力反彈。\n"
        else: report += "🧘 戰略：目前位階中性，建議分批佈局或等待關鍵突破。\n"
        
        return report
    except Exception as e: return f"❌ 台股指標獲取失敗: {e}"

def init_fubon():
    """主程式啟動時呼叫這個來連線"""
    global fubon_sdk, fubon_ready
    my_id = os.getenv("FUBON_ID")
    api_key = os.getenv("FUBON_API_KEY")
    cert_pwd = os.getenv("FUBON_CERT_PWD")
    cert_path = "./R124949189.pfx"

    try:
        print(f"🔌 正在連線富邦主機 (ID: {my_id})...")
        # 在這裡才真正建立 SDK 物件
        from fubon_neo.sdk import FubonSDK
        fubon_sdk = FubonSDK()
        
        accounts = fubon_sdk.apikey_login(my_id, api_key, cert_path, cert_pwd)
        if accounts.is_success:
            print("✅ 富邦帳戶登入成功！正在建立即時行情連線...")
            fubon_sdk.init_realtime() 
            fubon_ready = True
            print("🔥 富邦 V8 雙渦輪行情通道啟動完畢！")
        else:
            print(f"❌ 富邦登入失敗: {accounts.message}")
    except Exception as e:
        print(f"⚠️ 富邦 SDK 初始化異常 (可能伺服器維護中): {e}")
        fubon_ready = False
        fubon_sdk = None # 確保失敗時維持 None

# 👇 給 AI 用的 Tool 函數
# 👇 【深層戰術工具】抓取成交明細 (Intraday Trades)
def get_market_trades(symbol: str, limit: int = 20) -> str:
    global fubon_sdk, fubon_ready
    if not fubon_ready: return "⚠️ 富邦引擎未啟動。"
    symbol = symbol.upper().replace('.TW', '').replace('.TWO', '')
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        res = reststock.intraday.trades(symbol=symbol, limit=limit)
        data = res.get('data', []) if isinstance(res, dict) else getattr(res, 'data', [])
        if not data: return f"📊 {symbol} 目前無成交明細。"
        
        report = f"📜 【{symbol} 最近 {len(data)} 筆成交明細】\n"
        for d in data[:limit]:
            price = d.get('price')
            size = d.get('size')
            time_raw = d.get('time', 0)
            # 轉換時間格式 (微秒轉 HH:MM:SS)
            from datetime import datetime
            t_str = datetime.fromtimestamp(time_raw/1000000).strftime('%H:%M:%S')
            report += f"  [{t_str}] 價: {price} | 量: {size}\n"
        return report
    except Exception as e:
        return f"❌ 明細抓取異常: {e}"

# 👇 【深層戰術工具】抓取分價量表 (Intraday Volumes)
def get_price_volumes(symbol: str) -> str:
    global fubon_sdk, fubon_ready
    if not fubon_ready: return "⚠️ 富邦引擎未啟動。"
    symbol = symbol.upper().replace('.TW', '').replace('.TWO', '')
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        res = reststock.intraday.volumes(symbol=symbol)
        data = res.get('data', []) if isinstance(res, dict) else getattr(res, 'data', [])
        if not data: return f"📊 {symbol} 無分價量表數據。"
        
        # 排序：按價格由高到低
        data = sorted(data, key=lambda x: x.get('price', 0), reverse=True)
        report = f"🧱 【{symbol} 分價量表 - 壓力支撐觀測】\n"
        max_vol = max([d.get('volume', 1) for d in data])
        
        for d in data:
            price = d.get('price')
            vol = d.get('volume')
            bar_len = int((vol / max_vol) * 10)
            bar = "█" * bar_len
            report += f"  {price:>7.2f} | {bar} {vol}張\n"
        return report
    except Exception as e:
        return f"❌ 分價量表異常: {e}"

# 👇 【深層戰術工具】抓取 52 週高低價與基本資訊 (Historical Stats)
def get_historical_stats(symbol: str) -> str:
    global fubon_sdk, fubon_ready
    if not fubon_ready: return "⚠️ 富邦引擎未啟動。"
    symbol = symbol.upper().replace('.TW', '').replace('.TWO', '')
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        res = reststock.historical.stats(symbol=symbol)
        
        name = res.get('name', '未知')
        high52 = res.get('week52High', 0)
        low52 = res.get('week52Low', 0)
        curr_close = res.get('closePrice', 0)
        
        report = f"🏛️ 【{symbol} {name} 52週戰略位階】\n"
        report += f"  ● 52週最高: {high52}\n"
        report += f"  ● 52週最低: {low52}\n"
        report += f"  ● 最後收盤: {curr_close}\n"
        
        # 計算目前位階百分比
        pos = ((curr_close - low52) / (high52 - low52)) * 100 if (high52 - low52) != 0 else 0
        report += f"  ● 目前位階: {pos:.1f}% (0%為最低, 100%為最高)\n"
        return report
    except Exception as e:
        return f"❌ 52週數據異常: {e}"

# 把舊的 get_quote_and_orderbook 增強，加入更多總量資訊
def get_quote_and_orderbook(symbol: str) -> str:

    """
    【台股專用】獲取台股個股的即時五檔掛單 (Orderbook) 與買賣力道。
    當用戶詢問「五檔」、「掛單」、「大戶墊單」、「買賣壓」時，必須呼叫此工具。
    """
    global fubon_sdk, fubon_ready

    if not fubon_ready:
        return f"⚠️ 警告：富邦 V8 引擎未啟動。"

    # 🚨 一樣洗掉後綴
    symbol = symbol.upper().replace('.TW', '').replace('.TWO', '')
    
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        quote_data = reststock.intraday.quote(symbol=symbol)
        
        # 增加試撮數據判定 (支援開盤前/收盤前)
        current_price = quote_data.get('closePrice') or quote_data.get('lastPrice', 0)
        trial = quote_data.get('lastTrial', {})
        trial_price = trial.get('price', 0)
        
        status_msg = ""
        if quote_data.get('isTrial'):
            status_msg = f" (⚠️ 目前為試撮階段，價格: {trial_price})"
            current_price = trial_price
        
        bids = quote_data.get('bids', [])
        asks = quote_data.get('asks', [])

        report = f"📊 【{symbol} 即時報價與五檔觀測】\n現價: {current_price}{status_msg}\n\n"
        
        report += "🛑 [上方賣壓牆] (Asks):\n"
        if asks:
            for i, ask in enumerate(reversed(asks[:5])): 
                price = ask.get('price', 0) if isinstance(ask, dict) else getattr(ask, 'price', 0)
                size = ask.get('size', 0) if isinstance(ask, dict) else getattr(ask, 'size', 0)
                report += f"  賣{5-i}: 價格 {price} | 掛單 {size} 張\n"
        else:
            report += "  (目前無賣單數據)\n"
            
        report += "-----------------------\n🛡️ [下方防守牆] (Bids):\n"
        if bids:
            for i, bid in enumerate(bids[:5]):
                price = bid.get('price', 0) if isinstance(bid, dict) else getattr(bid, 'price', 0)
                size = bid.get('size', 0) if isinstance(bid, dict) else getattr(bid, 'size', 0)
                report += f"  買{i+1}: 價格 {price} | 掛單 {size} 張\n"
        else:
            report += "  (目前無買單數據)\n"

        return report

    except FugleAPIError as e:
        return f"❌ 取得 {symbol} 報價失敗 (狀態碼: {e.status_code})"
    except Exception as e:
        return f"❌ 五檔解析異常: {e}"

def get_market_hot_stocks() -> str:
    """
    【LLM 專用：台股資金熱點雷達】
    抓取今天的「成交值排行榜 (大資金)」與「漲幅排行榜 (強勢股)」。
    當用戶問「今天大盤在炒什麼」、「有什麼好標的」、「熱門股」、「換車」時呼叫。
    """
    global fubon_sdk, fubon_ready

    if not fubon_ready:
        return "⚠️ 富邦行情引擎未啟動，無法掃描熱門股。"

    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        
        # 1. 抓成交值排行
        actives = reststock.snapshot.actives(market='TSE', trade='value')
        # 2. 抓漲幅排行
        movers = reststock.snapshot.movers(market='TSE', direction='up', change='percent')

        report = "🔥 【台股資金熱點雷達 (即時快照)】\n\n"
        
        # 解析 Actives (成交值)
        actives_data = actives.get('data', []) if isinstance(actives, dict) else getattr(actives, 'data', [])
        report += "💰 [成交值排行榜] (大戶資金都在這):\n"
        if actives_data:
            for i, stock in enumerate(actives_data[:5]):
                sym = stock.get('symbol', '') if isinstance(stock, dict) else getattr(stock, 'symbol', '')
                name = stock.get('name', '') if isinstance(stock, dict) else getattr(stock, 'name', '')
                price = stock.get('closePrice', stock.get('lastPrice', 0)) if isinstance(stock, dict) else getattr(stock, 'closePrice', getattr(stock, 'lastPrice', 0))
                report += f"  {i+1}. {sym} {name} | 現價: {price}\n"
        else:
            report += "  (目前無成交值數據)\n"
        
        # 解析 Movers (漲幅)
        report += "\n🚀 [漲幅排行榜] (今天的強勢妖股):\n"
        movers_data = movers.get('data', []) if isinstance(movers, dict) else getattr(movers, 'data', [])
        if movers_data:
            for i, stock in enumerate(movers_data[:5]):
                sym = stock.get('symbol', '') if isinstance(stock, dict) else getattr(stock, 'symbol', '')
                name = stock.get('name', '') if isinstance(stock, dict) else getattr(stock, 'name', '')
                change = stock.get('changePercent', 0) if isinstance(stock, dict) else getattr(stock, 'changePercent', 0)
                report += f"  {i+1}. {sym} {name} | 漲幅: +{change}%\n"
        else:
            report += "  (目前無漲幅數據)\n"

        return report

    except FugleAPIError as e:
        return f"❌ 取得熱點雷達失敗 (狀態碼: {e.status_code})"
    except Exception as e:
        return f"❌ 熱點雷達解析異常: {e}"


def get_intraday_trend(symbol: str) -> str:
    """
    【台股專用】獲取台股個股的 5 分鐘 K 線盤中趨勢與量價資料。
    當用戶詢問「5分K」、「盤中趨勢」、「短線」、「今天走勢」時，必須呼叫此工具。
    """
    global fubon_sdk, fubon_ready

    if not fubon_ready:
        return f"⚠️ 警告：富邦 V8 引擎未啟動。"

    # 🚨 洗掉 AI 自作聰明的 Yahoo 後綴
    symbol = symbol.upper().replace('.TW', '').replace('.TWO', '')
    
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        
        # 1. 抓取 5 分 K 原始資料
        candles_data = reststock.intraday.candles(symbol=symbol, timeframe='5')
        
        # 🚨 [除錯專用] 把富邦吐出來的原始格式印出來看 (只印前1000字元避免洗版)
        print(f"👀 【除錯】富邦原始 API 回傳格式 (Raw Data 預覽):")
        if isinstance(candles_data, dict):
            print(json.dumps(candles_data, indent=2, ensure_ascii=False)[:1000] + "\n...(後面省略)")
        else:
            print(str(candles_data)[:1000] + "\n...(後面省略)")
        print("="*50)

        # 2. 提取資料陣列
        data_list = candles_data.get('data', []) if isinstance(candles_data, dict) else getattr(candles_data, 'data', [])
        
        if not data_list:
            return f"📊 【{symbol} 盤中趨勢】目前無 K 線數據 (可能尚未開盤)。"

        # 3. 硬核提取法：確保丟給 Pandas 的一定是乾淨的 Dictionary
        parsed_data = []
        for d in data_list:
            parsed_data.append({
                'date': d.get('date') if isinstance(d, dict) else getattr(d, 'date', ''),
                'open': d.get('open', 0) if isinstance(d, dict) else getattr(d, 'open', 0),
                'high': d.get('high', 0) if isinstance(d, dict) else getattr(d, 'high', 0),
                'low': d.get('low', 0) if isinstance(d, dict) else getattr(d, 'low', 0),
                'close': d.get('close', 0) if isinstance(d, dict) else getattr(d, 'close', 0),
                'volume': d.get('volume', 0) if isinstance(d, dict) else getattr(d, 'volume', 0)
            })

        # 4. 餵給 Pandas 處理
        df = pd.DataFrame(parsed_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        recent_df = df.tail(10)
        
        # 5. 組裝給 AI 的戰情報告
        report = f"📈 【{symbol} 盤中短線趨勢 (最新 {len(recent_df)} 根 5 分 K)】\n"
        report += "(註：時間由舊到新，可觀察底底高或頭頭低)\n\n"
        
        for index, row in recent_df.iterrows():
            time_str = row['date'].strftime('%H:%M')
            k_color = "🔴(紅)" if row['close'] > row['open'] else "🟢(綠)" if row['close'] < row['open'] else "⚪(平)"
            report += f"[{time_str}] {k_color} 開:{row['open']} | 高:{row['high']} | 低:{row['low']} | 收:{row['close']} | 量:{row['volume']}\n"

        avg_close = recent_df['close'].mean()
        last_close = recent_df['close'].iloc[-1]
        trend_status = "多頭強勢 (站上短均)" if last_close > avg_close else "空頭弱勢 (跌破短均)"
        
        report += f"\n💡 近期平均價: {avg_close:.2f} ({trend_status})"

        return report

    except FugleAPIError as e:
        return f"❌ 取得 {symbol} K 線失敗 (狀態碼: {e.status_code})"
    except Exception as e:
        return f"❌ 趨勢解析異常: {e}"
    