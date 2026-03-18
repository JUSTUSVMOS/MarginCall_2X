# fubon.py
import os
import json
import pandas as pd
from fubon_neo.sdk import FubonSDK
from fubon_neo.fugle_marketdata.rest.base_rest import FugleAPIError

# 把實體化宣告在這裡！這才是這個檔案的 global
fubon_sdk = FubonSDK()
fubon_ready = False

def init_fubon():
    """主程式啟動時呼叫這個來連線"""
    global fubon_sdk, fubon_ready
    my_id = os.getenv("FUBON_ID")
    api_key = os.getenv("FUBON_API_KEY")
    cert_pwd = os.getenv("FUBON_CERT_PWD")
    cert_path = ""

    try:
        print(f"🔌 正在連線富邦主機 (ID: {my_id})...")
        accounts = fubon_sdk.apikey_login(my_id, api_key, cert_path, cert_pwd)
        if accounts.is_success:
            print("✅ 富邦帳戶登入成功！正在建立即時行情連線...")
            fubon_sdk.init_realtime() 
            fubon_ready = True
            print("🔥 富邦 V8 雙渦輪行情通道啟動完畢！")
        else:
            print(f"❌ 富邦登入失敗: {accounts.message}")
    except Exception as e:
        print(f"❌ 富邦 SDK 初始化異常: {e}")

# 👇 給 AI 用的 Tool 函數
def get_quote_and_orderbook(symbol: str) -> str:
    global fubon_sdk, fubon_ready

    if not fubon_ready:
        return f"⚠️ 警告：富邦 V8 引擎未啟動。"

    symbol = symbol.upper()
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        quote_data = reststock.intraday.quote(symbol=symbol)
        
        is_dict = isinstance(quote_data, dict)
        current_price = quote_data.get('closePrice', quote_data.get('lastPrice', 0)) if is_dict else getattr(quote_data, 'closePrice', getattr(quote_data, 'lastPrice', 0))
        
        bids = quote_data.get('bids', []) if is_dict else getattr(quote_data, 'bids', [])
        asks = quote_data.get('asks', []) if is_dict else getattr(quote_data, 'asks', [])

        report = f"📊 【{symbol} 即時報價與五檔觀測】\n現價: {current_price}\n\n"
        
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
    global fubon_sdk, fubon_ready

    if not fubon_ready:
        return f"⚠️ 警告：富邦 V8 引擎未啟動。"

    symbol = symbol.upper()
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
    