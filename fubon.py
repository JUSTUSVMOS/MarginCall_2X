# fubon.py
import os
import json
import logging
import pandas as pd
from fubon_neo.sdk import FubonSDK
from fubon_neo.fugle_marketdata.rest.base_rest import FugleAPIError
from src.tools import tool

logger = logging.getLogger(__name__)

# 把實體化延後，避免 import 時就連線失敗崩潰
fubon_sdk = None
fubon_ready = False

from datetime import datetime, timedelta

def get_fubon_technical(symbol: str) -> str:
    if not fubon_ready: return "❌ 富邦 SDK 未啟動"
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        today_dt = datetime.now()
        today = today_dt.strftime('%Y-%m-%d')
        # 如果是周六或周日，把 today 往後推一天確保 API 不會報錯 (某些 API 要求的 to 必須包含最新交易日)
        # 或者確保 start_date 至少比 today 早
        start_date = (today_dt - timedelta(days=120)).strftime('%Y-%m-%d')
        
        # 1. 抓取 52 週高低與基本報價
        try:
            stats = reststock.historical.stats(symbol=symbol)
        except Exception as e:
            logger.warning(f"Stats fetch failed: {e}")
            stats = {}

        h52, l52 = stats.get('week52High', 0), stats.get('week52Low', 0)
        curr = stats.get('closePrice', 0)

        # 補抓即時報價，避免盤中拿到昨收
        try:
            quote_data = reststock.intraday.quote(symbol=symbol)
            live_price = quote_data.get('closePrice') or quote_data.get('lastPrice', 0)
            if live_price:
                curr = live_price
        except Exception as e:
            logger.warning(f"Quote fetch failed in technical: {e}")

        # 1.5 抓取分價量表 (Volume Profile)
        poc_price = 0
        vp_status = "N/A"
        net_buy_at_poc = 0
        try:
            vol_data = reststock.intraday.volumes(symbol=symbol)
            if vol_data and vol_data.get('data'):
                # 找出成交量最大的價格點 (POC)
                poc_item = max(vol_data['data'], key=lambda x: x['volume'])
                poc_price = poc_item['price']
                vp_status = "🛡️ 支撐" if curr > poc_price else "🧱 壓力"
                # 計算 POC 價位的積極買賣力道 (外盤 - 內盤)
                net_buy_at_poc = poc_item.get('volumeAtAsk', 0) - poc_item.get('volumeAtBid', 0)
        except Exception as e:
            logger.warning(f"Volume Profile fetch failed: {e}")
        
        # 2. 抓取歷史 K 線並計算 MA20, MA60
        # 確保 to >= from
        to_date = today
        from_date = start_date
        ma20, ma60 = 0, 0
        try:
            candles = reststock.historical.candles(symbol=symbol, to=to_date, **{'from': from_date})
            if candles.get('data'):
                df_hist = pd.DataFrame(candles['data'])
                # 確保按日期升冪排序計算均線
                df_hist['date'] = pd.to_datetime(df_hist['date'])
                df_hist = df_hist.sort_values('date', ascending=True)
                df_hist['ma20'] = df_hist['close'].rolling(window=20).mean()
                df_hist['ma60'] = df_hist['close'].rolling(window=60).mean()
                ma20 = df_hist['ma20'].iloc[-1] if len(df_hist) >= 20 else 0
                ma60 = df_hist['ma60'].iloc[-1] if len(df_hist) >= 60 else 0
        except Exception as e:
            logger.warning(f"Candles fetch failed: {e}")

        # 3. 抓取 RSI (週期改為國際標準 14)
        rsi = 0
        try:
            rsi_data = reststock.technical.rsi(symbol=symbol, timeframe='D', period=14, to=to_date, **{'from': from_date})
            rsi = rsi_data.get('data', [])[-1].get('rsi', 0) if rsi_data.get('data') else 0
        except Exception as e:
            logger.warning(f"RSI fetch failed: {e}")
        
        # 4. 抓取 MACD
        dif, dea, macd_hist = 0, 0, 0
        try:
            macd_data = reststock.technical.macd(symbol=symbol, timeframe='D', fast=12, slow=26, signal=9, to=to_date, **{'from': from_date})
            macd_last = macd_data.get('data', [])[-1] if macd_data.get('data') else {}
            dif, dea = macd_last.get('macdLine', 0), macd_last.get('signalLine', 0)
            macd_hist = dif - dea 
        except Exception as e:
            logger.warning(f"MACD fetch failed: {e}")
        
        # 5. 抓取布林通道
        upper, lower = 0, 0
        try:
            bb_data = reststock.technical.bb(symbol=symbol, timeframe='D', period=20, to=to_date, **{'from': from_date})
            bb_last = bb_data.get('data', [])[-1] if bb_data.get('data') else {}
            upper, lower = bb_last.get('upper', 0), bb_last.get('lower', 0)
        except Exception as e:
            logger.warning(f"BB fetch failed: {e}")

        # 6. 抓取 KDJ (9, 3, 3)
        vk, vd, vj = 50, 50, 50
        try:
            kdj_data = reststock.technical.kdj(symbol=symbol, to=to_date, timeframe='D', rPeriod=9, kPeriod=3, dPeriod=3, **{'from': from_date})
            kdj_last = kdj_data.get('data', [])[-1] if kdj_data.get('data') else {}
            vk, vd, vj = kdj_last.get('k', 50), kdj_last.get('d', 50), kdj_last.get('j', 50)
        except Exception as e:
            logger.warning(f"KDJ fetch failed: {e}")
        
        report = f"🇹🇼 === {symbol} 台股全武裝分析 ===\n"
        report += f"● 現價: {curr} | 52週高: {h52} | 52週低: {l52}\n"
        report += f"● 籌碼密集區 (POC): {poc_price} ({vp_status}) | 推力: {net_buy_at_poc:+} \n"
        report += f"● 均線位階: MA20:{ma20:.2f} | MA60:{ma60:.2f}\n"
        report += f"● KDJ(9,3,3): K:{vk:.1f} | D:{vd:.1f} | J:{vj:.1f}\n"
        report += f"● RSI(14): {rsi:.2f} ({'🔥極度超買' if rsi>75 else '❄️極度超跌' if rsi<25 else '⚖️中性'})\n"
        report += f"● MACD: DIF:{dif:.2f} | 柱狀體:{macd_hist:.2f} ({'📈多頭增強' if macd_hist>0 else '📉空頭衰退'})\n"
        report += f"● 布林通道: 上軌:{upper:.2f} | 下軌:{lower:.2f}\n"
        
        # 戰術建議
        if curr >= upper: report += "⚠️ 戰略：股價觸及布林上軌，短線噴發過頭，不建議追高。\n"
        elif curr <= lower: report += "🎯 戰略：股價觸及布林下軌，且 RSI 偏低，具備反彈潛力！\n"
        elif vk > vd and vk < 30: report += f"🚀 戰略：KDJ 低檔金叉 (K:{vk:.1f})，轉折噴發信號！\n"
        elif vk < vd and vk > 70: report += f"🥀 戰略：KDJ 高檔死叉 (K:{vk:.1f})，波段見頂信號。\n"
        elif vj > 100: report += "🔥 戰略：J 線噴發過度，留意隨時拉回。\n"
        elif vj < 0: report += "❄️ 戰略：J 線極度耗竭，空頭殺過頭，反彈將至。\n"
        elif vp_status == "🛡️ 支撐" and curr > poc_price and net_buy_at_poc > 0: report += f"💎 戰略：站穩 POC 密集區 ({poc_price}) 且積極買盤支撐，結構強韌。\n"
        elif vp_status == "🧱 壓力" and net_buy_at_poc < 0: report += f"🧱 戰略：上方 POC ({poc_price}) 賣壓沉重，主動賣盤強勁，留意拉回。\n"
        elif curr > ma20 and curr > ma60: report += "📈 戰略：股價站上月線與季線，多頭排列建立，回檔即買點。\n"
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
        logger.info(f"🔌 正在連線富邦主機 (ID: {my_id})...")
        # 在這裡才真正建立 SDK 物件
        from fubon_neo.sdk import FubonSDK
        fubon_sdk = FubonSDK()
        
        accounts = fubon_sdk.apikey_login(my_id, api_key, cert_path, cert_pwd)
        if accounts.is_success:
            logger.info("✅ 富邦帳戶登入成功！正在建立即時行情連線...")
            fubon_sdk.init_realtime() 
            fubon_ready = True
            logger.info("🔥 富邦 V8 雙渦輪行情通道啟動完畢！")
        else:
            logger.warning(f"❌ 富邦登入失敗: {accounts.message}")
    except Exception as e:
        logger.warning(f"⚠️ 富邦 SDK 初始化異常 (可能伺服器維護中): {e}")
        fubon_ready = False
        fubon_sdk = None
def get_fubon_inventories():
    """【深度掃描】獲取富邦實體帳戶庫存與成本字典 {symbol: {'shares': qty, 'cost': price}}"""
    global fubon_sdk, fubon_ready
    if not fubon_ready: return {}
    try:
        my_id = os.getenv("FUBON_ID")
        api_key = os.getenv("FUBON_API_KEY")
        cert_pwd = os.getenv("FUBON_CERT_PWD")
        cert_path = "./R124949189.pfx"

        login_res = fubon_sdk.apikey_login(my_id, api_key, cert_path, cert_pwd)
        if not login_res.is_success or not login_res.data:
            return {}

        acc = login_res.data[0]
        inventory_map = {}

        # 1. 從 inventories 抓取基礎庫存
        inv_res = fubon_sdk.accounting.inventories(acc)
        if inv_res.is_success:
            for item in inv_res.data:
                if item.today_qty > 0:
                    inventory_map[item.stock_no] = {'shares': item.today_qty, 'cost': 0.0}

        # 2. 從未實現損益補完數據 (包含 FMP 抓不到的成本資訊)
        unreal_res = fubon_sdk.accounting.unrealized_gains_and_loses(acc)
        if unreal_res.is_success:
            for item in unreal_res.data:
                s = item.stock_no
                qty = getattr(item, 'today_qty', 0)
                # 嘗試多種可能的成本欄位名稱
                cost = getattr(item, 'cost_price', getattr(item, 'buy_price', getattr(item, 'price_avg', 0.0)))

                if s in inventory_map:
                    inventory_map[s]['cost'] = cost
                elif qty > 0:
                    inventory_map[s] = {'shares': qty, 'cost': cost}

        return inventory_map
    except Exception as e:
        logger.warning(f"Fubon data fetch error: {e}")
        return {}


def get_fubon_bank_remain():
    """獲取富邦銀行可用餘額 (TWD)"""
    global fubon_sdk, fubon_ready
    if not fubon_ready: return None
    try:
        my_id = os.getenv("FUBON_ID")
        api_key = os.getenv("FUBON_API_KEY")
        cert_pwd = os.getenv("FUBON_CERT_PWD")
        cert_path = "./R124949189.pfx"
        login_res = fubon_sdk.apikey_login(my_id, api_key, cert_path, cert_pwd)
        if login_res.is_success and login_res.data:
            acc = login_res.data[0]
            res = fubon_sdk.accounting.bank_remain(acc)
            if res.is_success:
                # 確保回傳整數
                return int(res.data.available_balance)
        return None
    except Exception as e:
        logger.warning(f"Fubon data fetch error: {e}")
        return None

def _normalize_tw_symbol(symbol: str) -> str:
    clean_sym = symbol.upper().replace('.TW', '').replace('.TWO', '')
    # 簡單防呆：台股代碼 (含期權) 必然包含數字，若全為英文字母 (如 CRWV, AAPL)，直接擋下
    if not any(char.isdigit() for char in clean_sym):
        raise ValueError(f"❌ 此工具僅支援台股 (Taiwan Stocks)，美股 {symbol} 請改用美股專用工具。")
    return clean_sym


def _get_stock_rest_client():
    return fubon_sdk.marketdata.rest_client.stock


def _get_futopt_rest_client():
    return fubon_sdk.marketdata.rest_client.futopt


def build_market_trades_report(symbol: str, limit: int = 20) -> str:
    """Pure intraday-trades logic for direct callers and tests."""
    global fubon_ready
    if not fubon_ready:
        return "⚠️ 富邦引擎未啟動。"
    try:
        symbol = _normalize_tw_symbol(symbol)
        reststock = _get_stock_rest_client()
        res = reststock.intraday.trades(symbol=symbol, limit=limit)
        data = res.get('data', []) if isinstance(res, dict) else getattr(res, 'data', [])
        if not data:
            return f"📊 {symbol} 目前無成交明細。"

        report = f"📜 【{symbol} 最近 {len(data)} 筆成交明細】\n"
        for d in data[:limit]:
            price = d.get('price')
            size = d.get('size')
            time_raw = d.get('time', 0)
            t_str = datetime.fromtimestamp(time_raw / 1000000).strftime('%H:%M:%S')
            report += f"  [{t_str}] 價: {price} | 量: {size}\n"
        return report
    except Exception as e:
        logger.warning(f"Market trades fetch failed for {symbol}: {e}")
        return f"❌ 明細抓取異常: {e}"


def build_price_volumes_report(symbol: str) -> str:
    """Pure volume-profile logic for direct callers and tests."""
    global fubon_ready
    if not fubon_ready:
        return "⚠️ 富邦引擎未啟動。"
    try:
        symbol = _normalize_tw_symbol(symbol)
        reststock = _get_stock_rest_client()
        res = reststock.intraday.volumes(symbol=symbol)
        data = res.get('data', []) if isinstance(res, dict) else getattr(res, 'data', [])
        if not data:
            return f"📊 {symbol} 無分價量表數據。"

        data = sorted(data, key=lambda x: x.get('price', 0), reverse=True)
        report = f"🧱 【{symbol} 分價量表 - 壓力支撐觀測】\n"
        max_vol = max((d.get('volume', 1) for d in data), default=1) or 1

        for d in data:
            price = d.get('price')
            vol = d.get('volume')
            bar_len = int((vol / max_vol) * 10)
            bar = "█" * bar_len
            report += f"  {price:>7.2f} | {bar} {vol}張\n"
        return report
    except Exception as e:
        logger.warning(f"Price-volume fetch failed for {symbol}: {e}")
        return f"❌ 分價量表異常: {e}"


def build_historical_stats_report(symbol: str) -> str:
    """Pure historical-stats logic for direct callers and tests."""
    global fubon_ready
    if not fubon_ready:
        return "⚠️ 富邦引擎未啟動。"
    try:
        symbol = _normalize_tw_symbol(symbol)
        reststock = _get_stock_rest_client()
        res = reststock.historical.stats(symbol=symbol)

        name = res.get('name', '未知')
        high52 = res.get('week52High', 0)
        low52 = res.get('week52Low', 0)
        curr_close = res.get('closePrice', 0)

        # 補抓即時報價，避免盤中拿到昨收
        try:
            quote_data = reststock.intraday.quote(symbol=symbol)
            live_price = quote_data.get('closePrice') or quote_data.get('lastPrice', 0)
            if live_price:
                curr_close = live_price
        except Exception as e:
            logger.warning(f"Quote fetch failed in historical stats: {e}")

        report = f"🏛️ 【{symbol} {name} 52週戰略位階】\n"
        report += f"  ● 52週最高: {high52}\n"
        report += f"  ● 52週最低: {low52}\n"
        report += f"  ● 目前現價/收盤: {curr_close}\n"

        pos = ((curr_close - low52) / (high52 - low52)) * 100 if (high52 - low52) != 0 else 0
        report += f"  ● 目前位階: {pos:.1f}% (0%為最低, 100%為最高)\n"
        return report
    except Exception as e:
        logger.warning(f"Historical stats fetch failed for {symbol}: {e}")
        return f"❌ 52週數據異常: {e}"


def build_txo_sentiment_report() -> str:
    """Pure TXO-sentiment logic for direct callers and tests."""
    global fubon_ready
    if not fubon_ready:
        return "⚠️ 富邦引擎未啟動，無法抓取 TXO 數據。"

    try:
        restfut = _get_futopt_rest_client()
        res = restfut.snapshot.actives(market='TFE', trade='volume')
        data = res.get('data', []) if isinstance(res, dict) else getattr(res, 'data', [])

        if not data:
            return "📊 TXO 目前無交易數據。"

        calls_vol = 0
        puts_vol = 0
        report = "🔮 【TXO 台指期權戰報 - 市場情緒觀測】\n"

        for item in data:
            sym = (item.get('symbol', '') if isinstance(item, dict) else getattr(item, 'symbol', '')).upper()
            vol = item.get('volume', 0) if isinstance(item, dict) else getattr(item, 'volume', 0)

            if 'TXO' in sym:
                if 'C' in sym:
                    calls_vol += vol
                elif 'P' in sym:
                    puts_vol += vol
                elif len(sym) >= 10:
                    month_code = sym[-2]
                    if month_code in 'ABCDEFGHIJKL':
                        calls_vol += vol
                    elif month_code in 'MNOPQRSTUVWX':
                        puts_vol += vol

        pc_ratio = (puts_vol / calls_vol) if calls_vol > 0 else 0
        pc_label = "🔴 極度看空" if pc_ratio > 1.2 else "🟡 避險轉強" if pc_ratio > 1.0 else "🟢 多頭佔優" if pc_ratio < 0.8 else "⚖️ 中性偏多"

        report += f"  ● 今日熱門合約 P/C Ratio: {pc_ratio:.2f} ({pc_label})\n"
        report += f"  ● Call 總量: {calls_vol} | Put 總量: {puts_vol}\n"
        report += "\n💡 戰略提示：若 P/C Ratio 持續拉升，代表台股大盤壓力沉重，留意台積電是否同步走軟。"
        return report
    except Exception as e:
        logger.warning(f"TXO sentiment fetch failed: {e}")
        return f"❌ TXO 數據抓取失敗: {e} (請確認帳號具備期貨權限)"


def build_quote_and_orderbook_report(symbol: str) -> str:
    """Pure orderbook logic for direct callers and tests."""
    global fubon_ready
    if not fubon_ready:
        return "⚠️ 警告：富邦 V8 引擎未啟動。"

    try:
        symbol = _normalize_tw_symbol(symbol)
        reststock = _get_stock_rest_client()
        quote_data = reststock.intraday.quote(symbol=symbol)

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
        logger.warning(f"Quote/orderbook fetch failed for {symbol}: {e}")
        return f"❌ 取得 {symbol} 報價失敗 (狀態碼: {e.status_code})"
    except Exception as e:
        logger.warning(f"Quote/orderbook parse failed for {symbol}: {e}")
        return f"❌ 五檔解析異常: {e}"


def build_market_hot_stocks_report() -> str:
    """Pure hot-stocks logic for direct callers and tests."""
    global fubon_ready
    if not fubon_ready:
        return "⚠️ 富邦行情引擎未啟動，無法掃描熱門股。"

    try:
        reststock = _get_stock_rest_client()
        actives = reststock.snapshot.actives(market='TSE', trade='value')
        movers = reststock.snapshot.movers(market='TSE', direction='up', change='percent')

        report = "🔥 【台股資金熱點雷達 (即時快照)】\n\n"

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
        logger.warning(f"Hot stocks fetch failed: {e}")
        return f"❌ 取得熱點雷達失敗 (狀態碼: {e.status_code})"
    except Exception as e:
        logger.warning(f"Hot stocks parse failed: {e}")
        return f"❌ 熱點雷達解析異常: {e}"


def build_intraday_trend_report(symbol: str) -> str:
    """Pure intraday-trend logic for direct callers and tests."""
    global fubon_ready
    if not fubon_ready:
        return "⚠️ 警告：富邦 V8 引擎未啟動。"

    try:
        symbol = _normalize_tw_symbol(symbol)
        reststock = _get_stock_rest_client()
        candles_data = reststock.intraday.candles(symbol=symbol, timeframe='5')

        logger.info("👀 【除錯】富邦原始 API 回傳格式 (Raw Data 預覽):")
        if isinstance(candles_data, dict):
            logger.info("%s", json.dumps(candles_data, indent=2, ensure_ascii=False)[:1000] + "\n...(後面省略)")
        else:
            logger.info("%s", str(candles_data)[:1000] + "\n...(後面省略)")
        logger.info("%s", "=" * 50)

        data_list = candles_data.get('data', []) if isinstance(candles_data, dict) else getattr(candles_data, 'data', [])
        if not data_list:
            return f"📊 【{symbol} 盤中趨勢】目前無 K 線數據 (可能尚未開盤)。"

        parsed_data = []
        for d in data_list:
            parsed_data.append({
                'date': d.get('date') if isinstance(d, dict) else getattr(d, 'date', ''),
                'open': d.get('open', 0) if isinstance(d, dict) else getattr(d, 'open', 0),
                'high': d.get('high', 0) if isinstance(d, dict) else getattr(d, 'high', 0),
                'low': d.get('low', 0) if isinstance(d, dict) else getattr(d, 'low', 0),
                'close': d.get('close', 0) if isinstance(d, dict) else getattr(d, 'close', 0),
                'volume': d.get('volume', 0) if isinstance(d, dict) else getattr(d, 'volume', 0),
            })

        df = pd.DataFrame(parsed_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        recent_df = df.tail(10)
        report = f"📈 【{symbol} 盤中短線趨勢 (最新 {len(recent_df)} 根 5 分 K)】\n"
        report += "(註：時間由舊到新，可觀察底底高或頭頭低)\n\n"

        for _, row in recent_df.iterrows():
            time_str = row['date'].strftime('%H:%M')
            k_color = "🔴(紅)" if row['close'] > row['open'] else "🟢(綠)" if row['close'] < row['open'] else "⚪(平)"
            report += f"[{time_str}] {k_color} 開:{row['open']} | 高:{row['high']} | 低:{row['low']} | 收:{row['close']} | 量:{row['volume']}\n"

        avg_close = recent_df['close'].mean()
        last_close = recent_df['close'].iloc[-1]
        trend_status = "多頭強勢 (站上短均)" if last_close > avg_close else "空頭弱勢 (跌破短均)"
        report += f"\n💡 近期平均價: {avg_close:.2f} ({trend_status})"
        return report
    except FugleAPIError as e:
        logger.warning(f"Intraday trend fetch failed for {symbol}: {e}")
        return f"❌ 取得 {symbol} K 線失敗 (狀態碼: {e.status_code})"
    except Exception as e:
        logger.warning(f"Intraday trend parse failed for {symbol}: {e}")
        return f"❌ 趨勢解析異常: {e}"

# 👇 給 AI 用的 Tool 函數
# 👇 【深層戰術工具】抓取成交明細 (Intraday Trades)
@tool()
def get_market_trades(symbol: str, limit: int = 20) -> str:
    """
    Fetches the most recent intraday trade details (price and size) for a Taiwan stock.
    """
    return build_market_trades_report(symbol, limit)

# 👇 【深層戰術工具】抓取分價量表 (Intraday Volumes)
@tool()
def get_price_volumes(symbol: str) -> str:
    """
    Retrieves the volume profile (intraday volumes at price) for a Taiwan stock.
    Helps identify support and resistance levels based on volume concentration.
    """
    return build_price_volumes_report(symbol)

# 👇 【深層戰術工具】抓取 52 週高低價與基本資訊 (Historical Stats)
@tool()
def get_historical_stats(symbol: str) -> str:
    """
    Provides 52-week high/low prices and current percentile rank for a Taiwan stock.
    """
    return build_historical_stats_report(symbol)

# 👇 【TXO 期權戰術工具】抓取台指期權 (TXO) 戰報
@tool()
def get_txo_sentiment() -> str:
    """
    Calculates the Put/Call Ratio and market sentiment from Taiwan Index Options (TXO).
    A key indicator for detecting overall market direction and big player positioning.
    """
    return build_txo_sentiment_report()

# 把舊的 get_quote_and_orderbook 增強，加入更多總量資訊
@tool()
def get_quote_and_orderbook(symbol: str) -> str:

    """
    Fetches real-time bid/ask orderbook (Level 2) and recent price for a Taiwan stock.
    Used to analyze immediate supply/demand pressure and large order positioning.
    """
    return build_quote_and_orderbook_report(symbol)

@tool()
def get_market_hot_stocks() -> str:
    """
    Identifies Taiwan market heatspots by scanning for top stocks by trading value and percentage gain.
    Helps detect where the "big money" is moving during the trading session.
    """
    return build_market_hot_stocks_report()


@tool()
def get_intraday_trend(symbol: str) -> str:
    """
    Fetches 5-minute K-line data for a Taiwan stock to analyze intraday price and volume trends.
    Useful for short-term tactical decisions and monitoring momentum.
    """
    return build_intraday_trend_report(symbol)

def get_exhaustion_analysis(symbol: str) -> str:
    """
    【深層戰術工具：賣盤衰竭偵測 (ExhaustionScanner)】
    分析台股個股是否出現「賣不動」的信號。
    結合：
    1. 盤口掛單力道 (Bid/Ask Ratio)
    2. 冰山單吸收 (POC Volume vs Net Buy)
    3. 成交效率 (Tick Efficiency - 高量不跌)
    4. 技術超跌 (RSI/KDJ-J)
    """
    global fubon_sdk, fubon_ready
    if not fubon_ready: return "⚠️ 富邦引擎未啟動。"
    
    symbol = symbol.upper().replace('.TW', '').replace('.TWO', '')
    score = 0
    reasons = []
    
    try:
        reststock = fubon_sdk.marketdata.rest_client.stock
        
        # 1. 技術指標 (從現成工具抓取結果並解析)
        tech_report = get_fubon_technical(symbol)
        if "❄️極度超跌" in tech_report or "RSI < 25" in tech_report:
            score += 25
            reasons.append("✅ [超跌信號] RSI 進入極度超跌區，賣壓動能已釋放過度。")
        if "J 線極度耗竭" in tech_report:
            score += 15
            reasons.append("✅ [動能耗竭] KDJ-J 線 < 0，顯示空頭殺過頭，轉折將至。")
        if "觸及布林下軌" in tech_report:
            score += 10
            reasons.append("✅ [軌道支撐] 股價打入布林下軌，進入價值支撐區。")

        # 2. 盤口掛單 (Order Book)
        quote_res = reststock.intraday.quote(symbol=symbol)
        bids = quote_res.get('bids', [])
        asks = quote_res.get('asks', [])
        if bids and asks:
            total_bid_size = sum([b.get('size', 0) for b in bids])
            total_ask_size = sum([a.get('size', 0) for a in asks])
            ratio = total_bid_size / total_ask_size if total_ask_size > 0 else 10
            if ratio > 2.0:
                score += 15
                reasons.append(f"✅ [掛單支撐] 買盤掛單量是賣盤的 {ratio:.1f} 倍，下方墊單積極。")
            elif ratio > 1.5:
                score += 5

        # 3. 分價量表 (Volume Profile / POC)
        vol_res = reststock.intraday.volumes(symbol=symbol)
        data = vol_res.get('data', [])
        if data:
            poc_item = max(data, key=lambda x: x.get('volume', 0))
            poc_price = poc_item.get('price')
            curr_price = quote_res.get('closePrice') or quote_res.get('lastPrice', 0)
            
            if abs(curr_price - poc_price) / poc_price < 0.005: # Price at POC
                net_buy_at_poc = poc_item.get('volumeAtAsk', 0) - poc_item.get('volumeAtBid', 0)
                if net_buy_at_poc < 0: # 主動賣單多但價格撐住
                    score += 20
                    reasons.append(f"💎 [冰山單偵測] POC 價位 {poc_price} 出現大量主動賣單但價格未跌破，疑似有大戶吸收買盤。")
                else:
                    score += 10
                    reasons.append(f"✅ [籌碼密集] 股價回到今日成交最密集的 POC 區 ({poc_price})，具備支撐。")

        # 4. 成交明細效率 (Tick Efficiency)
        trades_res = reststock.intraday.trades(symbol=symbol, limit=50)
        t_data = trades_res.get('data', [])
        if len(t_data) >= 20:
            recent = t_data[:20]
            p_change = abs(recent[0].get('price', 0) - recent[-1].get('price', 0))
            v_total = sum([t.get('size', 0) for t in recent])
            efficiency = p_change / v_total if v_total > 0 else 0
            if efficiency < 0.001 and v_total > 50:
                score += 15
                reasons.append(f"✅ [賣壓衰竭] 近期成交 {v_total} 張，但價格波動極小 ({p_change})，賣方推動力道消失。")

        # 總結
        status = "🔴 賣壓仍重"
        if score >= 70: status = "🔥 極度衰竭 (底部形成中)"
        elif score >= 50: status = "🟢 賣盤衰竭 (分批佈局)"
        elif score >= 30: status = "🟡 賣壓減緩"
        
        report = f"🕵️ 【{symbol} 賣盤衰竭偵測報告】\n"
        report += f"📊 綜合評分: {score}/100 | 當前狀態: {status}\n"
        report += "------------------------------------------\n"
        report += "\n".join(reasons) if reasons else "⌛ 目前未偵測到明顯的賣盤衰竭信號。"
        report += "\n\n💡 建議：此工具偵測的是「賣方賣不動」的瞬間，仍需觀察 5 分 K 是否站上均線確認轉折。"
        
        return report

    except Exception as e:
        return f"❌ 衰竭掃描失敗: {e}"
    
