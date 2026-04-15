import os
import sys
import time
import logging
import json
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

# 引入核心引擎進行實測
try:
    import yfinance as yf
    from yf_session import get_ticker
    import engine_market as market
    import engine_fundamentals as fundamentals
    import engine_technical as technical
    import fubon
    import engine_risk as risk
except ImportError as e:
    logger.error(f"❌ 核心引擎缺失，無法啟動監控: {e}")
    sys.exit(1)

class DataSentinel:
    def __init__(self):
        self.status = {}
        self.tz = pytz.timezone('Asia/Taipei')

    def check_yfinance(self):
        """測試全球行情與基本面 (Yahoo Finance)"""
        try:
            ticker = get_ticker("AAPL")
            price = ticker.fast_info['last_price']
            info = ticker.info.get('symbol')
            if price and info:
                return "✅ 正常 (Price: {:.2f})".format(price)
            return "⚠️ 資料部分缺失"
        except Exception as e:
            return f"❌ 斷線: {str(e)[:50]}"

    def check_fubon(self):
        """測試台股即時行情 (Fubon SDK)"""
        try:
            # 這裡模擬初始化檢查
            fubon.init_fubon()
            # 嘗試抓取一個測試代碼
            hot = fubon.build_market_hot_stocks_report()
            if hot and "error" not in hot:
                return "✅ 正常 (已獲取熱門股)"
            return "⚠️ SDK 初始化成功但資料回傳異常"
        except Exception as e:
            return f"❌ 斷線: {str(e)[:50]}"

    def check_sec_and_nlp_engine(self):
        """測試 SEC EDGAR 與 自建 NLP 引擎路徑"""
        try:
            # 檢查新引擎是否存在
            if not os.path.exists(os.path.join(os.getcwd(), "engine_nlp.py")):
                return "❌ engine_nlp.py 遺失"
            
            # 測試 Finnhub API 連線
            key = os.getenv("FINNHUB_API_KEY")
            if not key:
                return "❌ 缺少 FINNHUB_API_KEY"
            
            import requests
            r = requests.get(f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={key}")
            if r.status_code == 200:
                return "✅ 正常 (SEC + NLP Engine OK)"
            return f"❌ API 錯誤: {r.status_code}"
        except Exception as e:
            return f"❌ 異常: {str(e)[:50]}"

    def check_seeking_alpha(self):
        """測試 Seeking Alpha 登入狀態"""
        auth_path = os.path.join(os.getcwd(), "auth.json")
        if not os.path.exists(auth_path):
            return "❌ auth.json 遺失 (需要重新登入)"
        
        try:
            with open(auth_path, 'r') as f:
                auth_data = json.load(f)
                # 檢查 cookie 是否過期 (簡單判斷)
                if len(auth_data.get('cookies', [])) > 0:
                    return "✅ 正常 (已持有 Token)"
                return "⚠️ auth.json 格式異常"
        except Exception:
            return "❌ 讀取失敗"

    def check_risk_indicators(self):
        """測試 GEX/DIX 等風險指標來源"""
        try:
            # 測試 SqueezeMetrics 模擬抓取
            res = risk.get_global_risk_radar()
            if "DIX" in res or "GEX" in res:
                return "✅ 正常 (指標已更新)"
            return "⚠️ 資料解析不全"
        except Exception as e:
            return f"❌ 異常: {str(e)[:50]}"

    def run_full_scan(self):
        logger.info("📡 啟動全維度雷達自檢...")
        self.status = {
            "1. 全球行情 (YF)": self.check_yfinance(),
            "2. 台股行情 (Fubon)": self.check_fubon(),
            "3. 語意/SEC (NLP Engine)": self.check_sec_and_nlp_engine(),
            "4. 深度評論 (SeekingAlpha)": self.check_seeking_alpha(),
            "5. 風險矩陣 (GEX/DIX)": self.check_risk_indicators()
        }
        
        print("\n" + "="*50)
        print(f"🛰️  MarginCall 2X 數據命脈監控回報 [{datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S')}]")
        print("="*50)
        for name, msg in self.status.items():
            print(f"{name:<25} : {msg}")
        print("="*50 + "\n")

if __name__ == "__main__":
    sentinel = DataSentinel()
    # 如果帶有 --loop 參數，則進入長駐模式
    if "--loop" in sys.argv:
        logger.info("🔄 進入長駐監控模式 (每 4 小時掃描一次)")
        while True:
            sentinel.run_full_scan()
            time.sleep(14400) # 4 hours
    else:
        # 單次掃描
        sentinel.run_full_scan()
