import yfinance as yf
import sqlite3
import pandas as pd
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# --- 緩存路徑設定 ---
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
DB_DAILY = CACHE_DIR / "yf_cache_daily.sqlite"
DB_HOURLY = CACHE_DIR / "yf_cache_hourly.sqlite"

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            symbol TEXT,
            interval TEXT,
            period TEXT,
            data_json TEXT,
            timestamp DATETIME,
            PRIMARY KEY (symbol, interval, period)
        )
    """)
    conn.commit()
    conn.close()

init_db(DB_DAILY)
init_db(DB_HOURLY)

class CachedTicker:
    """
    對標 yf.Ticker 的緩存包裝類。
    負責攔截 .history() 請求並先去本地 SQLite 找數據。
    """
    def __init__(self, symbol, cache_level):
        self.symbol = symbol.upper()
        self.cache_level = cache_level
        self.db_path = DB_DAILY if cache_level == "daily" else DB_HOURLY
        self._ticker = yf.Ticker(self.symbol)
        
        # 緩存過期時間 (日線 24小時, 小時線 1小時)
        self.ttl = 24 if cache_level == "daily" else 1

    @property
    def info(self):
        """info 無法有效緩存，直接透傳"""
        return self._ticker.info

    @property
    def options(self):
        """透傳 options 到 yf.Ticker"""
        return self._ticker.options

    @property
    def news(self):
        """透傳 news 到 yf.Ticker"""
        return self._ticker.news

    def __getattr__(self, item):
        """透傳其他屬性到 yf.Ticker (如 quarterly_income_stmt 等)"""
        return getattr(self._ticker, item)

    def history(self, period="1y", interval="1d", **kwargs):
        """
        攔截歷史數據請求。
        """
        # 只有特定的週期與間隔才啟用緩存，避免複雜化
        if interval not in ["1d", "1h", "1m"]:
            return self._ticker.history(period=period, interval=interval, **kwargs)

        # 1. 嘗試從本地讀取
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT data_json, timestamp FROM cache WHERE symbol=? AND interval=? AND period=?"
            df_local = pd.read_sql_query(query, conn, params=(self.symbol, interval, period))
            
            if not df_local.empty:
                cache_time = datetime.fromisoformat(df_local['timestamp'][0])
                if datetime.now() - cache_time < timedelta(hours=self.ttl):
                    logger.info(f"⚡ [Cache Hit] {self.symbol} ({interval})")
                    data = json.loads(df_local['data_json'][0])
                    df = pd.DataFrame(data)
                    df.index = pd.to_datetime(df.index)
                    return df
        except Exception as e:
            logger.error(f"⚠️ 緩存讀取異常: {e}")
        finally:
            conn.close()

        # 2. 本地沒有或已過期，去網絡抓取
        logger.info(f"🌐 [Network Fetch] {self.symbol} ({interval})")
        df_network = self._ticker.history(period=period, interval=interval, **kwargs)
        
        if not df_network.empty:
            # 3. 抓到後寫入本地緩存
            conn = sqlite3.connect(self.db_path)
            try:
                # 轉換為 JSON 格式存儲
                data_json = df_network.to_json(date_format='iso')
                conn.execute("""
                    INSERT OR REPLACE INTO cache (symbol, interval, period, data_json, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (self.symbol, interval, period, data_json, datetime.now().isoformat()))
                conn.commit()
            except Exception as e:
                logger.error(f"⚠️ 緩存寫入異常: {e}")
            finally:
                conn.close()
                
        return df_network

def get_ticker(symbol: str, cache_level: str = "daily") -> CachedTicker:
    """
    獲取帶有本地 SQLite 緩存功能的 Ticker 實例。
    """
    return CachedTicker(symbol, cache_level)

def get_download(tickers, interval="1d", period="1y", **kwargs):
    """
    批量下載目前尚未實作緩存邏輯，直接透傳。
    """
    return yf.download(tickers, interval=interval, period=period, **kwargs)
