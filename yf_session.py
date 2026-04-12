import yfinance as yf
import logging

logger = logging.getLogger(__name__)

def get_ticker(symbol: str, cache_level: str = "hourly") -> yf.Ticker:
    """
    獲取 Ticker。
    (註：由於最新版 yfinance 強制使用 curl_cffi 來繞過 Yahoo 防爬蟲機制，
    已不支援 requests-cache，因此在此層暫時停用網路快取，交由 YF 內建處理)
    """
    symbol = symbol.upper()
    return yf.Ticker(symbol)

def get_download(tickers, interval="1d", period="1y", **kwargs):
    """
    智能下載。
    (註：同上，停用 requests-cache session)
    """
    kwargs.pop('session', None)
    return yf.download(tickers, interval=interval, period=period, **kwargs)
