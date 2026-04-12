import yfinance as yf
from yf_session import get_ticker, get_download
import pandas as pd
import numpy as np
import logging
from typing import List, Union, Dict, Tuple

logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """
    OpenAlice 級別的技術分析公式引擎 (100% 對標)。
    支援多週期抓取與動態公式解析 (如: SMA(CLOSE('AAPL', '1d'), 200))。
    """
    def __init__(self):
        self.data_cache: Dict[str, pd.DataFrame] = {}

    # ==========================================
    # 1. 數據抓取的技術邊界 (Technical Boundaries)
    # ==========================================
    def _get_period_for_interval(self, interval: str) -> str:
        """對標 OpenAlice: 1d 抓 2年, 1h 抓 90天, 1m 抓 30天"""
        if interval == '1d': return "2y"
        elif interval == '1h': return "90d"
        elif interval == '1m': return "30d"
        elif interval == '1wk': return "5y"
        else: return "1y"

    def _fetch_data(self, symbol: str, interval: str) -> pd.DataFrame:
        """獲取並緩存 K 線數據"""
        symbol = symbol.upper()
        if symbol.isdigit(): symbol += ".TW"
        
        cache_key = f"{symbol}_{interval}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        period = self._get_period_for_interval(interval)
        cache_level = "live" if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "1h"] else "hourly"
        try:
            df = get_ticker(symbol, cache_level=cache_level).history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"無法獲取 {symbol} ({interval}) 的歷史數據")
            self.data_cache[cache_key] = df
            return df
        except Exception as e:
            raise ValueError(f"獲取數據失敗: {e}")

    # ==========================================
    # 2. 基礎價格數據 (Primitives)
    # ==========================================
    def _get_series(self, symbol: str, interval: str, col: str) -> np.ndarray:
        df = self._fetch_data(symbol, interval)
        if col not in df.columns:
            raise ValueError(f"找不到欄位 {col}")
        return df[col].values

    def CLOSE(self, symbol: str, interval: str) -> np.ndarray: return self._get_series(symbol, interval, 'Close')
    def OPEN(self, symbol: str, interval: str) -> np.ndarray: return self._get_series(symbol, interval, 'Open')
    def HIGH(self, symbol: str, interval: str) -> np.ndarray: return self._get_series(symbol, interval, 'High')
    def LOW(self, symbol: str, interval: str) -> np.ndarray: return self._get_series(symbol, interval, 'Low')
    def VOLUME(self, symbol: str, interval: str) -> np.ndarray: return self._get_series(symbol, interval, 'Volume')

    # ==========================================
    # 3. 統計與趨勢指標 (Statistics)
    # ==========================================
    def SMA(self, data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).rolling(window=period).mean().values

    def EMA(self, data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).ewm(span=period, adjust=False).mean().values

    def MAX(self, data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).rolling(window=period).max().values

    def MIN(self, data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).rolling(window=period).min().values

    def STDEV(self, data: np.ndarray, period: int) -> np.ndarray:
        return pd.Series(data).rolling(window=period).std().values

    # ==========================================
    # 4. 震盪與動能指標 (Technical) - 手刻算法
    # ==========================================
    def RSI(self, data: np.ndarray, period: int = 14) -> np.ndarray:
        """對標 Wilder's Smoothing RSI"""
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values

    def MACD(self, data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
        """回傳 dict，支援 MACD(..)['macd'] 切片"""
        exp1 = self.EMA(data, fast)
        exp2 = self.EMA(data, slow)
        macd = exp1 - exp2
        sig = self.EMA(macd, signal)
        hist = macd - sig
        return {'macd': macd, 'signal': sig, 'histogram': hist}

    def ATR(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """平均真實波幅"""
        tr1 = highs - lows
        tr2 = np.abs(highs - np.roll(closes, 1))
        tr3 = np.abs(lows - np.roll(closes, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan # 第一筆無前收盤價
        return pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values

    # ==========================================
    # 5. 形態分析 (Structure)
    # ==========================================
    def BBANDS(self, data: np.ndarray, period: int = 20, std: float = 2) -> Dict[str, np.ndarray]:
        """布林帶"""
        mid = self.SMA(data, period)
        dev = self.STDEV(data, period)
        upper = mid + (dev * std)
        lower = mid - (dev * std)
        return {'upper': upper, 'middle': mid, 'lower': lower}

    # ==========================================
    # 6. 高級邏輯：公式解析與陣列切片 (Evaluation Engine)
    # ==========================================
    def calculate(self, formula: str, precision: int = 4) -> str:
        """
        供 AI 呼叫的入口。
        支援陣列切片語法，例如: "CLOSE('AAPL', '1d')[-1] > SMA(CLOSE('AAPL', '1d'), 50)[-1]"
        """
        # 建立安全的執行環境 (Safe Environment)
        safe_env = {
            'CLOSE': self.CLOSE, 'OPEN': self.OPEN, 'HIGH': self.HIGH,
            'LOW': self.LOW, 'VOLUME': self.VOLUME,
            'SMA': self.SMA, 'EMA': self.EMA, 'MAX': self.MAX, 'MIN': self.MIN, 'STDEV': self.STDEV,
            'RSI': self.RSI, 'MACD': self.MACD, 'ATR': self.ATR, 'BBANDS': self.BBANDS,
            'np': np, 'pd': pd
        }

        try:
            # 執行 AI 傳入的公式
            result = eval(formula, {"__builtins__": {}}, safe_env)
            
            # 格式化輸出
            if isinstance(result, (np.ndarray, pd.Series)):
                # 如果回傳是整個陣列，為了不刷頻，我們只印出最後 5 個值 (類似 OpenAlice)
                vals = result[~np.isnan(result)][-5:].tolist()
                formatted = [round(float(v), precision) for v in vals]
                return f"陣列結果 (近5筆): {formatted}"
            elif isinstance(result, dict):
                # 處理 MACD 或 BBANDS 回傳的 dict
                res_str = ""
                for k, v in result.items():
                    vals = v[~np.isnan(v)][-1] if isinstance(v, (np.ndarray, pd.Series)) else v
                    res_str += f"{k}: {round(float(vals), precision)} | "
                return f"字典結果 (最新值): {res_str.strip(' | ')}"
            elif isinstance(result, (np.bool_, bool)):
                return f"邏輯判斷結果: {bool(result)}"
            else:
                return f"單一數值: {round(float(result), precision)}"
                
        except Exception as e:
            return f"❌ 公式計算錯誤: {e}\n請檢查語法，例如是否忘記加上 [-1] 獲取最新值？"

# --- 註冊給 main.py 使用的工具介面 ---
def calculate_indicator(formula: str) -> str:
    """
    Calculates technical indicators for any asset using formula expressions.
    
    Data Primitives: CLOSE('AAPL', '1d'), OPEN, HIGH, LOW, VOLUME (Intervals: '1m', '1h', '1d', '1wk')
    Statistics: SMA(data, period), EMA, MAX, MIN, STDEV
    Indicators: RSI(data, 14), MACD(data, fast, slow, sig), ATR(highs, lows, closes, 14), BBANDS(data, 20, 2)
    Logic & Slicing: Supports slicing [-1] and boolean logic (e.g., CLOSE(...) > SMA(...)).
    
    Example formula: "RSI(CLOSE('TSLA', '1h'), 14)[-1]"
    """
    calc = IndicatorCalculator()
    return calc.calculate(formula)

if __name__ == "__main__":
    # 實戰範例測試
    calc = IndicatorCalculator()
    print("1. 獲取最新收盤價:")
    print(calc.calculate("CLOSE('AAPL', '1d')[-1]"))
    
    print("\n2. 日線趨勢確認 (價格是否大於 50日均線):")
    print(calc.calculate("CLOSE('TSLA', '1d')[-1] > SMA(CLOSE('TSLA', '1d'), 50)[-1]"))
    
    print("\n3. 小時線動能分析 (RSI 是否超賣):")
    print(calc.calculate("RSI(CLOSE('TSLA', '1h'), 14)[-1]"))
    
    print("\n4. 布林帶最新數值:")
    print(calc.calculate("BBANDS(CLOSE('NVDA', '1d'), 20, 2)"))
