import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import engine_market
import engine_portfolio
import engine_risk
import nlp_worker
import engine_technical


def _make_ohlcv_frame(periods: int, close_start: float, close_step: float, freq: str = "D") -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq=freq)
    closes = [close_start + i * close_step for i in range(periods)]
    opens = [price - 0.5 for price in closes]
    highs = [price + 1.0 for price in closes]
    lows = [price - 1.0 for price in closes]
    volumes = [1000 + (i * 50) for i in range(periods)]
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=index,
    )


class MarketContextUpgradeChecks(unittest.TestCase):
    def test_reports_surface_vol_context_and_mtf_confluence(self):
        daily = _make_ohlcv_frame(periods=90, close_start=100.0, close_step=0.4)
        intraday = _make_ohlcv_frame(periods=24, close_start=120.0, close_step=0.2, freq="5min")
        calls = pd.DataFrame(
            {
                "strike": [135.0, 136.0],
                "impliedVolatility": [0.42, 0.44],
                "volume": [200, 180],
            }
        )
        puts = pd.DataFrame(
            {
                "strike": [134.0, 133.0],
                "impliedVolatility": [0.40, 0.39],
                "volume": [260, 220],
            }
        )

        class StubTicker:
            info = {
                "fiftyTwoWeekHigh": 180.0,
                "fiftyTwoWeekLow": 90.0,
                "bid": 135.0,
                "ask": 135.5,
                "bidSize": 150,
                "askSize": 100,
                "averageVolume": 1_000_000,
                "regularMarketVolume": 650_000,
            }
            options = ["2030-01-17"]

            def history(self, period, interval=None):
                if interval == "5m":
                    return intraday.copy()
                return daily.copy()

            def option_chain(self, _exp):
                return SimpleNamespace(calls=calls, puts=puts)

        with patch.object(engine_market, "get_ticker", return_value=StubTicker()):
            tech_report = engine_market.build_technical_report("NVDA")
            live_report = engine_market.build_realtime_insight("NVDA")

        self.assertIn("多時間框 RSI:", tech_report)
        self.assertIn("波動定價:", live_report)
        self.assertIn("VRP", live_report)

    def test_risk_snapshot_surfaces_breadth_gamma_and_correlations(self):
        frame = pd.DataFrame(
            [
                {
                    "SPX": 5200.0,
                    "SPX_10MA": 5100.0,
                    "SPX_20MA": 5000.0,
                    "SPX_200MA": 4900.0,
                    "dix_PR": 0.4,
                    "DXY_Z": 0.0,
                    "TNX_Z": 0.0,
                    "VIX_Z": 0.0,
                    "SKEW_PR": 0.0,
                    "gex": 1_000_000_000,
                }
            ]
        )

        with patch.object(engine_risk, "fetch_all_market_data", return_value=frame), patch.object(
            engine_risk.MacroEngine, "get_macro_dashboard", return_value={"Yield_Curve_10Y2Y": 0.2, "Fed_Funds_Rate": 4.5}
        ), patch.object(
            engine_risk, "get_market_sentiment_score", return_value=(0.0, "Neutral")
        ), patch.object(
            engine_risk, "_get_spx_trend_snapshot", return_value=(18.0, "ranging")
        ), patch.object(
            engine_risk, "get_market_breadth", return_value={"pct_above_50ma": 35.0, "pct_above_200ma": 25.0, "breadth_signal": "weak"}
        ), patch.object(
            engine_risk,
            "get_spy_gex_levels",
            return_value={
                "total_gex_billions": 0.6,
                "gamma_flip_level": 510.0,
                "max_pain": 505.0,
                "spot": 500.0,
                "below_flip": True,
                "above_flip": False,
            },
        ), patch.object(
            engine_risk.market,
            "build_option_volatility_context",
            return_value={
                "current_iv": 28.0,
                "realized_vol_30d": 33.0,
                "vrp": -5.0,
                "iv_vs_rv_percentile": 20.0,
                "signal": "⚠️ 波動低估",
            },
        ), patch.object(
            engine_risk, "get_rolling_correlations", return_value={"spyTltCorr60d": 0.25, "spyGldCorr60d": 0.1, "spyDxyCorr60d": -0.05}
        ):
            snapshot = engine_risk._build_global_risk_snapshot()
            report = engine_risk.format_global_risk_snapshot(snapshot)

        self.assertIn("Breadth:", report)
        self.assertIn("Gamma Levels:", report)
        self.assertIn("Corr60:", report)
        self.assertEqual(snapshot["signals"]["spyGammaFlipLevel"], 510.0)

    def test_nlp_time_decay_prefers_fresh_news(self):
        fresh = datetime.now(timezone.utc) - timedelta(hours=1)
        stale = datetime.now(timezone.utc) - timedelta(hours=30)
        fresh_weight, fresh_hours = nlp_worker._time_decay_weight(fresh)
        stale_weight, stale_hours = nlp_worker._time_decay_weight(stale)

        self.assertGreater(fresh_weight, stale_weight)
        self.assertLess(fresh_hours, stale_hours)

    def test_atr_position_size_uses_risk_budget(self):
        with patch.object(engine_portfolio, "build_portfolio_analysis", return_value={"total_current": 100_000.0}), patch.object(
            engine_portfolio, "fetch_exchange_rate", return_value=32.0
        ), patch.object(
            engine_technical.IndicatorCalculator, "HIGH", return_value=[101.0, 102.0]
        ), patch.object(
            engine_technical.IndicatorCalculator, "LOW", return_value=[99.0, 100.0]
        ), patch.object(
            engine_technical.IndicatorCalculator, "CLOSE", return_value=[100.0, 100.0]
        ), patch.object(
            engine_technical.IndicatorCalculator, "ATR", return_value=[0.5]
        ):
            report = engine_portfolio.build_position_size_report("AAPL", risk_pct=2.0)

        self.assertIn("ATR 倉位計算", report)
        self.assertIn("風險預算: NT$2,000", report)
        self.assertIn("建議股數: 31 股", report)


if __name__ == "__main__":
    unittest.main()
