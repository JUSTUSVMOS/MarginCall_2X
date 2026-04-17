import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

import engine_market
import engine_router
import engine_risk
import engine_technical
import fubon


def _make_intraday_frame(periods: int = 24) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 09:30", periods=periods, freq="5min")
    closes = [100 + (i * 0.4) for i in range(periods)]
    opens = [price - 0.2 for price in closes]
    highs = [price + 0.6 for price in closes]
    lows = [price - 0.6 for price in closes]
    volumes = [1000 + (i * 40) for i in range(periods)]
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=index,
    )


def _make_divergence_frame() -> pd.DataFrame:
    close = [
        120, 118, 121, 117, 119, 116, 118, 115, 117, 114,
        116, 113, 115, 112, 114, 111, 113, 110, 112, 109,
        111, 108, 110, 107, 109, 106, 108, 105, 107, 104,
        106, 103, 105, 102, 104, 101, 103, 100, 102, 101,
        103, 100.5, 104, 99.8, 105, 99.4, 106, 99.0, 107, 98.7,
        108, 98.5, 109, 98.4, 110, 98.3, 111, 98.25, 112, 98.2,
    ]
    index = pd.date_range("2024-01-01", periods=len(close), freq="D")
    opens = [price + (0.8 if i % 2 else -0.8) for i, price in enumerate(close)]
    highs = [max(o, c) + 1.5 for o, c in zip(opens, close)]
    lows = [min(o, c) - 1.5 for o, c in zip(opens, close)]
    volumes = [1500 + (i * 15) for i in range(len(close))]
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": close, "Volume": volumes},
        index=index,
    )


class TechnicalSignalUpgradeChecks(unittest.TestCase):
    def test_indicator_engine_detects_divergence_adx_and_obv(self):
        calc = engine_technical.IndicatorCalculator()
        price = pd.Series([10.0, 9.0, 10.5, 8.2, 10.0, 7.4, 11.0]).to_numpy()
        rsi = pd.Series([42.0, 24.0, 48.0, 27.0, 45.0, 33.0, 55.0]).to_numpy()

        divergence = calc.DIVERGENCE(price, rsi, lookback=7, order=1)
        adx = calc.ADX(
            pd.Series(range(101, 141)).to_numpy() + 1,
            pd.Series(range(101, 141)).to_numpy() - 1,
            pd.Series(range(101, 141)).to_numpy(),
        )
        obv = calc.OBV(
            pd.Series([10.0, 11.0, 10.0, 12.0]).to_numpy(),
            pd.Series([100.0, 120.0, 80.0, 150.0]).to_numpy(),
        )

        self.assertTrue(divergence["bullish_divergence"])
        self.assertEqual(adx["trend_regime"], "trending")
        self.assertEqual(obv.tolist(), [0.0, 120.0, 40.0, 190.0])

    def test_market_reports_surface_vwap_divergence_adx_and_obv(self):
        history = _make_divergence_frame()
        intraday = _make_intraday_frame()
        calls = pd.DataFrame({"volume": [100, 200]})
        puts = pd.DataFrame({"volume": [60, 80]})

        class StubTicker:
            info = {
                "fiftyTwoWeekHigh": 180.0,
                "fiftyTwoWeekLow": 90.0,
                "bid": 101.0,
                "ask": 101.5,
                "bidSize": 200,
                "askSize": 100,
                "averageVolume": 1_000_000,
                "regularMarketVolume": 500_000,
            }
            options = ["2030-01-17"]

            def history(self, period, interval=None):
                if interval == "5m":
                    return intraday.copy()
                return history.copy()

            def option_chain(self, _date):
                return SimpleNamespace(calls=calls, puts=puts)

        with patch.object(engine_market, "get_ticker", return_value=StubTicker()):
            tech_report = engine_market.build_technical_report("AAPL")
            live_report = engine_market.build_realtime_insight("AAPL")

        self.assertIn("RSI 背離:", tech_report)
        self.assertIn("ADX(14):", tech_report)
        self.assertIn("OBV 趨勢:", tech_report)
        self.assertIn("VWAP:", live_report)
        self.assertIn("雙錨點:", live_report)
        self.assertIn("波動定價:", live_report)

    def test_router_uses_soft_and_hard_alert_levels(self):
        fake_ticker = SimpleNamespace(history=lambda period, interval=None: pd.DataFrame({"Close": [10.0, 9.5]}))
        seed_nlp = {"nlp_alpha": -0.2}

        def run_case(technical_snapshot):
            captured_alerts = []
            with patch.object(engine_router.market, "normalize_ticker", return_value="TEST"), patch.object(
                engine_router.market, "get_asset_profile", return_value={"asset_type": "Tech_Momentum"}
            ), patch.object(engine_router, "fetch_nlp_alpha", return_value=dict(seed_nlp)), patch.object(
                engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
            ), patch.object(engine_router, "get_ticker", return_value=fake_ticker), patch.object(
                engine_router.risk, "calculate_buying_pressure", return_value=-0.95
            ), patch.object(
                engine_router.market, "build_technical_snapshot", return_value=technical_snapshot
            ), patch.object(
                engine_router.market, "build_technical_report", return_value="TECH"
            ), patch.object(
                engine_router.market, "build_realtime_insight", return_value="P/C Ratio: 1.60"
            ), patch.object(
                engine_router.market, "build_option_volatility_context", return_value={"summary": "N/A", "signal": "⚪ 無期權波動資料"}
            ), patch.object(engine_router, "_alert_callback", lambda msg: captured_alerts.append(msg)):
                engine_router.fetch_strat_data("test")
            return captured_alerts[0]

        soft_message = run_case(
            {
                "divergence": {"label": "⚪ 無明顯背離", "bearish_divergence": False},
                "adx": {"value": 18.4, "trend_regime": "ranging"},
                "obv": {"signal": "⚪ 量價中性"},
            }
        )
        hard_message = run_case(
            {
                "divergence": {"label": "🔴 頂背離", "bearish_divergence": True},
                "adx": {"value": 31.2, "trend_regime": "trending"},
                "obv": {"signal": "📉 價跌量弱，空方主導"},
            }
        )

        self.assertIn("盤中拋壓警報", soft_message)
        self.assertIn("硬體中斷", hard_message)

    def test_risk_snapshot_downgrades_ma_break_in_ranging_regime(self):
        frame = pd.DataFrame(
            [
                {
                    "SPX": 5000.0,
                    "SPX_10MA": 5050.0,
                    "SPX_20MA": 5100.0,
                    "SPX_200MA": 5200.0,
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
            engine_risk,
            "get_spy_gex_levels",
            return_value={"total_gex_billions": 1.0, "gamma_flip_level": None, "max_pain": None, "spot": 500.0, "below_flip": None, "above_flip": None},
        ), patch.object(
            engine_risk, "get_market_sentiment_score", return_value=(0.0, "Neutral")
        ), patch.object(
            engine_risk, "get_market_breadth", return_value={"pct_above_50ma": 60.0, "pct_above_200ma": 60.0, "breadth_signal": "deteriorating"}
        ), patch.object(
            engine_risk.market, "build_option_volatility_context", return_value={}
        ), patch.object(
            engine_risk, "get_rolling_correlations", return_value={}
        ), patch.object(engine_risk, "_get_spx_trend_snapshot", return_value=(18.0, "ranging")):
            ranging = engine_risk._build_global_risk_snapshot()

        with patch.object(engine_risk, "fetch_all_market_data", return_value=frame), patch.object(
            engine_risk.MacroEngine, "get_macro_dashboard", return_value={"Yield_Curve_10Y2Y": 0.2, "Fed_Funds_Rate": 4.5}
        ), patch.object(
            engine_risk,
            "get_spy_gex_levels",
            return_value={"total_gex_billions": 1.0, "gamma_flip_level": None, "max_pain": None, "spot": 500.0, "below_flip": None, "above_flip": None},
        ), patch.object(
            engine_risk, "get_market_sentiment_score", return_value=(0.0, "Neutral")
        ), patch.object(
            engine_risk, "get_market_breadth", return_value={"pct_above_50ma": 60.0, "pct_above_200ma": 60.0, "breadth_signal": "deteriorating"}
        ), patch.object(
            engine_risk.market, "build_option_volatility_context", return_value={}
        ), patch.object(
            engine_risk, "get_rolling_correlations", return_value={}
        ), patch.object(engine_risk, "_get_spx_trend_snapshot", return_value=(30.0, "trending")):
            trending = engine_risk._build_global_risk_snapshot()

        self.assertLess(ranging["riskMultiplier"], trending["riskMultiplier"])
        self.assertEqual(ranging["signals"]["spxTrendRegime"], "ranging")
        self.assertEqual(trending["signals"]["spxTrendRegime"], "trending")

    def test_fubon_exhaustion_adds_divergence_bonus(self):
        original_ready = fubon.fubon_ready
        original_sdk = fubon.fubon_sdk
        try:
            fubon.fubon_ready = True
            reststock = SimpleNamespace(
                intraday=SimpleNamespace(
                    quote=lambda symbol: {
                        "bids": [{"size": 20}],
                        "asks": [{"size": 8}],
                        "closePrice": 100.0,
                    },
                    volumes=lambda symbol: {
                        "data": [{"price": 100.0, "volume": 500, "volumeAtAsk": 150, "volumeAtBid": 220}]
                    },
                    trades=lambda symbol, limit=50: {
                        "data": [{"price": 100.0, "size": 5} for _ in range(20)]
                    },
                )
            )
            fubon.fubon_sdk = SimpleNamespace(marketdata=SimpleNamespace(rest_client=SimpleNamespace(stock=reststock)))

            tech_report = (
                "● RSI(14): 22.00 (❄️極度超跌)\n"
                "● RSI 背離: 🟢 底背離\n"
                "● 布林通道: 上軌:110.00 | 下軌:95.00\n"
            )
            with patch.object(fubon, "get_fubon_technical", return_value=tech_report):
                report = fubon.get_exhaustion_analysis("2330")
        finally:
            fubon.fubon_ready = original_ready
            fubon.fubon_sdk = original_sdk

        self.assertIn("[背離確認]", report)


if __name__ == "__main__":
    unittest.main()
