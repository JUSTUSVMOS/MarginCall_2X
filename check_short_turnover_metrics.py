import datetime as real_datetime
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import engine_fundamentals
import engine_market


def _make_fundamental_engine(info):
    engine = object.__new__(engine_fundamentals.FundamentalEngine)
    engine.symbol = "NVDA"
    engine.ticker = SimpleNamespace()
    engine._info = info
    return engine


def _make_intraday_frame():
    index = pd.date_range("2030-01-10 09:30", periods=12, freq="5min")
    return pd.DataFrame(
        {
            "Open": [100 + i * 0.5 for i in range(12)],
            "High": [100.5 + i * 0.5 for i in range(12)],
            "Low": [99.5 + i * 0.5 for i in range(12)],
            "Close": [100.2 + i * 0.5 for i in range(12)],
            "Volume": [100_000 + i * 10_000 for i in range(12)],
        },
        index=index,
    )


class FixedDateTime(real_datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        current = cls(2030, 1, 10, 12, 0, 0)
        if tz is not None:
            return tz.localize(current)
        return current


class FundamentalShortInterestTests(unittest.TestCase):
    def test_short_percent_uses_float_shares_and_adds_warning(self):
        engine = _make_fundamental_engine(
            {
                "sharesShort": 12_000_000,
                "floatShares": 60_000_000,
                "heldPercentInstitutions": 0.62,
            }
        )

        metrics = engine.get_valuation_metrics()

        self.assertEqual(metrics["放空比率(Short%)"], "20.00% ⚠️高軋空潛力")
        self.assertEqual(metrics["機構持倉比"], "62.0%")

    def test_short_percent_handles_zero_and_fallback_values(self):
        zero_short_engine = _make_fundamental_engine(
            {
                "sharesShort": 0,
                "floatShares": 100_000_000,
            }
        )
        self.assertEqual(zero_short_engine.get_valuation_metrics()["放空比率(Short%)"], "0.00%")

        fallback_engine = _make_fundamental_engine(
            {
                "sharesShort": float("nan"),
                "shortPercentOfFloat": 0.18,
            }
        )
        self.assertEqual(fallback_engine.get_valuation_metrics()["放空比率(Short%)"], "18.00% ⚠️高軋空潛力")

    def test_full_report_renders_short_percent_line(self):
        engine = _make_fundamental_engine({})
        engine.get_company_profile = lambda: {
            "公司名稱": "NVIDIA",
            "產業": "Technology / Semiconductors",
            "CEO": "Jensen Huang",
            "員工數": "29,600",
            "官方網站": "https://www.nvidia.com",
            "公司簡介": "GPU leader...",
        }
        engine.get_valuation_metrics = lambda: {
            "市值": "2.10B",
            "本益比(PE)": 30,
            "預估PE(Forward)": 25,
            "市淨率(PB)": 20,
            "企業倍數(EV/EBITDA)": 18,
            "股息殖利率": "0.10%",
            "空頭回補天數": 1.2,
            "放空比率(Short%)": "18.00% ⚠️高軋空潛力",
            "機構持倉比": "65.0%",
        }
        engine.get_financial_statements_summary = lambda: {}
        engine.get_quality_ratios = lambda: {}
        engine.get_earnings_and_estimates = lambda: {
            "近四季EPS(Trailing)": "2.00",
            "預估EPS(Forward)": "2.50",
        }
        engine.get_events_and_opinions = lambda: {}
        engine.get_insider_trading = lambda: "近期無申報紀錄"

        report = engine.get_full_fundamental_report()

        self.assertIn("放空比率: 18.00% ⚠️高軋空潛力", report)


class RealtimeTurnoverTests(unittest.TestCase):
    def _build_ticker(self, regular_market_volume, extra_info=None):
        calls = pd.DataFrame({"volume": [100, 200]})
        puts = pd.DataFrame({"volume": [50, 75]})
        info = {
            "bid": 101.0,
            "ask": 101.5,
            "bidSize": 200,
            "askSize": 100,
            "averageVolume": 1_000_000,
            "regularMarketVolume": regular_market_volume,
            "floatShares": 100_000_000,
        }
        if extra_info:
            info.update(extra_info)

        class StubTicker:
            options = ["2030-01-17"]

            def __init__(self, info_map):
                self.info = info_map

            def history(self, period, interval=None):
                return _make_intraday_frame().copy()

            def option_chain(self, _date):
                return SimpleNamespace(calls=calls, puts=puts)

        return StubTicker(info)

    def test_build_realtime_insight_includes_turnover_signal(self):
        ticker = self._build_ticker(6_000_000)

        with patch.object(engine_market, "get_ticker", return_value=ticker), patch.object(
            engine_market.datetime, "datetime", FixedDateTime
        ):
            result = engine_market.build_realtime_insight("NVDA")

        self.assertIn("換手率: 6.00% 🔥籌碼活躍", result)
        self.assertIn("成交量能比:", result)

    def test_build_realtime_insight_keeps_zero_volume_as_zero(self):
        ticker = self._build_ticker(0)

        with patch.object(engine_market, "get_ticker", return_value=ticker), patch.object(
            engine_market.datetime, "datetime", FixedDateTime
        ):
            result = engine_market.build_realtime_insight("NVDA")

        self.assertIn("成交量能比: 0.00x", result)
        self.assertIn("換手率: 0.00%", result)


if __name__ == "__main__":
    unittest.main()
