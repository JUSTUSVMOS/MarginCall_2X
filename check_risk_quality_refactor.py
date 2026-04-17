import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import config
import engine_market
import engine_portfolio
import engine_risk


class RiskQualityRefactorChecks(unittest.TestCase):
    def setUp(self):
        self.original_fx_cache = dict(engine_portfolio._fx_cache)
        self.original_risk_cache = dict(engine_risk._risk_cache)

    def tearDown(self):
        engine_portfolio._fx_cache.update(self.original_fx_cache)
        engine_risk._risk_cache.update(self.original_risk_cache)

    def test_dix_support_is_fixed_score_offset_in_snapshot_pipeline(self):
        base_row = {
            "gex": 1_000_000_000,
            "DXY_Z": 0.0,
            "TNX_Z": 0.0,
            "VIX_Z": 2.2,
            "SKEW_PR": 0.95,
            "SPX": 5200.0,
            "SPX_10MA": 5100.0,
            "SPX_20MA": 5000.0,
            "SPX_200MA": 4900.0,
        }
        df_with_dix = pd.DataFrame([{**base_row, "dix_PR": 0.9}])
        df_without_dix = pd.DataFrame([{**base_row, "dix_PR": 0.4}])

        with patch.object(engine_risk, "fetch_all_market_data", side_effect=[df_with_dix, df_without_dix]), patch.object(
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
        ):
            with_dix = engine_risk._build_global_risk_snapshot()
            without_dix = engine_risk._build_global_risk_snapshot()

        self.assertEqual(with_dix["signals"]["dixSupportOffset"], -engine_risk.DIX_SUPPORT_OFFSET_POINTS)
        self.assertEqual(without_dix["signals"]["dixSupportOffset"], 0)
        self.assertEqual(
            without_dix["riskScore"] - with_dix["riskScore"],
            engine_risk.DIX_SUPPORT_OFFSET_POINTS,
        )

    def test_global_risk_cache_round_trip_stays_consistent(self):
        engine_risk._risk_cache.update({"snapshot": None, "report": "", "timestamp": 0, "expiry": 1200})
        snapshot = {
            "generatedAt": "2026-01-01T00:00:00Z",
            "grossRiskScore": 66,
            "riskScore": 54,
            "state": "🔴 警戒",
            "riskMultiplier": 2.08,
            "scoreAdjustments": {"dixSupport": -12},
            "summary": "ok",
            "reasons": ["🟢 暗池吸籌，大戶提供下檔支撐"],
            "signals": {"dixPr": 0.9, "dixSupportOffset": -12, "gexBillions": 1.0},
        }

        with patch.object(engine_risk, "_build_global_risk_snapshot", return_value=snapshot), patch.object(
            engine_risk, "format_global_risk_snapshot", return_value="risk-report"
        ):
            fresh = engine_risk.get_global_risk_snapshot(force_refresh=True)
            cached = engine_risk.get_global_risk_snapshot()
            radar = engine_risk.get_global_risk_radar()

        self.assertFalse(fresh["cached"])
        self.assertTrue(cached["cached"])
        self.assertEqual(radar, "risk-report\n(⚡ DB-Cached)")

    def test_safe_float_rejects_non_finite_values(self):
        self.assertIsNone(engine_risk._safe_float(float("nan")))
        self.assertIsNone(engine_risk._safe_float(float("inf")))
        self.assertIsNone(engine_risk._safe_float(float("-inf")))
        self.assertEqual(engine_risk._safe_float("10.5", 1), 10.5)

    def test_concurrent_exchange_rate_refresh_returns_consistent_value(self):
        engine_portfolio._fx_cache.update({"rate": 31.5, "timestamp": 0})
        barrier = threading.Barrier(8)

        class StubTicker:
            fast_info = {"last_price": 32.15}

            def history(self, period):
                return pd.DataFrame()

        def call_fetch():
            barrier.wait()
            return engine_portfolio.fetch_exchange_rate()

        with patch.object(engine_portfolio, "get_ticker", return_value=StubTicker()):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _i: call_fetch(), range(8)))

        self.assertEqual(results, [32.15] * 8)
        self.assertEqual(engine_portfolio._fx_cache["rate"], 32.15)
        self.assertTrue(engine_portfolio._fx_cache["timestamp"] > 0)

    def test_system_prompt_loader_prefers_local_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_path = root / "system_prompt.txt"
            local_path = root / "system_prompt.local.txt"
            prompt_path.write_text("tracked prompt", encoding="utf-8")
            local_path.write_text("local prompt", encoding="utf-8")

            self.assertEqual(
                config._load_system_prompt(prompt_path=prompt_path, local_prompt_path=local_path),
                "local prompt",
            )

            local_path.unlink()

            self.assertEqual(
                config._load_system_prompt(prompt_path=prompt_path, local_prompt_path=local_path),
                "tracked prompt",
            )

    def test_fmp_parse_error_logs_and_falls_back_to_yfinance(self):
        class BadResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return [{"price": "bad"}]

        ticker = SimpleNamespace(
            info={"currentPrice": 123.456},
            history=lambda period="1d": pd.DataFrame(),
        )

        with patch.object(engine_market, "FMP_KEY", "demo"), patch.object(
            engine_market, "is_us_market_open", return_value=True
        ), patch.object(
            engine_market.requests, "get", return_value=BadResponse()
        ), patch.object(
            engine_market, "get_ticker", side_effect=lambda *args, **kwargs: ticker
        ):
            price = engine_market.fetch_live_price("AAPL")

        self.assertEqual(price, "123.46 (來源: YF)")


if __name__ == "__main__":
    unittest.main()
