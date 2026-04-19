import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import engine_portfolio
import engine_router
import fubon
from src import database


class RiskOverlayChecks(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db_file = database.DB_FILE
        database.DB_FILE = Path(self.tempdir.name) / "risk-overlay-checks.db"
        engine_portfolio.init_db()
        engine_router._nlp_ic_cache["entries"].clear()

    def tearDown(self):
        database.DB_FILE = self.original_db_file
        self.tempdir.cleanup()

    def test_portfolio_overlay_surfaces_drawdown_beta_and_scale(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("2330.TW", 100.0, 10.0, 1000.0, 0),
            )
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("CASH_TWD", 1.0, 200.0, 200.0, 0),
            )
            conn.executemany(
                """
                INSERT INTO portfolio_nav_history (
                    timestamp, nav_twd, total_cost_twd, gross_exposure_twd, cash_twd, pnl_pct, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("2026-01-02T00:00:00Z", 1000.0, 1000.0, 800.0, 200.0, 0.0, "seed"),
                    ("2026-01-06T00:00:00Z", 1200.0, 1000.0, 1000.0, 200.0, 20.0, "seed"),
                ],
            )
            conn.commit()

        class StubTicker:
            def __init__(self, price):
                self.fast_info = {"last_price": price}

            def history(self, period="1d", interval="1d"):
                return pd.DataFrame({"Close": [90.0]})

        with patch.object(fubon, "fubon_ready", False), patch.object(
            engine_portfolio, "get_ticker", return_value=StubTicker(90.0)
        ), patch.object(
            engine_portfolio,
            "compute_portfolio_beta_attribution",
            return_value={"portfolio_beta": 1.1, "methodology": "beta-ok"},
        ), patch.object(
            engine_portfolio,
            "_estimate_portfolio_volatility",
            return_value={"nav_vol_annual": 0.14, "observations": 60, "coverage_weight": 1.0, "skipped": {}},
        ), patch(
            "engine_risk.get_global_risk_snapshot",
            return_value={"state": "🔴 警戒", "riskScore": 66},
        ):
            report = engine_portfolio.build_portfolio_risk_overlay_report()

        self.assertIn("Portfolio Risk Overlay", report)
        self.assertIn("Defense Only", report)
        self.assertIn("Gross Scale 0.25x", report)

    def test_router_alpha_governor_scales_positive_alpha(self):
        fake_ticker = SimpleNamespace(history=lambda period, interval=None: pd.DataFrame({"Close": [10.0, 10.2]}))
        technical_snapshot = {
            "divergence": {"label": "⚪ 無明顯背離", "bearish_divergence": False},
            "adx": {"value": 18.4, "trend_regime": "ranging"},
            "obv": {"signal": "⚪ 量價中性"},
            "mtf_rsi": {"signal_label": "🟢 強超賣共振", "confluence_strength": 2, "signal_reliability": "HIGH"},
        }

        with patch.object(engine_router.market, "normalize_ticker", return_value="TEST"), patch.object(
            engine_router.market, "get_asset_profile", return_value={"asset_type": "Tech_Momentum"}
        ), patch.object(
            engine_router, "fetch_nlp_alpha", return_value={"nlp_alpha": 0.8}
        ), patch.object(
            engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
        ), patch.object(
            engine_router, "get_ticker", return_value=fake_ticker
        ), patch.object(
            engine_router.risk, "calculate_buying_pressure", return_value=0.2
        ), patch.object(
            engine_router.risk, "get_global_risk_snapshot", return_value={"state": "🔴 警戒", "riskScore": 66}
        ), patch.object(
            engine_router.market,
            "compute_nlp_signal_ic",
            return_value={"signal_quality": "weak", "directionality": "positive", "ic_rolling_mean": 0.03},
        ), patch(
            "engine_portfolio.compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "🟠 Risk-Off", "size_multiplier": 0.5, "recommended_gross_scale": 0.4},
        ), patch.object(
            engine_router.market, "build_technical_snapshot", return_value=technical_snapshot
        ), patch.object(
            engine_router.market, "build_technical_report", return_value="TECH"
        ), patch.object(
            engine_router.market, "build_realtime_insight", return_value="P/C Ratio: 0.80"
        ), patch.object(
            engine_router.market, "build_option_volatility_context", return_value={"summary": "N/A", "signal": "⚪ 無期權波動資料"}
        ), patch.object(engine_router, "_alert_callback", None):
            data = engine_router.fetch_strat_data("test")

        self.assertEqual(data["nlp_insights"]["nlp_alpha"], 0.8)
        self.assertAlmostEqual(data["leading_indicators"]["alpha_adjusted"], 0.195, places=3)
        self.assertEqual(data["leading_indicators"]["portfolio_trade_mode"], "🟠 Risk-Off")


if __name__ == "__main__":
    unittest.main()
