import unittest
from unittest.mock import patch

import engine_market
import engine_portfolio
import engine_router


class CandidateConstructorChecks(unittest.TestCase):
    def test_candidate_panel_ranks_better_signal_higher(self):
        snapshots = {
            "AAPL": {"mean_reversion": {"zscore": -1.5, "half_life_days": 6.0, "reversion_candidate": True}},
            "TSLA": {"mean_reversion": {"zscore": 0.7, "half_life_days": 22.0, "reversion_candidate": False}},
        }
        factors = {
            "AAPL": {"symbol": "AAPL", "momentum_12_1": 0.20, "reversal_1m": 0.03, "quality_raw": 0.25, "earnings_yield": 0.05, "book_price": 0.18},
            "TSLA": {"symbol": "TSLA", "momentum_12_1": 0.07, "reversal_1m": -0.02, "quality_raw": 0.06, "earnings_yield": 0.02, "book_price": 0.04},
        }
        ic_payloads = {
            "AAPL": {"signal_quality": "strong", "ic_rolling_mean": 0.06, "directionality": "positive"},
            "TSLA": {"signal_quality": "weak", "ic_rolling_mean": 0.02, "directionality": "positive"},
        }

        with patch("engine_risk.get_global_risk_snapshot", return_value={"state": "🟡 整理"}), patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={"trade_mode_label": "🟡 Soft Throttle", "recommended_gross_scale": 0.7, "size_multiplier": 0.7},
        ), patch.object(
            engine_router, "fetch_nlp_alpha", side_effect=lambda symbol: {"nlp_alpha": {"AAPL": 0.7, "TSLA": 0.2}[symbol]}
        ), patch.object(
            engine_router,
            "_build_alpha_confidence_overlay",
            side_effect=lambda symbol, *_args, **_kwargs: {
                "effective_alpha": {"AAPL": 0.55, "TSLA": 0.12}[symbol],
                "combined_multiplier": {"AAPL": 0.8, "TSLA": 0.6}[symbol],
            },
        ), patch.object(
            engine_market,
            "get_asset_profile",
            side_effect=lambda symbol: {"symbol": symbol, "asset_type": "Tech_Momentum", "sector": "Technology", "industry": "Software"},
        ), patch.object(
            engine_market, "compute_factor_snapshot", side_effect=lambda symbol: factors[symbol]
        ), patch.object(
            engine_market, "build_technical_snapshot", side_effect=lambda symbol: snapshots[symbol]
        ), patch.object(
            engine_market, "compute_nlp_signal_ic", side_effect=lambda symbol, **_kwargs: ic_payloads[symbol]
        ), patch.object(
            engine_market, "_compute_liquidity_proxy", side_effect=lambda symbol, period="6mo": ((15.0, 3_000_000.0) if symbol == "AAPL" else (14.0, 1_500_000.0))
        ), patch.object(
            engine_portfolio,
            "compute_portfolio_beta_attribution",
            side_effect=lambda holdings, **_kwargs: {
                "positions": {
                    next(iter(holdings)): {
                        "beta": 1.0 if next(iter(holdings)) == "AAPL" else 1.4,
                        "idio_vol": 0.20 if next(iter(holdings)) == "AAPL" else 0.42,
                    }
                }
            },
        ):
            payload = engine_market.compute_candidate_alpha_panel(["AAPL", "TSLA"])

        self.assertEqual(payload["rows"][0]["symbol"], "AAPL")
        self.assertGreater(payload["rows"][0]["expected_return_bps"], payload["rows"][1]["expected_return_bps"])

    def test_rebalance_plan_respects_caps_without_auto_execution(self):
        panel = {
            "generated_at": "2026-01-01 00:00:00",
            "rows": [
                {"symbol": "NVDA", "asset_type": "Tech_Momentum", "sector": "Technology", "expected_return_bps": 120.0, "forecast_confidence": 0.8, "final_alpha_score": 1.2},
                {"symbol": "AAPL", "asset_type": "Tech_Momentum", "sector": "Technology", "expected_return_bps": 90.0, "forecast_confidence": 0.7, "final_alpha_score": 0.9},
                {"symbol": "GLD", "asset_type": "Macro_Hedge", "sector": "Unknown", "expected_return_bps": 60.0, "forecast_confidence": 0.6, "final_alpha_score": 0.5},
            ],
        }
        snapshots = [
            {"symbol": "AAPL", "is_cash": False, "market_value_twd": 10000.0},
            {"symbol": "CASH_TWD", "is_cash": True, "market_value_twd": 90000.0},
        ]

        with patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={
                "trade_mode": "normal",
                "trade_mode_label": "🟢 Normal",
                "recommended_gross_scale": 0.6,
                "gross_exposure_ratio": 0.1,
                "current_nav_twd": 100000.0,
                "primary_constraint": "目前無強制降風險",
            },
        ), patch.object(
            engine_portfolio.market, "compute_candidate_alpha_panel", return_value=panel
        ):
            payload = engine_portfolio.compute_portfolio_rebalance_plan(
                symbols=["NVDA", "AAPL", "GLD"],
                candidate_panel=panel,
                snapshots=snapshots,
            )

        self.assertLessEqual(payload["target_weights"]["NVDA"], 0.13)
        self.assertLessEqual(payload["sector_allocations"]["Technology"], 0.30)
        self.assertIn("不會自動下單", payload["methodology"])

    def test_rebalance_plan_normalizes_raw_numeric_holdings(self):
        panel = {
            "generated_at": "2026-01-01 00:00:00",
            "rows": [
                {"symbol": "2330.TW", "asset_type": "Tech_Momentum", "sector": "Technology", "expected_return_bps": 80.0, "forecast_confidence": 0.6, "final_alpha_score": 0.7},
            ],
        }
        snapshots = [
            {"symbol": "2330", "is_cash": False, "market_value_twd": 10000.0},
            {"symbol": "CASH_TWD", "is_cash": True, "market_value_twd": 90000.0},
        ]

        with patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={
                "trade_mode": "normal",
                "trade_mode_label": "🟢 Normal",
                "recommended_gross_scale": 0.1,
                "gross_exposure_ratio": 0.1,
                "current_nav_twd": 100000.0,
                "primary_constraint": "目前無強制降風險",
            },
        ), patch.object(
            engine_portfolio.market, "_normalize_lookup_symbol", side_effect=lambda symbol: "2330.TW" if symbol == "2330" else symbol
        ):
            payload = engine_portfolio.compute_portfolio_rebalance_plan(
                symbols=["2330"],
                candidate_panel=panel,
                snapshots=snapshots,
            )

        symbols = [row["symbol"] for row in payload["recommendations"]]
        self.assertEqual(symbols, ["2330.TW"])
        self.assertEqual(payload["recommendations"][0]["action"], "hold")

    def test_rebalance_plan_reserves_gross_for_accumulation_only_holdings(self):
        panel = {
            "generated_at": "2026-01-01 00:00:00",
            "rows": [
                {"symbol": "NVDA", "asset_type": "Tech_Momentum", "sector": "Technology", "expected_return_bps": 120.0, "forecast_confidence": 0.8, "final_alpha_score": 1.2},
            ],
        }
        snapshots = [
            {"symbol": "0050_TRUST", "is_cash": False, "market_value_twd": 60000.0},
            {"symbol": "CASH_TWD", "is_cash": True, "market_value_twd": 40000.0},
        ]

        with patch.object(
            engine_portfolio,
            "compute_portfolio_risk_overlay",
            return_value={
                "trade_mode": "kill_switch",
                "trade_mode_label": "💀 Kill Switch",
                "recommended_gross_scale": 0.0,
                "gross_exposure_ratio": 0.6,
                "current_nav_twd": 100000.0,
                "primary_constraint": "drawdown governor 💀 Kill Switch",
            },
        ), patch.object(
            engine_portfolio.market, "_normalize_lookup_symbol", side_effect=lambda symbol: "0050.TW" if symbol == "0050" else symbol
        ), patch.object(
            engine_portfolio.market, "get_asset_profile", return_value={"sector": "ETF", "asset_type": "Value_Holding"}
        ):
            payload = engine_portfolio.compute_portfolio_rebalance_plan(
                symbols=["NVDA"],
                candidate_panel=panel,
                snapshots=snapshots,
            )

        self.assertAlmostEqual(payload["protected_gross_ratio"], 0.6, places=2)
        self.assertAlmostEqual(payload["allocated_gross_ratio"], 0.6, places=2)
        self.assertTrue(any(item["symbol"] == "ACCUMULATION_ONLY" for item in payload["blocked_by_risk"]))
        self.assertFalse(any(row["symbol"] == "NVDA" and row["action"] == "buy" for row in payload["recommendations"]))


if __name__ == "__main__":
    unittest.main()
