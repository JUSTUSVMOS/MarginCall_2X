import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import engine_portfolio
from src import database


class PreTradeRiskGateChecks(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db_file = database.DB_FILE
        database.DB_FILE = Path(self.tempdir.name) / "pretrade-risk-gate.db"
        engine_portfolio.init_db()

    def tearDown(self):
        database.DB_FILE = self.original_db_file
        self.tempdir.cleanup()

    def test_gate_blocks_averaging_down_in_risk_off(self):
        snapshots = [
            {"symbol": "AAPL", "is_cash": False, "market_value_twd": 8000.0, "pnl_value_twd": -900.0},
            {"symbol": "CASH_USD", "is_cash": True, "market_value_twd": 50000.0, "pnl_value_twd": 0.0},
        ]
        overlay = {
            "current_nav_twd": 58000.0,
            "trade_mode_label": "🟠 Risk-Off",
            "trade_mode": "risk_off",
            "allow_new_longs": True,
            "allow_average_down": False,
            "governor_message": "回撤超過 5%，新單砍半且禁止攤平虧損部位。",
            "recommended_gross_scale": 0.5,
            "gross_exposure_twd": 8000.0,
            "target_beta_band": [0.4, 0.7],
            "current_beta_to_nav": 0.3,
        }

        with patch.object(engine_portfolio, "_load_portfolio_rows", return_value=[]), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ), patch.object(
            engine_portfolio, "compute_portfolio_risk_overlay", return_value=overlay
        ):
            gate = engine_portfolio._apply_pretrade_risk_gate("AAPL", "buy", 1.0, 3200.0)

        self.assertFalse(gate["allowed"])
        self.assertIn("禁止攤平虧損部位", gate["message"])

    def test_gate_scales_order_to_position_cap(self):
        snapshots = [
            {"symbol": "CASH_USD", "is_cash": True, "market_value_twd": 100000.0, "pnl_value_twd": 0.0},
        ]
        overlay = {
            "current_nav_twd": 100000.0,
            "trade_mode_label": "🟢 Normal",
            "trade_mode": "normal",
            "allow_new_longs": True,
            "allow_average_down": True,
            "governor_message": "回撤仍在可接受區間。",
            "recommended_gross_scale": 1.0,
            "gross_exposure_twd": 0.0,
            "target_beta_band": [0.8, 1.1],
            "current_beta_to_nav": 0.2,
        }

        with patch.object(engine_portfolio, "_load_portfolio_rows", return_value=[]), patch.object(
            engine_portfolio, "_build_live_position_snapshots", return_value=snapshots
        ), patch.object(
            engine_portfolio, "compute_portfolio_risk_overlay", return_value=overlay
        ), patch.object(
            engine_portfolio.market, "get_asset_profile", return_value={"asset_type": "Value_Holding", "sector": "Unknown"}
        ), patch.object(
            engine_portfolio, "_estimate_symbol_beta", return_value={"symbol": "AAPL", "benchmark": "SPY", "beta": 0.8, "observations": 60}
        ):
            gate = engine_portfolio._apply_pretrade_risk_gate("AAPL", "buy", 200.0, 20000.0)

        self.assertTrue(gate["allowed"])
        self.assertAlmostEqual(gate["approved_shares"], 150.0, places=2)
        self.assertAlmostEqual(gate["approved_twd_total"], 15000.0, places=2)
        self.assertIn("單一持股上限 15.0% NAV", gate["message"])

    def test_scaled_buy_persists_risk_gate_note(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("CASH_USD", 1.0, 1000.0, 32000.0, 0),
            )
            conn.commit()

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=32.0), patch.object(
            engine_portfolio,
            "_apply_pretrade_risk_gate",
            return_value={
                "allowed": True,
                "approved_shares": 1.0,
                "approved_twd_total": 3200.0,
                "message": "⚠️ 風控縮倉：AAPL 由 2.0000 股縮至 1.0000 股 (單一持股上限 15.0% NAV)。",
                "note": "risk_gate:單一持股上限 15.0% NAV; requested_shares=2.0000; approved_shares=1.0000",
            },
        ):
            result = engine_portfolio.execute_position_update("AAPL", 100.0, 2.0, action="buy")

        self.assertIn("⚠️ 風控縮倉", result)
        with database.locked_connection() as conn:
            note = conn.execute("SELECT note FROM trade_log WHERE action = 'buy'").fetchone()[0]
        self.assertIn("approved_shares=1.0000", note)


if __name__ == "__main__":
    unittest.main()
