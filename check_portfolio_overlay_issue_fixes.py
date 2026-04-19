import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import engine_portfolio
from src import database


def _prices_from_returns(returns):
    series = 100 * pd.Series(returns, dtype=float).add(1.0).cumprod()
    return pd.DataFrame({"Close": series.to_numpy(dtype=float)})


class PortfolioOverlayIssueChecks(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db_file = database.DB_FILE
        database.DB_FILE = Path(self.tempdir.name) / "portfolio-overlay-issues.db"
        engine_portfolio.init_db()

    def tearDown(self):
        database.DB_FILE = self.original_db_file
        self.tempdir.cleanup()

    def test_init_db_creates_nav_history_table(self):
        with database.locked_connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_nav_history'"
            ).fetchone()

        self.assertEqual(row, ("portfolio_nav_history",))

    def test_numeric_taiwan_symbols_use_market_normalizer(self):
        histories = {
            "2330.TW": _prices_from_returns([0.008, -0.004, 0.007, -0.003, 0.006] * 40),
            "8069.TWO": _prices_from_returns([0.012, -0.006, 0.011, -0.004, 0.009] * 40),
        }

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="1d", interval="1d"):
                return histories[self.symbol].copy()

        with patch.object(
            engine_portfolio.market,
            "_normalize_lookup_symbol",
            side_effect=lambda symbol: {"2330": "2330.TW", "8069": "8069.TWO"}.get(symbol, symbol),
        ), patch.object(
            engine_portfolio,
            "get_ticker",
            side_effect=lambda symbol, **_kwargs: StubTicker(symbol),
        ) as mock_get_ticker:
            inverse_vol = engine_portfolio.compute_inverse_vol_weights(["8069"], lookback=60, period="6mo")
            beta = engine_portfolio.compute_portfolio_beta_attribution({"8069": 1.0}, benchmark="2330", period="6mo")
            vol = engine_portfolio._estimate_portfolio_volatility({"8069": 1.0}, invested_ratio=1.0, period="6mo")

        requested_symbols = [call.args[0] for call in mock_get_ticker.call_args_list]
        self.assertIn("8069.TWO", inverse_vol["weights"])
        self.assertIn("8069.TWO", beta["positions"])
        self.assertGreater(vol["coverage_weight"], 0.99)
        self.assertNotIn("8069.TW", requested_symbols)

    def test_volatility_estimate_skips_low_weight_tail(self):
        weights = {
            "AAA": 0.35,
            "BBB": 0.20,
            "CCC": 0.15,
            "DDD": 0.10,
            "EEE": 0.08,
            "FFF": 0.06,
            "GGG": 0.03,
            "HHH": 0.03,
        }
        history = _prices_from_returns([0.01, -0.008, 0.009, -0.007, 0.008, -0.006] * 30)

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol

            def history(self, period="1d", interval="1d"):
                return history.copy()

        with patch.object(engine_portfolio.market, "_normalize_lookup_symbol", side_effect=lambda symbol: symbol), patch.object(
            engine_portfolio,
            "get_ticker",
            side_effect=lambda symbol, **_kwargs: StubTicker(symbol),
        ) as mock_get_ticker:
            payload = engine_portfolio._estimate_portfolio_volatility(weights, invested_ratio=1.0, lookback=60)

        self.assertEqual(mock_get_ticker.call_count, 5)
        self.assertEqual(payload["selected_symbol_count"], 5)
        self.assertEqual(payload["requested_symbol_count"], 8)
        self.assertAlmostEqual(payload["coverage_weight"], 0.88, places=2)
        self.assertEqual(payload["skipped"]["FFF"], "低權重，未納入快速波動估算")


if __name__ == "__main__":
    unittest.main()
