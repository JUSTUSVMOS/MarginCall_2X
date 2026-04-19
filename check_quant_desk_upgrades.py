import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import engine_market
import engine_portfolio
from src import database


def _prices_from_returns(returns):
    series = 100 * np.cumprod(1 + np.asarray(returns, dtype=float))
    return pd.DataFrame({"Close": series})


class QuantDeskUpgradeChecks(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_db_file = database.DB_FILE
        database.DB_FILE = Path(self.tempdir.name) / "quant-desk-checks.db"
        engine_portfolio.init_db()

    def tearDown(self):
        database.DB_FILE = self.original_db_file
        self.tempdir.cleanup()

    def test_closed_book_analytics_and_beta_attribution(self):
        with database.locked_connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_log (
                    timestamp, symbol, action, price, shares, settle_currency,
                    settle_amount, fx_rate, realized_pnl
                ) VALUES (?, ?, 'sell', ?, ?, 'CASH_USD', ?, ?, ?)
                """,
                ("2026-01-02T14:30:00Z", "AAPL", 110.0, 1.0, 110.0, 1.0, 10.0),
            )
            conn.execute(
                """
                INSERT INTO trade_log (
                    timestamp, symbol, action, price, shares, settle_currency,
                    settle_amount, fx_rate, realized_pnl
                ) VALUES (?, ?, 'sell', ?, ?, 'CASH_USD', ?, ?, ?)
                """,
                ("2026-01-06T14:30:00Z", "AAPL", 100.0, 1.0, 100.0, 1.0, -10.0),
            )
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("AAPL", 100.0, 10.0, 1000.0, 0),
            )
            conn.execute(
                "INSERT OR REPLACE INTO portfolio (symbol, cost, shares, twd_cost, locked) VALUES (?, ?, ?, ?, ?)",
                ("MSFT", 50.0, 20.0, 1000.0, 0),
            )
            conn.commit()

        bench_returns = np.array([0.01, -0.02, 0.015, -0.005, 0.012, -0.01] * 8)
        aapl_returns = 0.002 + (1.5 * bench_returns)
        msft_returns = -0.001 + (0.5 * bench_returns)
        histories = {
            "SPY": _prices_from_returns(bench_returns),
            "AAPL": _prices_from_returns(aapl_returns),
            "MSFT": _prices_from_returns(msft_returns),
        }
        last_prices = {"AAPL": 110.0, "MSFT": 55.0, "SPY": 100.0}

        class StubTicker:
            def __init__(self, symbol):
                self.symbol = symbol
                self.fast_info = {"last_price": last_prices.get(symbol)}

            def history(self, period="1d", interval="1d"):
                return histories[self.symbol].copy()

        with patch.object(engine_portfolio, "fetch_exchange_rate", return_value=1.0), patch.object(
            engine_portfolio, "get_ticker", side_effect=lambda symbol, **_kwargs: StubTicker(symbol)
        ):
            analytics = engine_portfolio.compute_portfolio_analytics()
            analytics_report = engine_portfolio.build_portfolio_analytics_report()
            beta_report = engine_portfolio.build_portfolio_beta_report()

        self.assertEqual(analytics["closed_trade_count"], 2)
        self.assertAlmostEqual(analytics["win_rate"], 0.5)
        self.assertEqual(analytics["profit_factor"], 1.0)
        self.assertIn("Portfolio Quant Analytics", analytics_report)
        self.assertIn("Portfolio Beta: 1.00", beta_report)

    def test_mean_reversion_report_surfaces_half_life(self):
        base = 100 + (np.sin(np.linspace(0, 10 * np.pi, 120)) * 3)
        prices = base.copy()
        prices[-1] -= 6.0
        history = pd.DataFrame(
            {
                "Open": prices - 0.5,
                "High": prices + 1.0,
                "Low": prices - 1.0,
                "Close": prices,
                "Volume": np.linspace(1000, 2000, len(prices)),
            },
            index=pd.date_range("2024-01-01", periods=len(prices), freq="D"),
        )
        ticker = SimpleNamespace(
            history=lambda period, interval="1d": history.copy(),
            info={"fiftyTwoWeekHigh": 110.0, "fiftyTwoWeekLow": 90.0},
        )

        with patch.object(
            engine_market,
            "get_mtf_confluence",
            return_value={
                "rsi_by_timeframe": {"weekly": 55.0, "daily": 48.0, "intraday_1h": 42.0},
                "signal_label": "⚪ 中性",
                "signal_reliability": "NORMAL",
            },
        ), patch.object(engine_market, "get_ticker", return_value=ticker):
            report = engine_market.build_mean_reversion_report("AAPL", lookback=40)

        self.assertIn("均值回歸信號", report)
        self.assertIn("半衰期", report)


if __name__ == "__main__":
    unittest.main()
