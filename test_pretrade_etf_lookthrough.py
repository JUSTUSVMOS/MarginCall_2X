import unittest
from unittest.mock import patch, MagicMock
from engine_portfolio import _apply_pretrade_risk_gate

class TestPretradeETFLookthrough(unittest.TestCase):
    @patch('engine_portfolio.compute_portfolio_risk_overlay')
    @patch('engine_portfolio._build_live_position_snapshots')
    @patch('engine_portfolio._load_portfolio_rows')
    @patch('engine_portfolio.market.get_asset_profile')
    @patch('engine_portfolio.get_ticker')
    def test_etf_lookthrough_rejection(self, mock_get_ticker, mock_profile, mock_load, mock_snapshots, mock_overlay):
        mock_overlay.return_value = {
            "current_nav_twd": 100000.0,
            "allow_new_longs": True,
            "trade_mode": "normal",
            "gross_exposure_twd": 0,
            "recommended_gross_scale": 2.0,
            "target_beta_band": [0.5, 1.5]
        }
        
        mock_profile.return_value = {
            "asset_type": "ETF",
            "sector": "Equity",
            "is_etf": True
        }
        
        mock_snapshots.return_value = [
            {'symbol': 'NVDA', 'market_value_twd': 14000.0}
        ]
        
        def side_effect(symbol, *args, **kwargs):
            mock_ticker = MagicMock()
            if symbol == 'VOO':
                mock_ticker.get_holdings.return_value = [
                    {'Symbol': 'NVDA', 'Name': 'NVIDIA Corp', 'Percent': 0.07}
                ]
            else:
                mock_ticker.get_holdings.return_value = []
            return mock_ticker
            
        mock_get_ticker.side_effect = side_effect

        result = _apply_pretrade_risk_gate(symbol='VOO', action='buy', shares=2, actual_twd_total=20000.0)
        
        print(result)
        self.assertFalse(result['allowed'])
        self.assertIn("ETF 穿透風險檢查未通過", result['message'])
        self.assertIn("NVDA", result['message'])

if __name__ == '__main__':
    unittest.main()
