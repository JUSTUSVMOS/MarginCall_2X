import unittest
from unittest.mock import patch
import engine_portfolio
from engine_portfolio import get_portfolio_exposure_report

class TestExposureCalc(unittest.TestCase):
    @patch('engine_portfolio._load_portfolio_rows')
    @patch('engine_portfolio._build_live_position_snapshots')
    def test_get_portfolio_exposure_report(self, mock_snapshots, mock_load):
        mock_load.return_value = []
        # Mock portfolio with NVDA directly (10k) and VOO (100k)
        # VOO holds about 7% NVDA, so NVDA exposure should be 10k + ~7k = ~17k
        mock_snapshots.return_value = [
            {'symbol': 'NVDA', 'market_value_twd': 10000},
            {'symbol': 'VOO', 'market_value_twd': 100000}
        ]
        
        report = get_portfolio_exposure_report()
        print(report)
        self.assertIn("Portfolio True Exposure (Look-through)", report)
        self.assertIn("NVDA", report)

if __name__ == '__main__':
    unittest.main()
