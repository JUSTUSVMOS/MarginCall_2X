import sys
import pandas as pd
import numpy as np
try:
    from statsmodels.tsa.stattools import adfuller
    print("statsmodels available")
except ImportError:
    print("statsmodels missing")

from engine_market import compute_pair_trade_signal, compute_factor_snapshot
from engine_portfolio import compute_inverse_vol_weights

print("Checking pair trade signal (AAPL vs MSFT)...")
print(compute_pair_trade_signal("AAPL", "MSFT", lookback=60))

print("\nChecking factor snapshot (AAPL)...")
print(compute_factor_snapshot("AAPL"))

print("\nChecking inverse-vol weights (AAPL, MSFT)...")
print(compute_inverse_vol_weights(["AAPL", "MSFT"], lookback=60))
