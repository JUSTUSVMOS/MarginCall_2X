# Design Spec: ETF Look-through & Exposure Engine (Integrated Version)

## 1. Goal
Enable the system to "look through" ETF positions to identify true underlying stock exposure. This improves risk management by calculating aggregated weights of specific stocks (e.g., NVDA, 2330.TW) held both directly and indirectly via ETFs.

## 2. Architecture
Instead of a new module, we will extend the existing `yf_session.py` to handle ETF holdings caching.

### 2.1 Components
- **`yf_session.py` (Upgrade)**: 
    - Add `get_holdings()` method to `CachedTicker`.
    - Implement a specific caching table for ETF holdings (Top 10).
- **`engine_portfolio.py` (Upgrade)**: 
    - Integrates ETF decomposition into the portfolio risk calculation using `yf_session.get_ticker().get_holdings()`.
- **`engine_fundamentals.py` (Upgrade)**: 
    - Adds a tool for users to query ETF components directly.

## 3. Data Schema
Update `.cache/yf_cache_daily.sqlite` (via `yf_session.py`):
```sql
CREATE TABLE IF NOT EXISTS etf_holdings (
    symbol TEXT PRIMARY KEY,
    holdings_json TEXT,
    timestamp DATETIME
);
```

## 4. Logic Flow
### 4.1 ETF Component Fetching (inside `yf_session.py`)
1. `CachedTicker.get_holdings()` checks the `etf_holdings` table.
2. If data exists and is < 7 days old, return the parsed JSON.
3. Otherwise, call `self._ticker.get_funds_data().top_holdings`.
4. Parse the Top 10 results into JSON and update the cache.

### 4.2 Portfolio Exposure Calculation (inside `engine_portfolio.py`)
1. Iterate through all positions in the portfolio.
2. For each position:
   - If **Stock**: Add weight to the exposure map.
   - If **ETF**:
     - Fetch Top 10 holdings via `get_ticker(symbol).get_holdings()`.
     - For each holding: Add `(holding_symbol, weight * etf_weight)` to the exposure map.
3. Aggregate and identify concentration risks.

## 5. Tools & Interface
- **`get_deep_fundamentals(symbol)`**: Will be updated to show ETF components if the symbol is an ETF.
- **`get_portfolio_exposure()`**: New tool in `engine_portfolio.py` for true exposure reporting.

## 6. Testing Strategy
- Test with `VOO`, `0050.TW`, `00918.TW`, `FBCG`.
- Verify the single source of truth in `yf_session.py`.
