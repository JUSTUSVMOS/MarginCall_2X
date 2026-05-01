# ETF Look-through Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement true exposure calculation by look-through of ETF holdings, integrated directly into the existing `yf_session.py` caching layer.

**Architecture:** Extend `yf_session.py`'s `CachedTicker` with a `get_holdings()` method that caches Top 10 components in SQLite. Update `engine_portfolio.py` to aggregate these components for risk reporting.

**Tech Stack:** Python, yfinance, SQLite, pandas.

---

### Task 1: Database Schema & yf_session.py Upgrade

**Files:**
- Modify: `yf_session.py`
- Test: `test_yf_holdings.py` (New)

- [ ] **Step 1: Update `init_db` to include `etf_holdings` table.**
- [ ] **Step 2: Add `get_holdings()` method to `CachedTicker`.**
    - Include logic for SQLite check, `yf` fetch, and JSON storage.
- [ ] **Step 3: Write a test script `test_yf_holdings.py` to verify caching.**
    ```python
    from yf_session import get_ticker
    def test():
        t = get_ticker("VOO")
        h1 = t.get_holdings()
        print(f"Fetch 1: {len(h1)} items")
        h2 = t.get_holdings()
        print(f"Fetch 2 (Cache): {len(h2)} items")
        assert h1 == h2
    if __name__ == "__main__": test()
    ```
- [ ] **Step 4: Run test and verify.**
- [ ] **Step 5: Commit.**

### Task 2: Portfolio Exposure Engine

**Files:**
- Modify: `engine_portfolio.py`
- Test: `test_exposure_calc.py` (New)

- [ ] **Step 1: Implement `get_portfolio_exposure_report()` in `engine_portfolio.py`.**
    - Loop through positions, call `get_holdings()`, aggregate market value weighted by components.
- [ ] **Step 2: Register as a `@tool()`.**
- [ ] **Step 3: Write a test to verify exposure aggregation.**
    - Mock a portfolio with 1 stock and 1 ETF and check the sums.
- [ ] **Step 4: Commit.**

### Task 3: Fundamental Engine Integration

**Files:**
- Modify: `engine_fundamentals.py`

- [ ] **Step 1: Modify `get_full_fundamental_report` to check for ETF holdings first.**
- [ ] **Step 2: If holdings exist, return an ETF-specific summary instead of stock financials.**
- [ ] **Step 3: Verify with `00918.TW` and `FBCG`.**
- [ ] **Step 4: Commit.**
