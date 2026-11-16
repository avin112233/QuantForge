# QuantForge Day 5 — Portfolio Optimization and Risk Analytics

Builds monthly-rebalanced long-only ETF portfolios using equal weights, inverse volatility, minimum variance and maximum in-sample Sharpe. Uses a trailing 252-day daily return window, 10 bps turnover cost by default, and evaluates out-of-sample portfolio equity curves against SPY.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install scipy
```

## Run

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe day5.py
```

Outputs: `outputs/day5/equity_curves.csv`, `allocations.csv`, `summary.json`, and `portfolio_comparison.png`.

## Research limitations

- Uses adjusted or unadjusted close prices as provided by the Day 2 dataset; verify dividend/split treatment before interpreting performance.
- Assumes fractional ETF shares, daily close-to-close returns, no borrowing, no leverage and no market impact.
- Rebalance decisions use only previous-day observations. Orders are modeled at the first subsequent close-to-close return period; this is an idealized execution assumption.
- Turnover cost is charged as a fixed basis-point rate; no separate slippage or bid-ask model.
- The 252-day rolling window and hyperparameters are fixed; no tuning on the test results.
- VaR and CVaR are empirical daily historical estimates, not forecasts or guarantees.
- Annualized return and Sharpe use 252 trading days and are approximate.
