# QuantForge Day 4 — XGBoost walk-forward ETF research

Research-only, **not investment advice**. Uses the Day 2 cached `data/real_etfs.csv` and does not place trades.

## Setup (PowerShell, from repository root)

```powershell
.\.venv\Scripts\python.exe -m pip install "scikit-learn>=1.6,<2" "xgboost>=3,<4"
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe day4.py
```

Python 3.14 needs recent wheels. If a package fails to install, use Python 3.12/3.13 in a separate virtual environment instead of modifying system Python.

## Design

- Seven causal features computed at the end of day *t*; target = direction of close *t+1*.
- Independent model per ETF. Expanding-window XGBoost fits on prior labeled rows, with a one-row embargo before each 63-day test fold.
- Each out-of-sample prediction at *t* is applied to the *t → t+1* return; no same-day return capture.
- Long/cash decision when probability exceeds 0.55. 5 bps cost per position change; buy-and-hold pays initial entry cost.
- All metrics and charts use identical test dates for ML and buy-and-hold. Reports saved to `outputs/day4/`.
- No tuning on test results; parameters are fixed ex ante. Any future hyperparameter selection should use a nested time-series validation scheme.

## Limitations

Close-to-close execution is idealized: actual close execution with features based on that close is not directly achievable. A more realistic future iteration should use next-open execution with appropriately adjusted open prices, slippage, and liquidity constraints. Yahoo Finance data is research-grade, not exchange-certified. This example ignores tax, dividends not represented by the data, financing, and operational constraints. AUC and returns can be poor even when code is correct.
