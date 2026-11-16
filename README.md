# QuantForge Day 1 — Synthetic ETF Backtesting

A reproducible Python project for learning quantitative trading infrastructure.

**Important:** This dataset is synthetic, not real SPY/QQQ/IWM history. Backtest results are educational and are not evidence of live trading profitability.

## Features
- 1,500 daily OHLCV records (500 per ticker), deterministic synthetic data
- OHLCV validation
- 20/50-day SMA crossover, long/cash
- Next-day position execution approximation (signals shifted one bar)
- 5 bps cost per position change
- Equity curves, Sharpe ratio, drawdown, tests

## Windows PowerShell quick start
```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py
```

Outputs are created in `outputs/`. This is a simplified close-to-close backtest, not an executable brokerage simulator. It ignores slippage, spreads, dividends, market holidays, and realistic fill timing.
