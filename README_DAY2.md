# QuantForge — Day 2: Real market data and benchmarking

Day 2 adds `src/market_data.py`, `src/benchmark.py`, `day2.py`, and `tests/test_day2.py` without modifying Day 1.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install 'yfinance>=0.2.60,<2'
```

## Tests (no internet required)

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Run (internet required)

```powershell
.\.venv\Scripts\python.exe day2.py
```

The default period is 2020-01-01 to 2026-01-01 (end exclusive). To rerun without downloading:

```powershell
.\.venv\Scripts\python.exe day2.py --offline
```

Outputs: `data/real_etfs.csv` and `outputs/day2/` with comparison CSVs, charts and `summary.json`.

## Methodology and limitations

- yfinance is an unofficial data interface and may be rate-limited or unavailable. Check Yahoo's data licensing terms before redistribution.
- `auto_adjust=True` applies corporate-action adjustments to OHLC, so these are adjusted historical bars, not raw exchange OHLC.
- The strategy uses the previous session's close signal for the next close-to-close return, avoiding same-bar signal leakage. A production execution model would simulate next-open fills, slippage, spreads, and trading constraints.
- Buy-and-hold assumes purchase at the first closing price and a one-time entry fee. It is compared over the same date range as the strategy.
- No guarantee of profitability. Historical performance is not predictive of future results.
- Day 2 still uses fixed SMA parameters, not walk-forward out-of-sample optimization.
