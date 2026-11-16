# QuantForge Day 3 — Advanced strategies and out-of-sample comparison

## Install and run (PowerShell)

From `C:\QuantForge\QuantForge` (with Day 1 and Day 2 already installed):

```powershell
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe day3.py
Get-Content .\outputs\day3\strategy_summary.csv -TotalCount 10
Invoke-Item .\outputs\day3\SPY_strategies.png
```

Requires `data/real_etfs.csv` produced by Day 2. If missing, run `python day2.py` first.

## Methodology

- Five long-or-cash strategies: SMA 20/50, RSI(14) mean reversion (enter <30, exit >55), 63-day momentum, Bollinger(20, 2 std) mean reversion, buy-and-hold.
- All strategy signals use current and past closes only; holdings lag signals by one observation to prevent same-bar look-ahead.
- Chronological 70/30 split per ticker; indicator history before the split is used only for warm-up. Results are computed on the last 30% of observations.
- A 5-basis-point fee is applied per unit of position turnover. Simplified close-to-close fills; no bid/ask spread, slippage, tax, or market impact beyond this fee.
- Starting test equity is $10,000 for each independent ETF/strategy simulation. Positions may carry from the pre-test warm-up, so no separate fee is charged at the split boundary for an already-open position.
- This is an **out-of-sample evaluation split, not a trained/optimized model**. No parameters are fitted on the training period yet. Avoid tuning parameters against the test results.
- Yahoo Finance daily prices may be unadjusted; dividends/splits and total-return assumptions should be verified before making investment claims.

## Outputs

`outputs/day3/strategy_summary.csv`, `outputs/day3/config.json`, and per-ticker equity charts. Outputs are intentionally ignored by Git.
