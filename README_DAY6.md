# Day 6 — QuantForge Research Dashboard

Interactive Streamlit interface for the historical/simulated ETF research generated in Days 2–5.

## Setup (Windows PowerShell)

```powershell
cd C:\QuantForge\QuantForge
.\.venv\Scripts\python.exe -m pip install "streamlit>=1.40,<2"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Open the localhost URL displayed by Streamlit (normally http://localhost:8501).

## Inputs

- `outputs/day5/summary.json` — portfolio metrics
- `outputs/day5/equity_curves.csv` — portfolio equity curves
- `outputs/day5/allocations.csv` — portfolio weights
- `outputs/day4/summary.json` — ML results
- `outputs/day2` and `outputs/day3` — optional charts and CSVs

Missing inputs are handled gracefully. Re-run the corresponding earlier day scripts to regenerate artifacts. The dashboard does not fetch prices or train models.

## Limitations

The dashboard displays simulated backtests. It is not a live trading or brokerage application. Performance can be sensitive to fees, data handling, and execution assumptions. Allocation CSV visualization is schema-agnostic; inspect its original columns before interpreting the plotted values as weights.
