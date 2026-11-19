"""Read QuantForge research artifacts without re-running backtests."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def metric_table(summary: dict) -> pd.DataFrame:
    metrics = summary.get('metrics', {})
    if not isinstance(metrics, dict):
        return pd.DataFrame()
    rows = []
    for name, values in metrics.items():
        if isinstance(values, dict):
            rows.append({'strategy': name, **values})
    return pd.DataFrame(rows)


def ml_table(summary: dict) -> pd.DataFrame:
    results = summary.get('results', {})
    if not isinstance(results, dict):
        return pd.DataFrame()
    rows = []
    for ticker, values in results.items():
        if not isinstance(values, dict):
            continue
        ml = values.get('ml', {})
        hold = values.get('buy_hold', {})
        rows.append({
            'ticker': ticker,
            'ml_return_pct': ml.get('return_pct') if isinstance(ml, dict) else None,
            'hold_return_pct': hold.get('return_pct') if isinstance(hold, dict) else None,
            'roc_auc': values.get('roc_auc'),
        })
    return pd.DataFrame(rows)


def date_indexed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    for col in ('date', 'Date', 'dt', 'timestamp', 'Unnamed: 0'):
        if col in result.columns:
            parsed = pd.to_datetime(result[col], errors='coerce')
            if parsed.notna().sum() >= max(1, len(result) // 2):
                result.index = parsed
                return result.drop(columns=[col]).sort_index()
    return result


def numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
