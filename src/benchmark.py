"""Aligned buy-and-hold benchmark and risk reporting."""
import numpy as np
import pandas as pd
from src.strategy import backtest


def summary(returns: pd.Series, equity: pd.Series) -> dict:
    daily = returns.astype(float)
    vol = daily.std(ddof=1)
    dd = equity / equity.cummax() - 1
    return {
        'total_return_pct': round(float((equity.iloc[-1] / 10000 - 1) * 100), 2),
        'sharpe': round(float(np.sqrt(252) * daily.mean() / vol), 3) if pd.notna(vol) and vol > 0 else 0.0,
        'max_drawdown_pct': round(float(dd.min() * 100), 2),
        'final_equity_usd': round(float(equity.iloc[-1]), 2),
    }


def compare(df: pd.DataFrame, ticker: str, short=20, long=50, fee_bps=5):
    result = backtest(df, ticker=ticker, short=short, long=long, fee_bps=fee_bps)
    # Buy at the first close; first daily return is zero; no subsequent trades.
    # Apply a one-time entry cost to the benchmark for consistent fee assumptions.
    asset = result['asset_return'].copy()
    benchmark_returns = asset.copy()
    benchmark_returns.iloc[0] = -fee_bps / 10000
    result['benchmark_return'] = benchmark_returns
    result['benchmark_equity'] = 10000 * (1 + benchmark_returns).cumprod()
    strategy = summary(result['strategy_return'], result['equity'])
    benchmark = summary(result['benchmark_return'], result['benchmark_equity'])
    return result, {'ticker': ticker, 'strategy': strategy, 'buy_and_hold': benchmark}
