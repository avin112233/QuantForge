"""Causal long/cash ETF signals and close-to-close backtests."""
import numpy as np
import pandas as pd

STRATEGIES = ('sma', 'rsi', 'momentum', 'bollinger', 'buy_hold')


def signals(close: pd.Series, strategy: str) -> pd.Series:
    """Signal computed at close t, eligible to hold only from t+1."""
    p = close.astype(float)
    if p.isna().any() or (p <= 0).any():
        raise ValueError('Close prices must be positive and finite')
    if strategy == 'sma':
        a, b = p.rolling(20).mean(), p.rolling(50).mean()
        return (a > b).where(b.notna(), False).astype(int)
    if strategy == 'momentum':
        # 63-day return; stay invested only when positive.
        return (p.pct_change(63) > 0).astype(int)
    if strategy == 'rsi':
        # Wilder RSI(14); stateful entry <30 and exit >55.
        delta = p.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)
        holding = 0
        out = []
        for v in rsi:
            if pd.notna(v):
                if holding == 0 and v < 30:
                    holding = 1
                elif holding == 1 and v > 55:
                    holding = 0
            out.append(holding)
        return pd.Series(out, index=p.index, dtype=int)
    if strategy == 'bollinger':
        mean = p.rolling(20).mean()
        sd = p.rolling(20).std(ddof=0)
        lower = mean - 2 * sd
        holding = 0
        out = []
        for price, low, mid in zip(p, lower, mean):
            if pd.notna(low):
                if holding == 0 and price < low:
                    holding = 1
                elif holding == 1 and price >= mid:
                    holding = 0
            out.append(holding)
        return pd.Series(out, index=p.index, dtype=int)
    if strategy == 'buy_hold':
        return pd.Series(1, index=p.index, dtype=int)
    raise ValueError(f'Unknown strategy: {strategy}')


def backtest_signals(close: pd.Series, signal: pd.Series, fee_bps: float = 5, initial_cash: float = 10000) -> pd.DataFrame:
    """Trade at prior close for the next close-to-close interval (idealized execution).

    Costs charged when positions change; first return is zero. No shorting/leverage.
    """
    if fee_bps < 0 or initial_cash <= 0:
        raise ValueError('Invalid fee or initial capital')
    p = close.astype(float)
    if p.isna().any() or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Invalid close prices')
    if not signal.index.equals(p.index) or signal.isna().any() or not signal.isin([0, 1]).all():
        raise ValueError('Signals must be aligned binary values')
    position = signal.shift(1).fillna(0).astype(int)
    turnover = position.diff().abs().fillna(position.abs())
    asset_return = p.pct_change().fillna(0)
    daily_return = position * asset_return - turnover * fee_bps / 10000
    equity = initial_cash * (1 + daily_return).cumprod()
    return pd.DataFrame({'close': p, 'signal': signal, 'position': position,
                         'turnover': turnover, 'asset_return': asset_return,
                         'strategy_return': daily_return, 'equity': equity})


def statistics(result: pd.DataFrame, initial_cash: float = 10000) -> dict:
    r = result['strategy_return']
    equity = result['equity']
    std = r.std(ddof=1)
    sharpe = float(np.sqrt(252) * r.mean() / std) if pd.notna(std) and std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {'return_pct': round(float((equity.iloc[-1] / initial_cash - 1) * 100), 2),
            'sharpe': round(sharpe, 3),
            'max_drawdown_pct': round(float(drawdown.min() * 100), 2),
            'trades': int(result['turnover'].sum()),
            'final_equity_usd': round(float(equity.iloc[-1]), 2)}


def split_dates(dates: pd.Series, train_fraction: float = 0.7):
    """Return last training date and first test date for one ticker."""
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must be between 0 and 1')
    ordered = pd.to_datetime(dates).sort_values().drop_duplicates().reset_index(drop=True)
    cut = int(len(ordered) * train_fraction)
    if cut < 65 or len(ordered) - cut < 2:
        raise ValueError('Insufficient train/test history')
    return ordered.iloc[cut - 1], ordered.iloc[cut]
