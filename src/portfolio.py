"""Long-only, monthly-rebalanced ETF portfolio research utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252

def load_prices(path: str) -> pd.DataFrame:
    """Accept long-format (date,ticker,close) or wide-format ETF close prices."""
    raw = pd.read_csv(path)
    names = {str(c).lower().strip(): c for c in raw.columns}
    date_col = next((names[k] for k in ('date', 'datetime', 'timestamp') if k in names), raw.columns[0])
    ticker_col = next((names[k] for k in ('ticker', 'symbol', 'etf') if k in names), None)
    price_col = next((names[k] for k in ('close', 'adj_close', 'adj close', 'price') if k in names), None)
    if ticker_col is not None and price_col is not None:
        df = raw.pivot_table(index=date_col, columns=ticker_col, values=price_col, aggfunc='last')
    else:
        df = raw.set_index(date_col)
        # In wide OHLC tables, retain only requested ETF price columns.
        df = df.loc[:, [c for c in df.columns if str(c).upper() in {'SPY','QQQ','IWM','DIA','VTI'}]]
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.loc[df.index.notna()].sort_index()
    df.columns = df.columns.astype(str).str.upper()
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how='any')
    df = df.loc[(df > 0).all(axis=1)]
    if df.shape[1] < 2 or len(df) < 100:
        raise ValueError('Need at least two ETF columns and 100 complete positive-price observations')
    return df

def weights_for(returns: pd.DataFrame, method: str, risk_free_rate: float = 0.0) -> np.ndarray:
    """Estimate long-only fully-invested weights from historical daily returns."""
    if returns.empty or returns.isna().any().any():
        raise ValueError('Training returns must be nonempty and finite')
    n = returns.shape[1]
    equal = np.full(n, 1 / n)
    if method == 'equal':
        return equal
    vol = returns.std(ddof=1).to_numpy()
    if method == 'inverse_vol':
        inv = 1 / np.maximum(vol, 1e-8)
        return inv / inv.sum()
    if method not in ('min_vol', 'max_sharpe'):
        raise ValueError(f'Unknown method: {method}')
    cov = returns.cov().to_numpy() * TRADING_DAYS
    cov += np.eye(n) * 1e-8
    mu = returns.mean().to_numpy() * TRADING_DAYS
    def objective(w):
        variance = max(float(w @ cov @ w), 1e-12)
        if method == 'min_vol':
            return variance
        return -(float(w @ mu) - risk_free_rate) / np.sqrt(variance)
    result = minimize(objective, equal, method='SLSQP', bounds=[(0, 1)] * n,
                      constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                      options={'maxiter': 1000, 'ftol': 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        return equal  # explicit, deterministic fallback
    w = np.clip(result.x, 0, 1)
    return w / w.sum()

def backtest(prices: pd.DataFrame, method: str, train_days: int = 252,
             cost_bps: float = 10.0, risk_free_rate: float = 0.0) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly decision at prior close; next session's close-to-close returns accrue.

    Weight drift is accounted for between rebalances. Cost is charged on the
    first return day following each rebalance, including initial deployment.
    """
    if train_days < 30 or len(prices) <= train_days:
        raise ValueError('Insufficient history for chosen training window')
    if cost_bps < 0:
        raise ValueError('Transaction costs cannot be negative')
    returns = prices.pct_change(fill_method=None).dropna()
    if len(returns) <= train_days:
        raise ValueError('Insufficient return history')
    values, dates, records = [], [], []
    nav = 1.0
    held = np.zeros(returns.shape[1])
    prev_month = None
    for i in range(train_days, len(returns)):
        date = returns.index[i]
        month = (date.year, date.month)
        if month != prev_month:
            train = returns.iloc[i-train_days:i]  # never includes today's return
            target = weights_for(train, method, risk_free_rate)
            turnover = float(np.abs(target - held).sum())
            nav *= 1 - turnover * cost_bps / 10000
            if nav <= 0:
                raise ValueError('Costs exhausted portfolio value')
            held = target.copy()
            records.append({'date': date, 'method': method, 'turnover': turnover,
                            **dict(zip(returns.columns, target))})
            prev_month = month
        asset_r = returns.iloc[i].to_numpy()
        gross_r = float(held @ asset_r)
        nav *= 1 + gross_r
        held = held * (1 + asset_r) / (1 + gross_r)  # weights drift naturally
        dates.append(date)
        values.append(nav)
    return pd.Series(values, index=pd.DatetimeIndex(dates), name=method), pd.DataFrame(records)

def risk_report(nav: pd.Series, risk_free_rate: float = 0.0) -> dict:
    r = nav.pct_change().dropna()
    if r.empty:
        raise ValueError('Need at least two portfolio observations')
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann = float((nav.iloc[-1] / nav.iloc[0]) ** (TRADING_DAYS / len(r)) - 1)
    drawdown = nav / nav.cummax() - 1
    q = float(r.quantile(.05))
    return {'total_return_pct': round((nav.iloc[-1] - 1) * 100, 2),
            'annualized_return_pct': round(ann * 100, 2),
            'annualized_volatility_pct': round(vol * 100, 2),
            'sharpe': round((ann - risk_free_rate) / vol, 3) if vol > 0 else None,
            'max_drawdown_pct': round(float(drawdown.min()) * 100, 2),
            'daily_var_95_pct': round(-q * 100, 2),
            'daily_cvar_95_pct': round(-float(r[r <= q].mean()) * 100, 2)}
