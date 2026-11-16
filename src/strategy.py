import numpy as np
import pandas as pd

REQUIRED = {"date", "ticker", "open", "high", "low", "close", "volume"}


def load_prices(path):
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(REQUIRED - {"date", "ticker"})].isna().any().any():
        raise ValueError("Missing numeric prices or volumes")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate ticker/date records")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (df["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if ((df["high"] < df[["open", "close", "low"]].max(axis=1)) | (df["low"] > df[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Invalid OHLC bars")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def backtest(df, ticker="SPY", short=20, long=50, fee_bps=5):
    if not 0 < short < long:
        raise ValueError("Require 0 < short < long")
    if fee_bps < 0:
        raise ValueError("fee_bps must be nonnegative")
    x = df.loc[df["ticker"] == ticker].sort_values("date").copy()
    if len(x) < long + 2:
        raise ValueError("Not enough observations")
    x["sma_short"] = x["close"].rolling(short).mean()
    x["sma_long"] = x["close"].rolling(long).mean()
    x["signal"] = (x["sma_short"] > x["sma_long"]).astype(int)
    # Today's close determines tomorrow's position; avoids same-bar look-ahead.
    x["position"] = x["signal"].shift(1).fillna(0)
    x["asset_return"] = x["close"].pct_change().fillna(0)
    x["turnover"] = x["position"].diff().abs().fillna(x["position"].abs())
    x["strategy_return"] = x["position"] * x["asset_return"] - x["turnover"] * fee_bps / 10000
    x["equity"] = 10000 * (1 + x["strategy_return"]).cumprod()
    return x


def metrics(x):
    r = x["strategy_return"]
    equity = x["equity"]
    total = equity.iloc[-1] / equity.iloc[0] - 1
    vol = r.std(ddof=1)
    sharpe = float(np.sqrt(252) * r.mean() / vol) if vol > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {"total_return_pct": round(total * 100, 2), "sharpe": round(sharpe, 3), "max_drawdown_pct": round(drawdown.min() * 100, 2), "final_equity_usd": round(equity.iloc[-1], 2), "trades": int(x["turnover"].sum())}
