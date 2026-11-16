import pandas as pd
import pytest
from src.strategy import load_prices, backtest, metrics


def test_dataset():
    df = load_prices("data/synthetic_etfs.csv")
    assert len(df) == 1500
    assert set(df.ticker) == {"SPY", "QQQ", "IWM"}


def test_backtest():
    df = load_prices("data/synthetic_etfs.csv")
    x = backtest(df)
    assert len(x) == 500
    assert x["position"].iloc[0] == 0
    assert (x["position"].iloc[1:].to_numpy() == x["signal"].iloc[:-1].to_numpy()).all()
    assert x["equity"].min() > 0
    assert "sharpe" in metrics(x)


def test_invalid_windows():
    df = load_prices("data/synthetic_etfs.csv")
    with pytest.raises(ValueError):
        backtest(df, short=50, long=20)
