import pandas as pd
import pytest
from src.market_data import validate_bars
from src.benchmark import compare


def sample(n=80):
    dates = pd.bdate_range('2024-01-01', periods=n)
    closes = [100 + i * 0.2 for i in range(n)]
    return pd.DataFrame({'date': dates, 'ticker': 'SPY', 'open': closes,
                         'high': [p+1 for p in closes], 'low': [p-1 for p in closes],
                         'close': closes, 'volume': 1000})


def test_valid_bars():
    assert len(validate_bars(sample())) == 80


def test_duplicate_bars_rejected():
    df = sample()
    with pytest.raises(ValueError, match='Duplicate'):
        validate_bars(pd.concat([df, df.iloc[[0]]]))


def test_missing_prices_rejected():
    df = sample()
    df.loc[0, 'close'] = None
    with pytest.raises(ValueError, match='Missing'):
        validate_bars(df)


def test_invalid_ohlc_rejected():
    df = sample()
    df.loc[0, 'high'] = 1
    with pytest.raises(ValueError, match='Invalid OHLC'):
        validate_bars(df)


def test_negative_volume_rejected():
    df = sample()
    df.loc[0, 'volume'] = -1
    with pytest.raises(ValueError, match='Negative volume'):
        validate_bars(df)


def test_benchmark_is_aligned():
    bars = validate_bars(sample())
    result, report = compare(bars, 'SPY')
    assert len(result) == 80
    assert result['benchmark_equity'].iloc[-1] > 10000
    assert report['ticker'] == 'SPY'
    assert result['position'].iloc[0] == 0
