import numpy as np
import pandas as pd
import pytest
from src.advanced_strategies import STRATEGIES, signals, backtest_signals, statistics, split_dates


def prices(n=160):
    return pd.Series(np.linspace(100, 140, n), index=pd.date_range('2020-01-01', periods=n, freq='B'))


@pytest.mark.parametrize('name', STRATEGIES)
def test_signals_binary_and_aligned(name):
    p = prices()
    s = signals(p, name)
    assert s.index.equals(p.index)
    assert s.isin([0, 1]).all()


def test_signal_uses_next_period():
    p = pd.Series([100., 110., 121.])
    s = pd.Series([0, 1, 1])
    r = backtest_signals(p, s, fee_bps=0)
    assert r['position'].tolist() == [0, 0, 1]
    assert r['strategy_return'].iloc[1] == 0
    assert r['strategy_return'].iloc[2] == pytest.approx(0.1)


def test_cost_only_when_position_changes():
    p = pd.Series([100., 100., 100., 100.])
    s = pd.Series([1, 1, 0, 0])
    r = backtest_signals(p, s, fee_bps=10)
    assert r['turnover'].tolist() == [0, 1, 0, 1]
    assert r['strategy_return'].sum() == pytest.approx(-0.002)


def test_no_future_leakage_in_momentum():
    p = prices()
    original = signals(p, 'momentum')
    changed = p.copy()
    changed.iloc[-1] *= 2
    revised = signals(changed, 'momentum')
    pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])


def test_chronological_split():
    p = prices()
    train_end, test_start = split_dates(pd.Series(p.index))
    assert train_end < test_start
    assert test_start == p.index[int(len(p) * .7)]


def test_statistics():
    p = prices()
    r = backtest_signals(p, signals(p, 'buy_hold'))
    s = statistics(r)
    assert s['final_equity_usd'] > 10000
    assert s['trades'] == 1


def test_bad_signal_rejected():
    p = prices()
    with pytest.raises(ValueError):
        backtest_signals(p, pd.Series(2, index=p.index))
