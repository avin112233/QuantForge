import numpy as np
import pandas as pd
import pytest
from src.portfolio import weights_for, backtest, risk_report


def sample_prices(n=350):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2023-01-02', periods=n)
    returns = rng.normal(.0004, .01, (n, 3))
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates,
                        columns=['SPY', 'QQQ', 'IWM'])

@pytest.mark.parametrize('method', ['equal', 'inverse_vol', 'min_vol', 'max_sharpe'])
def test_weights(method):
    r = sample_prices().pct_change().dropna()
    w = weights_for(r, method)
    assert len(w) == 3
    assert np.all(w >= -1e-10)
    assert np.all(w <= 1 + 1e-10)
    assert np.isclose(w.sum(), 1)

def test_equal_weights():
    w = weights_for(sample_prices().pct_change().dropna(), 'equal')
    assert np.allclose(w, np.ones(3) / 3)

def test_unknown_method():
    with pytest.raises(ValueError):
        weights_for(sample_prices().pct_change().dropna(), 'invalid')

def test_cost_reduces_nav():
    p = sample_prices()
    free, _ = backtest(p, 'equal', 100, 0)
    paid, _ = backtest(p, 'equal', 100, 10)
    assert paid.iloc[-1] < free.iloc[-1]

def test_rebalance_monthly_and_no_lookahead():
    p = sample_prices()
    nav, alloc = backtest(p, 'equal', 100, 0)
    assert len(nav) > 0
    assert len(alloc) == len(set(zip(nav.index.year, nav.index.month)))
    # Changing the final price cannot affect any earlier portfolio NAV.
    changed = p.copy()
    changed.iloc[-1, 0] *= 1.5
    nav_changed, _ = backtest(changed, 'equal', 100, 0)
    pd.testing.assert_series_equal(nav.iloc[:-1], nav_changed.iloc[:-1])

def test_report():
    nav, _ = backtest(sample_prices(), 'inverse_vol', 100)
    result = risk_report(nav)
    assert 'max_drawdown_pct' in result
    assert 'daily_cvar_95_pct' in result
    assert result['max_drawdown_pct'] <= 0

def test_bad_cost():
    with pytest.raises(ValueError):
        backtest(sample_prices(), 'equal', 100, -1)
