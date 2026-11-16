import numpy as np
import pandas as pd
import pytest
from src.ml_research import make_features, walk_forward_predictions, evaluate


def prices(n=720):
    idx = pd.bdate_range('2021-01-01', periods=n)
    return pd.Series(100*np.exp(np.cumsum(np.random.default_rng(42).normal(.0002, .01, n))), index=idx)


def test_feature_labels_next_day():
    p = prices()
    f = make_features(p)
    assert f['target'].iloc[-1] != f['target'].iloc[-1]  # unknown future
    assert f['target'].iloc[100] == int(p.iloc[101] > p.iloc[100])


def test_feature_prefix_invariant():
    p = prices()
    a = make_features(p.iloc[:200])
    b = make_features(p)
    pd.testing.assert_frame_equal(a.iloc[:199][list(a.columns[:-1])], b.iloc[:199][list(b.columns[:-1])])


def test_bad_prices_rejected():
    p = prices()
    p.iloc[10] = -1
    with pytest.raises(ValueError):
        make_features(p)


def test_walk_forward_no_future_training():
    f = make_features(prices())
    pred = walk_forward_predictions(f, min_train=150, test_size=100)
    assert not pred.empty
    assert all(pd.to_datetime(pred.train_last_date) < pred.index)
    assert pred.signal.isin([0, 1]).all()


def test_walk_forward_reproducible():
    f = make_features(prices())
    a = walk_forward_predictions(f, min_train=150, test_size=150)
    b = walk_forward_predictions(f, min_train=150, test_size=150)
    pd.testing.assert_frame_equal(a, b)


def test_evaluation_costs_and_alignment():
    p = prices(150)
    dates = p.index[50:100]
    pred = pd.DataFrame({'prob_up': .8, 'target': (p.shift(-1) > p).loc[dates].astype(int), 'signal': 1}, index=dates)
    zero, _ = evaluate(p, pred, fee_bps=0)
    paid, result = evaluate(p, pred, fee_bps=10)
    assert paid['ml']['return_pct'] < zero['ml']['return_pct']
    assert result.index.equals(dates)
    assert paid['test_rows'] == 50


def test_invalid_walk_forward_settings():
    with pytest.raises(ValueError):
        walk_forward_predictions(make_features(prices()), min_train=20)
