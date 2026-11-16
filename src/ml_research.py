"""Leakage-aware, expanding-window ETF direction research (educational only)."""
import numpy as np
import pandas as pd

FEATURES = ('ret_1', 'ret_5', 'ret_20', 'vol_20', 'ma_gap_20', 'ma_gap_50', 'rsi_14')


def make_features(close: pd.Series) -> pd.DataFrame:
    """Features at close t; label is whether close t+1 exceeds close t."""
    p = close.astype(float)
    if p.isna().any() or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Close prices must be positive and finite')
    if not p.index.is_monotonic_increasing or p.index.has_duplicates:
        raise ValueError('Dates must be sorted and unique')
    daily = p.pct_change()
    delta = p.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = up / down.replace(0, np.nan)
    rsi = (100 - 100/(1+rs)).mask((down == 0) & (up > 0), 100).mask((down == 0) & (up == 0), 50)
    f = pd.DataFrame(index=p.index)
    f['ret_1'] = daily
    f['ret_5'] = p.pct_change(5)
    f['ret_20'] = p.pct_change(20)
    f['vol_20'] = daily.rolling(20).std()
    f['ma_gap_20'] = p / p.rolling(20).mean() - 1
    f['ma_gap_50'] = p / p.rolling(50).mean() - 1
    f['rsi_14'] = rsi / 100
    # Last row has no known label and is excluded from historical evaluation.
    future = p.shift(-1)
    f['target'] = (future > p).astype(float).where(future.notna())
    return f.replace([np.inf, -np.inf], np.nan)


def walk_forward_predictions(features: pd.DataFrame, min_train: int = 504, test_size: int = 63,
                             threshold: float = 0.55) -> pd.DataFrame:
    """Non-overlapping expanding-window folds; no fitting on current/future labels.

    A row at t is only eligible for training when its t+1 label is observed
    by the close immediately preceding the test fold.
    """
    from xgboost import XGBClassifier
    if min_train < 100 or test_size < 1 or not 0 < threshold < 1:
        raise ValueError('Invalid walk-forward parameters')
    x = features.copy()
    if not x.index.is_monotonic_increasing or x.index.has_duplicates:
        raise ValueError('Feature index must be sorted and unique')
    x = x.dropna(subset=list(FEATURES) + ['target'])
    if len(x) < min_train + 2:
        raise ValueError('Not enough labeled observations')
    predictions = []
    # Reserve one observation between training and test to avoid using
    # the immediately preceding day's next-day label at decision time.
    for start in range(min_train + 1, len(x), test_size):
        train = x.iloc[:start-1]
        test = x.iloc[start:start+test_size]
        if test.empty:
            continue
        if train['target'].nunique() < 2:
            raise ValueError('Training fold contains only one target class')
        model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05,
                              subsample=0.9, colsample_bytree=0.9, objective='binary:logistic',
                              eval_metric='logloss', random_state=42, n_jobs=1)
        model.fit(train[list(FEATURES)], train['target'].astype(int))
        prob = model.predict_proba(test[list(FEATURES)])[:, 1]
        predictions.append(pd.DataFrame({'prob_up': prob, 'target': test['target'].astype(int),
                                         'signal': (prob >= threshold).astype(int),
                                         'train_last_date': str(train.index[-1].date())}, index=test.index))
    if not predictions:
        raise ValueError('No test folds generated')
    return pd.concat(predictions)


def evaluate(close: pd.Series, pred: pd.DataFrame, fee_bps: float = 5.0) -> tuple[dict, pd.DataFrame]:
    """Test next-day returns: signal at close t acts over t->t+1.

    The final prediction with no next close is excluded. A turnover cost is
    charged on each signal change, including initial entry; no terminal exit.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    if fee_bps < 0:
        raise ValueError('fee_bps must be nonnegative')
    p = close.astype(float)
    next_ret = p.shift(-1) / p - 1
    joined = pred.join(next_ret.rename('next_return'), how='left').dropna(subset=['next_return']).copy()
    if joined.empty:
        raise ValueError('No test predictions with observed future prices')
    position = joined['signal'].astype(int)
    turnover = position.diff().abs().fillna(position.iloc[0]).astype(float)
    joined['ml_return'] = position * joined['next_return'] - turnover * fee_bps / 10000
    # Benchmark uses the same exact test intervals; initial entry cost included.
    joined['hold_return'] = joined['next_return']
    joined.iloc[0, joined.columns.get_loc('hold_return')] -= fee_bps / 10000
    joined['ml_equity'] = 10000 * (1 + joined['ml_return']).cumprod()
    joined['hold_equity'] = 10000 * (1 + joined['hold_return']).cumprod()
    def metrics(r, eq):
        sd = r.std(ddof=1)
        return {'return_pct': round(float((eq.iloc[-1] / 10000 - 1) * 100), 2),
                'sharpe': round(float(np.sqrt(252)*r.mean()/sd), 3) if pd.notna(sd) and sd > 0 else 0.0,
                'max_drawdown_pct': round(float((eq / eq.cummax() - 1).min() * 100), 2)}
    y = joined['target'].astype(int)
    report = {'test_rows': len(joined), 'accuracy': round(float(accuracy_score(y, position)), 4),
              'precision': round(float(precision_score(y, position, zero_division=0)), 4),
              'recall': round(float(recall_score(y, position, zero_division=0)), 4),
              'roc_auc': round(float(roc_auc_score(y, joined['prob_up'])), 4) if y.nunique() == 2 else None,
              'trades': int(turnover.sum()), 'ml': metrics(joined['ml_return'], joined['ml_equity']),
              'buy_hold': metrics(joined['hold_return'], joined['hold_equity'])}
    return report, joined
