import json
import pandas as pd
from src.dashboard_data import read_json, read_csv, metric_table, ml_table, date_indexed, numeric_columns


def test_missing_json(tmp_path):
    assert read_json(tmp_path / 'missing.json') == {}


def test_invalid_json(tmp_path):
    p = tmp_path / 'bad.json'
    p.write_text('{bad')
    assert read_json(p) == {}


def test_json_round_trip(tmp_path):
    p = tmp_path / 'ok.json'
    p.write_text(json.dumps({'metrics': {'equal': {'sharpe': .6}}}))
    assert read_json(p)['metrics']['equal']['sharpe'] == .6


def test_missing_csv(tmp_path):
    assert read_csv(tmp_path / 'missing.csv').empty


def test_metric_table():
    frame = metric_table({'metrics': {'equal': {'sharpe': .6}, 'SPY_buy_hold': {'sharpe': .7}}})
    assert len(frame) == 2 and 'strategy' in frame


def test_metric_table_invalid():
    assert metric_table({'metrics': []}).empty


def test_ml_table():
    frame = ml_table({'results': {'SPY': {'ml': {'return_pct': 12}, 'buy_hold': {'return_pct': 59}, 'roc_auc': .49}}})
    assert frame.iloc[0]['ml_return_pct'] == 12


def test_ml_table_invalid():
    assert ml_table({'results': None}).empty


def test_date_indexed():
    frame = date_indexed(pd.DataFrame({'date': ['2025-01-01', '2025-01-02'], 'x': [1, 2]}))
    assert isinstance(frame.index, pd.DatetimeIndex)


def test_numeric_columns():
    frame = numeric_columns(pd.DataFrame({'a': ['1', '2'], 'b': ['x', 'y']}))
    assert list(frame.columns) == ['a']
