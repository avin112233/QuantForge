"""Offline Day 3 strategy comparison using Day 2 cached ETF prices."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from src.market_data import validate_bars
from src.advanced_strategies import STRATEGIES, signals, backtest_signals, statistics, split_dates

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(ROOT / 'data' / 'real_etfs.csv'))
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'])
    parser.add_argument('--train-fraction', type=float, default=0.7)
    parser.add_argument('--fee-bps', type=float, default=5)
    args = parser.parse_args()

    data = validate_bars(pd.read_csv(args.input))
    data['date'] = pd.to_datetime(data['date'])
    out = ROOT / 'outputs' / 'day3'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ticker in args.tickers:
        frame = data.loc[data['ticker'] == ticker].sort_values('date').set_index('date')
        if frame.empty:
            raise ValueError(f'No data for {ticker}')
        _, test_start = split_dates(pd.Series(frame.index), args.train_fraction)
        print(f'{ticker}: test starts {test_start.date()}')
        fig, ax = plt.subplots(figsize=(11, 5))
        for strategy in STRATEGIES:
            # Compute indicators on full history for warm-up; evaluate only on test rows.
            sig = signals(frame['close'], strategy)
            result = backtest_signals(frame['close'], sig, args.fee_bps)
            # Test is a fresh $10k portfolio, with position already established by
            # training history. Rebase equity using test-period daily returns.
            test = result.loc[result.index >= test_start].copy()
            test['equity'] = 10000 * (1 + test['strategy_return']).cumprod()
            report = statistics(test)
            rows.append({'ticker': ticker, 'strategy': strategy, 'test_start': str(test_start.date()),
                         'test_end': str(test.index[-1].date()), **report})
            ax.plot(test.index, test['equity'], label=strategy)
        ax.set(title=f'{ticker}: out-of-sample strategy comparison', ylabel='Simulated equity ($)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f'{ticker}_strategies.png', dpi=130)
        plt.close(fig)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / 'strategy_summary.csv', index=False)
    (out / 'config.json').write_text(json.dumps({'input': args.input, 'train_fraction': args.train_fraction,
                                                  'fee_bps': args.fee_bps, 'strategies': STRATEGIES}, indent=2))
    print('\nOut-of-sample returns (%):')
    print(summary.pivot(index='ticker', columns='strategy', values='return_pct').round(2).to_string())
    print(f'\nSaved results to {out}')


if __name__ == '__main__':
    main()
