"""Run real-data ingestion and benchmark comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.market_data import download_universe, validate_bars
from src.benchmark import compare
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', default='2026-01-01', help='Exclusive end date')
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'])
    parser.add_argument('--offline', action='store_true', help='Reuse previously downloaded CSV')
    args = parser.parse_args()
    path = ROOT / 'data' / 'real_etfs.csv'
    if args.offline:
        if not path.exists():
            raise SystemExit('No cached CSV. Run without --offline first.')
        data = validate_bars(pd.read_csv(path))
    else:
        data = download_universe(args.tickers, args.start, args.end, path)
    out = ROOT / 'outputs' / 'day2'
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for ticker in args.tickers:
        if ticker not in set(data['ticker']):
            raise ValueError(f'{ticker} not present in data')
        result, report = compare(data, ticker)
        result.to_csv(out / f'{ticker}_comparison.csv', index=False)
        plt.figure(figsize=(10, 5))
        plt.plot(result['date'], result['equity'], label='SMA 20/50')
        plt.plot(result['date'], result['benchmark_equity'], label='Buy & hold')
        plt.title(f'{ticker}: SMA crossover vs buy-and-hold')
        plt.ylabel('Portfolio value ($)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f'{ticker}_comparison.png', dpi=130)
        plt.close()
        reports.append(report)
        print(f"{ticker}: strategy {report['strategy']['total_return_pct']}% | buy-and-hold {report['buy_and_hold']['total_return_pct']}%")
    (out / 'summary.json').write_text(json.dumps({'start': args.start, 'end_exclusive': args.end, 'fee_bps': 5, 'results': reports}, indent=2))
    print(f'Saved results to {out}')


if __name__ == '__main__':
    main()
