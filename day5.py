"""Day 5 portfolio allocation and risk comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.portfolio import load_prices, backtest, risk_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/real_etfs.csv')
    parser.add_argument('--train-days', type=int, default=252)
    parser.add_argument('--cost-bps', type=float, default=10.0)
    parser.add_argument('--capital', type=float, default=10000)
    args = parser.parse_args()
    if args.capital <= 0:
        parser.error('--capital must be positive')
    prices = load_prices(args.data)
    print('Loaded ETFs:', ', '.join(prices.columns))
    out = Path('outputs/day5')
    out.mkdir(parents=True, exist_ok=True)
    methods = ['equal', 'inverse_vol', 'min_vol', 'max_sharpe']
    curves = {}
    allocation_tables = []
    metrics = {}
    for method in methods:
        nav, allocations = backtest(prices, method, args.train_days, args.cost_bps)
        curves[method] = nav * args.capital
        allocation_tables.append(allocations)
        metrics[method] = risk_report(nav)
        print(f"{method:12s}: return {metrics[method]['total_return_pct']:7.2f}% | Sharpe {metrics[method]['sharpe']}")
    # Same OOS dates; SPY baseline starts at the same first trade date.
    first_index = next(iter(curves.values())).index
    spy = prices['SPY'].pct_change(fill_method=None).reindex(first_index)
    if spy.isna().any():
        raise ValueError('SPY baseline missing dates')
    baseline = (1 + spy).cumprod() * args.capital
    baseline.name = 'SPY_buy_hold'
    curves['SPY_buy_hold'] = baseline
    metrics['SPY_buy_hold'] = risk_report(baseline / args.capital)
    frame = pd.DataFrame(curves)
    frame.to_csv(out / 'equity_curves.csv', index_label='date')
    pd.concat(allocation_tables, ignore_index=True).to_csv(out / 'allocations.csv', index=False)
    (out / 'summary.json').write_text(json.dumps({'configuration': vars(args), 'metrics': metrics}, indent=2), encoding='utf-8')
    ax = frame.plot(figsize=(12, 6), title='Day 5: Out-of-sample portfolio comparison')
    ax.set(ylabel='Portfolio value ($)', xlabel='Date')
    ax.figure.tight_layout()
    chart_path = out.resolve() / "portfolio_comparison.png"
    ax.figure.savefig(str(chart_path), format="png", dpi=120)
    plt.close(ax.figure)
    print(f'Saved results to {out.resolve()}')


if __name__ == '__main__':
    main()
