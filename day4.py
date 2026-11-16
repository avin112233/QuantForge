"""Run reproducible, per-ETF walk-forward XGBoost research on cached Day 2 CSV."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from src.market_data import validate_bars
from src.ml_research import make_features, walk_forward_predictions, evaluate

ROOT = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=str(ROOT / 'data' / 'real_etfs.csv'))
    ap.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'])
    ap.add_argument('--min-train', type=int, default=504)
    ap.add_argument('--test-size', type=int, default=63)
    ap.add_argument('--threshold', type=float, default=0.55)
    ap.add_argument('--fee-bps', type=float, default=5.0)
    args = ap.parse_args()
    data = validate_bars(pd.read_csv(args.input))
    out = ROOT / 'outputs' / 'day4'
    out.mkdir(parents=True, exist_ok=True)
    reports = {}
    for ticker in args.tickers:
        df = data.loc[data.ticker == ticker].sort_values('date').set_index('date')
        if df.empty:
            raise ValueError(f'Missing ETF {ticker}')
        close = df['close']
        features = make_features(close)
        pred = walk_forward_predictions(features, args.min_train, args.test_size, args.threshold)
        report, equity = evaluate(close, pred, args.fee_bps)
        reports[ticker] = report
        equity.to_csv(out / f'{ticker}_predictions.csv', index_label='date')
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(equity.index, equity.ml_equity, label='XGBoost long/cash')
        ax.plot(equity.index, equity.hold_equity, label='Buy and hold')
        
        ax.set(title=f'{ticker}: walk-forward out-of-sample', ylabel='Simulated portfolio ($)')
        ax.legend()
        fig.tight_layout()

        chart_path = out / f"{str(ticker).strip()}_ml_vs_hold.png"
        fig.savefig(str(chart_path), format="png", dpi=120)
        plt.close(fig)

        print(f"{ticker}: ML {report['ml']['return_pct']}% | Hold {report['buy_hold']['return_pct']}% | AUC {report['roc_auc']}")

    (out / 'summary.json').write_text(
        json.dumps({'configuration': vars(args), 'results': reports}, indent=2)
    )
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
