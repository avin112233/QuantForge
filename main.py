from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.strategy import load_prices, backtest, metrics


def main():
    root = Path(__file__).resolve().parent
    df = load_prices(root / "data" / "synthetic_etfs.csv")
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    for ticker in ["SPY", "QQQ", "IWM"]:
        result = backtest(df, ticker=ticker)
        result.to_csv(output / f"{ticker}_backtest.csv", index=False)
        print(ticker, metrics(result))
        plt.figure(figsize=(9, 4))
        plt.plot(result["date"], result["equity"], label="Strategy equity")
        plt.title(f"{ticker} synthetic-data SMA crossover backtest")
        plt.ylabel("Portfolio value (USD)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{ticker}_equity.png", dpi=130)
        plt.close()
    print(f"Results saved in: {output}")


if __name__ == "__main__":
    main()
