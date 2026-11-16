"""Download and validate daily adjusted OHLCV bars."""
from pathlib import Path
import pandas as pd

COLUMNS = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']


def validate_bars(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    x = df[COLUMNS].copy()
    x['date'] = pd.to_datetime(x['date'], errors='coerce')
    if x['date'].isna().any() or x['ticker'].isna().any() or x['ticker'].astype(str).str.strip().eq('').any():
        raise ValueError('Missing or invalid date/ticker')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        x[col] = pd.to_numeric(x[col], errors='coerce')
    if x[['open','high','low','close','volume']].isna().any().any():
        raise ValueError('Missing or nonnumeric OHLCV values')
    if x.duplicated(['ticker','date']).any():
        raise ValueError('Duplicate ticker/date bars')
    if (x[['open','high','low','close']] <= 0).any().any():
        raise ValueError('Nonpositive prices')
    if (x['volume'] < 0).any():
        raise ValueError('Negative volume')
    if ((x['high'] < x[['open','low','close']].max(axis=1)) | (x['low'] > x[['open','high','close']].min(axis=1))).any():
        raise ValueError('Invalid OHLC range')
    return x.sort_values(['ticker','date']).reset_index(drop=True)


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    # auto_adjust applies corporate-action adjustments to OHLC, not just close.
    raw = yf.download(ticker, start=start, end=end, auto_adjust=False,
                      progress=False, threads=False, actions=False)
    if raw.empty:
        raise ValueError(f'No price history returned for {ticker}')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw.columns = [str(c).lower() for c in raw.columns]
    raw['ticker'] = ticker
    return validate_bars(raw)


def download_universe(tickers: list[str], start: str, end: str, destination: Path) -> pd.DataFrame:
    if not tickers or len(tickers) != len(set(tickers)):
        raise ValueError('Provide unique, nonempty ticker symbols')
    frames = []
    for ticker in tickers:
        print(f'Downloading {ticker}...')
        frames.append(download_ticker(ticker, start, end))
    combined = validate_bars(pd.concat(frames, ignore_index=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False, date_format='%Y-%m-%d')
    return combined
