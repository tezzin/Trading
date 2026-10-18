"""Download and store OHLCV candles. Public market data only, no API keys."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
DATA_DIR = Path("data")


def candle_path(symbol: str, timeframe: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{symbol.replace('/', '-')}_{timeframe}.csv"


def fetch_ohlcv(symbol: str, timeframe: str, since: str, exchange_id: str = "binance",
                until: str | None = None, limit: int = 1000) -> pd.DataFrame:
    """Page through the exchange's candle history from `since` (ISO date) to `until` or now."""
    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    until_ms = exchange.parse8601(f"{until}T00:00:00Z") if until else exchange.milliseconds()
    step_ms = exchange.parse_timeframe(timeframe) * 1000

    rows: list[list] = []
    cursor = since_ms
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step_ms
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df[df["timestamp"] < until_ms]
    return clean_candles(df)


def clean_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, de-duplicate and index candles by UTC time."""
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.index.name = "time"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def save_candles(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    epoch = pd.Timestamp(0, tz="UTC")
    out.insert(0, "timestamp", (out.index - epoch) // pd.Timedelta(milliseconds=1))
    out.to_csv(path, index=False)


def load_candles(path: Path) -> pd.DataFrame:
    return clean_candles(pd.read_csv(path))
