"""Technical indicators. Each value at row i uses only rows <= i (no look-ahead)."""
from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    change = series.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)
