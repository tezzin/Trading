"""Baseline strategy: moving-average trend following with an ATR volatility filter.

This is the yardstick that later ML strategies must beat after fees.
"""
from __future__ import annotations

import pandas as pd

from .config import StrategyConfig
from .indicators import atr, sma


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Return per-candle `signal` (1 = want to be long, 0 = flat) and `stop_distance`.

    Both are computed from the candle's close, so they may only be acted on
    from the next candle onward.
    """
    if cfg.fast_period >= cfg.slow_period:
        raise ValueError("fast_period must be shorter than slow_period")

    fast = sma(df["close"], cfg.fast_period)
    slow = sma(df["close"], cfg.slow_period)
    vol = atr(df, cfg.atr_period)

    trending_up = fast > slow
    enough_volatility = (vol / df["close"]) >= cfg.min_atr_pct
    ready = slow.notna() & vol.notna()

    out = pd.DataFrame(index=df.index)
    out["signal"] = (trending_up & enough_volatility & ready).astype(int)
    # Hold through quiet periods once in: exit only when the trend flips.
    out.loc[ready & trending_up & ~enough_volatility, "signal"] = -1  # -1 = hold, no new entry
    out["stop_distance"] = vol * cfg.stop_atr_mult
    return out
