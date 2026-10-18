"""Rule-based strategies. Later layers (the ML filter) must beat these after fees.

Every strategy returns, per candle, a `signal` and a `stop_distance`:
  1 = enter long (or stay long), 0 = exit (or stay flat), -1 = keep the current position.
Both are computed from the candle's close, so they may only be acted on from the next candle.
"""
from __future__ import annotations

import pandas as pd

from .config import StrategyConfig
from .indicators import atr, sma


def ma_cross(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Phase 1 baseline: long while the fast average is above the slow one."""
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
    out.loc[ready & trending_up & ~enough_volatility, "signal"] = -1
    out["stop_distance"] = vol * cfg.stop_atr_mult
    return out


def breakout(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Channel breakout: buy new highs in an uptrend, sell when price breaks the recent low.

    Trades far less often than ma_cross, so fees eat less of each win.
    """
    if cfg.exit_period >= cfg.entry_period:
        raise ValueError("exit_period must be shorter than entry_period")

    close = df["close"]
    prior_high = df["high"].rolling(cfg.entry_period).max().shift(1)
    prior_low = df["low"].rolling(cfg.exit_period).min().shift(1)
    trend = sma(close, cfg.trend_period)
    vol = atr(df, cfg.atr_period)
    ready = prior_high.notna() & trend.notna() & vol.notna()

    enter = ready & (close > prior_high) & (close > trend) & ((vol / close) >= cfg.min_atr_pct)
    leave = ready & (close < prior_low)

    out = pd.DataFrame(index=df.index)
    out["signal"] = -1
    out.loc[enter, "signal"] = 1
    out.loc[leave | ~ready, "signal"] = 0
    out["stop_distance"] = vol * cfg.stop_atr_mult
    return out


STRATEGIES = {"ma_cross": ma_cross, "breakout": breakout}


def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    try:
        strategy = STRATEGIES[cfg.name]
    except KeyError:
        raise ValueError(f"Unknown strategy {cfg.name!r}, choose from {sorted(STRATEGIES)}") from None
    return strategy(df, cfg)
