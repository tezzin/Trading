"""Machine-learning entry filter, trained walk-forward so it never sees the future.

The model predicts the probability that an entry made now gains at least
`min_return` over the next `horizon` candles. Rule-based entries go ahead only
when that probability is at least `threshold`; exits are never blocked.

Walk-forward: every `retrain_every` candles a fresh model is trained on all
earlier candles whose outcome was already known at that time, then used for
the next block. Until `min_train` candles of history exist, no entries pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MLConfig
from .indicators import atr, rsi, sma


def features(df: pd.DataFrame) -> pd.DataFrame:
    close, volume = df["close"], df["volume"]
    out = pd.DataFrame(index=df.index)
    for n in (1, 6, 24, 72):
        out[f"ret_{n}"] = close.pct_change(n)
    out["atr_pct"] = atr(df, 14) / close
    out["rsi"] = rsi(close, 14)
    out["dist_sma50"] = close / sma(close, 50) - 1
    out["dist_sma200"] = close / sma(close, 200) - 1
    vol_mean = volume.rolling(50).mean()
    out["volume_z"] = (volume - vol_mean) / volume.rolling(50).std()
    hi, lo = df["high"].rolling(55).max(), df["low"].rolling(55).min()
    out["channel_pos"] = (close - lo) / (hi - lo)
    return out.replace([np.inf, -np.inf], np.nan)


def labels(df: pd.DataFrame, horizon: int, min_return: float) -> pd.Series:
    """1 if the close `horizon` candles ahead is at least `min_return` higher. NaN at the end."""
    future = df["close"].shift(-horizon) / df["close"] - 1
    return (future >= min_return).astype(float).where(future.notna())


def walk_forward_proba(df: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = features(df), labels(df, cfg.horizon, cfg.min_return)
    proba = pd.Series(np.nan, index=df.index, name="ml_proba")
    n = len(df)

    for start in range(cfg.min_train, n, cfg.retrain_every):
        # Only rows whose label was fully known by `start` (its horizon ended before it).
        known = slice(0, max(start - cfg.horizon, 0))
        X_train, y_train = X.iloc[known], y.iloc[known]
        mask = y_train.notna()
        X_train, y_train = X_train[mask], y_train[mask]
        usable = X_train.columns[X_train.notna().any()]  # e.g. volume_z on flat test data
        X_train = X_train[usable]
        if len(y_train) < cfg.min_train // 2 or y_train.nunique() < 2:
            continue
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                               l2_regularization=1.0, random_state=0)
        model.fit(X_train, y_train)
        block = slice(start, min(start + cfg.retrain_every, n))
        proba.iloc[block] = model.predict_proba(X.iloc[block][usable])[:, 1]
    return proba


def apply_filter(signals: pd.DataFrame, proba: pd.Series, threshold: float) -> pd.DataFrame:
    """Turn entries the model doesn't back into 'keep current position'."""
    out = signals.copy()
    weak = (out["signal"] == 1) & ~(proba >= threshold)
    out.loc[weak, "signal"] = -1
    out["ml_proba"] = proba
    return out
