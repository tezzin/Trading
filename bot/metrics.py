"""Performance metrics for an equity curve."""
from __future__ import annotations

import math

import pandas as pd

PERIODS_PER_YEAR = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "1h": 8_760, "4h": 2_190, "1d": 365}


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def sharpe(equity: pd.Series, timeframe: str) -> float:
    returns = equity.pct_change().dropna()
    if returns.std() == 0 or returns.empty:
        return 0.0
    return float(returns.mean() / returns.std() * math.sqrt(PERIODS_PER_YEAR.get(timeframe, 8_760)))


def summarize(equity: pd.Series, trades: list[dict], closes: pd.Series, timeframe: str,
              fees_paid: float) -> dict:
    wins = [t for t in trades if t["pnl"] > 0]
    return {
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(float(equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "buy_and_hold_pct": round(float(closes.iloc[-1] / closes.iloc[0] - 1) * 100, 2),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "sharpe": round(sharpe(equity, timeframe), 2),
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "fees_paid": round(float(fees_paid), 2),
    }
