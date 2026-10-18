"""Performance metrics for an equity curve."""
from __future__ import annotations

import math

import pandas as pd

# Below this many trades a result is mostly luck.
MIN_TRADES_FOR_GATE = 30

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
    hold = closes / closes.iloc[0]
    return {
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(float(equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "buy_and_hold_pct": round(float(closes.iloc[-1] / closes.iloc[0] - 1) * 100, 2),
        "hold_sharpe": round(sharpe(hold, timeframe), 2),
        "hold_max_drawdown_pct": round(max_drawdown(hold) * 100, 2),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "sharpe": round(sharpe(equity, timeframe), 2),
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
        "fees_paid": round(float(fees_paid), 2),
    }


def gate_failures(stats: dict) -> list[str]:
    """Reasons a result fails the risk-adjusted gate; empty means it passes.

    The bot holds at most part of its money in a coin, so it is judged per unit
    of risk: a better Sharpe ratio and a smaller worst drop than holding, a
    profit, and enough trades for the result not to be luck.
    """
    reasons = []
    if stats["total_return_pct"] <= 0:
        reasons.append("not profitable")
    if stats["sharpe"] <= stats["hold_sharpe"]:
        reasons.append("Sharpe not above holding")
    if stats["max_drawdown_pct"] <= stats["hold_max_drawdown_pct"]:
        reasons.append("drawdown not smaller than holding")
    if stats["trades"] < MIN_TRADES_FOR_GATE:
        reasons.append(f"fewer than {MIN_TRADES_FOR_GATE} trades")
    return reasons
