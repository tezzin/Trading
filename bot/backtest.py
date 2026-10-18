"""Candle-by-candle backtester for a long-only spot strategy.

Rules that keep results honest:
- Signals are computed on a candle's close and filled at the NEXT candle's open.
- Every fill pays the fee and slippage.
- Stop-losses trigger on the candle's low; a gap below the stop fills at the open.
- Every entry is sized and approved by the same RiskManager used live.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import Config
from .metrics import summarize
from .ml import apply_filter, walk_forward_proba
from .risk import RiskManager
from .strategy import generate_signals


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[dict]
    stats: dict
    risk_events: list[str] = field(default_factory=list)


def build_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Strategy signals, filtered by the ML model when it is enabled."""
    signals = generate_signals(df, cfg.strategy)
    if cfg.ml.enabled:
        signals = apply_filter(signals, walk_forward_proba(df, cfg.ml), cfg.ml.threshold)
    return signals


def run_backtest(df: pd.DataFrame, cfg: Config, signals: pd.DataFrame | None = None) -> BacktestResult:
    """Backtest `df`. Pass `signals` computed on a longer history to keep indicator
    warm-up and ML training data from before the tested period."""
    if len(df) < 2:
        raise ValueError("Need at least 2 candles to backtest")
    signals = build_signals(df, cfg) if signals is None else signals.loc[df.index]
    bt, risk = cfg.backtest, RiskManager(cfg.risk)
    fee, slip = bt.fee_rate, bt.slippage

    opens, lows, closes = df["open"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    sig, stop_dist = signals["signal"].to_numpy(), signals["stop_distance"].to_numpy()
    times = df.index

    cash, qty, stop = bt.starting_balance, 0.0, 0.0
    entry: dict | None = None
    pending: tuple | None = None  # ("buy", stop_price) or ("sell", reason)
    fees_paid = 0.0
    trades: list[dict] = []
    equity = []

    def close_position(price: float, i: int, reason: str) -> None:
        nonlocal cash, qty, fees_paid, entry
        fill = price * (1 - slip)
        cost = qty * fill * fee
        cash += qty * fill - cost
        fees_paid += cost
        trades.append({**entry, "exit_time": times[i], "exit_price": fill, "reason": reason,
                       "pnl": qty * fill - cost - entry["cost_basis"]})
        qty, entry = 0.0, None

    for i in range(len(df)):
        # 1. Fill the order decided on the previous candle's close.
        if pending is not None:
            kind, arg = pending
            pending = None
            if kind == "buy" and qty == 0:
                fill = opens[i] * (1 + slip)
                equity_now = cash
                decision = risk.size_entry(equity_now, cash, fill, arg, closes[i - 1], fee)
                if decision.approved:
                    qty, stop = decision.qty, arg
                    cost = qty * fill * fee
                    cash -= qty * fill + cost
                    fees_paid += cost
                    entry = {"entry_time": times[i], "entry_price": fill, "qty": qty,
                             "stop": stop, "cost_basis": qty * fill + cost}
            elif kind == "sell" and qty > 0:
                close_position(opens[i], i, arg)

        # 2. Stop-loss inside this candle.
        if qty > 0 and lows[i] <= stop:
            close_position(min(opens[i], stop), i, "stop")

        # 3. Mark to market and update risk state.
        eq = cash + qty * closes[i]
        equity.append(eq)
        risk.update(eq, times[i].date())

        # 4. Decide what to do at the next open.
        if qty > 0 and (risk.killed or sig[i] == 0):
            pending = ("sell", "kill switch" if risk.killed else "signal")
        elif qty == 0 and sig[i] == 1 and risk.can_open and i + 1 < len(df):
            pending = ("buy", closes[i] - stop_dist[i])

    equity_curve = pd.Series(equity, index=times, name="equity")
    stats = summarize(equity_curve, trades, df["close"], cfg.timeframe, fees_paid)
    return BacktestResult(equity_curve, trades, stats, risk.events)
