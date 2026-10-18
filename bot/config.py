"""Settings loaded from config.yaml, with the plan's starting limits as defaults."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class StrategyConfig:
    name: str = "breakout"       # "breakout" or "ma_cross" (the Phase 1 baseline)
    # breakout
    entry_period: int = 55       # enter on a close above the highest high of this many candles
    exit_period: int = 20        # exit on a close below the lowest low of this many candles
    trend_period: int = 200      # only enter above this moving average
    # ma_cross
    fast_period: int = 20
    slow_period: int = 100
    # both
    atr_period: int = 14
    stop_atr_mult: float = 2.5
    min_atr_pct: float = 0.003


@dataclass
class MLConfig:
    enabled: bool = False
    horizon: int = 12            # candles ahead the model predicts
    min_return: float = 0.01     # a "good" entry gains at least this much over the horizon
    threshold: float = 0.55      # only enter when the model's probability is at least this
    retrain_every: int = 180     # candles between retrains (walk-forward)
    min_train: int = 1000        # candles of history before the first model


@dataclass
class BacktestConfig:
    starting_balance: float = 500.0
    fee_rate: float = 0.001
    slippage: float = 0.0005


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.20
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.15
    max_price_deviation: float = 0.05


@dataclass
class Config:
    exchange: str = "binance"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "4h"
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    ml: MLConfig = field(default_factory=MLConfig)


def _build(cls, values: dict | None):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path | None = None) -> Config:
    """Load config from YAML. Missing file or keys fall back to defaults."""
    if path is None or not Path(path).exists():
        return Config()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config(
        exchange=raw.get("exchange", "binance"),
        symbols=raw.get("symbols", ["BTC/USDT", "ETH/USDT"]),
        timeframe=raw.get("timeframe", "4h"),
        strategy=_build(StrategyConfig, raw.get("strategy")),
        backtest=_build(BacktestConfig, raw.get("backtest")),
        risk=_build(RiskConfig, raw.get("risk")),
        ml=_build(MLConfig, raw.get("ml")),
    )
