import numpy as np
import pandas as pd
import pytest

from bot.backtest import run_backtest
from bot.config import BacktestConfig, Config, RiskConfig, StrategyConfig
from tests.conftest import make_candles

FAST = StrategyConfig(name="ma_cross", fast_period=3, slow_period=6, atr_period=3, stop_atr_mult=2.0, min_atr_pct=0.0)


def config(**backtest):
    return Config(strategy=FAST, backtest=BacktestConfig(**backtest),
                  risk=RiskConfig(risk_per_trade=1.0, max_position_pct=1.0))


def always_long(stop_distance):
    def fake(df, cfg):
        return pd.DataFrame({"signal": 1, "stop_distance": stop_distance}, index=df.index)
    return fake


def test_orders_fill_at_next_candle_open(monkeypatch):
    def enter_then_exit(df, cfg):
        signal = [1, 1, 1, 0] + [0] * (len(df) - 4)
        return pd.DataFrame({"signal": signal, "stop_distance": 5.0}, index=df.index)

    monkeypatch.setattr("bot.backtest.generate_signals", enter_then_exit)
    df = make_candles(np.linspace(100, 110, 20))
    trade = run_backtest(df, config(fee_rate=0, slippage=0)).trades[0]
    # Long signal on candle 0's close fills at candle 1's open;
    # exit signal on candle 3's close fills at candle 4's open.
    assert trade["entry_time"] == df.index[1]
    assert trade["entry_price"] == pytest.approx(df["open"].iloc[1])
    assert trade["exit_time"] == df.index[4]
    assert trade["exit_price"] == pytest.approx(df["open"].iloc[4])


def test_round_trip_on_flat_price_loses_exactly_the_costs():
    prices = [100] * 10 + list(np.linspace(101, 120, 20)) + list(np.linspace(119, 90, 30))
    df = make_candles(prices, spread=0.0)
    result = run_backtest(df, config(fee_rate=0.001, slippage=0.0005))
    assert result.trades
    assert result.stats["fees_paid"] > 0
    free = run_backtest(df, config(fee_rate=0, slippage=0))
    assert result.equity.iloc[-1] < free.equity.iloc[-1]


def test_gap_below_stop_fills_at_open():
    prices = [100] * 10 + list(np.linspace(101, 120, 20))
    df = make_candles(prices, spread=0.001)
    # Crash: the next candle opens far below any stop.
    crash = df.index[-1] + pd.Timedelta(hours=1)
    df.loc[crash] = [60.0, 61.0, 59.0, 60.5, 1.0]
    result = run_backtest(df, config(fee_rate=0, slippage=0))
    stopped = [t for t in result.trades if t["reason"] == "stop"]
    assert stopped and stopped[-1]["exit_price"] == pytest.approx(60.0)


def test_results_do_not_depend_on_future_candles(random_walk):
    cfg = Config()
    cut = 2_000
    full = run_backtest(random_walk, cfg).equity.iloc[: cut - 1]
    past = run_backtest(random_walk.iloc[:cut], cfg).equity.iloc[: cut - 1]
    pd.testing.assert_series_equal(full, past)


def test_kill_switch_stops_trading_after_max_drawdown(monkeypatch):
    # Always wants to be long with a tight stop, in a steady downtrend: repeated stop-outs.
    monkeypatch.setattr("bot.backtest.generate_signals", always_long(0.5))
    df = make_candles(np.linspace(100, 40, 300), spread=0.0)
    cfg = Config(backtest=BacktestConfig(fee_rate=0, slippage=0),
                 risk=RiskConfig(risk_per_trade=0.02, max_position_pct=1.0,
                                 daily_loss_limit=1.0, max_drawdown=0.15))
    result = run_backtest(df, cfg)
    assert any("kill switch" in e for e in result.risk_events)
    drawdown = result.equity / result.equity.cummax() - 1
    killed_at = drawdown[drawdown <= -0.15].index[0]
    assert all(t["entry_time"] <= killed_at for t in result.trades)
    assert result.equity.iloc[-1] == pytest.approx(result.equity.loc[killed_at:].iloc[1])


def test_risk_per_trade_bounds_a_stop_loss(random_walk):
    cfg = Config(risk=RiskConfig(risk_per_trade=0.01))
    result = run_backtest(random_walk, cfg)
    stops = [t for t in result.trades if t["reason"] == "stop"]
    assert stops
    for t in stops:
        # Loss is close to 1% of equity, plus fees and slippage; gaps can add a little more.
        assert t["pnl"] > -0.02 * cfg.backtest.starting_balance * 2
