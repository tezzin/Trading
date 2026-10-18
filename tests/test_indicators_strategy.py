import pandas as pd

from bot.config import StrategyConfig
from bot.indicators import atr, sma
from bot.strategy import generate_signals


def test_indicators_ignore_future_candles(random_walk):
    cut = 1_500
    full, past = random_walk, random_walk.iloc[:cut]
    pd.testing.assert_series_equal(sma(full["close"], 50).iloc[:cut], sma(past["close"], 50))
    pd.testing.assert_series_equal(atr(full, 14).iloc[:cut], atr(past, 14))


def test_signals_ignore_future_candles(random_walk):
    cfg = StrategyConfig()
    cut = 2_000
    pd.testing.assert_frame_equal(generate_signals(random_walk, cfg).iloc[:cut],
                                  generate_signals(random_walk.iloc[:cut], cfg))


def test_no_signal_before_indicators_are_ready(random_walk):
    cfg = StrategyConfig(name="ma_cross", slow_period=100)
    signals = generate_signals(random_walk, cfg)
    assert (signals["signal"].iloc[:99] == 0).all()
