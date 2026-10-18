import numpy as np
import pandas as pd
import pytest

from bot.config import MLConfig, StrategyConfig
from bot.ml import apply_filter, features, labels, walk_forward_proba
from bot.strategy import generate_signals
from tests.conftest import make_candles

CFG = MLConfig(enabled=True, horizon=6, retrain_every=200, min_train=600)


def test_features_ignore_future_candles(random_walk):
    cut = 1_500
    pd.testing.assert_frame_equal(features(random_walk).iloc[:cut], features(random_walk.iloc[:cut]))


def test_labels_look_exactly_horizon_ahead():
    df = make_candles([100, 100, 102, 100, 100])
    y = labels(df, horizon=2, min_return=0.01)
    assert list(y.iloc[:3]) == [1.0, 0.0, 0.0]
    assert y.iloc[3:].isna().all()


def test_predictions_do_not_depend_on_future_candles(random_walk):
    cut = 2_100
    full = walk_forward_proba(random_walk, CFG).iloc[:cut]
    past = walk_forward_proba(random_walk.iloc[:cut], CFG)
    pd.testing.assert_series_equal(full, past)


def test_no_predictions_before_min_train(random_walk):
    proba = walk_forward_proba(random_walk, CFG)
    assert proba.iloc[: CFG.min_train].isna().all()
    assert proba.iloc[CFG.min_train:].notna().all()


def test_model_finds_a_real_pattern():
    # After two up candles in a row, price keeps rising; otherwise it falls back.
    rng = np.random.default_rng(1)
    closes, ups = [100.0], 0
    for _ in range(4_000):
        drift = 0.004 if ups >= 2 else -0.001
        step = drift + rng.normal(0, 0.003)
        ups = ups + 1 if step > 0 else 0
        closes.append(closes[-1] * (1 + step))
    df = make_candles(closes)
    cfg = MLConfig(horizon=3, min_return=0.003, retrain_every=500, min_train=1_500)
    proba = walk_forward_proba(df, cfg)
    y = labels(df, cfg.horizon, cfg.min_return)
    scored = proba.notna() & y.notna()
    high, low = proba[scored] >= 0.5, proba[scored] < 0.5
    assert y[scored][high].mean() > y[scored][low].mean() + 0.2


def test_filter_blocks_only_unbacked_entries():
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    signals = pd.DataFrame({"signal": [1, 1, 0, -1], "stop_distance": 1.0}, index=idx)
    proba = pd.Series([0.7, 0.4, 0.1, np.nan], index=idx)
    out = apply_filter(signals, proba, threshold=0.55)
    assert list(out["signal"]) == [1, -1, 0, -1]


def test_breakout_enters_on_new_high_and_exits_on_new_low():
    prices = list(np.linspace(100, 110, 60)) + [125] + list(np.linspace(124, 90, 30))
    df = make_candles(prices, spread=0.0)
    cfg = StrategyConfig(name="breakout", entry_period=20, exit_period=10, trend_period=30,
                         min_atr_pct=0.0)
    sig = generate_signals(df, cfg)["signal"]
    assert sig.iloc[60] == 1
    assert (sig.iloc[61:] == 0).any()
    assert set(sig.unique()) <= {-1, 0, 1}


def test_breakout_ignores_future_candles(random_walk):
    cfg = StrategyConfig(name="breakout")
    cut = 2_000
    pd.testing.assert_frame_equal(generate_signals(random_walk, cfg).iloc[:cut],
                                  generate_signals(random_walk.iloc[:cut], cfg))


def test_unknown_strategy_is_an_error(random_walk):
    with pytest.raises(ValueError):
        generate_signals(random_walk, StrategyConfig(name="moon"))
