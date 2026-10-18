import pandas as pd
import pytest

from bot.data import load_candles, save_candles

from bot.config import load_config


def test_defaults_when_file_missing(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.risk.max_drawdown == 0.15
    assert cfg.backtest.fee_rate == 0.001


def test_example_config_loads():
    cfg = load_config("config.example.yaml")
    assert cfg.symbols == ["BTC/USDT", "ETH/USDT"]
    assert cfg.risk.daily_loss_limit == 0.03


def test_unknown_key_is_an_error(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("risk:\n  max_drawdwn: 0.5\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_candles_survive_save_and_load(tmp_path, random_walk):
    path = tmp_path / "BTC-USDT_1h.csv"
    save_candles(random_walk, path)
    loaded = load_candles(path)
    pd.testing.assert_frame_equal(loaded, random_walk, check_freq=False, check_index_type=False)


def test_gate_judges_risk_not_raw_return():
    from bot.metrics import gate_failures

    base = {"total_return_pct": 8.0, "buy_and_hold_pct": 40.0, "sharpe": 1.2, "hold_sharpe": 0.9,
            "max_drawdown_pct": -9.0, "hold_max_drawdown_pct": -30.0, "trades": 45}
    assert gate_failures(base) == []  # lower return than holding is fine if risk-adjusted is better
    assert gate_failures({**base, "sharpe": 0.8}) == ["Sharpe not above holding"]
    assert gate_failures({**base, "max_drawdown_pct": -35.0}) == ["drawdown not smaller than holding"]
    assert gate_failures({**base, "trades": 10}) == ["fewer than 30 trades"]
    assert gate_failures({**base, "total_return_pct": -1.0}) == ["not profitable"]
