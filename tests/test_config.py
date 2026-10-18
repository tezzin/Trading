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
