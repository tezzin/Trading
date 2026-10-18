import numpy as np
import pandas as pd
import pytest


def make_candles(closes, spread=0.002, start="2024-01-01"):
    """Candles where open = previous close and high/low sit a small spread around."""
    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * (1 + spread)
    lows = np.minimum(opens, closes) * (1 - spread)
    index = pd.date_range(start, periods=len(closes), freq="1h", tz="UTC", name="time")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": 1.0}, index=index)


@pytest.fixture
def random_walk():
    rng = np.random.default_rng(7)
    closes = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.01, 3_000)))
    return make_candles(closes)
