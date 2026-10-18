# Trading

A crypto trading bot for Binance, built in phases. Each phase must pass a gate before the next one starts, and **no real order is placed until the owner explicitly says go**.

| Phase | What | Gate to pass |
| --- | --- | --- |
| 1. Build and backtest (this code) | Data download, baseline strategy, risk rules, backtester | Beats buy-and-hold after fees on data it was not tuned on |
| 2. Testnet | Same code against the Binance Spot Testnet | Clean run of 2+ weeks, kill switch proven, owner's go |
| 3. Small live | Dry run on live prices, then 200 to 500 USDT on a sub-account | 3 months close to backtest results |
| 4. Review and grow | ML signal, more coins, larger size | Only if it is working |

Trading bots usually lose money once fees and slippage are counted. Treat any money given to this bot as money you can afford to lose.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # optional, defaults match it
```

Phase 1 needs no API keys: candle history is public.

## Use

```bash
# Download 4-hour BTC/USDT and ETH/USDT candles from Binance into data/
python -m bot download --since 2021-01-01

# Compare every strategy on the tuning period and the held-out test period
python -m bot compare --split 2025-04-01

# Backtest the configured strategy, or only a period you held back for testing
python -m bot backtest
python -m bot backtest --start 2025-04-01 --save   # writes results/
```

`compare` also applies the risk-adjusted gate to every result. Because the bot holds at most 20% of its money in a coin, it is not expected to out-earn holding in a rally. Instead, to pass it must be profitable, have a higher Sharpe ratio and a smaller worst drop than holding the coin, and make at least 30 trades. Settings are frozen before a gate run, and coins the strategy was never tuned on (e.g. `--symbols SOL/USDT XRP/USDT BNB/USDT ADA/USDT`) are the fairest test.

The backtest prints net return, buy-and-hold return, max drawdown, Sharpe ratio, trade count, win rate and fees paid. Change settings only by looking at the tuning period; the test period is the honest check, and every look at it makes it a little less honest.

## How it works

- `bot/data.py` pages through candle history with [ccxt](https://github.com/ccxt/ccxt) and stores it as CSV.
- `bot/strategy.py` has two rule-based strategies, both with a stop-loss 2.5 x ATR below entry:
  - `breakout` (default): buy when the close breaks the highest high of the last 55 candles while above the 200-candle average; sell when it breaks the lowest low of the last 20. It trades rarely, so fees take less.
  - `ma_cross`: the Phase 1 baseline, long while the 20-candle average is above the 100-candle average. It lost to buy-and-hold on real 2021-2026 data.
- `bot/ml.py` is an optional filter (`ml.enabled: true`). A gradient-boosted model estimates the chance that an entry gains 1% within 12 candles, and entries below 55% are skipped. It is retrained every 180 candles on past data only (walk-forward), so it never sees the future.
- `bot/risk.py` approves and sizes every entry: 1% of equity at risk per trade, max 20% of equity per coin, no new entries after a 3% daily loss, and a permanent halt (close everything) after a 15% drop from peak.
- `bot/backtest.py` replays candles one at a time. Signals use a candle's close and fill at the next candle's open, every fill pays 0.1% fee plus 0.05% slippage, and stops that gap fill at the open price.

All limits live in `config.yaml`.

## Tests

```bash
python -m pytest -q
```

They check there is no look-ahead (future candles never change past results), that fills, fees, stops and gaps behave as described, and that every risk limit fires.
