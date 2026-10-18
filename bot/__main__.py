"""Command line: `python -m bot download` and `python -m bot backtest`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import load_config
from .data import candle_path, fetch_ohlcv, load_candles, save_candles


def cmd_download(args, cfg) -> None:
    for symbol in args.symbols or cfg.symbols:
        df = fetch_ohlcv(symbol, cfg.timeframe, args.since, cfg.exchange, args.until)
        path = candle_path(symbol, cfg.timeframe)
        save_candles(df, path)
        print(f"{symbol}: {len(df)} candles {df.index[0]} -> {df.index[-1]} saved to {path}")


def cmd_backtest(args, cfg) -> None:
    for symbol in args.symbols or cfg.symbols:
        path = candle_path(symbol, cfg.timeframe)
        if not path.exists():
            raise SystemExit(f"No data at {path}. Run: python -m bot download --since 2021-01-01")
        df = load_candles(path)
        if args.start:
            df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
        if args.end:
            df = df[df.index < pd.Timestamp(args.end, tz="UTC")]
        result = run_backtest(df, cfg)
        print(f"\n== {symbol} ==")
        print(json.dumps(result.stats, indent=2))
        for event in result.risk_events:
            print("risk:", event)
        if args.save:
            out = Path("results")
            out.mkdir(exist_ok=True)
            name = symbol.replace("/", "-")
            result.equity.to_csv(out / f"{name}_equity.csv")
            pd.DataFrame(result.trades).to_csv(out / f"{name}_trades.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(prog="bot")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download candle history (public data, no keys)")
    dl.add_argument("--since", required=True, help="start date, e.g. 2021-01-01")
    dl.add_argument("--until", help="end date (default: now)")
    dl.add_argument("--symbols", nargs="*")

    bt = sub.add_parser("backtest", help="backtest the strategy on downloaded candles")
    bt.add_argument("--symbols", nargs="*")
    bt.add_argument("--start", help="only use candles from this date")
    bt.add_argument("--end", help="only use candles before this date")
    bt.add_argument("--save", action="store_true", help="write equity and trades to results/")

    args = parser.parse_args()
    cfg = load_config(args.config)
    {"download": cmd_download, "backtest": cmd_backtest}[args.command](args, cfg)


if __name__ == "__main__":
    main()
