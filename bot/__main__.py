"""Command line: `python -m bot download` and `python -m bot backtest`."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import build_signals, run_backtest
from .config import load_config
from .metrics import gate_failures
from .data import candle_path, fetch_ohlcv, load_candles, save_candles


def cmd_download(args, cfg) -> None:
    for symbol in args.symbols or cfg.symbols:
        df = fetch_ohlcv(symbol, cfg.timeframe, args.since, cfg.exchange, args.until)
        path = candle_path(symbol, cfg.timeframe)
        save_candles(df, path)
        print(f"{symbol}: {len(df)} candles {df.index[0]} -> {df.index[-1]} saved to {path}")


def cmd_backtest(args, cfg) -> None:
    for symbol in args.symbols or cfg.symbols:
        full = load_or_exit(symbol, cfg.timeframe)
        signals = build_signals(full, cfg)  # on full history: warm-up and ML training stay intact
        df = full
        if args.start:
            df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
        if args.end:
            df = df[df.index < pd.Timestamp(args.end, tz="UTC")]
        result = run_backtest(df, cfg, signals)
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


def load_or_exit(symbol: str, timeframe: str) -> pd.DataFrame:
    path = candle_path(symbol, timeframe)
    if not path.exists():
        raise SystemExit(f"No data at {path}. Run: python -m bot download --since 2021-01-01")
    return load_candles(path)


def cmd_compare(args, cfg) -> None:
    """Score each strategy variant on the tuning period and the held-out test period."""
    split = pd.Timestamp(args.split, tz="UTC")
    variants = {
        "ma_cross (Phase 1)": replace(cfg, strategy=replace(cfg.strategy, name="ma_cross"),
                                      ml=replace(cfg.ml, enabled=False)),
        "breakout": replace(cfg, strategy=replace(cfg.strategy, name="breakout"),
                            ml=replace(cfg.ml, enabled=False)),
        "breakout + ML": replace(cfg, strategy=replace(cfg.strategy, name="breakout"),
                                 ml=replace(cfg.ml, enabled=True)),
    }
    cols = ["total_return_pct", "buy_and_hold_pct", "sharpe", "hold_sharpe", "max_drawdown_pct",
            "hold_max_drawdown_pct", "trades"]
    passes: dict[str, list[str]] = {name: [] for name in variants}
    symbols = args.symbols or cfg.symbols
    for symbol in symbols:
        full = load_or_exit(symbol, cfg.timeframe)
        rows = []
        for name, vcfg in variants.items():
            signals = build_signals(full, vcfg)
            for period, df in (("tuning", full[full.index < split]), ("test", full[full.index >= split])):
                if len(df) < 2:
                    continue
                result = run_backtest(df, vcfg, signals)
                failures = gate_failures(result.stats)
                if period == "test" and not failures:
                    passes[name].append(symbol)
                rows.append({"period": period, "strategy": name,
                             **{c: result.stats[c] for c in cols},
                             "kill_switch": "yes" if any("kill switch" in e for e in result.risk_events) else "",
                             "gate": "PASS" if not failures else "; ".join(failures)})
        table = pd.DataFrame(rows).sort_values(["period", "strategy"], ascending=[False, True])
        print(f"\n== {symbol} {cfg.timeframe}, test period from {args.split} ==")
        print(table.to_string(index=False))

    print(f"\nRisk-adjusted gate, test period ({len(symbols)} coins):")
    for name, ok in passes.items():
        print(f"  {name}: passes on {len(ok)} of {len(symbols)}" + (f" ({', '.join(ok)})" if ok else ""))


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

    cmp = sub.add_parser("compare", help="compare strategies on a tuning and a held-out test period")
    cmp.add_argument("--split", default="2025-04-01", help="test period starts here")
    cmp.add_argument("--symbols", nargs="*")

    args = parser.parse_args()
    cfg = load_config(args.config)
    {"download": cmd_download, "backtest": cmd_backtest, "compare": cmd_compare}[args.command](args, cfg)


if __name__ == "__main__":
    main()
