from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.lightgbm_model import (
    RETURN_HORIZONS,
    build_return_training_examples_for_ticker,
    save_return_models,
    train_return_models,
)

DEFAULT_HORIZONS = (30, 180)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "lightgbm_return_models"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save per-ticker LightGBM forward-return models.")
    parser.add_argument("ticker", help="Ticker symbol to train saved return models for.")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        choices=list(RETURN_HORIZONS),
        help="Forward-return horizons to train. Defaults to the live-ensemble horizons (30 180).",
    )
    parser.add_argument("--period", default="5y", help="Price-history period passed to get_stock_data (default: 5y).")
    parser.add_argument(
        "--interval",
        default="1d",
        help="Price-history interval passed to get_stock_data (default: 1d).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1260,
        help="Feature-table lookback window in trading days (default: 1260).",
    )
    parser.add_argument(
        "--min-rows-per-horizon",
        type=int,
        default=50,
        help="Minimum labeled rows required to train a horizon model (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base directory where per-ticker model folders are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ticker = str(args.ticker).upper()
    training_examples = build_return_training_examples_for_ticker(
        ticker=ticker,
        period=args.period,
        interval=args.interval,
        lookback_days=int(args.lookback_days),
        horizons=tuple(int(value) for value in args.horizons),
    )
    models = train_return_models(training_examples, min_rows_per_horizon=int(args.min_rows_per_horizon))
    if not models:
        print(f"No LightGBM return models were trained for {ticker}.")
        return 1

    output_dir = Path(args.output_dir) / ticker
    saved_paths = save_return_models(models, output_dir)
    print(f"Saved {len(saved_paths)} LightGBM return model(s) for {ticker} under {output_dir}")
    for path in saved_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
