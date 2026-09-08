from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.data_fetcher import get_stock_data
from modules.feature_engineering import build_feature_table
from modules.lightgbm_model import (
    build_return_training_examples,
    compare_feature_row_to_training_ranges,
    latest_lightgbm_feature_row,
    load_return_models,
    predict_forward_return,
    summarize_training_feature_ranges,
)
from modules.polygon_client import is_polygon_configured


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a live LightGBM feature row against reconstructed training feature ranges."
    )
    parser.add_argument("ticker", help="Ticker symbol to inspect.")
    parser.add_argument("--horizon", type=int, default=30, choices=(30, 180, 720), help="Return-model horizon to inspect.")
    parser.add_argument("--period", default="5y", help="Price-history period passed to get_stock_data (default: 5y).")
    parser.add_argument("--interval", default="1d", help="Price-history interval passed to get_stock_data (default: 1d).")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=1260,
        help="Feature-table lookback window passed to build_feature_table for training reconstruction.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "data" / "lightgbm_return_models"),
        help="Base directory containing per-ticker saved LightGBM model folders.",
    )
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=5.0,
        help="Flag features beyond this absolute z-score in addition to min/max range checks.",
    )
    return parser.parse_args()


def _render_frame(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(none)"
    return frame.to_string()


def main() -> int:
    args = _parse_args()
    ticker = str(args.ticker).upper()
    if not is_polygon_configured():
        print(
            "POLYGON_API_KEY is not configured. This sandbox cannot reconstruct AIM's real training/live feature rows.\n"
            "Run this script from the repository root with Polygon configured and your saved model artifacts present."
        )
        return 1

    price_data = get_stock_data(ticker, period=args.period, interval=args.interval)
    if price_data is None or price_data.empty:
        print(f"No price history was returned for {ticker}.")
        return 1

    live_feature_table = build_feature_table(ticker, price_data, lookback_days=len(price_data))
    latest_feature_row = latest_lightgbm_feature_row(live_feature_table)
    if latest_feature_row is None:
        print(f"Latest live feature row was unavailable for {ticker}.")
        return 1

    training_feature_table = build_feature_table(ticker, price_data, lookback_days=int(args.lookback_days))
    training_examples = build_return_training_examples(
        ticker=ticker,
        price_data=price_data,
        feature_table=training_feature_table,
        lookback_days=int(args.lookback_days),
        horizons=(int(args.horizon),),
    )
    if int(args.horizon) not in training_examples:
        print(f"No reconstructed training examples were available for {ticker} horizon {int(args.horizon)}d.")
        return 1

    x_train, y_train = training_examples[int(args.horizon)]
    training_ranges = summarize_training_feature_ranges(x_train)
    comparison = compare_feature_row_to_training_ranges(
        latest_feature_row,
        training_ranges,
        std_threshold=float(args.std_threshold),
    )
    flagged = comparison.loc[comparison["outside_training_range"] | comparison["beyond_std_threshold"]]

    print(
        f"{ticker} horizon={int(args.horizon)}d reconstructed training rows={len(x_train)} "
        f"features={x_train.shape[1]} live_feature_date={latest_feature_row.name}"
    )
    print(
        f"Training labels: min={float(y_train.min()):.6f} max={float(y_train.max()):.6f} "
        f"mean={float(y_train.mean()):.6f} std={float(y_train.std(ddof=0)):.6f}"
    )
    print(f"\nFlagged features (outside min/max or |z| > {float(args.std_threshold):.1f}):")
    print(_render_frame(flagged))

    print("\nFull per-feature comparison:")
    print(_render_frame(comparison))

    model_dir = Path(args.model_dir) / ticker
    models = load_return_models(model_dir, horizons=(int(args.horizon),))
    model = models.get(int(args.horizon))
    if model is None:
        print(
            f"\nSaved model not found at {model_dir / f'lightgbm_return_h{int(args.horizon)}.joblib'}.\n"
            "The training/live feature comparison above is still valid, but the original saved-model prediction "
            "cannot be reproduced in this run."
        )
        return 0

    predicted_return = predict_forward_return(model, latest_feature_row)
    if predicted_return is None:
        print("\nSaved-model inference returned no prediction.")
        return 0
    current_price = float(pd.to_numeric(price_data["Close"], errors="coerce").dropna().iloc[-1])
    projected_price = current_price * (1.0 + float(predicted_return))
    print(
        f"\nSaved-model inference: predicted_forward_return={float(predicted_return):.6f} "
        f"current_price={current_price:.6f} projected_price={projected_price:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
