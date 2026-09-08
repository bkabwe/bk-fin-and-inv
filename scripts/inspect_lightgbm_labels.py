from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.feature_engineering import normalize_daily_index
from modules.polygon_client import (
    PolygonNotConfiguredError,
    get_aggregates_range,
    get_reference_dividends,
    get_reference_splits,
    is_polygon_configured,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw/adjusted price gaps and forward-return labels for a ticker.")
    parser.add_argument("ticker", help="Ticker symbol to inspect.")
    parser.add_argument("--start", required=True, help="Inclusive start date (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive end date (YYYY-MM-DD).")
    parser.add_argument("--horizon", type=int, default=30, help="Forward-return horizon in trading rows (default: 30).")
    parser.add_argument(
        "--focus-date",
        action="append",
        default=[],
        help="Date to print with a +/- 3-row window. Repeat for multiple suspicious dates.",
    )
    parser.add_argument(
        "--gap-threshold-pct",
        type=float,
        default=20.0,
        help="Absolute one-day close gap threshold in percent for flagging suspicious bars (default: 20).",
    )
    parser.add_argument(
        "--label-low",
        type=float,
        default=-0.5,
        help="Lower bound used when printing extreme forward-return labels (default: -0.5).",
    )
    parser.add_argument(
        "--label-high",
        type=float,
        default=1.0,
        help="Upper bound used when printing extreme forward-return labels (default: 1.0).",
    )
    return parser.parse_args()


def _normalize_close(price_data: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(price_data.get("Close"), errors="coerce").astype("float64")
    frame = pd.DataFrame({"Close": close.values}, index=normalize_daily_index(price_data.index))
    frame = frame[~frame.index.isna()].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame["Close"]


def _fetch_daily_range(ticker: str, start: str, end: str, *, adjusted: bool) -> pd.DataFrame:
    return get_aggregates_range(
        ticker,
        multiplier=1,
        timespan="day",
        from_date=start,
        to_date=end,
        adjusted=adjusted,
    )


def _gap_frame(price_data: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    close = _normalize_close(price_data)
    gaps = close.pct_change() * 100.0
    frame = pd.DataFrame({"Close": close, "gap_pct": gaps})
    return frame.loc[gaps.abs() >= float(threshold_pct)]


def _label_frame(price_data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close = _normalize_close(price_data)
    future_close = close.shift(-horizon)
    future_date = close.index.to_series().shift(-horizon)
    labels = (future_close - close) / close
    frame = pd.DataFrame(
        {
            "current_close": close,
            "future_close": future_close,
            "future_date": future_date,
            "forward_return": labels,
        }
    )
    return frame.dropna(subset=["current_close", "future_close", "forward_return"])


def _print_focus_windows(name: str, frame: pd.DataFrame, focus_dates: list[str]) -> None:
    if frame.empty or not focus_dates:
        return
    print(f"\n{name} focus-date windows:")
    for raw_date in focus_dates:
        ts = pd.Timestamp(raw_date)
        if ts not in frame.index:
            nearest = frame.index.get_indexer([ts], method="nearest")[0]
            ts = frame.index[nearest]
        start = max(0, frame.index.get_loc(ts) - 3)
        end = min(len(frame), frame.index.get_loc(ts) + 4)
        print(f"\nAround {raw_date} (nearest row {frame.index[frame.index.get_loc(ts)].date()}):")
        print(frame.iloc[start:end].to_string())


def main() -> int:
    args = _parse_args()
    ticker = str(args.ticker).upper()
    if not is_polygon_configured():
        print("POLYGON_API_KEY is not configured. Export it or put it in the repo-root .env before running this diagnostic.")
        return 1

    try:
        adjusted = _fetch_daily_range(ticker, args.start, args.end, adjusted=True)
        unadjusted = _fetch_daily_range(ticker, args.start, args.end, adjusted=False)
        dividends = get_reference_dividends(ticker, limit=100)
        splits = get_reference_splits(ticker, execution_date_gte=args.start, limit=100)
    except PolygonNotConfiguredError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"Failed to fetch Polygon data for {ticker}: {exc}")
        return 1

    print(f"Adjusted rows: {len(adjusted)} | Unadjusted rows: {len(unadjusted)}")
    print("\nReference splits in range:")
    split_rows = [event for event in splits if args.start <= str(event.get("execution_date") or "") <= args.end]
    print(pd.DataFrame(split_rows).to_string(index=False) if split_rows else "(none)")
    print("\nReference dividends in range:")
    dividend_rows = [event for event in dividends if args.start <= str(event.get("ex_dividend_date") or "") <= args.end]
    print(pd.DataFrame(dividend_rows).to_string(index=False) if dividend_rows else "(none)")

    adjusted_gaps = _gap_frame(adjusted, args.gap_threshold_pct)
    unadjusted_gaps = _gap_frame(unadjusted, args.gap_threshold_pct)
    print(f"\nAdjusted close gaps >= {args.gap_threshold_pct:.1f}%:")
    print(adjusted_gaps.to_string())
    print(f"\nUnadjusted close gaps >= {args.gap_threshold_pct:.1f}%:")
    print(unadjusted_gaps.to_string())

    adjusted_labels = _label_frame(adjusted, int(args.horizon))
    unadjusted_labels = _label_frame(unadjusted, int(args.horizon))
    adjusted_extreme = adjusted_labels.loc[
        (adjusted_labels["forward_return"] < float(args.label_low)) | (adjusted_labels["forward_return"] > float(args.label_high))
    ]
    unadjusted_extreme = unadjusted_labels.loc[
        (unadjusted_labels["forward_return"] < float(args.label_low))
        | (unadjusted_labels["forward_return"] > float(args.label_high))
    ]
    print(f"\nAdjusted {args.horizon}d labels outside [{args.label_low:.2f}, {args.label_high:.2f}]:")
    print(adjusted_extreme.to_string())
    print(f"\nUnadjusted {args.horizon}d labels outside [{args.label_low:.2f}, {args.label_high:.2f}]:")
    print(unadjusted_extreme.to_string())

    _print_focus_windows("Adjusted prices", adjusted[["Open", "High", "Low", "Close", "Volume"]], list(args.focus_date))
    _print_focus_windows("Unadjusted prices", unadjusted[["Open", "High", "Low", "Close", "Volume"]], list(args.focus_date))
    _print_focus_windows("Adjusted labels", adjusted_labels[["current_close", "future_close", "future_date", "forward_return"]], list(args.focus_date))
    _print_focus_windows("Unadjusted labels", unadjusted_labels[["current_close", "future_close", "future_date", "forward_return"]], list(args.focus_date))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
