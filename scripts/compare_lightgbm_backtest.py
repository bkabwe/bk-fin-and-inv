from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.backtest_comparison import format_comparison_summary, run_lightgbm_backtest_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare walk-forward RMSE for ARIMA, trend, and LightGBM.")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of tickers to evaluate (default: 30)")
    parser.add_argument("--period", type=str, default="2y", help="History period to fetch per ticker (default: 2y)")
    parser.add_argument("--interval", type=str, default="1d", help="Bar interval (default: 1d)")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    args = parser.parse_args()

    output = run_lightgbm_backtest_comparison(
        tickers=args.tickers,
        sample_size=max(1, int(args.sample_size)),
        period=args.period,
        interval=args.interval,
    )
    print(f"Evaluated {len(output['sample_tickers'])} ticker(s)")
    print(format_comparison_summary(output["summary"]))


if __name__ == "__main__":
    main()
