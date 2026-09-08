from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.backtest_comparison import (
    format_comparison_summary,
    format_per_ticker_diagnostics_table,
    run_lightgbm_backtest_comparison,
)
from modules.lightgbm_model import RETURN_HORIZONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare walk-forward RMSE for ARIMA, trend, LightGBM, and validation diagnostics."
    )
    parser.add_argument("--sample-size", type=int, default=30, help="Number of tickers to evaluate (default: 30)")
    parser.add_argument(
        "--period",
        type=str,
        default=None,
        help="History period to fetch per ticker (default: 2y for 30d, 5y for 180d/720d)",
    )
    parser.add_argument("--interval", type=str, default="1d", help="Bar interval (default: 1d)")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional explicit ticker list")
    parser.add_argument(
        "--horizon",
        type=int,
        choices=list(RETURN_HORIZONS),
        default=30,
        help="Forecast horizon / walk-forward test window in days (default: 30)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed used to shuffle the available ticker universe before sampling",
    )
    parser.add_argument(
        "--ticker-offset",
        type=int,
        default=0,
        help="Optional offset applied after any shuffle and before taking the sample (default: 0)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output = run_lightgbm_backtest_comparison(
        tickers=args.tickers,
        sample_size=max(1, int(args.sample_size)),
        period=args.period,
        interval=args.interval,
        horizon=int(args.horizon),
        random_seed=args.random_seed,
        ticker_offset=max(0, int(args.ticker_offset)),
        evaluate_naive_baseline=True,
        lightgbm_diagnostics=True,
    )
    config = output["window_config"]
    print(
        f"Evaluated {len(output['sample_tickers'])} ticker(s) for {int(output['horizon'])}d horizon "
        f"using period={output['period']}"
    )
    print(
        "Walk-forward window config: "
        f"train_len={int(config['train_len'])}, "
        f"test_len={int(config['test_len'])}, "
        f"stride={int(config['stride'])}, "
        f"max_history_rows={int(config['max_history_rows'])}"
    )
    print(format_comparison_summary(output["summary"]))
    print("")
    print(format_per_ticker_diagnostics_table(output["per_ticker"]))


if __name__ == "__main__":
    main()
