from __future__ import annotations

import argparse
import subprocess
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
from modules.experiment_tracker import compare_latest_two_runs, record_experiment_run
from modules.lightgbm_model import RETURN_HORIZONS


def _detect_git_commit() -> str | None:
    try:
        output = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return output.stdout.strip() or None
    except Exception:
        return None


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
    parser.add_argument(
        "--path-rmse",
        action="store_true",
        help=(
            "Also compute each model's day-over-day log-return path RMSE, a supplementary metric to the "
            "existing terminal-point-oriented price-level RMSE (see README's 'Known modeling limitations')"
        ),
    )
    parser.add_argument(
        "--transaction-cost-aware",
        action="store_true",
        help=(
            "Also simulate a simple long/flat trading rule per model with a round-trip transaction cost "
            "(see --transaction-cost-bps), reporting net-of-cost return/trade-rate/hit-rate plus a "
            "buy-and-hold baseline, directly addressing the 'no transaction costs or slippage' RMSE limitation"
        ),
    )
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="Round-trip transaction cost in basis points applied to each simulated trade (default: 10.0)",
    )
    parser.add_argument(
        "--record-experiment",
        type=str,
        default=None,
        metavar="RUN_NAME",
        help="If set, record this run's summary metrics to the experiment-tracking store (data/experiments.db) "
        "under this run name, so future runs can be compared against it",
    )
    parser.add_argument(
        "--git-commit",
        type=str,
        default=None,
        help="Git commit SHA to associate with a recorded experiment run (default: auto-detected via "
        "'git rev-parse HEAD' when --record-experiment is set)",
    )
    parser.add_argument(
        "--compare-latest",
        action="store_true",
        help="After recording, print metric deltas versus the previous recorded run with the same "
        "--record-experiment run name and horizon (requires --record-experiment)",
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
        evaluate_path_rmse=bool(args.path_rmse),
        evaluate_transaction_cost_aware=bool(args.transaction_cost_aware),
        transaction_cost_bps=float(args.transaction_cost_bps),
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

    if args.record_experiment:
        git_commit = args.git_commit or _detect_git_commit()
        run_id = record_experiment_run(
            run_name=args.record_experiment,
            horizon=int(output["horizon"]),
            git_commit=git_commit,
            config={
                "sample_size": max(1, int(args.sample_size)),
                "period": output["period"],
                "interval": args.interval,
                "window_config": config,
                "evaluate_path_rmse": bool(args.path_rmse),
                "evaluate_transaction_cost_aware": bool(args.transaction_cost_aware),
                "transaction_cost_bps": float(args.transaction_cost_bps),
            },
            metrics=output["summary"],
            sample_tickers=output["sample_tickers"],
        )
        print(f"\nRecorded experiment run '{args.record_experiment}' (#{run_id}, commit={git_commit or 'n/a'})")
        if args.compare_latest:
            comparison = compare_latest_two_runs(args.record_experiment, int(output["horizon"]))
            if comparison is None:
                print("No prior recorded run available for comparison yet.")
            else:
                print("Metric deltas vs. previous run (positive = increase, negative = decrease):")
                deltas = comparison["metric_deltas"]
                if not deltas:
                    print("  (no shared numeric model metrics to compare)")
                for metric_name, delta in sorted(deltas.items()):
                    print(f"  {metric_name}: {delta:+.6f}")


if __name__ == "__main__":
    main()
