from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.backtest_comparison import format_comparison_summary, run_lightgbm_backtest_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare walk-forward RMSE for ARIMA, trend, LightGBM, and validation diagnostics."
    )
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
        evaluate_naive_baseline=True,
        lightgbm_diagnostics=True,
    )
    print(f"Evaluated {len(output['sample_tickers'])} ticker(s)")
    print(format_comparison_summary(output["summary"]))
    print("")
    print("Per-ticker RMSE and LightGBM prediction variance diagnostics")
    print(
        "Ticker  ARIMA RMSE  Trend RMSE  LightGBM RMSE  Naive RMSE  "
        "LightGBM Pred Return Std  LightGBM Pred Return Min  LightGBM Pred Return Max"
    )
    print("------  ----------  ----------  -------------  ----------  ----------------------  -----------------------  -----------------------")
    for row in output["per_ticker"]:
        diagnostics = row.get("lightgbm_diagnostics") or {}
        stats = diagnostics.get("prediction_stats") if isinstance(diagnostics, dict) else None
        pred_std = stats.get("std") if isinstance(stats, dict) else None
        pred_min = stats.get("min") if isinstance(stats, dict) else None
        pred_max = stats.get("max") if isinstance(stats, dict) else None
        std_text = f"{pred_std:.8f}" if pred_std is not None else "n/a"
        min_text = f"{pred_min:.8f}" if pred_min is not None else "n/a"
        max_text = f"{pred_max:.8f}" if pred_max is not None else "n/a"
        print(
            f"{row.get('ticker', ''):<6}  "
            f"{float(row.get('arima_rmse', 1.0)):>10.6f}  "
            f"{float(row.get('trend_rmse', 1.0)):>10.6f}  "
            f"{float(row.get('lightgbm_rmse', 1.0)):>13.6f}  "
            f"{float(row.get('naive_rmse', 1.0)):>10.6f}  "
            f"{std_text:>22}  {min_text:>23}  {max_text:>23}"
        )


if __name__ == "__main__":
    main()
