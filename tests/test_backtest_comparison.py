from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import backtester
from modules.backtest_comparison import run_lightgbm_backtest_comparison, summarize_backtest_results


def _constant_price_frame(length: int = 120, value: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=length, freq="D")
    close = np.full(length, value, dtype=float)
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Volume": np.full(length, 1000.0, dtype=float),
        },
        index=index,
    )


class WalkForwardLightGBMTests(unittest.TestCase):
    def test_walk_forward_includes_lightgbm_rmse(self):
        data = _constant_price_frame()
        feature_table = pd.DataFrame({"feature_a": np.arange(60, dtype=float)}, index=data.index[:60])
        training_examples = {30: (feature_table.copy(), pd.Series(np.linspace(0.0, 0.01, num=60), index=feature_table.index))}

        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", True),
            patch(
                "modules.backtester.build_feature_table",
                side_effect=lambda _ticker, price_data, lookback_days=0: pd.DataFrame(
                    {"feature_a": np.arange(len(price_data), dtype=float)},
                    index=price_data.index,
                ),
            ),
            patch("modules.backtester.build_return_training_examples", return_value=training_examples),
            patch("modules.backtester.train_return_models", return_value={30: object()}),
            patch("modules.backtester.predict_forward_return", return_value=0.0),
        ):
            result = backtester.run_walk_forward("AAPL-LGBM-RMSE", data, evaluate_lightgbm=True)

        self.assertEqual(result["lightgbm_windows"], 2)
        self.assertEqual(result["lightgbm_rmse"], 0.0)

    def test_walk_forward_lightgbm_failure_is_graceful(self):
        data = _constant_price_frame()
        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", True),
            patch("modules.backtester.build_feature_table", side_effect=RuntimeError("boom")),
        ):
            result = backtester.run_walk_forward("AAPL-LGBM-FAIL", data, evaluate_lightgbm=True)

        self.assertEqual(result["lightgbm_rmse"], 1.0)
        self.assertEqual(result["lightgbm_windows"], 0)


class BacktestComparisonSummaryTests(unittest.TestCase):
    def test_summarize_backtest_results(self):
        rows = [
            {
                "ticker": "AAA",
                "arima_rmse": 2.0,
                "trend_rmse": 2.5,
                "lightgbm_rmse": 1.5,
                "arima_windows": 3,
                "trend_windows": 3,
                "lightgbm_windows": 3,
            },
            {
                "ticker": "BBB",
                "arima_rmse": 1.0,
                "trend_rmse": 0.8,
                "lightgbm_rmse": 1.2,
                "arima_windows": 2,
                "trend_windows": 2,
                "lightgbm_windows": 2,
            },
            {
                "ticker": "CCC",
                "arima_rmse": 1.1,
                "trend_rmse": 1.3,
                "lightgbm_rmse": 1.0,
                "arima_windows": 1,
                "trend_windows": 1,
                "lightgbm_windows": 0,
            },
        ]
        summary = summarize_backtest_results(rows)
        self.assertEqual(summary["models"]["arima"]["windows_evaluated"], 6)
        self.assertEqual(summary["models"]["trend"]["windows_evaluated"], 6)
        self.assertEqual(summary["models"]["lightgbm"]["windows_evaluated"], 5)
        self.assertEqual(summary["models"]["lightgbm"]["tickers_evaluated"], 2)
        self.assertEqual(summary["lightgbm_wins_vs_both"]["wins"], 1)
        self.assertEqual(summary["lightgbm_wins_vs_both"]["comparable_tickers"], 2)
        self.assertEqual(summary["lightgbm_wins_vs_both"]["win_pct"], 50.0)

    def test_run_comparison_respects_explicit_empty_ticker_list(self):
        with patch("modules.backtest_comparison.get_sp500_tickers") as sp500_mock:
            result = run_lightgbm_backtest_comparison(tickers=[], sample_size=5)
        sp500_mock.assert_not_called()
        self.assertEqual(result["sample_tickers"], [])
        self.assertEqual(result["summary"]["total_tickers"], 0)


if __name__ == "__main__":
    unittest.main()
