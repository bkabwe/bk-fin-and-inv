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


def _realistic_price_frame(length: int = 252) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=length)
    drift = np.linspace(0.0, 24.0, num=length)
    wave = 2.5 * np.sin(np.arange(length) / 6.0)
    close = 100.0 + drift + wave
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 1.2,
            "Low": close - 1.2,
            "Volume": 1_000_000.0 + (np.arange(length) % 15) * 5_000.0,
        },
        index=index,
    )


class WalkForwardLightGBMTests(unittest.TestCase):
    def setUp(self):
        if hasattr(backtester.run_walk_forward, "clear"):
            backtester.run_walk_forward.clear()

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

    @unittest.skipUnless(backtester.LIGHTGBM_AVAILABLE, "lightgbm not installed")
    def test_walk_forward_trains_lightgbm_with_object_typed_macro_columns(self):
        index = pd.bdate_range("2024-01-02", periods=180)
        close = np.linspace(100.0, 130.0, num=len(index)) + np.sin(np.arange(len(index)) / 4.0)
        data = pd.DataFrame(
            {
                "Close": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Volume": np.full(len(index), 1000.0, dtype=float),
            },
            index=index,
        )

        def _macro_table(start_date, end_date):
            macro_index = pd.date_range(start_date, end_date, freq="B")
            return pd.DataFrame(
                {
                    "dgs10_level": pd.Series([None] * len(macro_index), dtype="object"),
                    "dgs10_delta_5d": pd.Series([None] * len(macro_index), dtype="object"),
                    "dgs10_pct_change_5d": pd.Series([None] * len(macro_index), dtype="object"),
                    "dgs10_delta_30d": pd.Series([None] * len(macro_index), dtype="object"),
                    "dgs10_pct_change_30d": pd.Series([None] * len(macro_index), dtype="object"),
                    "cpiaucsl_level": pd.Series([None] * len(macro_index), dtype="object"),
                    "cpiaucsl_delta_5d": pd.Series([None] * len(macro_index), dtype="object"),
                    "cpiaucsl_pct_change_5d": pd.Series([None] * len(macro_index), dtype="object"),
                    "cpiaucsl_delta_30d": pd.Series([None] * len(macro_index), dtype="object"),
                    "cpiaucsl_pct_change_30d": pd.Series([None] * len(macro_index), dtype="object"),
                    "fedfunds_level": pd.Series([None] * len(macro_index), dtype="object"),
                    "fedfunds_delta_5d": pd.Series([None] * len(macro_index), dtype="object"),
                    "fedfunds_pct_change_5d": pd.Series([None] * len(macro_index), dtype="object"),
                    "fedfunds_delta_30d": pd.Series([None] * len(macro_index), dtype="object"),
                    "fedfunds_pct_change_30d": pd.Series([None] * len(macro_index), dtype="object"),
                },
                index=macro_index,
            )

        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", side_effect=_macro_table),
        ):
            result = backtester.run_walk_forward("AAPL-LGBM-REAL", data, evaluate_lightgbm=True)

        self.assertGreater(result["lightgbm_windows"], 0)
        self.assertNotEqual(result["lightgbm_rmse"], 1.0)
        self.assertTrue(np.isfinite(float(result["lightgbm_rmse"])))

    def test_walk_forward_realistic_history_yields_non_fallback_lightgbm_windows(self):
        data = _realistic_price_frame(length=252)
        captured_training_row_indexes: list[pd.Index] = []

        def _macro_table(start_date, end_date):
            macro_index = pd.date_range(start_date, end_date, freq="B")
            return pd.DataFrame(
                {
                    "dgs10_level": np.linspace(4.0, 4.4, num=len(macro_index)),
                    "dgs10_delta_5d": np.zeros(len(macro_index)),
                    "dgs10_pct_change_5d": np.zeros(len(macro_index)),
                    "dgs10_delta_30d": np.zeros(len(macro_index)),
                    "dgs10_pct_change_30d": np.zeros(len(macro_index)),
                    "cpiaucsl_level": np.linspace(300.0, 302.0, num=len(macro_index)),
                    "cpiaucsl_delta_5d": np.zeros(len(macro_index)),
                    "cpiaucsl_pct_change_5d": np.zeros(len(macro_index)),
                    "cpiaucsl_delta_30d": np.zeros(len(macro_index)),
                    "cpiaucsl_pct_change_30d": np.zeros(len(macro_index)),
                    "fedfunds_level": np.linspace(5.0, 5.1, num=len(macro_index)),
                    "fedfunds_delta_5d": np.zeros(len(macro_index)),
                    "fedfunds_pct_change_5d": np.zeros(len(macro_index)),
                    "fedfunds_delta_30d": np.zeros(len(macro_index)),
                    "fedfunds_pct_change_30d": np.zeros(len(macro_index)),
                },
                index=macro_index,
            )

        def _train_return_models(training_examples, min_rows_per_horizon=50, random_state=42):
            dataset = training_examples.get(30)
            if not dataset:
                return None
            x_train, y_train = dataset
            captured_training_row_indexes.append(x_train.index)
            return {30: object()} if len(y_train) >= int(min_rows_per_horizon) else None

        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", True),
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", side_effect=_macro_table),
            patch("modules.backtester.train_return_models", side_effect=_train_return_models),
            patch("modules.backtester.predict_forward_return", return_value=0.01),
        ):
            result = backtester.run_walk_forward("AAPL-LGBM-WARMUP", data, evaluate_lightgbm=True)

        self.assertGreater(result["lightgbm_windows"], 0)
        self.assertNotEqual(result["lightgbm_rmse"], 1.0)
        self.assertTrue(np.isfinite(float(result["lightgbm_rmse"])))
        self.assertGreater(len(captured_training_row_indexes), 0)
        close_index = data["Close"].dropna().astype(float).tail(252).index
        expected_starts = list(range(0, max(1, len(close_index) - (60 + 30) + 1), 30))[:10]
        expected_windows = [close_index[start : start + 60] for start in expected_starts]
        self.assertEqual(len(captured_training_row_indexes), len(expected_windows))
        for captured_index, expected_index in zip(captured_training_row_indexes, expected_windows):
            self.assertTrue(captured_index.equals(expected_index))

    def test_walk_forward_lightgbm_failure_is_logged(self):
        data = _constant_price_frame()
        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", True),
            patch("modules.backtester.build_feature_table", side_effect=RuntimeError("boom")),
        ):
            with self.assertLogs("modules.backtester", level="WARNING") as captured:
                result = backtester.run_walk_forward("AAPL-LGBM-FAIL", data, evaluate_lightgbm=True)

        self.assertEqual(result["lightgbm_rmse"], 1.0)
        self.assertEqual(result["lightgbm_windows"], 0)
        self.assertTrue(any("LightGBM backtest window failed for AAPL-LGBM-FAIL" in message for message in captured.output))

    @unittest.skipUnless(backtester.ARIMA_AVAILABLE, "statsmodels not installed")
    def test_walk_forward_timezone_aware_history_keeps_all_arima_windows(self):
        index = pd.bdate_range("2024-01-02", periods=252).tz_localize("America/New_York")
        close = 100.0 + np.linspace(0.0, 24.0, num=len(index)) + 1.5 * np.sin(np.arange(len(index)) / 6.0)
        data = pd.DataFrame(
            {
                "Close": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Volume": np.full(len(index), 1000.0, dtype=float),
            },
            index=index,
        )

        with (
            patch("modules.backtester._select_arima_order", return_value=(1, 0, 0)),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", False),
        ):
            result = backtester.run_walk_forward("AAPL-ARIMA-TZ", data, evaluate_lightgbm=False)

        self.assertEqual(result["n_windows"], 6)
        self.assertEqual(result["arima_windows"], 6)


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
