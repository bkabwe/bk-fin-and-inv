from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import backtester
from modules.backtest_comparison import (
    format_per_ticker_diagnostics_table,
    run_lightgbm_backtest_comparison,
    select_sample_tickers,
    summarize_backtest_results,
)


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


def _constant_business_price_frame(length: int = 120, value: float = 100.0) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-02", periods=length)
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
        # Embargo: the last `horizon` (30) rows of each 60-row train window are
        # excluded because their forward-return label's target close falls
        # inside the held-out test window (see run_walk_forward's embargo
        # comment). Only the first 30 rows of each window remain trainable.
        expected_windows = [close_index[start : start + 30] for start in expected_starts]
        self.assertEqual(len(captured_training_row_indexes), len(expected_windows))
        for captured_index, expected_index in zip(captured_training_row_indexes, expected_windows, strict=False):
            self.assertTrue(captured_index.equals(expected_index))

    def test_walk_forward_normalizes_timezone_aware_lightgbm_training_window_indexes(self):
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
            result = backtester.run_walk_forward("AAPL-LGBM-TZ", data, evaluate_lightgbm=True)

        self.assertGreater(result["lightgbm_windows"], 0)
        self.assertNotEqual(result["lightgbm_rmse"], 1.0)
        self.assertGreater(len(captured_training_row_indexes), 0)

        close_index = data["Close"].dropna().astype(float).tail(252).index
        normalized_index = backtester._normalize_lightgbm_price_frame(data.copy().reindex(close_index)).index
        expected_starts = list(range(0, max(1, len(normalized_index) - (60 + 30) + 1), 30))[:10]
        # Embargo: only the first 30 (of 60) train rows survive -- see the
        # embargo comment in run_walk_forward / the analogous non-tz test above.
        expected_windows = [normalized_index[start : start + 30] for start in expected_starts]

        self.assertEqual(len(captured_training_row_indexes), len(expected_windows))
        for captured_index, expected_index in zip(captured_training_row_indexes, expected_windows, strict=False):
            self.assertIsNone(getattr(captured_index, "tz", None))
            self.assertEqual(len(captured_index), 30)
            self.assertTrue(captured_index.equals(expected_index))

    def test_lightgbm_training_examples_reindex_against_normalized_timezone_aware_train_window(self):
        index = pd.bdate_range("2024-01-02", periods=120).tz_localize("America/New_York")
        close = 100.0 + np.linspace(0.0, 12.0, num=len(index))
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
                    "dgs10_level": np.linspace(4.0, 4.2, num=len(macro_index)),
                    "dgs10_delta_5d": np.zeros(len(macro_index)),
                    "dgs10_pct_change_5d": np.zeros(len(macro_index)),
                    "dgs10_delta_30d": np.zeros(len(macro_index)),
                    "dgs10_pct_change_30d": np.zeros(len(macro_index)),
                    "cpiaucsl_level": np.linspace(300.0, 301.0, num=len(macro_index)),
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

        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", side_effect=_macro_table),
        ):
            train_price = backtester._normalize_lightgbm_price_frame(data.iloc[:60])
            history = backtester._normalize_lightgbm_price_frame(data.iloc[:90])
            features = backtester.build_feature_table("AAPL", history, lookback_days=len(history))
            datasets = backtester.build_return_training_examples(
                ticker="AAPL",
                price_data=history,
                feature_table=features,
                lookback_days=len(history),
                horizons=(30,),
            )

        x_train_all, y_train_all = datasets[30]
        x_train = x_train_all.reindex(train_price.index).dropna(how="all")
        y_train = y_train_all.reindex(x_train.index).dropna()
        x_train = x_train.reindex(y_train.index)

        self.assertEqual(len(train_price), 60)
        self.assertEqual(len(x_train), 60)
        self.assertEqual(len(y_train), 60)
        self.assertTrue(x_train.index.equals(train_price.index))
        self.assertIsNone(getattr(x_train.index, "tz", None))

    def test_walk_forward_lightgbm_failure_is_logged(self):
        data = _constant_price_frame()
        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", True),
            patch("modules.backtester.build_feature_table", side_effect=RuntimeError("boom")),
            self.assertLogs("modules.backtester", level="WARNING") as captured,
        ):
            result = backtester.run_walk_forward("AAPL-LGBM-FAIL", data, evaluate_lightgbm=True)

        self.assertIsNone(result["lightgbm_rmse"])
        self.assertEqual(result["lightgbm_windows"], 0)
        self.assertTrue(any("LightGBM backtest window failed for AAPL-LGBM-FAIL" in message for message in captured.output))

    def test_walk_forward_reports_none_lightgbm_rmse_with_zero_lightgbm_windows_and_valid_trend_windows(self):
        data = _constant_price_frame(length=150, value=100.0)

        class _FakeLinearRegression:
            def fit(self, x, y):
                return self

            def predict(self, future):
                return np.log(np.full(len(future), 100.0))

        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", True),
            patch("modules.backtester.LinearRegression", _FakeLinearRegression),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", True),
            patch("modules.backtester.build_feature_table", side_effect=RuntimeError("boom")),
        ):
            result = backtester.run_walk_forward("AAPL-LGBM-NONE", data, evaluate_lightgbm=True)

        self.assertGreater(result["trend_windows"], 0)
        self.assertEqual(result["lightgbm_windows"], 0)
        self.assertIsNone(result["lightgbm_rmse"])

    def test_walk_forward_reports_naive_no_change_baseline_rmse(self):
        index = pd.date_range("2024-01-01", periods=120, freq="D")
        close = 100.0 + np.arange(120, dtype=float)
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
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", False),
        ):
            result = backtester.run_walk_forward(
                "AAPL-NAIVE",
                data,
                evaluate_lightgbm=False,
                evaluate_naive_baseline=True,
            )

        expected_rmse = float(np.sqrt(np.mean(np.square(np.arange(1, 31, dtype=float)))))
        self.assertEqual(result["naive_windows"], 2)
        self.assertAlmostEqual(result["naive_rmse"], round(expected_rmse, 6), places=6)

    def test_walk_forward_default_horizon_matches_explicit_30d_behavior(self):
        data = _constant_price_frame()

        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", False),
        ):
            default_result = backtester.run_walk_forward(
                "AAPL-DEFAULT-HORIZON",
                data,
                evaluate_lightgbm=False,
                evaluate_naive_baseline=True,
            )
            explicit_result = backtester.run_walk_forward(
                "AAPL-EXPLICIT-30D",
                data,
                horizon=30,
                evaluate_lightgbm=False,
                evaluate_naive_baseline=True,
            )

        self.assertEqual(default_result, explicit_result)

    def test_walk_forward_horizon_specific_window_counts_fit_realistic_five_year_history(self):
        data = _constant_business_price_frame(length=1258)

        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", False),
        ):
            result_180 = backtester.run_walk_forward(
                "AAPL-H180",
                data,
                horizon=180,
                evaluate_lightgbm=False,
                evaluate_naive_baseline=True,
            )
            result_720 = backtester.run_walk_forward(
                "AAPL-H720",
                data,
                horizon=720,
                evaluate_lightgbm=False,
                evaluate_naive_baseline=True,
            )

        self.assertEqual(result_180["naive_windows"], 4)
        self.assertEqual(result_180["n_windows"], 4)
        self.assertEqual(result_720["naive_windows"], 1)
        self.assertEqual(result_720["n_windows"], 1)
        self.assertLess(result_720["naive_windows"], result_180["naive_windows"])

    def test_walk_forward_window_configs_keep_30d_and_180d_unchanged(self):
        config_30 = backtester.get_walk_forward_window_config(30)
        config_180 = backtester.get_walk_forward_window_config(180)
        config_720 = backtester.get_walk_forward_window_config(720)

        self.assertEqual((config_30.train_len, config_30.test_len, config_30.stride, config_30.max_history_rows), (60, 30, 30, 252))
        self.assertEqual((config_180.train_len, config_180.test_len, config_180.stride, config_180.max_history_rows), (360, 180, 180, 1260))
        self.assertEqual((config_720.train_len, config_720.test_len, config_720.stride, config_720.max_history_rows), (450, 720, 720, 1260))

    @unittest.skipUnless(backtester.LIGHTGBM_AVAILABLE, "lightgbm not installed")
    def test_walk_forward_lightgbm_diagnostics_capture_near_constant_predictions(self):
        data = _realistic_price_frame(length=252)

        def _build_features(_ticker, price_data, lookback_days=0):
            return pd.DataFrame({"feature_a": np.arange(len(price_data), dtype=float)}, index=price_data.index)

        def _build_examples(**kwargs):
            feature_table = kwargs["feature_table"]
            x_train = pd.DataFrame({"feature_a": np.zeros(len(feature_table), dtype=float)}, index=feature_table.index)
            y_train = pd.Series(np.full(len(feature_table), 0.0025, dtype=float), index=feature_table.index)
            return {30: (x_train, y_train)}

        with (
            patch("modules.backtester.ARIMA_AVAILABLE", False),
            patch("modules.backtester.SKLEARN_AVAILABLE", False),
            patch("modules.backtester.LIGHTGBM_AVAILABLE", True),
            patch("modules.backtester.build_feature_table", side_effect=_build_features),
            patch("modules.backtester.build_return_training_examples", side_effect=_build_examples),
        ):
            result = backtester.run_walk_forward(
                "AAPL-LGBM-DIAG",
                data,
                evaluate_lightgbm=True,
                lightgbm_diagnostics=True,
            )

        diagnostics = result["lightgbm_diagnostics"]
        prediction_stats = diagnostics["prediction_stats"]
        per_window = diagnostics["per_window"]
        self.assertGreater(result["lightgbm_windows"], 0)
        self.assertEqual(len(per_window), result["lightgbm_windows"])
        self.assertIsNotNone(prediction_stats)
        self.assertEqual(prediction_stats["std"], 0.0)
        self.assertEqual(prediction_stats["min"], prediction_stats["max"])
        realized_returns = {round(float(row["realized_return"]), 8) for row in per_window}
        self.assertGreater(len(realized_returns), 1)

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

    def test_select_sample_tickers_preserves_default_order_without_seed(self):
        tickers = [f"TICKER{i:03d}" for i in range(10)]
        self.assertEqual(select_sample_tickers(tickers, sample_size=4), tickers[:4])

    def test_run_comparison_reuses_same_seeded_sample(self):
        tickers = [f"TICKER{i:03d}" for i in range(100)]
        stub_result = {"arima_rmse": 1.0, "trend_rmse": 1.0, "lightgbm_rmse": 1.0, "n_windows": 0}

        with (
            patch("modules.backtest_comparison.get_sp500_tickers", return_value=tickers),
            patch("modules.backtest_comparison.get_stock_data", return_value=_constant_price_frame()),
            patch("modules.backtest_comparison.run_walk_forward", return_value=stub_result),
        ):
            first = run_lightgbm_backtest_comparison(sample_size=8, random_seed=42)
            second = run_lightgbm_backtest_comparison(sample_size=8, random_seed=42)

        self.assertEqual(first["sample_tickers"], second["sample_tickers"])
        self.assertEqual([row["ticker"] for row in first["per_ticker"]], first["sample_tickers"])
        self.assertEqual([row["ticker"] for row in second["per_ticker"]], second["sample_tickers"])

    def test_run_comparison_changes_sample_with_different_seed(self):
        tickers = [f"TICKER{i:03d}" for i in range(100)]
        stub_result = {"arima_rmse": 1.0, "trend_rmse": 1.0, "lightgbm_rmse": 1.0, "n_windows": 0}

        with (
            patch("modules.backtest_comparison.get_sp500_tickers", return_value=tickers),
            patch("modules.backtest_comparison.get_stock_data", return_value=_constant_price_frame()),
            patch("modules.backtest_comparison.run_walk_forward", return_value=stub_result),
        ):
            first = run_lightgbm_backtest_comparison(sample_size=8, random_seed=7)
            second = run_lightgbm_backtest_comparison(sample_size=8, random_seed=13)

        self.assertNotEqual(first["sample_tickers"], second["sample_tickers"])
        self.assertNotEqual(set(first["sample_tickers"]), set(second["sample_tickers"]))

    def test_run_comparison_applies_ticker_offset_after_default_ordering(self):
        tickers = [f"TICKER{i:03d}" for i in range(10)]
        stub_result = {"arima_rmse": 1.0, "trend_rmse": 1.0, "lightgbm_rmse": 1.0, "n_windows": 0}

        with (
            patch("modules.backtest_comparison.get_sp500_tickers", return_value=tickers),
            patch("modules.backtest_comparison.get_stock_data", return_value=_constant_price_frame()),
            patch("modules.backtest_comparison.run_walk_forward", return_value=stub_result),
        ):
            result = run_lightgbm_backtest_comparison(sample_size=3, ticker_offset=4)

        self.assertEqual(result["sample_tickers"], tickers[4:7])

    def test_run_comparison_uses_horizon_specific_defaults_and_forwards_horizon(self):
        tickers = ["TICKER001"]
        stub_result = {"arima_rmse": 1.0, "trend_rmse": 1.0, "lightgbm_rmse": 1.0, "n_windows": 0}

        with (
            patch("modules.backtest_comparison.get_sp500_tickers", return_value=tickers),
            patch("modules.backtest_comparison.get_stock_data", return_value=_constant_price_frame()) as stock_data_mock,
            patch("modules.backtest_comparison.run_walk_forward", return_value=stub_result) as walk_forward_mock,
        ):
            result = run_lightgbm_backtest_comparison(sample_size=1, horizon=720)

        stock_data_mock.assert_called_once_with("TICKER001", period="5y", interval="1d")
        walk_forward_mock.assert_called_once()
        self.assertEqual(walk_forward_mock.call_args.kwargs["horizon"], 720)
        self.assertEqual(result["period"], "5y")
        self.assertEqual(result["window_config"]["train_len"], 450)
        self.assertEqual(result["window_config"]["test_len"], 720)
        self.assertEqual(result["window_config"]["stride"], 720)

    def test_format_per_ticker_diagnostics_table_renders_na_for_zero_window_rmse(self):
        table = format_per_ticker_diagnostics_table(
            [
                {
                    "ticker": "ZERO",
                    "arima_rmse": 1.0,
                    "trend_rmse": 1.0,
                    "lightgbm_rmse": 1.0,
                    "naive_rmse": 1.0,
                    "arima_windows": 0,
                    "trend_windows": 0,
                    "lightgbm_windows": 0,
                    "naive_windows": 0,
                    "lightgbm_diagnostics": {"prediction_stats": None},
                }
            ]
        )

        self.assertIn("ZERO", table)
        self.assertIn("n/a", table)
        self.assertNotIn("1.000000", table)

    def test_summarize_backtest_results_includes_naive_model_and_comparison(self):
        rows = [
            {
                "ticker": "AAA",
                "arima_rmse": 2.0,
                "trend_rmse": 2.5,
                "lightgbm_rmse": 1.5,
                "naive_rmse": 1.7,
                "arima_windows": 3,
                "trend_windows": 3,
                "lightgbm_windows": 3,
                "naive_windows": 3,
            },
            {
                "ticker": "BBB",
                "arima_rmse": 1.0,
                "trend_rmse": 0.8,
                "lightgbm_rmse": 1.2,
                "naive_rmse": 1.1,
                "arima_windows": 2,
                "trend_windows": 2,
                "lightgbm_windows": 2,
                "naive_windows": 2,
            },
        ]
        summary = summarize_backtest_results(rows)
        self.assertIn("naive", summary["models"])
        self.assertEqual(summary["models"]["naive"]["windows_evaluated"], 5)
        self.assertEqual(summary["lightgbm_wins_vs_naive"]["wins"], 1)
        self.assertEqual(summary["lightgbm_wins_vs_naive"]["comparable_tickers"], 2)
        self.assertEqual(summary["lightgbm_wins_vs_naive"]["win_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
