from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import lightgbm_model


def _sample_price_data(length: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=length, freq="D")
    close = np.linspace(100.0, 200.0, num=length)
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Volume": np.full(length, 1000.0),
        },
        index=idx,
    )


def _sample_feature_table(length: int = 900, include_nans: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=length, freq="D")
    table = pd.DataFrame(
        {
            "technical_feature_a": np.linspace(0.0, 1.0, num=length),
            "macro_feature_b": np.linspace(1.0, 2.0, num=length),
        },
        index=idx,
    )
    if include_nans and length > 5:
        table.iloc[5, 0] = np.nan
    if include_nans and length > 7:
        table.iloc[7, 1] = np.nan
    return table


class LightGBMModelTests(unittest.TestCase):
    def test_forward_return_training_example_calculation(self):
        price_data = _sample_price_data(length=100)
        feature_table = _sample_feature_table(length=100, include_nans=False)
        examples = lightgbm_model.build_return_training_examples(
            "AAPL",
            price_data,
            feature_table=feature_table,
            horizons=(30,),
        )

        self.assertIn(30, examples)
        x_train, y_train = examples[30]
        self.assertEqual(len(x_train), 70)
        expected = (price_data["Close"].iloc[30] - price_data["Close"].iloc[0]) / price_data["Close"].iloc[0]
        self.assertAlmostEqual(float(y_train.iloc[0]), float(expected), places=12)

    def test_training_examples_keep_nan_features_for_lightgbm_native_handling(self):
        price_data = _sample_price_data(length=100)
        feature_table = _sample_feature_table(length=100, include_nans=True)
        examples = lightgbm_model.build_return_training_examples(
            "AAPL",
            price_data,
            feature_table=feature_table,
            horizons=(30,),
        )
        x_train, _ = examples[30]
        self.assertEqual(len(x_train), 70)
        self.assertTrue(x_train.isna().any().any())

    def test_training_examples_align_timezone_aware_close_with_daily_feature_index(self):
        base_index = pd.bdate_range("2024-01-02", periods=120)
        price_index = base_index.tz_localize("America/New_York")
        close = np.linspace(100.0, 130.0, num=len(price_index))
        price_data = pd.DataFrame(
            {
                "Close": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Volume": np.full(len(price_index), 1000.0),
            },
            index=price_index,
        )
        feature_table = pd.DataFrame(
            {"technical_feature_a": np.linspace(0.0, 1.0, num=len(base_index))},
            index=base_index,
        )

        examples = lightgbm_model.build_return_training_examples(
            "AAPL",
            price_data,
            feature_table=feature_table,
            horizons=(30,),
        )

        self.assertIn(30, examples)
        _, y_train = examples[30]
        self.assertGreater(len(y_train), 0)

    def test_prepare_feature_frame_normalizes_and_deduplicates_index(self):
        feature_table = pd.DataFrame(
            {"technical_feature_a": [1.0, 2.0, 3.0]},
            index=pd.DatetimeIndex(
                [
                    pd.Timestamp("2024-01-02 16:00:00", tz="America/New_York"),
                    pd.Timestamp("2024-01-02 18:00:00", tz="America/New_York"),
                    pd.Timestamp("2024-01-03 16:00:00", tz="America/New_York"),
                ]
            ),
        )

        prepared = lightgbm_model.prepare_lightgbm_feature_frame(feature_table)

        self.assertEqual(list(prepared.index), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")])
        self.assertEqual(float(prepared.loc[pd.Timestamp("2024-01-02"), "technical_feature_a"]), 2.0)

    def test_latest_lightgbm_feature_row_skips_all_nan_tail_rows(self):
        feature_table = pd.DataFrame(
            {
                "technical_feature_a": [1.0, np.nan, np.nan],
                "macro_feature_b": [2.0, np.nan, np.nan],
            },
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )

        latest_row = lightgbm_model.latest_lightgbm_feature_row(feature_table)

        self.assertIsNotNone(latest_row)
        self.assertEqual(latest_row.name, pd.Timestamp("2024-01-01"))
        self.assertEqual(float(latest_row["technical_feature_a"]), 1.0)

    def test_build_examples_for_ticker_reuses_stock_data_fetch(self):
        price_data = _sample_price_data(length=120)
        with patch("modules.lightgbm_model.get_stock_data", return_value=price_data) as stock_data_mock:
            examples = lightgbm_model.build_return_training_examples_for_ticker("AAPL", horizons=(30,))
        stock_data_mock.assert_called_once_with("AAPL", period="5y", interval="1d")
        self.assertIn(30, examples)

    def test_training_examples_exclude_extreme_forward_return_labels_and_keep_surrounding_examples(self):
        price_data = _sample_price_data(length=100)
        anomalous_date = price_data.index[20]
        price_data.loc[anomalous_date, "Close"] = 10.0
        feature_table = _sample_feature_table(length=100, include_nans=False)

        with self.assertLogs("modules.lightgbm_model", level="WARNING") as logs:
            examples = lightgbm_model.build_return_training_examples(
                "AIM",
                price_data,
                feature_table=feature_table,
                horizons=(30,),
            )

        x_train, y_train = examples[30]
        self.assertEqual(len(x_train), 69)
        self.assertEqual(len(y_train), 69)
        self.assertNotIn(anomalous_date, x_train.index)
        self.assertIn(price_data.index[19], x_train.index)
        self.assertIn(price_data.index[21], x_train.index)
        self.assertLessEqual(float(y_train.max()), lightgbm_model.MAX_FORWARD_RETURN_LABEL_30D)
        self.assertTrue(any("Excluding LightGBM training label for AIM horizon=30d" in message for message in logs.output))

    def test_compare_feature_row_to_training_ranges_flags_out_of_range_values(self):
        x_train = pd.DataFrame(
            {
                "feature_a": np.linspace(0.0, 1.0, num=20),
                "feature_b": np.linspace(10.0, 20.0, num=20),
            }
        )
        summary = lightgbm_model.summarize_training_feature_ranges(x_train)
        comparison = lightgbm_model.compare_feature_row_to_training_ranges(
            pd.Series({"feature_a": 9.0, "feature_b": 15.0}),
            summary,
            std_threshold=3.0,
        )

        self.assertTrue(bool(comparison.loc["feature_a", "outside_training_range"]))
        self.assertTrue(bool(comparison.loc["feature_a", "beyond_std_threshold"]))
        self.assertFalse(bool(comparison.loc["feature_b", "outside_training_range"]))
        self.assertFalse(bool(comparison.loc["feature_b", "beyond_std_threshold"]))

    @unittest.skipUnless(lightgbm_model.LIGHTGBM_AVAILABLE, "lightgbm not installed")
    def test_train_save_load_and_infer(self):
        price_data = _sample_price_data(length=900)
        feature_table = _sample_feature_table(length=900)
        examples = lightgbm_model.build_return_training_examples(
            "AAPL",
            price_data,
            feature_table=feature_table,
            horizons=(30,),
        )

        models = lightgbm_model.train_return_models(examples, min_rows_per_horizon=50)
        self.assertIsNotNone(models)
        self.assertIn(30, models or {})

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = lightgbm_model.save_return_models(models or {}, tmpdir)
            self.assertEqual(len(saved), 1)
            loaded = lightgbm_model.load_return_models(tmpdir, horizons=(30,))
            self.assertIn(30, loaded)
            pred = lightgbm_model.predict_forward_return(loaded[30], feature_table.iloc[-1])
            reordered_with_extra = pd.Series(
                {
                    "macro_feature_b": feature_table.iloc[-1]["macro_feature_b"],
                    "technical_feature_a": feature_table.iloc[-1]["technical_feature_a"],
                    "unexpected_feature": 999.0,
                }
            )
            pred_reordered = lightgbm_model.predict_forward_return(loaded[30], reordered_with_extra)

        self.assertIsNotNone(pred)
        self.assertIsNotNone(pred_reordered)
        self.assertTrue(np.isfinite(float(pred)))
        self.assertTrue(np.isfinite(float(pred_reordered)))
        self.assertAlmostEqual(float(pred), float(pred_reordered), places=10)
        self.assertLess(abs(float(pred)), 5.0)

    def test_graceful_handling_for_missing_lightgbm(self):
        example = {30: (_sample_feature_table(length=120), pd.Series(np.linspace(0.01, 0.03, num=120)))}
        with patch("modules.lightgbm_model.LIGHTGBM_AVAILABLE", False):
            result = lightgbm_model.train_return_models(example, min_rows_per_horizon=10)
        self.assertIsNone(result)

    def test_graceful_handling_for_insufficient_training_data(self):
        x_small = _sample_feature_table(length=20)
        y_small = pd.Series(np.linspace(0.01, 0.02, num=20))
        result = lightgbm_model.train_return_models({30: (x_small, y_small)}, min_rows_per_horizon=50)
        self.assertIsNone(result)

    def test_inference_gracefully_handles_missing_model(self):
        pred = lightgbm_model.predict_forward_return(None, _sample_feature_table(length=1).iloc[0])
        self.assertIsNone(pred)


if __name__ == "__main__":
    unittest.main()
