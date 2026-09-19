from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import feature_engineering


def _sample_price_frame(length: int = 40, start: str = "2024-01-01") -> pd.DataFrame:
    close = np.arange(1, length + 1, dtype=float)
    index = pd.date_range(start=start, periods=length, freq="D")
    return pd.DataFrame(
        {
            "Close": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Volume": np.full(length, 1000.0),
        },
        index=index,
    )


class FeatureEngineeringTechnicalTests(unittest.TestCase):
    def test_rsi_ema_and_rolling_volatility(self):
        price_data = _sample_price_frame(40)

        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()),
        ):
            table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=40)

        close = price_data["Close"].values
        alpha = 2.0 / (12.0 + 1.0)
        ema_manual = close[0]
        for value in close[1:]:
            ema_manual = (alpha * value) + ((1 - alpha) * ema_manual)
        ema_ratio_manual = (close[-1] / ema_manual) - 1.0

        returns = price_data["Close"].pct_change().dropna().values
        vol10_manual = float(np.std(returns[-10:], ddof=1))

        self.assertAlmostEqual(float(table["technical_close_to_ema_12d"].iloc[-1]), float(ema_ratio_manual), places=6)
        self.assertAlmostEqual(float(table["technical_volatility_10d"].iloc[-1]), vol10_manual, places=12)
        self.assertGreater(float(table["technical_rsi_14d"].iloc[-1]), 99.9)
        self.assertTrue(table["technical_close_to_ema_12d"].iloc[:11].isna().all())
        self.assertTrue(table["technical_close_to_ema_26d"].iloc[:25].isna().all())
        # Raw price/level features are never exposed to the (tree-based)
        # LightGBM model -- only stationary ratios/returns are.
        self.assertNotIn("technical_ema_12d", table.columns)
        self.assertNotIn("technical_ema_26d", table.columns)
        self.assertNotIn("technical_vwap_daily_approx", table.columns)

    def test_lagged_return_and_rolling_mean_return_features(self):
        price_data = _sample_price_frame(90)
        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()),
        ):
            table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=90)

        returns = price_data["Close"].pct_change()
        # Compare from index 55 onward so every rolling/lag window (up to 50d) is
        # fully warmed up and NaN-free on both sides.
        self.assertTrue(np.allclose(table["technical_return_lag_1d"].iloc[55:].values, returns.shift(1).iloc[55:].values))
        self.assertTrue(np.allclose(table["technical_return_lag_2d"].iloc[55:].values, returns.shift(2).iloc[55:].values))
        self.assertTrue(np.allclose(table["technical_return_lag_5d"].iloc[55:].values, returns.shift(5).iloc[55:].values))
        self.assertTrue(
            np.allclose(table["technical_return_mean_5d"].iloc[55:].values, returns.rolling(5).mean().iloc[55:].values)
        )
        self.assertTrue(
            np.allclose(table["technical_return_mean_20d"].iloc[55:].values, returns.rolling(20).mean().iloc[55:].values)
        )
        self.assertTrue(
            np.allclose(table["technical_return_mean_50d"].iloc[55:].values, returns.rolling(50).mean().iloc[55:].values)
        )

    def test_rsi_flat_series_is_neutral_after_warmup(self):
        price_data = _sample_price_frame(30)
        price_data["Close"] = 10.0
        price_data["High"] = 10.5
        price_data["Low"] = 9.5
        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()),
        ):
            table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=30)

        self.assertAlmostEqual(float(table["technical_rsi_14d"].iloc[-1]), 50.0, places=6)



class FeatureEngineeringFundamentalTests(unittest.TestCase):
    def test_fundamentals_forward_fill_until_next_filing(self):
        price_data = _sample_price_frame(120, start="2024-01-01")
        company_facts = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"end": "2023-12-31", "filed": "2024-01-15", "form": "10-Q", "val": 100.0},
                                {"end": "2024-03-31", "filed": "2024-04-15", "form": "10-Q", "val": 150.0},
                            ]
                        }
                    },
                    "GrossProfit": {
                        "units": {
                            "USD": [
                                {"end": "2023-12-31", "filed": "2024-01-15", "form": "10-Q", "val": 40.0},
                                {"end": "2024-03-31", "filed": "2024-04-15", "form": "10-Q", "val": 75.0},
                            ]
                        }
                    },
                    "Liabilities": {
                        "units": {
                            "USD": [
                                {"end": "2023-12-31", "filed": "2024-01-15", "form": "10-Q", "val": 50.0},
                                {"end": "2024-03-31", "filed": "2024-04-15", "form": "10-Q", "val": 60.0},
                            ]
                        }
                    },
                    "StockholdersEquity": {
                        "units": {
                            "USD": [
                                {"end": "2023-12-31", "filed": "2024-01-15", "form": "10-Q", "val": 100.0},
                                {"end": "2024-03-31", "filed": "2024-04-15", "form": "10-Q", "val": 120.0},
                            ]
                        }
                    },
                }
            }
        }

        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value=company_facts),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()),
        ):
            table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=120)

        self.assertAlmostEqual(float(table.loc["2024-01-20", "fundamental_gross_margin"]), 0.40, places=6)
        self.assertAlmostEqual(float(table.loc["2024-04-10", "fundamental_gross_margin"]), 0.40, places=6)
        self.assertAlmostEqual(float(table.loc["2024-04-16", "fundamental_gross_margin"]), 0.50, places=6)
        self.assertTrue(pd.isna(table.loc["2024-01-10", "fundamental_gross_margin"]))

        # Feature follows existing SEC adapter scaling (ratio * 100).
        self.assertAlmostEqual(float(table.loc["2024-01-20", "fundamental_debt_to_equity"]), 50.0, places=6)
        self.assertAlmostEqual(float(table.loc["2024-04-16", "fundamental_debt_to_equity"]), 50.0, places=6)
        self.assertTrue(pd.isna(table.loc["2024-01-20", "fundamental_revenue_growth"]))
        self.assertAlmostEqual(float(table.loc["2024-04-16", "fundamental_revenue_growth"]), 0.5, places=6)


class FeatureEngineeringGracefulDegradationTests(unittest.TestCase):
    def test_missing_fundamental_and_macro_data_returns_nan_columns(self):
        price_data = _sample_price_frame(40)
        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", side_effect=RuntimeError("fred unavailable")),
        ):
            table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=40)

        self.assertFalse(table.empty)
        self.assertTrue(table["fundamental_gross_margin"].isna().all())
        self.assertTrue(table["fundamental_debt_to_equity"].isna().all())
        self.assertTrue(table["fundamental_revenue_growth"].isna().all())
        self.assertNotIn("macro_dgs10_level", table.columns)
        self.assertTrue(table["macro_cpiaucsl_delta_5d"].isna().all())
        self.assertTrue(table["macro_fedfunds_pct_change_30d"].isna().all())

    def test_insufficient_price_history_keeps_undercomputable_features_nan(self):
        price_data = _sample_price_frame(5)
        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()),
        ):
            table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=5)

        self.assertEqual(len(table), 5)
        self.assertTrue(table["technical_volatility_30d"].isna().all())
        self.assertTrue(table["technical_close_to_ema_26d"].isna().all())
        self.assertTrue(table["technical_volume_vs_avg_20d"].isna().all())
        self.assertTrue(table["technical_rsi_14d"].isna().all())

    def test_feature_table_uses_cached_result_for_same_inputs(self):
        price_data = _sample_price_frame(40)
        if hasattr(feature_engineering._build_feature_table_cached, "clear"):
            feature_engineering._build_feature_table_cached.clear()
        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}) as sec_mock,
            patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()) as macro_mock,
        ):
            first = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=40)
            second = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=40)

        self.assertEqual(sec_mock.call_count, 1)
        self.assertEqual(macro_mock.call_count, 1)
        self.assertEqual(len(first), len(second))

    def test_feature_table_uses_shared_macro_table_without_refetching_fred(self):
        price_data = _sample_price_frame(40)
        shared_macro = pd.DataFrame(
            {
                "dgs10_level": np.linspace(4.0, 4.2, num=60),
                "dgs10_delta_5d": np.linspace(0.0, 0.1, num=60),
                "dgs10_pct_change_5d": np.linspace(0.0, 0.01, num=60),
                "dgs10_delta_30d": np.linspace(0.0, 0.2, num=60),
                "dgs10_pct_change_30d": np.linspace(0.0, 0.02, num=60),
                "cpiaucsl_level": np.linspace(300.0, 302.0, num=60),
                "cpiaucsl_delta_5d": np.linspace(0.0, 0.3, num=60),
                "cpiaucsl_pct_change_5d": np.linspace(0.0, 0.01, num=60),
                "cpiaucsl_delta_30d": np.linspace(0.0, 1.0, num=60),
                "cpiaucsl_pct_change_30d": np.linspace(0.0, 0.03, num=60),
                "fedfunds_level": np.linspace(5.0, 5.1, num=60),
                "fedfunds_delta_5d": np.linspace(0.0, 0.05, num=60),
                "fedfunds_pct_change_5d": np.linspace(0.0, 0.01, num=60),
                "fedfunds_delta_30d": np.linspace(0.0, 0.1, num=60),
                "fedfunds_pct_change_30d": np.linspace(0.0, 0.02, num=60),
            },
            index=pd.date_range("2023-12-15", periods=60, freq="D"),
        )
        with (
            patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}),
            patch("modules.feature_engineering.fred_client.get_macro_feature_table") as macro_mock,
        ):
            table = feature_engineering.build_feature_table(
                "AAPL",
                price_data,
                lookback_days=40,
                shared_macro_table=shared_macro,
            )

        macro_mock.assert_not_called()
        self.assertFalse(table.empty)
        self.assertNotIn("macro_dgs10_level", table.columns)
        self.assertFalse(table["macro_dgs10_delta_5d"].isna().all())


if __name__ == "__main__":
    unittest.main()
