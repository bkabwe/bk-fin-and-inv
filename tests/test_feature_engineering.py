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

        with patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}):
            with patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()):
                table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=40)

        close = price_data["Close"].values
        alpha = 2.0 / (12.0 + 1.0)
        ema_manual = close[0]
        for value in close[1:]:
            ema_manual = (alpha * value) + ((1 - alpha) * ema_manual)

        returns = price_data["Close"].pct_change().dropna().values
        vol10_manual = float(np.std(returns[-10:], ddof=1))

        self.assertAlmostEqual(float(table["technical_ema_12d"].iloc[-1]), float(ema_manual), places=6)
        self.assertAlmostEqual(float(table["technical_volatility_10d"].iloc[-1]), vol10_manual, places=12)
        self.assertGreater(float(table["technical_rsi_14d"].iloc[-1]), 99.9)


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

        with patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value=company_facts):
            with patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()):
                table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=120)

        self.assertAlmostEqual(float(table.loc["2024-01-20", "fundamental_gross_margin"]), 0.40, places=6)
        self.assertAlmostEqual(float(table.loc["2024-04-10", "fundamental_gross_margin"]), 0.40, places=6)
        self.assertAlmostEqual(float(table.loc["2024-04-16", "fundamental_gross_margin"]), 0.50, places=6)

        self.assertAlmostEqual(float(table.loc["2024-01-20", "fundamental_debt_to_equity"]), 50.0, places=6)
        self.assertAlmostEqual(float(table.loc["2024-04-16", "fundamental_debt_to_equity"]), 50.0, places=6)
        self.assertTrue(pd.isna(table.loc["2024-01-20", "fundamental_revenue_growth"]))
        self.assertAlmostEqual(float(table.loc["2024-04-16", "fundamental_revenue_growth"]), 0.5, places=6)


class FeatureEngineeringGracefulDegradationTests(unittest.TestCase):
    def test_missing_fundamental_and_macro_data_returns_nan_columns(self):
        price_data = _sample_price_frame(40)
        with patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}):
            with patch("modules.feature_engineering.fred_client.get_macro_feature_table", side_effect=RuntimeError("fred unavailable")):
                table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=40)

        self.assertFalse(table.empty)
        self.assertTrue(table["fundamental_gross_margin"].isna().all())
        self.assertTrue(table["fundamental_debt_to_equity"].isna().all())
        self.assertTrue(table["fundamental_revenue_growth"].isna().all())
        self.assertTrue(table["macro_dgs10_level"].isna().all())
        self.assertTrue(table["macro_cpiaucsl_delta_5d"].isna().all())
        self.assertTrue(table["macro_fedfunds_pct_change_30d"].isna().all())

    def test_insufficient_price_history_keeps_undercomputable_features_nan(self):
        price_data = _sample_price_frame(5)
        with patch("modules.feature_engineering.sec_edgar_client.get_company_facts", return_value={}):
            with patch("modules.feature_engineering.fred_client.get_macro_feature_table", return_value=pd.DataFrame()):
                table = feature_engineering.build_feature_table("AAPL", price_data, lookback_days=5)

        self.assertEqual(len(table), 5)
        self.assertTrue(table["technical_volatility_30d"].isna().all())
        self.assertTrue(table["technical_ema_26d"].isna().all())
        self.assertTrue(table["technical_volume_vs_avg_20d"].isna().all())


if __name__ == "__main__":
    unittest.main()
