from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules import price_format, profit_opportunities, scoring_engine


class FormatPriceTests(unittest.TestCase):
    def test_whole_cents_for_dollar_prices(self):
        self.assertEqual(price_format.format_price(1234.5), "$1,234.50")
        self.assertEqual(price_format.format_price(1), "$1.00")

    def test_sub_dollar_prices_keep_four_decimals_and_sub_cent_prices_six(self):
        self.assertEqual(price_format.format_price(0.2083), "$0.2083")
        self.assertEqual(price_format.format_price(0.0031234), "$0.003123")

    def test_missing_and_invalid_values(self):
        self.assertEqual(price_format.format_price(None), "—")
        self.assertEqual(price_format.format_price(""), "—")
        self.assertEqual(price_format.format_price(float("nan")), "—")
        self.assertEqual(price_format.format_price("n/a"), "n/a")

    def test_numeric_strings_are_formatted(self):
        self.assertEqual(price_format.format_price("12.3456"), "$12.35")

    def test_scoring_engine_reuses_the_shared_helpers(self):
        self.assertIs(scoring_engine.price_decimals, price_format.price_decimals)
        self.assertIs(scoring_engine.round_price, price_format.round_price)


class PriceDecimalsTests(unittest.TestCase):
    def test_keeps_cents_for_prices_of_one_dollar_and_up(self):
        self.assertEqual(scoring_engine.price_decimals(1.0), 2)
        self.assertEqual(scoring_engine.price_decimals(190.55), 2)

    def test_uses_four_decimals_between_one_cent_and_one_dollar(self):
        self.assertEqual(scoring_engine.price_decimals(0.9999), 4)
        self.assertEqual(scoring_engine.price_decimals(0.124), 4)
        self.assertEqual(scoring_engine.price_decimals(0.01), 4)

    def test_uses_six_decimals_below_one_cent(self):
        self.assertEqual(scoring_engine.price_decimals(0.0099), 6)
        self.assertEqual(scoring_engine.price_decimals(0.0), 6)
        self.assertEqual(scoring_engine.price_decimals(None), 6)

    def test_round_price_defaults_to_the_value_s_own_precision(self):
        self.assertEqual(scoring_engine.round_price(0.123456), 0.1235)
        self.assertEqual(scoring_engine.round_price(123.456), 123.46)
        self.assertEqual(scoring_engine.round_price(0.0031234567), 0.003123)

    def test_round_price_can_use_a_separate_reference_price(self):
        self.assertEqual(scoring_engine.round_price(0.0031234567, 150.0), 0.0)
        self.assertEqual(scoring_engine.round_price(0.123456, 0.004), 0.123456)


class SubDollarRoundingHelperTests(unittest.TestCase):
    def test_cap_target_does_not_round_a_penny_stock_target_to_the_nearest_cent(self):
        # price $0.124, +200% cap = $0.372. Cent rounding gave $0.37 (an
        # upside of 198.4%) and, for other prices, rounded *past* the cap.
        self.assertEqual(scoring_engine._cap_target(0.9, 0.124, 0.372), 0.372)
        self.assertEqual(scoring_engine._cap_target(0.1, 0.124, 0.372), 0.124)

    def test_cap_target_keeps_cent_rounding_for_normal_prices(self):
        self.assertEqual(scoring_engine._cap_target(123.456, 100.0, 300.0), 123.46)

    def test_confidence_bounds_keeps_a_sub_cent_lower_edge_instead_of_collapsing_to_zero(self):
        # Price $0.004 with a $0.0035 GARCH lower edge: cent rounding turned
        # the edge into exactly $0.00, which downstream code read as "missing".
        low, high = scoring_engine._confidence_bounds(
            target=0.006,
            values=[0.005, 0.007],
            current_price=0.004,
            cap_value=0.012,
            garch_low=0.0035,
            garch_high=0.009,
        )

        self.assertEqual(low, 0.0035)
        self.assertEqual(high, 0.009)

    def test_confidence_bounds_keeps_cent_rounding_for_normal_prices(self):
        low, high = scoring_engine._confidence_bounds(
            target=110.0,
            values=[108.0, 112.0],
            current_price=100.0,
            cap_value=None,
            garch_low=95.123,
            garch_high=120.456,
        )

        self.assertEqual(low, 95.12)
        self.assertEqual(high, 120.46)

    def test_market_cap_padding_keeps_sub_cent_precision(self):
        low, high = scoring_engine._apply_market_cap_confidence_padding(0.0035, 0.009, 0.004, "Micro Cap")

        padding = 0.004 * scoring_engine._MARKET_CAP_CONFIDENCE_PADDING["Micro Cap"]
        self.assertAlmostEqual(low, max(0.0, 0.0035 - padding), places=6)
        self.assertAlmostEqual(high, 0.009 + padding, places=6)
        self.assertNotEqual(high, round(high, 2))


class _FakeLinearRegression:
    def fit(self, x, y):
        return self

    def predict(self, future):
        return np.full(len(future), 4.7)


def _sub_cent_price_history(length: int = 260) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=length)
    close = np.linspace(0.0030, 0.0040, num=length)
    return pd.DataFrame(
        {"Close": close, "High": close * 1.02, "Low": close * 0.98, "Volume": np.full(length, 1_000_000.0)},
        index=index,
    )


class SubCentProjectionIntegrationTests(unittest.TestCase):
    """End-to-end: a sub-cent ticker must produce a coherent upside / band /
    Risk-Adjusted Upside, not the 8.0-style artifact that cent rounding caused."""

    def _projections(self, data: pd.DataFrame) -> dict:
        info = {"exchange": "NASDAQ", "marketCap": 5_000_000, "currentPrice": float(data["Close"].iloc[-1])}
        technical = {"resistance_levels": [0.02], "indicators": {"bb_high": 0.019}}
        with (
            patch("modules.scoring_engine.LIGHTGBM_AVAILABLE", False),
            patch("modules.scoring_engine.load_return_models", return_value={}),
            patch(
                "modules.scoring_engine.run_walk_forward",
                return_value={
                    "arima_rmse": 2.0,
                    "trend_rmse": 1.0,
                    "lightgbm_rmse": None,
                    "n_windows": 0,
                    "arima_windows": 0,
                    "trend_windows": 0,
                    "lightgbm_windows": 0,
                },
            ),
            patch("modules.scoring_engine.LinearRegression", _FakeLinearRegression),
            patch("modules.scoring_engine._garch_confidence_from_returns", return_value=(0.0030, 0.0090)),
            patch("modules.scoring_engine.analyze_technical", return_value=technical),
            patch(
                "modules.scoring_engine.get_macro_regime",
                return_value={"risk_free_rate": 0.045, "bullish_sectors": [], "bearish_sectors": [], "market_regime": "neutral"},
            ),
            patch("modules.scoring_engine.analyze_fundamentals", return_value={"metrics": {}, "fundamental_score": 50}),
        ):
            return scoring_engine._get_price_projections_core("PENNY", info=info, data=data)

    def test_sub_cent_projection_keeps_price_target_and_band_precision(self):
        data = _sub_cent_price_history()
        projections = self._projections(data)

        price = float(data["Close"].iloc[-1])
        self.assertEqual(projections["current_price"], round(price, 6))
        # +200% cap on a $0.004 price is $0.012 -- representable only with
        # sub-cent precision (cent rounding gave $0.01 / +150%).
        self.assertAlmostEqual(projections["short_term_target"], 0.012, places=6)
        self.assertAlmostEqual(projections["short_term_upside"], 200.0, places=1)
        self.assertGreater(projections["short_term_low"], 0.0)
        self.assertGreaterEqual(projections["short_term_high"], projections["short_term_target"])

    def test_sub_cent_row_risk_adjusted_upside_stays_within_documented_range(self):
        data = _sub_cent_price_history()
        projections = self._projections(data)
        analysis = {
            "company": "Penny Co",
            "score": 70,
            "current_price": float(data["Close"].iloc[-1]),
            "projections": projections,
        }

        row = profit_opportunities.profit_row_from_analysis("PENNY", "short_term", analysis)

        self.assertIsNotNone(row)
        self.assertGreater(row["Risk-Adjusted Upside"], 0.0)
        self.assertLessEqual(row["Risk-Adjusted Upside"], 1.0)
        self.assertLessEqual(row["Projected Upside %"], 200.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
