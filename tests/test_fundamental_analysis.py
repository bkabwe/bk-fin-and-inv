from __future__ import annotations

import unittest

from modules.fundamental_analysis import analyze_fundamentals


class DcfEstimateUnitsTests(unittest.TestCase):
    """Regression tests for the Graham-formula DCF unit bug.

    ``risk_free_rate`` is a decimal fraction everywhere in this codebase
    (e.g. 0.045 for 4.5%), but Graham's revised formula expects the
    denominator ``Y`` to be a percentage *number* (e.g. 4.4). Dividing by the
    raw decimal inflates the estimate ~100x.
    """

    def test_dcf_estimate_is_not_inflated_by_decimal_risk_free_rate(self):
        info = {
            "trailingEps": 5.0,
            "forwardEps": 5.5,
            "trailingPE": 18.0,
            "forwardPE": 16.0,
            "earningsGrowth": 0.10,
            "debtToEquity": 50.0,
            "returnOnEquity": 0.18,
            "sector": "Technology",
            "marketCap": 50_000_000_000,
            "currentPrice": 100.0,
        }

        result = analyze_fundamentals(info, current_price=100.0, risk_free_rate=0.045)
        dcf_estimate = result["metrics"]["dcf_estimate"]

        # eps_for_dcf(5.5) * (8.5 + 2*10) * (4.4 / 4.5) ~= 154.7
        expected = 5.5 * (8.5 + 2 * 10.0) * (4.4 / 4.5)
        self.assertAlmostEqual(dcf_estimate, expected, places=2)

        # Before the fix this would have been ~100x larger (dividing by 0.045
        # instead of 4.5), i.e. > 10,000. Guard against the regression coming back.
        self.assertLess(dcf_estimate, 1000.0)

    def test_dcf_estimate_scales_with_risk_free_rate_as_percentage(self):
        info = {
            "trailingEps": 5.0,
            "forwardEps": 5.0,
            "earningsGrowth": 0.05,
        }

        low_rate = analyze_fundamentals(info, current_price=100.0, risk_free_rate=0.02)
        high_rate = analyze_fundamentals(info, current_price=100.0, risk_free_rate=0.08)

        # Higher discount rate should produce a *lower* intrinsic value estimate.
        self.assertGreater(
            low_rate["metrics"]["dcf_estimate"],
            high_rate["metrics"]["dcf_estimate"],
        )


if __name__ == "__main__":
    unittest.main()
