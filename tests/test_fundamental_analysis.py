from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.fundamental_analysis import _load_refreshed_sector_pe, analyze_fundamentals, get_sector_benchmark_pe


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


class ComparableValuationEnsembleTests(unittest.TestCase):
    """Tests for the sector-P/E comparables valuation added alongside Graham DCF."""

    def test_comparable_value_estimate_uses_eps_times_sector_pe(self):
        info = {
            "trailingEps": 4.0,
            "forwardEps": 4.5,
            "earningsGrowth": 0.08,
            "sector": "Technology",
        }

        result = analyze_fundamentals(info, current_price=100.0, risk_free_rate=0.045)
        metrics = result["metrics"]

        # eps_for_dcf prefers forwardEps (4.5); Technology's default benchmark P/E is 28.
        self.assertAlmostEqual(metrics["comparable_value_estimate"], 4.5 * 28, places=2)

    def test_valuation_estimate_averages_dcf_and_comparables_when_both_available(self):
        info = {
            "trailingEps": 4.0,
            "forwardEps": 4.5,
            "earningsGrowth": 0.08,
            "sector": "Technology",
        }

        result = analyze_fundamentals(info, current_price=100.0, risk_free_rate=0.045)
        metrics = result["metrics"]

        self.assertIsNotNone(metrics["dcf_estimate"])
        self.assertIsNotNone(metrics["comparable_value_estimate"])
        expected = (metrics["dcf_estimate"] + metrics["comparable_value_estimate"]) / 2
        self.assertAlmostEqual(metrics["valuation_estimate"], expected, places=2)

    def test_valuation_estimate_falls_back_to_whichever_input_is_available(self):
        # No growth data => Graham DCF is unavailable, only comparables should populate.
        info = {"trailingEps": 4.0, "sector": "Technology"}

        result = analyze_fundamentals(info, current_price=100.0, risk_free_rate=0.045)
        metrics = result["metrics"]

        self.assertIsNone(metrics["dcf_estimate"])
        self.assertIsNotNone(metrics["comparable_value_estimate"])
        self.assertAlmostEqual(metrics["valuation_estimate"], metrics["comparable_value_estimate"], places=2)

    def test_valuation_estimate_none_when_no_eps_data(self):
        result = analyze_fundamentals({"sector": "Technology"}, current_price=100.0, risk_free_rate=0.045)
        metrics = result["metrics"]

        self.assertIsNone(metrics["dcf_estimate"])
        self.assertIsNone(metrics["comparable_value_estimate"])
        self.assertIsNone(metrics["valuation_estimate"])


class SectorBenchmarkPeRefreshTests(unittest.TestCase):
    """Tests for the sector P/E refresh mechanism (scripts/refresh_sector_pe.py output)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.refresh_file = Path(self._tmp.name) / "sector_benchmark_pe.json"
        patcher = patch(
            "modules.fundamental_analysis.SECTOR_BENCHMARK_PE_FILE",
            self.refresh_file,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(_load_refreshed_sector_pe.cache_clear)

    def test_falls_back_to_hardcoded_default_when_no_refresh_file(self):
        self.assertFalse(self.refresh_file.exists())
        self.assertAlmostEqual(get_sector_benchmark_pe("Technology"), 28.0)

    def test_falls_back_to_default_arg_for_unrecognized_sector(self):
        self.assertAlmostEqual(get_sector_benchmark_pe("Not A Real Sector", default=17.5), 17.5)

    def test_prefers_refreshed_value_over_hardcoded_default(self):
        self.refresh_file.write_text(
            json.dumps({"generated_at": "2026-01-01T00:00:00+00:00", "sectors": {"Technology": 33.4}}),
            encoding="utf-8",
        )
        self.assertAlmostEqual(get_sector_benchmark_pe("Technology"), 33.4)

    def test_refreshed_file_only_overrides_sectors_it_contains(self):
        self.refresh_file.write_text(
            json.dumps({"generated_at": "2026-01-01T00:00:00+00:00", "sectors": {"Technology": 33.4}}),
            encoding="utf-8",
        )
        # Healthcare isn't in the refreshed file, so the hardcoded default (22) still applies.
        self.assertAlmostEqual(get_sector_benchmark_pe("Healthcare"), 22.0)

    def test_malformed_refresh_file_falls_back_gracefully(self):
        self.refresh_file.write_text("not valid json", encoding="utf-8")
        self.assertAlmostEqual(get_sector_benchmark_pe("Technology"), 28.0)


if __name__ == "__main__":
    unittest.main()
