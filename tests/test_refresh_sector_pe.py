from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import refresh_sector_pe


class CollectSectorPePairsTests(unittest.TestCase):
    """collect_sector_pe_pairs fetches (sector, trailing_pe) per ticker, tolerating failures."""

    def test_collects_sector_and_pe_for_each_ticker(self):
        infos = {
            "AAA": {"sector": "Technology", "trailingPE": 25.0},
            "BBB": {"sector": "Financial Services", "trailingPE": "18.5"},
        }

        with patch("scripts.refresh_sector_pe.get_stock_info", side_effect=lambda t: infos.get(t, {})):
            pairs = refresh_sector_pe.collect_sector_pe_pairs(["AAA", "BBB"], max_workers=2)

        self.assertIn(("Technology", 25.0), pairs)
        # "Financial Services" is a Yahoo-style alias normalized to "Financials".
        self.assertIn(("Financials", 18.5), pairs)

    def test_tolerates_per_ticker_fetch_failures(self):
        def _raise(ticker):
            raise RuntimeError("boom")

        with patch("scripts.refresh_sector_pe.get_stock_info", side_effect=_raise):
            pairs = refresh_sector_pe.collect_sector_pe_pairs(["AAA"], max_workers=1)

        self.assertEqual(pairs, [(None, None)])

    def test_handles_missing_or_unparseable_pe(self):
        infos = {
            "AAA": {"sector": "Technology", "trailingPE": None},
            "BBB": {"sector": "Technology", "trailingPE": "not-a-number"},
            "CCC": {},
        }

        with patch("scripts.refresh_sector_pe.get_stock_info", side_effect=lambda t: infos.get(t, {})):
            pairs = refresh_sector_pe.collect_sector_pe_pairs(["AAA", "BBB", "CCC"], max_workers=3)

        for _sector, pe in pairs:
            self.assertIsNone(pe)


class ComputeSectorMediansTests(unittest.TestCase):
    def test_computes_median_pe_per_sector(self):
        pairs = [("Technology", pe) for pe in (20.0, 25.0, 30.0, 22.0, 28.0)]
        result = refresh_sector_pe.compute_sector_medians(pairs, min_samples=5)
        self.assertEqual(result["Technology"]["median_pe"], 25.0)
        self.assertEqual(result["Technology"]["sample_count"], 5)

    def test_omits_sectors_below_min_samples(self):
        pairs = [("Healthcare", 20.0), ("Healthcare", 22.0)]
        result = refresh_sector_pe.compute_sector_medians(pairs, min_samples=5)
        self.assertNotIn("Healthcare", result)

    def test_filters_out_of_range_and_missing_values(self):
        pairs = [
            ("Technology", 25.0),
            ("Technology", None),
            (None, 30.0),
            ("Technology", -5.0),
            ("Technology", 500.0),
            ("Technology", 26.0),
            ("Technology", 27.0),
            ("Technology", 28.0),
        ]
        result = refresh_sector_pe.compute_sector_medians(pairs, min_samples=4)
        # Only the 4 in-range Technology values (25, 26, 27, 28) should count.
        self.assertEqual(result["Technology"]["sample_count"], 4)
        self.assertEqual(result["Technology"]["median_pe"], 26.5)


class RefreshSectorPeMainTests(unittest.TestCase):
    """End-to-end test of main() writing the refreshed benchmarks JSON."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_file = Path(self._tmp.name) / "sector_benchmark_pe.json"

    def test_main_writes_sector_medians_to_output_file(self):
        tickers = [f"T{i}" for i in range(6)]
        infos = {t: {"sector": "Technology", "trailingPE": 20.0 + i} for i, t in enumerate(tickers)}

        with (
            patch("scripts.refresh_sector_pe.get_sp500_tickers", return_value=tickers),
            patch("scripts.refresh_sector_pe.get_stock_info", side_effect=lambda t: infos.get(t, {})),
            patch("sys.argv", ["refresh_sector_pe.py", "--output", str(self.output_file), "--min-samples", "5"]),
        ):
            refresh_sector_pe.main()

        self.assertTrue(self.output_file.exists())
        payload = json.loads(self.output_file.read_text(encoding="utf-8"))
        self.assertIn("Technology", payload["sectors"])
        self.assertEqual(payload["source_ticker_count"], len(tickers))
        self.assertIn("generated_at", payload)

    def test_main_leaves_existing_file_untouched_when_no_sector_qualifies(self):
        self.output_file.write_text(json.dumps({"sectors": {"Technology": 99.0}}), encoding="utf-8")

        with (
            patch("scripts.refresh_sector_pe.get_sp500_tickers", return_value=["AAA"]),
            patch(
                "scripts.refresh_sector_pe.get_stock_info",
                return_value={"sector": "Technology", "trailingPE": 20.0},
            ),
            patch("sys.argv", ["refresh_sector_pe.py", "--output", str(self.output_file), "--min-samples", "5"]),
        ):
            refresh_sector_pe.main()

        # Only one valid sample (below default min_samples=5), so nothing should be overwritten.
        payload = json.loads(self.output_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["sectors"]["Technology"], 99.0)


if __name__ == "__main__":
    unittest.main()
