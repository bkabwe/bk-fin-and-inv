from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts import lightgbm_batch_pipeline as pipeline


class LightGBMBatchPipelineTests(unittest.TestCase):
    def test_discover_applies_max_tickers_cap_by_fast_score(self):
        rows = [
            {"ticker": "AAA", "type": "CS", "primary_exchange": "XNAS"},
            {"ticker": "BBB", "type": "CS", "primary_exchange": "XNYS"},
            {"ticker": "CCC", "type": "CS", "primary_exchange": "XNAS"},
        ]
        scores = {"AAA": 45, "BBB": 92, "CCC": 71}

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "discovery.json"
            macro_output = Path(tmpdir) / "macro.joblib"
            args = argparse.Namespace(
                output=str(output),
                macro_output=str(macro_output),
                price_floor=0.1,
                fast_screen_min_score=50,
                fast_screen_margin=15,
                fast_screen_period="1y",
                fast_screen_interval="1d",
                matrix_jobs=4,
                macro_lookback_days=365,
                max_tickers=2,
            )

            with (
                patch.object(pipeline, "get_all_active_ticker_details", return_value=rows),
                patch.object(
                    pipeline,
                    "get_stock_data",
                    return_value=pd.DataFrame({"Close": [10.0, 11.0]}),
                ),
                patch.object(
                    pipeline,
                    "fast_screen_score",
                    side_effect=lambda ticker, **kwargs: (scores[ticker], None),
                ),
                patch.object(pipeline, "get_macro_feature_table", return_value=pd.DataFrame()),
            ):
                rc = pipeline.discover(args)

            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([item["ticker"] for item in payload["tickers"]], ["BBB", "CCC"])
            self.assertEqual(payload["skipped"]["AAA"]["reason"], "max_tickers_cap")

    def test_discover_cap_respects_max_with_duplicate_tickers(self):
        rows = [
            {"ticker": "AAA", "type": "CS", "primary_exchange": "XNAS"},
            {"ticker": "AAA", "type": "CS", "primary_exchange": "XNYS"},
            {"ticker": "BBB", "type": "CS", "primary_exchange": "XNAS"},
        ]
        scores = {"AAA": 90, "BBB": 40}

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "discovery.json"
            macro_output = Path(tmpdir) / "macro.joblib"
            args = argparse.Namespace(
                output=str(output),
                macro_output=str(macro_output),
                price_floor=0.1,
                fast_screen_min_score=50,
                fast_screen_margin=15,
                fast_screen_period="1y",
                fast_screen_interval="1d",
                matrix_jobs=4,
                macro_lookback_days=365,
                max_tickers=1,
            )

            with (
                patch.object(pipeline, "get_all_active_ticker_details", return_value=rows),
                patch.object(
                    pipeline,
                    "get_stock_data",
                    return_value=pd.DataFrame({"Close": [10.0, 11.0]}),
                ),
                patch.object(
                    pipeline,
                    "fast_screen_score",
                    side_effect=lambda ticker, **kwargs: (scores[ticker], None),
                ),
                patch.object(pipeline, "get_macro_feature_table", return_value=pd.DataFrame()),
            ):
                rc = pipeline.discover(args)

            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["tickers"]), 1)
            self.assertEqual(payload["tickers"][0]["ticker"], "AAA")
            self.assertEqual(payload["tickers"][0]["exchange"], "XNAS")
            capped = [entry for entry in payload["skipped"].values() if entry.get("reason") == "max_tickers_cap"]
            self.assertEqual(len(capped), 2)


if __name__ == "__main__":
    unittest.main()
