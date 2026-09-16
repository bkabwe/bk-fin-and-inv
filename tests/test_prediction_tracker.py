from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from modules import prediction_tracker


class PredictionTrackerCoreTests(unittest.TestCase):
    """Unit tests for prediction_tracker's persistence, dedupe, and resolution logic."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_dir = Path(self._tmp.name)
        self.predictions_file = tmp_dir / "predictions.json"
        patcher_dir = patch.object(prediction_tracker, "DATA_DIR", tmp_dir)
        patcher_file = patch.object(prediction_tracker, "PREDICTIONS_FILE", self.predictions_file)
        patcher_dir.start()
        patcher_file.start()
        self.addCleanup(patcher_dir.stop)
        self.addCleanup(patcher_file.stop)

    def _record(self, **overrides):
        defaults = dict(
            ticker="AAPL",
            company="Apple Inc.",
            horizon="short_term",
            current_price=100.0,
            target_price=110.0,
            target_low=105.0,
            target_high=115.0,
            projected_upside_pct=10.0,
            score=75,
            data_quality="High",
            models_used="ARIMA, Trend",
            source="profit_opportunities",
        )
        defaults.update(overrides)
        return prediction_tracker.record_prediction(**defaults)

    # -- persistence / atomic write -----------------------------------

    def test_record_prediction_persists_to_disk_and_round_trips(self):
        pid = self._record()
        self.assertIsNotNone(pid)
        self.assertTrue(self.predictions_file.exists())

        records = prediction_tracker.get_all_predictions()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], pid)
        self.assertEqual(records[0]["ticker"], "AAPL")
        self.assertEqual(records[0]["status"], "pending")

    # -- dedupe ----------------------------------------------------------

    def test_record_prediction_dedupes_same_ticker_horizon_same_day(self):
        first = self._record()
        second = self._record()

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(prediction_tracker.get_all_predictions()), 1)

    def test_record_prediction_allows_different_horizon_same_day(self):
        first = self._record(horizon="short_term")
        second = self._record(horizon="medium_term")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(prediction_tracker.get_all_predictions()), 2)

    def test_record_prediction_allows_new_prediction_after_prior_one_resolved(self):
        first = self._record()
        records = prediction_tracker.get_all_predictions()
        records[0]["status"] = "resolved"
        prediction_tracker._save_all(records)

        second = self._record()

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)

    # -- resolution transitions ------------------------------------------

    def test_resolve_pending_predictions_marks_resolved_when_target_date_passed(self):
        pid = self._record()
        records = prediction_tracker.get_all_predictions()
        records[0]["target_date"] = "2020-01-10"
        records[0]["scan_date"] = "2020-01-01"
        prediction_tracker._save_all(records)

        frame = pd.DataFrame(
            {"Close": [108.0, 112.0]},
            index=pd.to_datetime(["2020-01-09", "2020-01-11"]),
        )
        with patch("modules.data_fetcher.get_stock_data", return_value=frame):
            summary = prediction_tracker.resolve_pending_predictions()

        self.assertEqual(summary["resolved"], 1)
        record = prediction_tracker.get_all_predictions()[0]
        self.assertEqual(record["id"], pid)
        self.assertEqual(record["status"], "resolved")
        self.assertEqual(record["actual_price_at_target_date"], 112.0)
        self.assertAlmostEqual(record["actual_return_pct"], 12.0, places=4)
        self.assertTrue(record["hit_target"])
        self.assertIsNotNone(record["resolved_at"])

    def test_resolve_pending_predictions_leaves_future_target_date_pending(self):
        self._record()
        records = prediction_tracker.get_all_predictions()
        records[0]["target_date"] = "2999-01-01"
        prediction_tracker._save_all(records)

        summary = prediction_tracker.resolve_pending_predictions()

        self.assertEqual(summary["still_pending"], 1)
        self.assertEqual(summary["resolved"], 0)
        self.assertEqual(prediction_tracker.get_all_predictions()[0]["status"], "pending")

    def test_resolve_pending_predictions_marks_no_data_after_week_of_failures(self):
        self._record()
        records = prediction_tracker.get_all_predictions()
        records[0]["target_date"] = "2020-01-01"
        prediction_tracker._save_all(records)

        with patch("modules.data_fetcher.get_stock_data", return_value=pd.DataFrame()):
            summary = prediction_tracker.resolve_pending_predictions()

        self.assertEqual(summary["no_data"], 1)
        self.assertEqual(prediction_tracker.get_all_predictions()[0]["status"], "unresolved_no_data")

    def test_resolve_pending_predictions_keeps_recent_target_pending_on_transient_failure(self):
        from datetime import date, timedelta

        self._record()
        records = prediction_tracker.get_all_predictions()
        records[0]["target_date"] = (date.today() - timedelta(days=2)).isoformat()
        prediction_tracker._save_all(records)

        with patch("modules.data_fetcher.get_stock_data", return_value=pd.DataFrame()):
            summary = prediction_tracker.resolve_pending_predictions()

        self.assertEqual(summary["still_pending"], 1)
        self.assertEqual(summary["no_data"], 0)
        self.assertEqual(prediction_tracker.get_all_predictions()[0]["status"], "pending")

    # -- score-vs-outcome validation --------------------------------------

    def test_compute_score_validation_stats_returns_none_below_min_samples(self):
        for i in range(3):
            self._record(ticker=f"T{i}")
        records = prediction_tracker.get_all_predictions()
        for i, rec in enumerate(records):
            rec["status"] = "resolved"
            rec["actual_return_pct"] = float(i)
        prediction_tracker._save_all(records)

        stats = prediction_tracker.compute_score_validation_stats(min_samples=5)

        self.assertEqual(stats["overall"]["count"], 3)
        self.assertIsNone(stats["overall"]["information_coefficient"])

    def test_compute_score_validation_stats_detects_positive_correlation(self):
        # Higher score -> higher realized return, and top-scored hit their target.
        samples = [
            {"score": 90, "actual_return_pct": 20.0, "hit_target": True},
            {"score": 80, "actual_return_pct": 15.0, "hit_target": True},
            {"score": 70, "actual_return_pct": 10.0, "hit_target": True},
            {"score": 40, "actual_return_pct": -5.0, "hit_target": False},
            {"score": 30, "actual_return_pct": -8.0, "hit_target": False},
            {"score": 20, "actual_return_pct": -10.0, "hit_target": False},
        ]
        for idx, sample in enumerate(samples):
            self._record(ticker=f"T{idx}", score=sample["score"])
        records = prediction_tracker.get_all_predictions()
        for rec, sample in zip(records, samples, strict=False):
            rec["status"] = "resolved"
            rec["actual_return_pct"] = sample["actual_return_pct"]
            rec["hit_target"] = sample["hit_target"]
        prediction_tracker._save_all(records)

        stats = prediction_tracker.compute_score_validation_stats(min_samples=5)

        overall = stats["overall"]
        self.assertEqual(overall["count"], 6)
        self.assertGreater(overall["information_coefficient"], 0.9)
        self.assertEqual(overall["precision_at_top_third"], 100.0)
        self.assertEqual(overall["precision_at_bottom_third"], 0.0)


if __name__ == "__main__":
    unittest.main()
