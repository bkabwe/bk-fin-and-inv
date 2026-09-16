"""FastAPI `TestClient` tests for the simplest, dependency-free API routes.

These cover routes backed only by local JSON files (notifications,
track-record) plus the `/health` check, redirecting each module's data file
to a temp directory the same way `tests/test_prediction_tracker.py` does, so
no network or Streamlit runtime is required.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from modules import notifications, prediction_tracker


class ApiRoutesTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_dir = Path(self._tmp.name)

        self.notifications_file = tmp_dir / "notifications.json"
        notifications_patchers = [
            patch.object(notifications, "DATA_DIR", tmp_dir),
            patch.object(notifications, "NOTIFICATIONS_FILE", self.notifications_file),
        ]

        self.predictions_file = tmp_dir / "predictions.json"
        prediction_tracker_patchers = [
            patch.object(prediction_tracker, "DATA_DIR", tmp_dir),
            patch.object(prediction_tracker, "PREDICTIONS_FILE", self.predictions_file),
        ]

        for patcher in [*notifications_patchers, *prediction_tracker_patchers]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_health_check(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_notifications_round_trip_and_mark_read(self):
        empty = self.client.get("/api/notifications")
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json(), [])

        notifications.add_notification("Alert", "AAPL crossed target", "AAPL", type="price")

        listed = self.client.get("/api/notifications")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()), 1)
        self.assertFalse(listed.json()[0]["read"])

        read_response = self.client.post("/api/notifications/read")
        self.assertEqual(read_response.status_code, 200)
        self.assertEqual(read_response.json(), {"status": "ok"})

        after_read = self.client.get("/api/notifications")
        self.assertTrue(after_read.json()[0]["read"])

    def test_track_record_summary_with_no_predictions(self):
        response = self.client.get("/api/track-record/summary")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total_predictions"], 0)
        self.assertEqual(body["total_resolved"], 0)
        self.assertIsNone(body["overall_hit_rate_target"])

    def test_track_record_predictions_filters_by_status(self):
        prediction_tracker.record_prediction(
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
            source="stock_analysis",
        )

        all_records = self.client.get("/api/track-record/predictions")
        self.assertEqual(all_records.status_code, 200)
        self.assertEqual(len(all_records.json()), 1)

        pending_only = self.client.get("/api/track-record/predictions", params={"status": "pending"})
        self.assertEqual(len(pending_only.json()), 1)

        resolved_only = self.client.get("/api/track-record/predictions", params={"status": "resolved"})
        self.assertEqual(resolved_only.json(), [])


if __name__ == "__main__":
    unittest.main()
