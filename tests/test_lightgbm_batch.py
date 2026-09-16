from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from modules import lightgbm_batch


class LightGBMBatchTests(unittest.TestCase):
    def test_get_batch_models_for_ticker_returns_requested_horizons(self):
        batch = lightgbm_batch.create_return_model_batch(
            batch_id="20260914T000000Z",
            models={"aapl": {30: object(), 180: object(), 720: object()}},
        )

        models = lightgbm_batch.get_batch_models_for_ticker(batch, "AAPL")

        self.assertEqual(set(models.keys()), {30, 180})

    def test_merge_current_and_previous_batches_carries_forward_recent_tickers(self):
        current_model = object()
        previous_model = object()
        current_batch = lightgbm_batch.create_return_model_batch(
            batch_id="current",
            models={"AAPL": {30: current_model, 180: current_model}},
            training_metadata={"AAPL": {"row_counts": {"30": 60, "180": 60}, "exchange": "XNAS", "last_trained": "2026-09-14T00:00:00Z"}},
            created_at="2026-09-14T00:00:00Z",
        )
        previous_batch = lightgbm_batch.create_return_model_batch(
            batch_id="previous",
            models={"MSFT": {30: previous_model, 180: previous_model}},
            training_metadata={"MSFT": {"row_counts": {"30": 55, "180": 55}, "exchange": "XNAS", "last_trained": "2026-09-07T00:00:00Z"}},
            created_at="2026-09-07T00:00:00Z",
        )
        previous_manifest = {
            "tickers": {
                "MSFT": {"last_trained": "2026-09-07T00:00:00Z", "cycles_since_training": 1, "exchange": "XNAS"}
            }
        }

        models, metadata, carried_forward, dropped = lightgbm_batch.merge_current_and_previous_batches(
            current_batch,
            previous_manifest=previous_manifest,
            previous_batch=previous_batch,
            max_stale_cycles=4,
        )

        self.assertEqual(set(models.keys()), {"AAPL", "MSFT"})
        self.assertEqual(carried_forward, ["MSFT"])
        self.assertEqual(dropped, [])
        self.assertEqual(metadata["MSFT"]["cycles_since_training"], 2)

    def test_merge_current_and_previous_batches_drops_stale_tickers(self):
        previous_model = object()
        current_batch = lightgbm_batch.create_return_model_batch(batch_id="current", models={})
        previous_batch = lightgbm_batch.create_return_model_batch(
            batch_id="previous",
            models={"MSFT": {30: previous_model, 180: previous_model}},
            training_metadata={"MSFT": {"row_counts": {"30": 55, "180": 55}, "exchange": "XNAS", "last_trained": "2026-09-07T00:00:00Z"}},
        )
        previous_manifest = {"tickers": {"MSFT": {"cycles_since_training": 4, "last_trained": "2026-09-07T00:00:00Z"}}}

        models, metadata, carried_forward, dropped = lightgbm_batch.merge_current_and_previous_batches(
            current_batch,
            previous_manifest=previous_manifest,
            previous_batch=previous_batch,
            max_stale_cycles=4,
        )

        self.assertEqual(models, {})
        self.assertEqual(metadata, {})
        self.assertEqual(carried_forward, [])
        self.assertEqual(dropped, ["MSFT"])

    def test_build_live_manifest_points_to_release_asset(self):
        manifest = lightgbm_batch.build_live_manifest(
            repository="bkabwe/bk-fin-and-inv",
            batch_id="20260914T000000Z",
            batch_release_tag="lightgbm-batch-20260914T000000Z",
            batch_asset_name="lightgbm_return_model_batch.joblib",
            created_at="2026-09-14T00:00:00Z",
            training_metadata={"AAPL": {"exchange": "XNAS", "last_trained": "2026-09-14T00:00:00Z", "cycles_since_training": 0, "trained_horizons": [30, 180], "row_counts": {"30": 60, "180": 60}}},
        )

        self.assertEqual(manifest["latest_batch"]["asset_name"], "lightgbm_return_model_batch.joblib")
        self.assertIn("/releases/download/lightgbm-batch-20260914T000000Z/", manifest["latest_batch"]["asset_url"])
        self.assertEqual(manifest["latest_batch"]["asset_count"], 1)
        self.assertEqual(manifest["latest_batch"]["asset_urls"], [manifest["latest_batch"]["asset_url"]])
        self.assertEqual(manifest["tickers"]["AAPL"]["cycles_since_training"], 0)

    def test_build_live_manifest_splits_multi_part_batch_asset_urls(self):
        manifest = lightgbm_batch.build_live_manifest(
            repository="bkabwe/bk-fin-and-inv",
            batch_id="20260914T000000Z",
            batch_release_tag="lightgbm-batch-20260914T000000Z",
            batch_asset_name="lightgbm_return_model_batch.joblib",
            created_at="2026-09-14T00:00:00Z",
            training_metadata={},
            asset_part_count=3,
        )

        latest_batch = manifest["latest_batch"]
        self.assertEqual(latest_batch["asset_count"], 3)
        self.assertEqual(len(latest_batch["asset_urls"]), 3)
        self.assertTrue(latest_batch["asset_urls"][0].endswith("lightgbm_return_model_batch.joblib.part001of003"))
        self.assertTrue(latest_batch["asset_urls"][2].endswith("lightgbm_return_model_batch.joblib.part003of003"))
        self.assertEqual(latest_batch["asset_url"], latest_batch["asset_urls"][0])

    def test_batch_asset_urls_from_manifest_prefers_multi_part_list(self):
        manifest = {"latest_batch": {"asset_url": "https://example/one", "asset_urls": ["https://example/one", "https://example/two"]}}

        self.assertEqual(
            lightgbm_batch.batch_asset_urls_from_manifest(manifest),
            ["https://example/one", "https://example/two"],
        )

    def test_batch_asset_urls_from_manifest_falls_back_to_single_asset_url(self):
        manifest = {"latest_batch": {"asset_url": "https://example/one"}}

        self.assertEqual(lightgbm_batch.batch_asset_urls_from_manifest(manifest), ["https://example/one"])

    def test_batch_asset_part_count_uses_ceil_division(self):
        self.assertEqual(lightgbm_batch.batch_asset_part_count(0, chunk_size=100), 1)
        self.assertEqual(lightgbm_batch.batch_asset_part_count(100, chunk_size=100), 1)
        self.assertEqual(lightgbm_batch.batch_asset_part_count(101, chunk_size=100), 2)
        self.assertEqual(lightgbm_batch.batch_asset_part_count(250, chunk_size=100), 3)

    def test_load_return_model_batch_from_urls_concatenates_parts(self):
        batch = lightgbm_batch.create_return_model_batch(batch_id="current", models={"AAPL": {30: {"stub": True}}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/batch.joblib"
            joblib.dump(batch, path)
            raw_bytes = Path(path).read_bytes()
        midpoint = len(raw_bytes) // 2
        part_one, part_two = raw_bytes[:midpoint], raw_bytes[midpoint:]

        class _FakeStreamedResponse:
            def __init__(self, payload: bytes):
                self._payload = payload

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size):
                for start in range(0, len(self._payload), chunk_size):
                    yield self._payload[start : start + chunk_size]

        with patch("modules.lightgbm_batch.requests.get", side_effect=[_FakeStreamedResponse(part_one), _FakeStreamedResponse(part_two)]) as get_mock:
            loaded = lightgbm_batch.load_return_model_batch_from_urls(["https://example/one", "https://example/two"])

        self.assertEqual(get_mock.call_count, 2)
        self.assertEqual(loaded["batch_id"], "current")
        self.assertIn("AAPL", loaded["models"])

    def test_default_live_manifest_url_uses_repository_default(self):
        url = lightgbm_batch.default_live_manifest_url()

        self.assertEqual(url, "https://github.com/bkabwe/bk-fin-and-inv/releases/download/lightgbm-model-live/latest.json")

    def test_load_return_model_batch_round_trips_joblib_file(self):
        batch = lightgbm_batch.create_return_model_batch(batch_id="current", models={"AAPL": {30: {"stub": True}}})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/batch.joblib"
            joblib.dump(batch, path)
            loaded = lightgbm_batch.load_return_model_batch(path)

        self.assertEqual(loaded["batch_id"], "current")
        self.assertIn("AAPL", loaded["models"])


if __name__ == "__main__":
    unittest.main()
