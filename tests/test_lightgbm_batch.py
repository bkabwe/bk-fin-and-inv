from __future__ import annotations

import tempfile
import unittest

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
        self.assertEqual(manifest["tickers"]["AAPL"]["cycles_since_training"], 0)

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
