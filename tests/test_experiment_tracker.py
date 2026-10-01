from __future__ import annotations

import shutil
import sqlite3
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from modules import experiment_tracker

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / ".test_artifacts"


def _make_scratch_dir(prefix: str) -> Path:
    path = TEST_ARTIFACTS_DIR / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class ExperimentTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_dir = _make_scratch_dir("experiment_tracker")
        self.addCleanup(lambda: shutil.rmtree(self.scratch_dir, ignore_errors=True))

        self.data_dir = self.scratch_dir / "data"
        self.db_path = self.data_dir / "experiments.db"
        patchers = [
            patch.object(experiment_tracker, "DATA_DIR", self.data_dir),
            patch.object(experiment_tracker, "EXPERIMENTS_DB", self.db_path),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _record(self, **overrides: object) -> int:
        payload = {
            "run_name": "walk_forward",
            "horizon": 30,
            "git_commit": "abc123",
            "config": {"train_len": 60, "test_len": 30},
            "metrics": {
                "models": {
                    "lightgbm": {"mean_rmse": 1.2, "median_rmse": 1.1, "windows_evaluated": 5},
                    "trend": {"mean_rmse": 1.5, "windows_evaluated": 5},
                }
            },
            "sample_tickers": ["AAPL", "MSFT"],
        }
        payload.update(overrides)
        return experiment_tracker.record_experiment_run(**payload)

    def test_record_and_list_round_trip(self):
        row_id = self._record(
            config={"train_len": 90, "test_len": 30, "evaluate_lightgbm": True},
            metrics={"models": {"lightgbm": {"mean_rmse": 0.95, "windows_evaluated": 8}}},
            sample_tickers=["AAPL", "NVDA"],
        )

        runs = experiment_tracker.list_experiment_runs()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["id"], row_id)
        self.assertEqual(runs[0]["run_name"], "walk_forward")
        self.assertEqual(runs[0]["horizon"], 30)
        self.assertEqual(runs[0]["git_commit"], "abc123")
        self.assertEqual(runs[0]["config"], {"train_len": 90, "test_len": 30, "evaluate_lightgbm": True})
        self.assertEqual(runs[0]["metrics"], {"models": {"lightgbm": {"mean_rmse": 0.95, "windows_evaluated": 8}}})
        self.assertEqual(runs[0]["sample_tickers"], ["AAPL", "NVDA"])

    def test_list_filters_by_run_name_and_horizon(self):
        self._record(run_name="walk_forward", horizon=30)
        self._record(run_name="walk_forward", horizon=180)
        self._record(run_name="feature_ablation", horizon=30)

        self.assertEqual(len(experiment_tracker.list_experiment_runs(run_name="walk_forward")), 2)
        self.assertEqual(len(experiment_tracker.list_experiment_runs(horizon=30)), 2)
        self.assertEqual(
            [run["run_name"] for run in experiment_tracker.list_experiment_runs(run_name="walk_forward", horizon=180)],
            ["walk_forward"],
        )
        self.assertEqual(
            [run["horizon"] for run in experiment_tracker.list_experiment_runs(run_name="walk_forward", horizon=180)],
            [180],
        )

    def test_list_orders_most_recent_first(self):
        first_id = self._record(git_commit="first")
        second_id = self._record(git_commit="second")

        runs = experiment_tracker.list_experiment_runs(run_name="walk_forward", horizon=30)

        self.assertEqual([run["id"] for run in runs[:2]], [second_id, first_id])
        self.assertEqual([run["git_commit"] for run in runs[:2]], ["second", "first"])

    def test_compare_latest_two_runs_returns_none_with_fewer_than_two_runs(self):
        self.assertIsNone(experiment_tracker.compare_latest_two_runs("walk_forward", 30))

        self._record(run_name="walk_forward", horizon=30)

        self.assertIsNone(experiment_tracker.compare_latest_two_runs("walk_forward", 30))

    def test_compare_latest_two_runs_returns_metric_deltas(self):
        first_id = self._record(
            metrics={
                "models": {
                    "lightgbm": {"mean_rmse": 1.2, "median_rmse": 1.1, "windows_evaluated": 5},
                    "trend": {"mean_rmse": 1.5, "windows_evaluated": 5},
                    "naive": {"mean_rmse": 1.9},
                },
                "total_tickers": 12,
            }
        )
        second_id = self._record(
            metrics={
                "models": {
                    "lightgbm": {"mean_rmse": 1.0, "median_rmse": 1.05, "windows_evaluated": 6},
                    "trend": {"mean_rmse": 1.4, "windows_evaluated": 6},
                    "arima": {"mean_rmse": 1.7},
                },
                "total_tickers": 12,
            }
        )

        comparison = experiment_tracker.compare_latest_two_runs("walk_forward", 30)

        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertEqual(comparison["previous"]["id"], first_id)
        self.assertEqual(comparison["latest"]["id"], second_id)
        self.assertAlmostEqual(comparison["metric_deltas"]["models.lightgbm.mean_rmse"], -0.2)
        self.assertAlmostEqual(comparison["metric_deltas"]["models.lightgbm.median_rmse"], -0.05)
        self.assertAlmostEqual(comparison["metric_deltas"]["models.lightgbm.windows_evaluated"], 1.0)
        self.assertAlmostEqual(comparison["metric_deltas"]["models.trend.mean_rmse"], -0.1)
        self.assertAlmostEqual(comparison["metric_deltas"]["models.trend.windows_evaluated"], 1.0)
        self.assertNotIn("models.naive.mean_rmse", comparison["metric_deltas"])
        self.assertNotIn("models.arima.mean_rmse", comparison["metric_deltas"])
        self.assertNotIn("total_tickers", comparison["metric_deltas"])

    def test_malformed_rows_are_skipped_with_warning(self):
        first_id = self._record(git_commit="first")
        second_id = self._record(git_commit="second")

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO experiment_runs (
                    run_name,
                    horizon,
                    git_commit,
                    recorded_at,
                    config_json,
                    metrics_json,
                    sample_tickers_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "walk_forward",
                    30,
                    "broken",
                    "2026-10-01T00:00:00+00:00",
                    "{not valid json",
                    "{}",
                    None,
                ),
            )
            connection.commit()

        with self.assertLogs(experiment_tracker.logger.name, level="WARNING") as captured:
            runs = experiment_tracker.list_experiment_runs(run_name="walk_forward", horizon=30, limit=10)

        self.assertEqual([run["id"] for run in runs], [second_id, first_id])
        self.assertTrue(any("Skipping malformed experiment run row" in message for message in captured.output))


if __name__ == "__main__":
    unittest.main()
