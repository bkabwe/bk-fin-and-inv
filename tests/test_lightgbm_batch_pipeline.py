from __future__ import annotations

import argparse
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import joblib
import pandas as pd
import requests

from scripts import lightgbm_batch_pipeline as pipeline
from scripts.lightgbm_batch_pipeline import GitHubReleaseClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | list | Exception | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = {} if payload is None else payload
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class GitHubReleaseClientUploadTests(unittest.TestCase):
    def test_upload_asset_streams_file_handle_instead_of_reading_all_bytes(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        observed: dict[str, bool] = {}

        def _post_side_effect(*args, **kwargs):
            data = kwargs["data"]
            observed["is_open_during_post"] = not data.closed
            observed["is_file_like"] = hasattr(data, "read")
            return _FakeResponse(201, {"ok": True})

        client.session.post = MagicMock(side_effect=_post_side_effect)
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            client.upload_asset(release, asset_path, "asset.bin")

        self.assertIsNotNone(client.session.post.call_args)
        self.assertTrue(observed.get("is_file_like"))
        self.assertTrue(observed.get("is_open_during_post"))

    def test_upload_asset_retries_retryable_http_errors(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(
            side_effect=[
                _FakeResponse(503, text="temporary outage"),
                _FakeResponse(201, {"ok": True}),
            ]
        )
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir, patch("scripts.lightgbm_batch_pipeline.time.sleep") as sleep_mock:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            payload = client.upload_asset(release, asset_path, "asset.bin")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(client.session.post.call_count, 2)
        sleep_mock.assert_called_once_with(2)

    def test_upload_asset_retries_http_429(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(
            side_effect=[
                _FakeResponse(429, text="rate limited"),
                _FakeResponse(201, {"ok": True}),
            ]
        )
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir, patch("scripts.lightgbm_batch_pipeline.time.sleep") as sleep_mock:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            payload = client.upload_asset(release, asset_path, "asset.bin")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(client.session.post.call_count, 2)
        sleep_mock.assert_called_once_with(2)

    def test_upload_asset_retries_request_exception(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(
            side_effect=[
                requests.Timeout("timed out"),
                _FakeResponse(201, {"ok": True}),
            ]
        )
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir, patch("scripts.lightgbm_batch_pipeline.time.sleep") as sleep_mock:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            payload = client.upload_asset(release, asset_path, "asset.bin")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(client.session.post.call_count, 2)
        sleep_mock.assert_called_once_with(2)

    def test_upload_asset_raises_after_retryable_http_failures(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(
            side_effect=[
                _FakeResponse(503, text="temporary outage"),
                _FakeResponse(503, text="still down"),
                _FakeResponse(503, text="final failure"),
            ]
        )
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir, patch("scripts.lightgbm_batch_pipeline.time.sleep") as sleep_mock:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            with self.assertRaisesRegex(RuntimeError, "GitHub asset upload failed: 503"):
                client.upload_asset(release, asset_path, "asset.bin")

        self.assertEqual(client.session.post.call_count, 3)
        self.assertEqual(sleep_mock.call_args_list, [call(2), call(4)])

    def test_upload_asset_raises_after_retryable_request_exceptions(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(
            side_effect=[
                requests.Timeout("timed out"),
                requests.Timeout("still timed out"),
                requests.Timeout("timed out again"),
            ]
        )
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir, patch("scripts.lightgbm_batch_pipeline.time.sleep") as sleep_mock:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            with self.assertRaisesRegex(RuntimeError, "GitHub asset upload failed for asset.bin"):
                client.upload_asset(release, asset_path, "asset.bin")

        self.assertEqual(client.session.post.call_count, 3)
        self.assertEqual(sleep_mock.call_args_list, [call(2), call(4)])

    def test_upload_asset_raises_on_unexpected_success_status(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(return_value=_FakeResponse(302, {"ok": True}, text="redirect"))
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            with self.assertRaisesRegex(RuntimeError, "unexpected status: 302"):
                client.upload_asset(release, asset_path, "asset.bin")

    def test_upload_asset_raises_when_response_json_invalid(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(return_value=_FakeResponse(201, ValueError("bad json")))
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            with self.assertRaisesRegex(RuntimeError, "returned invalid JSON"):
                client.upload_asset(release, asset_path, "asset.bin")

    def test_upload_asset_raises_when_response_json_is_not_object(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(return_value=_FakeResponse(201, payload=[]))
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            with self.assertRaisesRegex(RuntimeError, "unexpected JSON payload type"):
                client.upload_asset(release, asset_path, "asset.bin")


def _discover_args(tmpdir: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        output=str(tmpdir / "discovery.json"),
        macro_output=str(tmpdir / "macro.joblib"),
        price_floor=0.10,
        fast_screen_min_score=pipeline.DEFAULT_FAST_SCREEN_MIN_SCORE,
        fast_screen_margin=pipeline.DEFAULT_FAST_SCREEN_MARGIN,
        fast_screen_period="1y",
        fast_screen_interval="1d",
        matrix_jobs=4,
        macro_lookback_days=365 * 5,
        discover_workers=4,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class DiscoverParallelizationTests(unittest.TestCase):
    """discover() parallelizes per-ticker Polygon/fast-screen work across threads.

    _chunk_for_shard() routes tickers to shards using their index in the
    `filtered` list (idx % shard_count), so the order of `filtered` must stay
    deterministic (matching the input tickers' order) regardless of which
    thread finishes first.
    """

    def test_filtered_order_matches_input_order_despite_out_of_order_completion(self):
        tickers = [{"ticker": t, "primary_exchange": "XNYS", "type": "CS"} for t in ["AAA", "BBB", "CCC", "DDD", "EEE"]]

        # Stagger fake "network" latency so tickers complete in reverse of
        # their input order (EEE finishes first, AAA finishes last).
        delays = {"AAA": 0.08, "BBB": 0.06, "CCC": 0.04, "DDD": 0.02, "EEE": 0.0}

        def _fake_get_stock_data(ticker, period=None, interval=None):
            time.sleep(delays[ticker])
            return pd.DataFrame({"Close": [10.0, 11.0]})

        def _fake_fast_screen_score(ticker, period=None, interval=None, data=None):
            return 80, None

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            args = _discover_args(tmpdir)
            with (
                patch.object(pipeline, "get_all_active_ticker_details", return_value=tickers),
                patch.object(pipeline, "get_stock_data", side_effect=_fake_get_stock_data),
                patch.object(pipeline, "fast_screen_score", side_effect=_fake_fast_screen_score),
                patch.object(pipeline, "get_macro_feature_table", return_value=pd.DataFrame()),
            ):
                pipeline.discover(args)

            payload = json.loads(Path(args.output).read_text(encoding="utf-8"))
            self.assertEqual([row["ticker"] for row in payload["tickers"]], ["AAA", "BBB", "CCC", "DDD", "EEE"])

    def test_failed_ticker_is_skipped_without_aborting_other_tickers(self):
        tickers = [{"ticker": t, "primary_exchange": "XNYS", "type": "CS"} for t in ["AAA", "BBB", "CCC"]]

        def _fake_get_stock_data(ticker, period=None, interval=None):
            if ticker == "BBB":
                raise RuntimeError("boom")
            return pd.DataFrame({"Close": [10.0, 11.0]})

        def _fake_fast_screen_score(ticker, period=None, interval=None, data=None):
            return 80, None

        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            args = _discover_args(tmpdir)
            with (
                patch.object(pipeline, "get_all_active_ticker_details", return_value=tickers),
                patch.object(pipeline, "get_stock_data", side_effect=_fake_get_stock_data),
                patch.object(pipeline, "fast_screen_score", side_effect=_fake_fast_screen_score),
                patch.object(pipeline, "get_macro_feature_table", return_value=pd.DataFrame()),
            ):
                pipeline.discover(args)

            payload = json.loads(Path(args.output).read_text(encoding="utf-8"))
            self.assertEqual([row["ticker"] for row in payload["tickers"]], ["AAA", "CCC"])
            self.assertEqual(payload["skipped"]["BBB"]["reason"], "discover_error")


def _write_partial_shard(
    path: Path,
    *,
    ticker: str,
    row_counts: dict[str, int] | None = None,
) -> None:
    row_counts = row_counts if row_counts is not None else {"30": 60, "180": 60}
    batch = pipeline.create_return_model_batch(
        batch_id="20260915T172900Z",
        models={ticker: {30: object(), 180: object()}},
        training_metadata={
            ticker: {
                "exchange": "XNAS",
                "row_counts": row_counts,
                "trained_horizons": [30, 180],
                "last_trained": "2026-09-15T17:29:00Z",
                "fast_score": 80,
                "current_price": 100.0,
            }
        },
        created_at="2026-09-15T17:29:00Z",
    )
    payload = {
        **batch,
        "shard_index": 0,
        "shard_count": 1,
        "expected_tickers": [ticker],
        "assigned_tickers": [ticker],
        "skipped": {},
    }
    joblib.dump(payload, path)


def _reduce_args(
    partial_dir: Path, tmpdir: Path, sanity_tickers: tuple[str, ...], *, scan_shard_count: int = 1
) -> argparse.Namespace:
    return argparse.Namespace(
        partial_dir=str(partial_dir),
        macro_file=None,
        output_batch=str(tmpdir / "batch.joblib"),
        output_manifest=str(tmpdir / "manifest.json"),
        batch_id="20260915T172900Z",
        repository="",
        min_rows_per_horizon=50,
        max_stale_cycles=4,
        sanity_tickers=list(sanity_tickers),
        scan_shard_count=scan_shard_count,
        promote=False,
    )


class ReduceAndValidateMemoryTests(unittest.TestCase):
    """Regression tests for the OOM-causing redundant reload/deepcopy that was removed.

    The original reduce_and_validate() re-loaded the just-dumped merged batch from
    disk via load_return_model_batch() purely to assert it was non-empty, holding a
    second full in-memory copy of the (potentially multi-GB) model set. These tests
    assert that emptiness-validation reload is gone (load_return_model_batch is only
    invoked once per partial shard file up through smoke testing) while the
    emptiness validation it used to provide is still enforced. A later, deliberate
    reload of the merged batch was reintroduced further downstream (see
    ReduceAndValidatePreviousBatchMemoryTests below) so the just-freed current model
    set and a previous live batch are never both fully resident in memory at once;
    that reload only happens after this test's intentional early failure.
    """

    def test_reduce_and_validate_does_not_reload_merged_batch_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            partial_dir = tmpdir / "partials"
            partial_dir.mkdir()
            _write_partial_shard(partial_dir / "shard-0.joblib", ticker="AAA")
            _write_partial_shard(partial_dir / "shard-1.joblib", ticker="BBB")

            args = _reduce_args(partial_dir, tmpdir, sanity_tickers=("ZZZ",))

            # ZZZ is not part of the merged batch, so the smoke test intentionally
            # fails fast (before any network/model-prediction calls) once past the
            # merged-batch validation this test is targeting.
            with (
                patch.object(
                    pipeline, "load_return_model_batch", side_effect=pipeline.load_return_model_batch
                ) as load_mock,
                self.assertRaisesRegex(RuntimeError, "No sanity tickers were present"),
            ):
                pipeline.reduce_and_validate(args)

            # Exactly one load per partial shard file, and no additional reload of the
            # merged current_batch that was just written to disk.
            self.assertEqual(load_mock.call_count, 2)

    def test_reduce_and_validate_raises_when_merged_batch_has_no_models(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            partial_dir = tmpdir / "partials"
            partial_dir.mkdir()
            empty_batch = pipeline.create_return_model_batch(batch_id="20260915T172900Z", models={})
            payload = {
                **empty_batch,
                "shard_index": 0,
                "shard_count": 1,
                "expected_tickers": [],
                "assigned_tickers": [],
                "skipped": {},
            }
            joblib.dump(payload, partial_dir / "shard-0.joblib")

            args = _reduce_args(partial_dir, tmpdir, sanity_tickers=("AAPL", "MSFT"))

            with self.assertRaisesRegex(RuntimeError, "Merged batch is empty or failed to persist"):
                pipeline.reduce_and_validate(args)


class ReduceAndValidatePreviousBatchMemoryTests(unittest.TestCase):
    """Regression tests for the OOM fixed by never holding both the current and
    previous live batches fully deserialized in memory at the same time.

    Once a previous `lightgbm-model-live` release exists, the original
    reduce_and_validate() downloaded and deserialized it unconditionally while the
    just-built current batch was still fully resident, and merged them together --
    doubling peak memory for a full-universe batch. These tests assert the
    expensive previous-batch download is skipped when nothing would be carried
    forward, and that a full run correctly reloads the current batch from disk
    (rather than keeping the original in-memory copy alive) to combine with a
    small carried-forward delta extracted from the previous batch.
    """

    def _previous_manifest(self, *, ticker: str, cycles_since_training: int) -> dict:
        return {
            "latest_batch": {
                "asset_url": "https://github.com/bkabwe/bk-fin-and-inv/releases/download/lightgbm-batch-20260908T172900Z/return_model_batch.joblib",
            },
            "tickers": {
                ticker: {
                    "exchange": "XNAS",
                    "last_trained": "2026-09-08T17:29:00Z",
                    "cycles_since_training": cycles_since_training,
                    "trained_horizons": [30, 180],
                    "row_counts": {"30": 60, "180": 60},
                    "batch_id": "20260908T172900Z",
                }
            }
        }

    def test_needs_previous_batch_download_true_when_carry_forward_candidate_exists(self):
        manifest = self._previous_manifest(ticker="CCC", cycles_since_training=0)
        self.assertTrue(
            pipeline._needs_previous_batch_download(manifest, current_tickers={"AAA", "BBB"}, max_stale_cycles=4)
        )

    def test_needs_previous_batch_download_false_when_ticker_already_current(self):
        manifest = self._previous_manifest(ticker="AAA", cycles_since_training=0)
        self.assertFalse(
            pipeline._needs_previous_batch_download(manifest, current_tickers={"AAA", "BBB"}, max_stale_cycles=4)
        )

    def test_needs_previous_batch_download_false_when_beyond_stale_cycles(self):
        manifest = self._previous_manifest(ticker="CCC", cycles_since_training=4)
        self.assertFalse(
            pipeline._needs_previous_batch_download(manifest, current_tickers={"AAA", "BBB"}, max_stale_cycles=4)
        )

    def test_load_previous_live_state_skips_download_when_nothing_to_carry_forward(self):
        manifest = self._previous_manifest(ticker="AAA", cycles_since_training=0)
        with (
            patch.object(pipeline, "fetch_live_manifest", return_value=manifest),
            patch.object(pipeline, "load_return_model_batch_from_urls") as download_mock,
        ):
            returned_manifest, returned_batch = pipeline._load_previous_live_state(
                "bkabwe/bk-fin-and-inv", current_tickers={"AAA", "BBB"}, max_stale_cycles=4
            )
        download_mock.assert_not_called()
        self.assertEqual(returned_manifest, manifest)
        self.assertEqual(returned_batch, {})

    def test_reduce_and_validate_reloads_current_batch_to_combine_with_carried_forward(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            partial_dir = tmpdir / "partials"
            partial_dir.mkdir()
            _write_partial_shard(partial_dir / "shard-0.joblib", ticker="AAA")
            _write_partial_shard(partial_dir / "shard-1.joblib", ticker="BBB")

            previous_batch = pipeline.create_return_model_batch(
                batch_id="20260908T172900Z",
                models={"CCC": {30: object(), 180: object()}},
                training_metadata={
                    "CCC": {
                        "exchange": "XNAS",
                        "row_counts": {"30": 60, "180": 60},
                        "trained_horizons": [30, 180],
                        "last_trained": "2026-09-08T17:29:00Z",
                        "fast_score": 70,
                        "current_price": 50.0,
                    }
                },
                created_at="2026-09-08T17:29:00Z",
            )
            previous_manifest = self._previous_manifest(ticker="CCC", cycles_since_training=0)

            args = _reduce_args(partial_dir, tmpdir, sanity_tickers=("AAA",))

            with (
                patch.object(pipeline, "fetch_live_manifest", return_value=previous_manifest),
                patch.object(pipeline, "load_return_model_batch_from_urls", return_value=previous_batch) as download_mock,
                patch.object(
                    pipeline, "load_return_model_batch", side_effect=pipeline.load_return_model_batch
                ) as load_mock,
                # Smoke testing (price history/feature/prediction) is exercised by
                # other tests; it is irrelevant to the previous-batch memory
                # behavior under test here, so stub it out as a no-op.
                patch.object(pipeline, "_smoke_test_predictions"),
            ):
                result = pipeline.reduce_and_validate(args)

            self.assertEqual(result, 0)
            # The previous live batch is only fetched once, and only because CCC
            # was a genuine carry-forward candidate.
            download_mock.assert_called_once()
            # One load per partial shard file, plus exactly one reload of the
            # merged current batch to combine it with the carried-forward delta
            # once the previous batch's own reference has already been dropped.
            self.assertEqual(load_mock.call_count, 3)

            manifest = json.loads(Path(args.output_manifest).read_text(encoding="utf-8"))
            self.assertEqual(manifest["current_run"]["trained_tickers"], ["AAA", "BBB"])
            self.assertEqual(manifest["current_run"]["carried_forward_tickers"], ["CCC"])
            self.assertIn("CCC", manifest["tickers"])

            live_batch = pipeline.load_return_model_batch(Path(args.output_batch))
            self.assertEqual(set(live_batch["models"].keys()), {"AAA", "BBB", "CCC"})


class ScanShardWritingTests(unittest.TestCase):
    """Tests for the ticker-partitioned scan-shard storage feature.

    `reduce_and_validate` promotes the live batch as usual, but when
    `scan_shard_count > 1` it additionally writes out ticker-partitioned
    "scan shard" files so a scan/scoring consumer can download only the
    slice of tickers it needs instead of the full combined batch.
    """

    def test_write_scan_shards_returns_no_files_when_shard_count_is_one(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            live_models = {"AAA": {30: object(), 180: object()}, "BBB": {30: object(), 180: object()}}
            live_meta = {"AAA": {"exchange": "XNAS"}, "BBB": {"exchange": "XNAS"}}

            ticker_shards, shard_files = pipeline._write_scan_shards(
                live_models,
                live_meta,
                batch_id="20260915T172900Z",
                created_at="2026-09-15T17:29:00Z",
                scan_shard_count=1,
                output_dir=tmpdir / "scan_shards",
            )

            self.assertEqual(ticker_shards, {"AAA": 0, "BBB": 0})
            self.assertEqual(shard_files, [])
            self.assertFalse((tmpdir / "scan_shards").exists())

    def test_write_scan_shards_partitions_tickers_across_shard_files(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            live_models = {
                "AAA": {30: object(), 180: object()},
                "BBB": {30: object(), 180: object()},
                "CCC": {30: object(), 180: object()},
            }
            live_meta = {ticker: {"exchange": "XNAS"} for ticker in live_models}
            output_dir = tmpdir / "scan_shards"

            ticker_shards, shard_files = pipeline._write_scan_shards(
                live_models,
                live_meta,
                batch_id="20260915T172900Z",
                created_at="2026-09-15T17:29:00Z",
                scan_shard_count=2,
                output_dir=output_dir,
            )

            self.assertEqual(len(shard_files), 2)
            self.assertEqual({index for index, _ in shard_files}, {0, 1})
            all_shard_tickers: set[str] = set()
            for shard_index, shard_path in shard_files:
                self.assertTrue(shard_path.exists())
                shard_batch = pipeline.load_return_model_batch(shard_path)
                shard_tickers = set(shard_batch["models"].keys())
                all_shard_tickers |= shard_tickers
                for ticker in shard_tickers:
                    self.assertEqual(ticker_shards[ticker], shard_index)
            self.assertEqual(all_shard_tickers, set(live_models.keys()))

    def test_promote_outputs_uploads_each_shard_under_its_own_asset_name_and_cleans_up(self):
        client = MagicMock()
        client.ensure_release.return_value = {"id": 99}
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            batch_path = tmpdir / "batch.joblib"
            batch_path.write_bytes(b"combined batch")
            manifest_path = tmpdir / "manifest.json"
            manifest_path.write_text("{}", encoding="utf-8")
            shard0_path = tmpdir / "scan_shard_000.joblib"
            shard0_path.write_bytes(b"shard 0")
            shard1_path = tmpdir / "scan_shard_001.joblib"
            shard1_path.write_bytes(b"shard 1")

            args = argparse.Namespace(repository="bkabwe/bk-fin-and-inv")
            with patch.object(pipeline, "GitHubReleaseClient", return_value=client), patch.dict(
                "os.environ", {"GITHUB_TOKEN": "token"}
            ):
                pipeline._promote_outputs(
                    args,
                    batch_path,
                    manifest_path,
                    "lightgbm-batch-20260915T172900Z",
                    shard_files=[(0, shard0_path), (1, shard1_path)],
                    scan_shard_count=2,
                )

            uploaded_names = [call_args.args[2] for call_args in client.upload_asset.call_args_list]
            self.assertIn(pipeline.DEFAULT_BATCH_ASSET_NAME, uploaded_names)
            self.assertIn("lightgbm_return_model_batch.shard000of002.joblib", uploaded_names)
            self.assertIn("lightgbm_return_model_batch.shard001of002.joblib", uploaded_names)
            self.assertIn(pipeline.DEFAULT_LIVE_MANIFEST_ASSET_NAME, uploaded_names)
            # Shard files are temporary local artifacts and must be cleaned up
            # after their upload, same as the byte-range split parts.
            self.assertFalse(shard0_path.exists())
            self.assertFalse(shard1_path.exists())
            # The combined batch and manifest files are not shard-specific
            # temporary files and must be left untouched.
            self.assertTrue(batch_path.exists())
            self.assertTrue(manifest_path.exists())

    def test_reduce_and_validate_writes_shard_manifest_when_scan_shard_count_greater_than_one(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            partial_dir = tmpdir / "partials"
            partial_dir.mkdir()
            _write_partial_shard(partial_dir / "shard-0.joblib", ticker="AAA")
            _write_partial_shard(partial_dir / "shard-1.joblib", ticker="BBB")

            args = _reduce_args(partial_dir, tmpdir, sanity_tickers=("AAA",), scan_shard_count=2)

            with (
                patch.object(pipeline, "_smoke_test_predictions"),
                patch.object(pipeline, "fetch_live_manifest", return_value={}),
            ):
                result = pipeline.reduce_and_validate(args)

            self.assertEqual(result, 0)
            manifest = json.loads(Path(args.output_manifest).read_text(encoding="utf-8"))
            self.assertEqual(manifest["latest_batch"]["shard_count"], 2)
            self.assertEqual(len(manifest["latest_batch"]["shards"]), 2)
            self.assertIn("shard_index", manifest["tickers"]["AAA"])
            self.assertIn("shard_index", manifest["tickers"]["BBB"])

            scan_shards_dir = tmpdir / "scan_shards"
            shard_files = sorted(scan_shards_dir.glob("*.joblib"))
            self.assertEqual(len(shard_files), 2)


class BatchAssetSplittingTests(unittest.TestCase):
    """Regression tests for GitHub's 2 GiB per-release-asset limit.

    A full-universe batch file previously failed to promote once it exceeded
    GitHub's per-asset size limit; these tests confirm large batches are split
    into parts before upload and reassembled unchanged after download.
    """

    def test_split_large_file_returns_original_path_when_under_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "batch.joblib"
            path.write_bytes(b"x" * 100)

            parts = pipeline._split_large_file(path, chunk_size=1000)

        self.assertEqual(parts, [path])

    def test_split_large_file_splits_into_ordered_parts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "batch.joblib"
            payload = bytes(range(256)) * 4  # 1024 bytes
            path.write_bytes(payload)

            parts = pipeline._split_large_file(path, chunk_size=300)

            self.assertEqual([part.name for part in parts], [
                "batch.joblib.part001of004",
                "batch.joblib.part002of004",
                "batch.joblib.part003of004",
                "batch.joblib.part004of004",
            ])
            reassembled = b"".join(part.read_bytes() for part in parts)
            self.assertEqual(reassembled, payload)

    def test_upload_batch_asset_uploads_single_part_for_small_files(self):
        client = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.joblib"
            batch_path.write_bytes(b"small file")

            pipeline._upload_batch_asset(client, {"id": 1}, batch_path)

            client.upload_asset.assert_called_once_with({"id": 1}, batch_path, pipeline.DEFAULT_BATCH_ASSET_NAME)
            # The original (un-split) file must still exist and be untouched.
            self.assertTrue(batch_path.exists())

    def test_upload_batch_asset_uses_custom_base_asset_name_for_shards(self):
        client = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_path = Path(tmpdir) / "scan_shard_000.joblib"
            shard_path.write_bytes(b"shard contents")

            pipeline._upload_batch_asset(
                client, {"id": 1}, shard_path, base_asset_name="lightgbm_return_model_batch.shard000of002.joblib"
            )

            client.upload_asset.assert_called_once_with(
                {"id": 1}, shard_path, "lightgbm_return_model_batch.shard000of002.joblib"
            )

    def test_upload_batch_asset_splits_and_cleans_up_parts_for_large_files(self):
        client = MagicMock()
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.joblib"
            batch_path.write_bytes(b"y" * 1000)

            pipeline._upload_batch_asset(client, {"id": 1}, batch_path, chunk_size=300)

            self.assertEqual(client.upload_asset.call_count, 4)
            uploaded_names = [call_args.args[2] for call_args in client.upload_asset.call_args_list]
            self.assertEqual(
                uploaded_names,
                [
                    "lightgbm_return_model_batch.joblib.part001of004",
                    "lightgbm_return_model_batch.joblib.part002of004",
                    "lightgbm_return_model_batch.joblib.part003of004",
                    "lightgbm_return_model_batch.joblib.part004of004",
                ],
            )
            # Part files are temporary and must be cleaned up after upload.
            for call_args in client.upload_asset.call_args_list:
                self.assertFalse(call_args.args[1].exists())
            # The original merged batch file itself must be left untouched.
            self.assertTrue(batch_path.exists())


if __name__ == "__main__":
    unittest.main()
