from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import joblib
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


def _reduce_args(partial_dir: Path, tmpdir: Path, sanity_tickers: tuple[str, ...]) -> argparse.Namespace:
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
        promote=False,
    )


class ReduceAndValidateMemoryTests(unittest.TestCase):
    """Regression tests for the OOM-causing redundant reload/deepcopy that was removed.

    The original reduce_and_validate() re-loaded the just-dumped merged batch from
    disk via load_return_model_batch() purely to assert it was non-empty, holding a
    second full in-memory copy of the (potentially multi-GB) model set. These tests
    assert that call is gone (load_return_model_batch is only ever invoked once per
    partial shard file, never again for the merged output) while the emptiness
    validation it used to provide is still enforced.
    """

    def test_reduce_and_validate_does_not_reload_merged_batch_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            partial_dir = tmpdir / "partials"
            partial_dir.mkdir()
            _write_partial_shard(partial_dir / "shard-0.joblib", ticker="AAA")
            _write_partial_shard(partial_dir / "shard-1.joblib", ticker="BBB")

            args = _reduce_args(partial_dir, tmpdir, sanity_tickers=("ZZZ",))

            with patch.object(
                pipeline, "load_return_model_batch", side_effect=pipeline.load_return_model_batch
            ) as load_mock:
                # ZZZ is not part of the merged batch, so the smoke test intentionally
                # fails fast (before any network/model-prediction calls) once past the
                # merged-batch validation this test is targeting.
                with self.assertRaisesRegex(RuntimeError, "No sanity tickers were present"):
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
