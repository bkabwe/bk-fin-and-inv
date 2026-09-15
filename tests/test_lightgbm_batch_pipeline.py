from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import requests

from scripts.lightgbm_batch_pipeline import GitHubReleaseClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | Exception | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
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

        call = client.session.post.call_args
        self.assertIsNotNone(call)
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


if __name__ == "__main__":
    unittest.main()
