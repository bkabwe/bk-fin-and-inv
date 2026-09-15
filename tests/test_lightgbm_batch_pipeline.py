from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from scripts.lightgbm_batch_pipeline import GitHubReleaseClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class GitHubReleaseClientUploadTests(unittest.TestCase):
    def test_upload_asset_streams_file_handle_instead_of_reading_all_bytes(self):
        client = GitHubReleaseClient("bkabwe/bk-fin-and-inv", "test-token")
        client.delete_asset_if_exists = MagicMock()
        client.session.post = MagicMock(return_value=_FakeResponse(201, {"ok": True}))
        release = {"upload_url": "https://uploads.github.test/upload{?name}"}

        with tempfile.TemporaryDirectory() as tmpdir:
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            client.upload_asset(release, asset_path, "asset.bin")

        call = client.session.post.call_args
        self.assertIsNotNone(call)
        self.assertTrue(hasattr(call.kwargs["data"], "read"))
        self.assertNotIsInstance(call.kwargs["data"], (bytes, bytearray))

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

        with tempfile.TemporaryDirectory() as tmpdir, patch("scripts.lightgbm_batch_pipeline.time.sleep"):
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            payload = client.upload_asset(release, asset_path, "asset.bin")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(client.session.post.call_count, 2)

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

        with tempfile.TemporaryDirectory() as tmpdir, patch("scripts.lightgbm_batch_pipeline.time.sleep"):
            asset_path = Path(tmpdir) / "asset.bin"
            asset_path.write_bytes(b"hello world")
            payload = client.upload_asset(release, asset_path, "asset.bin")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(client.session.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
