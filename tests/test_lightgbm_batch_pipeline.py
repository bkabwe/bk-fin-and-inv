from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import requests

from scripts import lightgbm_batch_pipeline as pipeline
from scripts.lightgbm_batch_pipeline import GitHubReleaseClient


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
            self.assertIn(
                ("AAA", "max_tickers_cap"),
                {(entry.get("ticker"), entry.get("reason")) for entry in payload["skipped"].values()},
            )

    def test_discover_cap_counts_unique_tickers_with_duplicate_rows(self):
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
            self.assertEqual([item["ticker"] for item in payload["tickers"]], ["AAA", "BBB"])
            self.assertEqual(payload["tickers"][0]["exchange"], "XNAS")
            duplicate = [entry for entry in payload["skipped"].values() if entry.get("reason") == "duplicate_ticker"]
            self.assertEqual(len(duplicate), 1)
            self.assertEqual(duplicate[0]["ticker"], "AAA")

    def test_discover_cap_keeps_unique_tickers_when_duplicate_scores_are_higher(self):
        rows = [
            {"ticker": "AAA", "type": "CS", "primary_exchange": "XNAS"},
            {"ticker": "AAA", "type": "CS", "primary_exchange": "XNYS"},
            {"ticker": "BBB", "type": "CS", "primary_exchange": "XNAS"},
            {"ticker": "CCC", "type": "CS", "primary_exchange": "XNAS"},
        ]
        scores = {"AAA": 95, "BBB": 90, "CCC": 80}

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
                patch.object(pipeline, "get_stock_data", return_value=pd.DataFrame({"Close": [10.0, 11.0]})),
                patch.object(pipeline, "fast_screen_score", side_effect=lambda ticker, **kwargs: (scores[ticker], None)),
                patch.object(pipeline, "get_macro_feature_table", return_value=pd.DataFrame()),
            ):
                rc = pipeline.discover(args)

            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([item["ticker"] for item in payload["tickers"]], ["AAA", "BBB"])
            skipped_reasons = [(entry.get("ticker"), entry.get("reason")) for entry in payload["skipped"].values()]
            self.assertIn(("AAA", "duplicate_ticker"), skipped_reasons)
            self.assertIn(("CCC", "max_tickers_cap"), skipped_reasons)

    def test_discover_cap_preserves_discovery_order_for_fast_score_ties(self):
        rows = [
            {"ticker": "AAA", "type": "CS", "primary_exchange": "XNAS"},
            {"ticker": "BBB", "type": "CS", "primary_exchange": "XNAS"},
            {"ticker": "CCC", "type": "CS", "primary_exchange": "XNAS"},
        ]
        scores = {"AAA": 90, "BBB": 90, "CCC": 80}

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
                patch.object(pipeline, "get_stock_data", return_value=pd.DataFrame({"Close": [10.0, 11.0]})),
                patch.object(pipeline, "fast_screen_score", side_effect=lambda ticker, **kwargs: (scores[ticker], None)),
                patch.object(pipeline, "get_macro_feature_table", return_value=pd.DataFrame()),
            ):
                rc = pipeline.discover(args)

            self.assertEqual(rc, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([item["ticker"] for item in payload["tickers"]], ["AAA", "BBB"])


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | Exception | None = None, text: str = ""):
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

        post_call = client.session.post.call_args
        self.assertIsNotNone(post_call)
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


if __name__ == "__main__":
    unittest.main()
