from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from scripts import snapshot_universe

TEST_ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / ".test_artifacts"


def _make_scratch_dir(prefix: str) -> Path:
    path = TEST_ARTIFACTS_DIR / f"{prefix}_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


class SnapshotUniverseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch_dir = _make_scratch_dir("snapshot_universe")
        self.addCleanup(lambda: shutil.rmtree(self.scratch_dir, ignore_errors=True))

        self.snapshot_dir = self.scratch_dir / "snapshots"
        self.universe_file = self.snapshot_dir / "universe_membership.jsonl"
        self.sentiment_dir = self.snapshot_dir / "sentiment"

        patchers = [
            patch.object(snapshot_universe, "DATA_DIR", self.scratch_dir),
            patch.object(snapshot_universe, "SNAPSHOT_DIR", self.snapshot_dir),
            patch.object(snapshot_universe, "UNIVERSE_SNAPSHOT_FILE", self.universe_file),
            patch.object(snapshot_universe, "SENTIMENT_SNAPSHOT_DIR", self.sentiment_dir),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_first_run_creates_daily_universe_snapshot(self):
        with (
            patch.object(snapshot_universe, "_today_iso", return_value="2026-10-01"),
            patch.object(snapshot_universe, "get_sp500_tickers", return_value=["AAPL", "MSFT"]),
        ):
            exit_code = snapshot_universe.main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            self._read_jsonl(self.universe_file),
            [
                {
                    "snapshot_date": "2026-10-01",
                    "source": "sp500",
                    "tickers": ["AAPL", "MSFT"],
                }
            ],
        )

    def test_second_run_same_day_does_not_duplicate_snapshot(self):
        with (
            patch.object(snapshot_universe, "_today_iso", return_value="2026-10-01"),
            patch.object(snapshot_universe, "get_sp500_tickers", return_value=["AAPL", "MSFT"]),
        ):
            first_exit_code = snapshot_universe.main([])
            second_exit_code = snapshot_universe.main([])

        self.assertEqual(first_exit_code, 0)
        self.assertEqual(second_exit_code, 0)
        self.assertEqual(len(self._read_jsonl(self.universe_file)), 1)

    def test_tickers_override_with_sentiment_writes_expected_records(self):
        sentiment_payloads = {
            "TSLA": {
                "sentiment_score": 0.42,
                "sentiment_label": "Positive",
                "headlines": [{"headline": "TSLA rallies"}, {"headline": "Analysts upbeat"}],
                "short_ratio": 1.2,
                "short_pct_float": 0.8,
                "put_call_ratio": 0.7,
                "options_sentiment": "Bullish options positioning",
            },
            "NVDA": {
                "sentiment_score": -0.15,
                "sentiment_label": "Negative",
                "headlines": [{"headline": "NVDA cools"}],
                "short_ratio": 2.3,
                "short_pct_float": 1.1,
                "put_call_ratio": 1.4,
                "options_sentiment": "Neutral",
            },
        }

        with (
            patch.object(snapshot_universe, "_today_iso", return_value="2026-10-01"),
            patch.object(snapshot_universe, "get_sp500_tickers", return_value=["AAPL", "MSFT"]),
            patch.object(
                snapshot_universe.sentiment_analysis,
                "analyze_sentiment",
                side_effect=lambda ticker: sentiment_payloads[ticker],
            ),
        ):
            exit_code = snapshot_universe.main(["--with-sentiment", "--tickers", "tsla", "NVDA"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            self._read_jsonl(self.universe_file),
            [
                {
                    "snapshot_date": "2026-10-01",
                    "source": "sp500",
                    "tickers": ["AAPL", "MSFT"],
                }
            ],
        )
        self.assertEqual(
            self._read_jsonl(self.sentiment_dir / "2026-10-01.jsonl"),
            [
                {
                    "snapshot_date": "2026-10-01",
                    "ticker": "TSLA",
                    "sentiment_score": 0.42,
                    "sentiment_label": "Positive",
                    "headline_count": 2,
                    "short_ratio": 1.2,
                    "short_pct_float": 0.8,
                    "put_call_ratio": 0.7,
                    "options_sentiment": "Bullish options positioning",
                },
                {
                    "snapshot_date": "2026-10-01",
                    "ticker": "NVDA",
                    "sentiment_score": -0.15,
                    "sentiment_label": "Negative",
                    "headline_count": 1,
                    "short_ratio": 2.3,
                    "short_pct_float": 1.1,
                    "put_call_ratio": 1.4,
                    "options_sentiment": "Neutral",
                },
            ],
        )

    def test_sentiment_failure_does_not_block_other_tickers(self):
        def _sentiment_side_effect(ticker: str) -> dict:
            if ticker == "TSLA":
                raise RuntimeError("boom")
            return {
                "sentiment_score": 0.5,
                "sentiment_label": "Positive",
                "headlines": [{"headline": "MSFT steady"}],
                "short_ratio": None,
                "short_pct_float": None,
                "put_call_ratio": None,
                "options_sentiment": "Neutral",
            }

        with (
            patch.object(snapshot_universe, "_today_iso", return_value="2026-10-01"),
            patch.object(snapshot_universe, "get_sp500_tickers", return_value=["AAPL", "MSFT"]),
            patch.object(snapshot_universe.sentiment_analysis, "analyze_sentiment", side_effect=_sentiment_side_effect),
        ):
            exit_code = snapshot_universe.main(["--with-sentiment", "--tickers", "TSLA", "MSFT"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            self._read_jsonl(self.sentiment_dir / "2026-10-01.jsonl"),
            [
                {
                    "snapshot_date": "2026-10-01",
                    "ticker": "MSFT",
                    "sentiment_score": 0.5,
                    "sentiment_label": "Positive",
                    "headline_count": 1,
                    "short_ratio": None,
                    "short_pct_float": None,
                    "put_call_ratio": None,
                    "options_sentiment": "Neutral",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
