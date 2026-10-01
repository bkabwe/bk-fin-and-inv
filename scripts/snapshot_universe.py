from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.data_fetcher import get_sp500_tickers
from modules.logger import get_logger
from modules import sentiment_analysis

logger = get_logger(__name__)
DATA_DIR = PROJECT_ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
UNIVERSE_SNAPSHOT_FILE = SNAPSHOT_DIR / "universe_membership.jsonl"
SENTIMENT_SNAPSHOT_DIR = SNAPSHOT_DIR / "sentiment"


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSONL line %s in %s: %s", line_number, path, exc)
                continue
            if isinstance(payload, dict):
                records.append(payload)
            else:
                logger.warning("Skipping non-object JSONL line %s in %s", line_number, path)
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=False))
        handle.write("\n")


def _snapshot_exists(path: Path, snapshot_date: str) -> bool:
    return any(record.get("snapshot_date") == snapshot_date for record in _iter_jsonl(path))


def _normalize_tickers(values: list[str] | None) -> list[str]:
    if not values:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            ticker = part.strip().upper().replace(".", "-")
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            normalized.append(ticker)
    return normalized


def _build_universe_record(snapshot_date: str, tickers: list[str]) -> dict[str, Any]:
    return {
        "snapshot_date": snapshot_date,
        "source": "sp500",
        "tickers": list(tickers),
    }


def _build_sentiment_record(snapshot_date: str, ticker: str, sentiment: dict[str, Any]) -> dict[str, Any]:
    headlines = sentiment.get("headlines")
    return {
        "snapshot_date": snapshot_date,
        "ticker": ticker.upper(),
        "sentiment_score": sentiment.get("sentiment_score"),
        "sentiment_label": sentiment.get("sentiment_label"),
        "headline_count": len(headlines) if isinstance(headlines, list) else 0,
        "short_ratio": sentiment.get("short_ratio"),
        "short_pct_float": sentiment.get("short_pct_float"),
        "put_call_ratio": sentiment.get("put_call_ratio"),
        "options_sentiment": sentiment.get("options_sentiment"),
    }


def _write_sentiment_snapshot(snapshot_date: str, tickers: list[str]) -> int:
    output_path = SENTIMENT_SNAPSHOT_DIR / f"{snapshot_date}.jsonl"
    written = 0
    for ticker in tickers:
        try:
            sentiment = sentiment_analysis.analyze_sentiment(ticker)
        except Exception as exc:  # pragma: no cover - exercised via unit test
            logger.warning("Sentiment snapshot failed for %s: %s", ticker, exc)
            continue
        _append_jsonl(output_path, _build_sentiment_record(snapshot_date, ticker, sentiment or {}))
        written += 1

    logger.info("Wrote %s sentiment snapshot record(s) to %s", written, output_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append a point-in-time S&P 500 universe snapshot and optional sentiment summaries."
    )
    parser.add_argument(
        "--with-sentiment",
        action="store_true",
        help="Also snapshot per-ticker sentiment summaries (slower FinBERT-backed path).",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional explicit ticker list for sentiment snapshotting only; defaults to the fetched S&P 500 list.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    snapshot_date = _today_iso()
    universe_tickers = get_sp500_tickers()

    if _snapshot_exists(UNIVERSE_SNAPSHOT_FILE, snapshot_date):
        logger.info("Universe snapshot for %s already exists at %s; skipping.", snapshot_date, UNIVERSE_SNAPSHOT_FILE)
        return 0

    _append_jsonl(
        UNIVERSE_SNAPSHOT_FILE,
        _build_universe_record(snapshot_date=snapshot_date, tickers=universe_tickers),
    )
    logger.info(
        "Wrote universe snapshot for %s with %s ticker(s) to %s",
        snapshot_date,
        len(universe_tickers),
        UNIVERSE_SNAPSHOT_FILE,
    )

    if args.with_sentiment:
        sentiment_tickers = _normalize_tickers(args.tickers) or list(universe_tickers)
        _write_sentiment_snapshot(snapshot_date, sentiment_tickers)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
