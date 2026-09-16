"""Refresh data/sector_benchmark_pe.json from the current ticker universe.

This is the refresh mechanism for modules.fundamental_analysis.SECTOR_BENCHMARK_PE,
which is otherwise a static, hand-maintained dict that drifts stale as sector
multiples move with rate cycles. Run this periodically (e.g. quarterly, or
whenever sector benchmarks look obviously stale in the app):

    python scripts/refresh_sector_pe.py

It fetches trailing P/E and sector for each ticker in the chosen universe
(default: S&P 500), computes the per-sector median (filtering out missing or
extreme outlier P/Es), and writes the result to
data/sector_benchmark_pe.json. modules.fundamental_analysis.get_sector_benchmark_pe
prefers this refreshed file over the hardcoded defaults, and the app works
unmodified if the file is absent, stale, or missing a given sector.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.data_fetcher import get_sp500_tickers, get_stock_info  # noqa: E402
from modules.fundamental_analysis import SECTOR_BENCHMARK_PE_FILE, normalize_sector_name  # noqa: E402
from modules.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_MAX_WORKERS = 8
# Sanity bounds applied before a ticker's P/E contributes to a sector median,
# so a handful of distressed-earnings or speculative names don't skew the
# benchmark used for every other stock in that sector.
MIN_VALID_PE = 0.0
MAX_VALID_PE = 200.0
DEFAULT_MIN_SAMPLES = 5


def collect_sector_pe_pairs(
    tickers: list[str],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[tuple[str | None, float | None]]:
    """Fetch (sector, trailing_pe) for each ticker, tolerating per-ticker failures."""

    def _fetch(ticker: str) -> tuple[str | None, float | None]:
        try:
            info = get_stock_info(ticker)
        except Exception as exc:
            logger.warning("Failed to fetch info for %s: %s", ticker, exc)
            return None, None
        if not info:
            return None, None
        sector = normalize_sector_name(info.get("sector"))
        pe = info.get("trailingPE")
        try:
            pe = float(pe) if pe is not None else None
        except (TypeError, ValueError):
            pe = None
        return sector, pe

    pairs: list[tuple[str | None, float | None]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(_fetch, tickers):
            pairs.append(result)
    return pairs


def compute_sector_medians(
    sector_pe_pairs: list[tuple[str | None, float | None]],
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict[str, dict[str, float | int]]:
    """Group (sector, pe) pairs by sector and compute the median valid P/E.

    Sectors with fewer than `min_samples` valid (non-null, in-range) P/E
    observations are omitted entirely -- callers should keep the hardcoded
    default for those rather than publish a benchmark from a tiny sample.
    """
    by_sector: dict[str, list[float]] = {}
    for sector, pe in sector_pe_pairs:
        if not sector or pe is None:
            continue
        if not (MIN_VALID_PE < pe <= MAX_VALID_PE):
            continue
        by_sector.setdefault(sector, []).append(pe)

    result: dict[str, dict[str, float | int]] = {}
    for sector, values in by_sector.items():
        if len(values) < min_samples:
            continue
        result[sector] = {"median_pe": round(median(values), 2), "sample_count": len(values)}
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh sector P/E benchmarks from the current ticker universe.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Thread-pool concurrency for per-ticker fetches (default: {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"Minimum valid tickers required to publish a sector's median P/E (default: {DEFAULT_MIN_SAMPLES}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SECTOR_BENCHMARK_PE_FILE,
        help="Output path for the refreshed benchmarks JSON (default: data/sector_benchmark_pe.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tickers = get_sp500_tickers()
    logger.info("Refreshing sector P/E benchmarks from %d tickers", len(tickers))

    pairs = collect_sector_pe_pairs(tickers, max_workers=args.max_workers)
    sector_stats = compute_sector_medians(pairs, min_samples=args.min_samples)
    if not sector_stats:
        logger.error("No sector produced enough valid P/E samples; leaving existing benchmarks file untouched.")
        return

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_ticker_count": len(tickers),
        "sectors": {sector: stats["median_pe"] for sector, stats in sector_stats.items()},
        "sector_sample_counts": {sector: stats["sample_count"] for sector, stats in sector_stats.items()},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote refreshed sector P/E benchmarks for %d sectors to %s", len(sector_stats), args.output)


if __name__ == "__main__":
    main()
