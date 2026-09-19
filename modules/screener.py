from __future__ import annotations

import concurrent.futures
import threading
from typing import Callable

import pandas as pd

from modules.data_fetcher import get_nasdaq_tickers, get_nyseamerican_tickers, get_otc_tickers, get_sp500_tickers
from modules.logger import get_logger
from modules.scoring_engine import analyze_stock, fast_screen_score
from modules.validators import sanitize_ticker_list

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default thread-pool concurrency for I/O-bound Polygon API fetches.
_DEFAULT_MAX_WORKERS = 8

# Default safety margin for the two-tier fast-screen pre-filter.
# A ticker whose normalized technical subscore is below ``min_score - margin``
# is skipped from the expensive trend/GARCH/backtest full analysis.
# Increase this value to make the pre-filter more permissive (fewer false
# negatives); set ``use_fast_screen=False`` to disable entirely.
DEFAULT_FAST_SCREEN_MARGIN = 15


def screener_row_from_analysis(
    result: dict,
    min_score: int,
    *,
    label: str = "",
) -> dict | None:
    """Build a screener result row from an already-computed
    :func:`modules.scoring_engine.analyze_stock` result.

    Returns ``None`` when ``result["score"]`` doesn't meet ``min_score``.
    Factored out of ``run_screener``'s per-ticker worker so callers that
    already have a shared ``analyze_stock`` result for this ticker (e.g. the
    sharded scan-email path, which derives both a screener row and a
    profit-opportunities row from a single analysis) can build this row
    without re-running the full analysis.
    """
    if result["score"] < min_score:
        return None
    entry, current = result["entry_price"], result["current_price"]
    pct = ((current - entry) / entry * 100) if entry and current else None
    return {
        "Ticker": result["ticker"],
        "Company": result["company"],
        "Score": result["score"],
        "Recommendation": result["recommendation"],
        "Time Horizon": result["time_horizon"],
        "Sector Trend": str(result.get("sector_trend") or "unknown").replace("_", " ").title(),
        "Market Cap Tier": result.get("market_cap_tier") or "unknown",
        "Long-Term Stage": result.get("longterm_stage") or "",
        "Entry Price": entry,
        "Target Price": result["target_price"],
        "Stop Loss": result["stop_loss"],
        "Current Price": current,
        "% from Entry": round(pct, 2) if pct is not None else None,
        **({"Index": label} if label else {}),
    }


def _get_tickers(universe: str, custom_tickers: list[str] | None = None) -> list[str]:
    universe = (universe or "sp500").lower()
    if universe == "nasdaq":
        return get_nasdaq_tickers()
    if universe == "nyseamerican":
        return get_nyseamerican_tickers()
    if universe == "otc":
        return get_otc_tickers()
    if universe == "custom":
        return sanitize_ticker_list(custom_tickers or [])
    return get_sp500_tickers()


def run_screener(
    universe: str = "sp500",
    min_score: int = 50,
    max_results: int = 25,
    label: str = "",
    custom_tickers: list[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    use_fast_screen: bool = True,
    fast_screen_margin: int = DEFAULT_FAST_SCREEN_MARGIN,
    on_ticker_processed: Callable[[str, dict], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
    on_total_known: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """Run the stock screener across the selected universe.

    Parameters
    ----------
    universe:
        Ticker universe (``sp500``, ``nasdaq``, ``nyseamerican``, ``otc``,
        ``custom``).
    min_score:
        Minimum composite score (0-100) a ticker must achieve to appear in
        results.
    max_results:
        Maximum number of qualifying tickers to return, sorted by score.
    label:
        Optional index label added to each result row (e.g. ``"S&P 500"``).
    custom_tickers:
        Explicit list of tickers used when ``universe="custom"``.
    progress_callback:
        Optional callable receiving a float in ``[0.0, 1.0]`` after each
        ticker is processed.
    max_workers:
        Number of threads for parallel I/O-bound Polygon fetches.
    use_fast_screen:
        When ``True`` (default), apply a cheap technical-subscore pre-filter
        before running the expensive full analysis (trend/GARCH +
        walk-forward backtest).  Set to ``False`` for small universes (e.g.
        NASDAQ 100 custom lists) where speed is less of a concern.
    fast_screen_margin:
        Safety margin (in score points) applied to the fast-screen cutoff.
        A ticker passes to full analysis if its normalized technical subscore
        ≥ ``min_score - fast_screen_margin``.  Widen this margin to reduce
        false negatives; the default of 15 is deliberately conservative.
    on_ticker_processed:
        Optional callable invoked after each ticker finishes processing with
        ``(ticker, counts)`` where ``counts`` is a dict containing
        ``screened``, ``total``, ``qualified``, ``fast_filtered``,
        ``fully_analyzed`` and ``failed_count``. Used by callers (e.g. the
        FastAPI screener route) that need richer incremental progress than
        the single-float ``progress_callback``.
    stop_check:
        Optional callable returning ``True`` once processing should stop
        early (e.g. a user-requested cancellation). Checked before each
        ticker starts and between completed futures; when triggered, the
        in-progress thread pool is shut down without waiting for pending
        futures and the function returns the partial results gathered so
        far with ``results.attrs["stopped"] = True``.
    on_total_known:
        Optional callable invoked once with the resolved ticker-universe
        size as soon as it is known (before any ticker is processed), so
        callers can surface an accurate progress denominator immediately.
    """
    try:
        tickers = _get_tickers(universe, custom_tickers)
    except RuntimeError as exc:
        raise RuntimeError(f"Unable to fetch ticker universe '{universe}': {exc}") from exc
    if not tickers:
        return pd.DataFrame()

    total = len(tickers)
    logger.info("Scanning %s tickers in universe '%s' (fast_screen=%s, workers=%s)", total, universe, use_fast_screen, max_workers)
    if on_total_known:
        on_total_known(total)

    fast_screen_threshold = min_score - fast_screen_margin  # normalized 0-100

    # Shared mutable state (thread-safe via lock)
    rows: list[dict] = []
    failed_tickers: list[dict] = []
    fast_filtered_count = 0
    fully_analyzed_count = 0
    processed_count = 0
    _lock = threading.Lock()

    def _process_ticker(ticker: str) -> None:
        nonlocal fast_filtered_count, fully_analyzed_count, processed_count
        if stop_check is not None and stop_check():
            return
        try:
            if use_fast_screen:
                fast_score, err = fast_screen_score(ticker)
                if err is not None:
                    with _lock:
                        failed_tickers.append({"ticker": ticker, "reason": f"fast-screen error: {err}"})
                    return
                if fast_score < fast_screen_threshold:
                    # Technical subscore is well below threshold — even with
                    # perfect fundamentals + sentiment it is unlikely to qualify.
                    with _lock:
                        fast_filtered_count += 1
                    logger.debug("Fast-filtered %s (tech_score=%s < %s)", ticker, fast_score, fast_screen_threshold)
                    return

            result = analyze_stock(ticker)
            with _lock:
                fully_analyzed_count += 1
            row = screener_row_from_analysis(result, min_score, label=label)
            if row is not None:
                with _lock:
                    rows.append(row)
        except Exception as exc:
            reason = str(exc) or type(exc).__name__
            with _lock:
                failed_tickers.append({"ticker": ticker, "reason": reason})
            logger.debug("Failed %s: %s", ticker, reason)
        finally:
            # Always increment processed_count exactly once (even for early returns).
            with _lock:
                processed_count += 1
                pct_done = processed_count / total
                snapshot = {
                    "screened": processed_count,
                    "total": total,
                    "qualified": len(rows),
                    "fast_filtered": fast_filtered_count,
                    "fully_analyzed": fully_analyzed_count,
                    "failed_count": len(failed_tickers),
                }
            if progress_callback:
                progress_callback(pct_done)
            if on_ticker_processed:
                on_ticker_processed(ticker, snapshot)

    stopped = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_ticker, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            if stop_check is not None and stop_check():
                stopped = True
                executor.shutdown(wait=False, cancel_futures=True)
                break
            try:
                future.result()
            except Exception as exc:
                ticker = futures[future]
                logger.warning("Unexpected error for %s: %s", ticker, exc)

    if not rows:
        results = pd.DataFrame()
    else:
        results = pd.DataFrame(rows).sort_values("Score", ascending=False).head(max_results).reset_index(drop=True)

    results.attrs["stopped"] = stopped
    results.attrs["source_ticker_count"] = total
    results.attrs["qualified_count"] = len(rows)
    results.attrs["universe"] = universe
    results.attrs["fallback_used"] = False
    results.attrs["fast_filtered_count"] = fast_filtered_count
    results.attrs["fully_analyzed_count"] = fully_analyzed_count
    results.attrs["failed_count"] = len(failed_tickers)
    results.attrs["failed_tickers"] = failed_tickers
    results.attrs["revalidation_requested"] = False
    results.attrs["revalidation_status"] = "not_requested"
    results.attrs["revalidation_count"] = 0
    results.attrs["revalidation_verified_count"] = 0
    results.attrs["revalidation_unavailable_count"] = 0

    logger.info(
        "Screener complete | universe=%s | total=%s | fast_filtered=%s | fully_analyzed=%s | failed=%s | qualified=%s | displayed=%s",
        universe,
        total,
        fast_filtered_count,
        fully_analyzed_count,
        len(failed_tickers),
        len(rows),
        len(results),
    )
    if failed_tickers:
        logger.warning(
            "Tickers that failed analysis (%s): %s",
            len(failed_tickers),
            [f["ticker"] for f in failed_tickers[:20]],
        )
    return results
