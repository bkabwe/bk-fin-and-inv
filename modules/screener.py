from __future__ import annotations

from typing import Callable

import pandas as pd

from modules.data_fetcher import get_nasdaq100_tickers, get_otc_tickers, get_russell2000_tickers, get_sp500_tickers
from modules.logger import get_logger
from modules.scoring_engine import analyze_stock
from modules.validators import sanitize_ticker_list

logger = get_logger(__name__)


def _get_tickers(universe: str, custom_tickers: list[str] | None = None) -> list[str]:
    universe = (universe or "sp500").lower()
    if universe == "nasdaq100":
        return get_nasdaq100_tickers()
    if universe == "russell2000":
        return get_russell2000_tickers()
    if universe == "otc":
        return get_otc_tickers()
    if universe == "custom":
        return sanitize_ticker_list(custom_tickers or [])
    return get_sp500_tickers()


def run_screener(
    universe: str = "sp500",
    min_score: int = 50,
    max_results: int = 25,
    batch_size: int = 50,
    label: str = "",
    custom_tickers: list[str] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    logger.info("Starting screener | universe=%s | batch_size=%s | min_score=%s | max_results=%s", universe, batch_size, min_score, max_results)
    rows = []
    try:
        tickers = _get_tickers(universe, custom_tickers)
    except RuntimeError as exc:
        raise RuntimeError(f"Unable to fetch ticker universe '{universe}': {exc}") from exc
    if not tickers:
        return pd.DataFrame()
    if batch_size:
        tickers = tickers[: max(1, int(batch_size))]
    for i, ticker in enumerate(tickers, start=1):
        try:
            result = analyze_stock(ticker)
            if result["score"] >= min_score:
                entry, current = result["entry_price"], result["current_price"]
                pct = ((current - entry) / entry * 100) if entry and current else None
                rows.append(
                    {
                        "Ticker": result["ticker"],
                        "Company": result["company"],
                        "Score": result["score"],
                        "Recommendation": result["recommendation"],
                        "Time Horizon": result["time_horizon"],
                        "Entry Price": entry,
                        "Target Price": result["target_price"],
                        "Stop Loss": result["stop_loss"],
                        "Current Price": current,
                        "% from Entry": round(pct, 2) if pct is not None else None,
                        **({"Index": label} if label else {}),
                    }
                )
        except Exception:
            pass
        if progress_callback:
            progress_callback(i / len(tickers))
    results = pd.DataFrame(rows).sort_values("Score", ascending=False).head(max_results) if rows else pd.DataFrame()
    results.attrs["source_ticker_count"] = len(tickers)
    results.attrs["universe"] = universe
    results.attrs["fallback_used"] = False
    logger.info("Completed screener | universe=%s | screened=%s | matched=%s", universe, len(tickers), len(results))
    return results
