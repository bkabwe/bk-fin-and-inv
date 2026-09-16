"""Shared "profit opportunities" scanning logic.

This module consolidates the per-ticker fast-screen + full-analysis +
target-date-estimation logic that was previously duplicated (with slowly
diverging behaviour) across three call sites:

* ``pages/6_Profit_Opportunities.py`` (interactive Streamlit page)
* ``scripts/scan_email_report.py`` (scheduled CI scan -> email report)
* ``api/routes/profit.py`` (FastAPI/Celery background task for the React UI)

Only the Streamlit page previously computed an ARIMA/trend-based
``estimate_target_date()``; the other two skipped it entirely. All three
now share the same horizon configuration, ticker-universe collection, and
per-ticker analysis helpers defined here.
"""

from __future__ import annotations

import threading
import time
from datetime import date, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from modules.data_fetcher import (
    get_nasdaq_tickers,
    get_nyseamerican_tickers,
    get_otc_tickers,
    get_sp500_tickers,
    get_stock_data,
)
from modules.logger import get_logger
from modules.scoring_engine import analyze_stock, fast_screen_score
from modules.validators import sanitize_ticker_list

logger = get_logger(__name__)

try:  # pragma: no cover - exercised implicitly wherever Streamlit runs
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover

    def cache_data(ttl: int | None = None):  # type: ignore[misc]
        """TTL-aware, stampede-safe cache fallback for non-Streamlit usage."""

        def decorator(func):
            _cache: dict = {}
            _inflight: dict = {}
            _lock = threading.Lock()

            def wrapper(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                while True:
                    now = time.monotonic()
                    with _lock:
                        entry = _cache.get(key)
                        if entry is not None:
                            value, ts = entry
                            if ttl is None or (now - ts) < ttl:
                                return value
                        event = _inflight.get(key)
                        if event is None:
                            ev = threading.Event()
                            _inflight[key] = ev
                            break
                    event.wait(timeout=300)

                try:
                    result = func(*args, **kwargs)
                    with _lock:
                        _cache[key] = (result, time.monotonic())
                    return result
                finally:
                    with _lock:
                        ev = _inflight.pop(key, None)
                    if ev is not None:
                        ev.set()

            return wrapper

        return decorator


try:  # pragma: no cover
    from statsmodels.tsa.arima.model import ARIMA

    ARIMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    ARIMA_AVAILABLE = False

try:  # pragma: no cover
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Horizon configuration (canonical keys shared by all callers)
# ---------------------------------------------------------------------------
HORIZON_SETTINGS: dict[str, dict[str, Any]] = {
    "short_term": {
        "label": "Short-Term (1–4 weeks)",
        "target_key": "short_term_target",
        "target_low_key": "short_term_low",
        "target_high_key": "short_term_high",
        "upside_key": "short_term_upside",
        "basis_key": "short_term_basis",
        "date_windows": {"High": 4, "Medium": 6, "Low": 8},
    },
    "medium_term": {
        "label": "Medium-Term (1–6 months)",
        "target_key": "medium_term_target",
        "target_low_key": "medium_term_low",
        "target_high_key": "medium_term_high",
        "upside_key": "medium_term_upside",
        "basis_key": "medium_term_basis",
        "date_windows": {"High": 14, "Medium": 21, "Low": 30},
    },
    "long_term": {
        "label": "Long-Term (6–24 months)",
        "target_key": "long_term_target",
        "target_low_key": "long_term_low",
        "target_high_key": "long_term_high",
        "upside_key": "long_term_upside",
        "basis_key": "long_term_basis",
        "date_windows": {"High": 30, "Medium": 45, "Low": 60},
    },
}

# Reverse lookup for callers still working with Streamlit's human-readable
# radio-button labels (e.g. "Short-Term (1–4 weeks)") instead of canonical
# horizon keys (e.g. "short_term").
HORIZON_LABEL_TO_KEY: dict[str, str] = {v["label"]: k for k, v in HORIZON_SETTINGS.items()}

_UNIVERSE_FETCHERS: dict[str, Callable[[], list[str]]] = {
    "S&P 500": get_sp500_tickers,
    "sp500": get_sp500_tickers,
    "NASDAQ": get_nasdaq_tickers,
    "nasdaq": get_nasdaq_tickers,
    "NYSE American": get_nyseamerican_tickers,
    "nyseamerican": get_nyseamerican_tickers,
    "OTC": get_otc_tickers,
    "otc": get_otc_tickers,
}

DEFAULT_MAX_WORKERS = 8
DEFAULT_FAST_SCREEN_BASE = 30
DEFAULT_FAST_SCREEN_MARGIN = 15
DEFAULT_MIN_UPSIDE_PCT = 15.0


def collect_universe_tickers(
    universes: list[str],
    custom_tickers: list[str] | None = None,
    portfolio_tickers: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Resolve a list of universe names into a de-duplicated ticker list plus
    a ``ticker -> "Universe A, Universe B"`` source-label map.

    ``universes`` entries may be display names (``"S&P 500"``, ``"My
    Portfolio"``) or lowercase keys (``"sp500"``, ``"custom"``).
    """
    source_map: dict[str, set[str]] = {}
    ordered: list[str] = []

    def _add(raw_ticker: str, label: str) -> None:
        clean = str(raw_ticker).strip().upper().replace(".", "-")
        if not clean:
            return
        if clean not in source_map:
            source_map[clean] = set()
            ordered.append(clean)
        source_map[clean].add(label)

    for universe_name in universes:
        if universe_name in ("My Portfolio", "portfolio"):
            for ticker in portfolio_tickers or []:
                _add(ticker, universe_name)
            continue
        if universe_name == "custom":
            for ticker in sanitize_ticker_list(custom_tickers or []):
                _add(ticker, universe_name)
            continue
        fetcher = _UNIVERSE_FETCHERS.get(universe_name)
        if not fetcher:
            continue
        try:
            values = fetcher()
        except Exception as exc:
            logger.warning("Could not load %s universe: %s", universe_name, exc)
            continue
        for ticker in values:
            _add(ticker, universe_name)

    return ordered, {k: ", ".join(sorted(v)) for k, v in source_map.items()}


def analyze_ticker_for_horizon(
    ticker: str,
    horizon: str,
    *,
    use_fast_screen: bool = True,
    fast_screen_threshold: float = DEFAULT_FAST_SCREEN_BASE - DEFAULT_FAST_SCREEN_MARGIN,
    price_data: pd.DataFrame | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Run the fast-screen pre-filter (optional) + full stock analysis for a
    single ticker/horizon pair.

    Returns a status dict:
    ``{"row": dict | None, "fast_filtered": bool, "fully_analyzed": bool,
       "failed": bool, "reason": str | None}``.

    ``row`` is populated whenever ``analyze_stock`` succeeds and yields a
    positive current price — *regardless* of whether the projected upside
    clears any particular minimum threshold. Callers decide whether/when to
    apply a minimum-upside filter: batch jobs (email report, API task) apply
    it immediately via :func:`filter_by_upside`, while the Streamlit page
    re-applies it at display time so users can adjust the threshold without
    triggering a full re-scan.
    """
    settings = HORIZON_SETTINGS[horizon]
    outcome: dict[str, Any] = {"row": None, "fast_filtered": False, "fully_analyzed": False, "failed": False, "reason": None}
    try:
        if use_fast_screen:
            fast_score, err = fast_screen_score(ticker, data=price_data)
            if err is not None:
                outcome["failed"] = True
                outcome["reason"] = f"fast-screen error: {err}"
                return outcome
            if fast_score < fast_screen_threshold:
                outcome["fast_filtered"] = True
                return outcome

        analyze_kwargs: dict[str, Any] = {"investment_horizon": horizon}
        if price_data is not None:
            analyze_kwargs["data_override"] = price_data
            analyze_kwargs["projection_data_override"] = price_data
        analysis = analyze_stock(ticker, **analyze_kwargs)
        outcome["fully_analyzed"] = True

        projections = analysis.get("projections") or {}
        current_price = float(projections.get("current_price") or analysis.get("current_price") or 0)
        if current_price <= 0:
            return outcome

        target_price = float(projections.get(settings["target_key"]) or 0)
        upside = float(projections.get(settings["upside_key"]) or 0)
        rsi = (analysis.get("technical") or {}).get("indicators", {}).get("rsi")

        outcome["row"] = {
            "Ticker": ticker,
            "Company": analysis.get("company") or ticker,
            "Score": int(analysis.get("score") or 0),
            "Current Price": round(current_price, 4),
            "Target Price": round(target_price, 4),
            "Target Low": round(float(projections.get(settings["target_low_key"]) or target_price * 0.95), 4),
            "Target High": round(float(projections.get(settings["target_high_key"]) or target_price * 1.05), 4),
            "Projected Upside %": round(upside, 4),
            "Sector Trend": str(analysis.get("sector_trend") or "unknown").replace("_", " ").title(),
            "Market Cap Tier": analysis.get("market_cap_tier") or "unknown",
            "Long-Term Stage": analysis.get("longterm_stage") or "",
            "Confidence": str(projections.get("data_quality") or "Limited"),
            "Basis": str(projections.get(settings["basis_key"]) or ""),
            "_rsi": rsi,
            **({"Index": label} if label else {}),
        }
        return outcome
    except Exception as exc:
        outcome["failed"] = True
        outcome["reason"] = str(exc) or type(exc).__name__
        return outcome


def scan_profit_opportunities(
    tickers: list[str],
    horizon: str,
    *,
    use_fast_screen: bool = True,
    fast_screen_margin: int = DEFAULT_FAST_SCREEN_MARGIN,
    fast_screen_base: int = DEFAULT_FAST_SCREEN_BASE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    parallel: bool = True,
    source_labels: dict[str, str] | None = None,
    prefetch_price_period: str | None = None,
    on_ticker_processed: Callable[[str, dict], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
    on_total_known: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """Scan ``tickers`` for the given horizon and return every candidate with
    a positive current price (upside filtering is left to
    :func:`filter_by_upside`).

    Set ``parallel=False`` for a plain sequential loop (used by the
    scheduled email-report script and the Streamlit "Thorough" mode).
    ``prefetch_price_period`` (e.g. ``"1y"``), when set, fetches OHLCV data
    once per ticker and reuses it for both the fast-screen and full-analysis
    calls, avoiding a duplicate Polygon fetch.
    """
    source_labels = source_labels or {}
    fast_screen_threshold = fast_screen_base - fast_screen_margin
    total = len(tickers)
    if on_total_known:
        on_total_known(total)

    rows: list[dict] = []
    failed_tickers: list[dict] = []
    fast_filtered_count = 0
    fully_analyzed_count = 0
    processed_count = 0
    _lock = threading.Lock()

    def _process(ticker: str) -> None:
        nonlocal fast_filtered_count, fully_analyzed_count, processed_count
        if stop_check is not None and stop_check():
            return
        price_data = None
        if prefetch_price_period:
            try:
                price_data = get_stock_data(ticker, period=prefetch_price_period, interval="1d")
            except Exception:
                price_data = None
        outcome = analyze_ticker_for_horizon(
            ticker,
            horizon,
            use_fast_screen=use_fast_screen,
            fast_screen_threshold=fast_screen_threshold,
            price_data=price_data,
            label=source_labels.get(ticker, ""),
        )
        with _lock:
            if outcome["fast_filtered"]:
                fast_filtered_count += 1
            if outcome["fully_analyzed"]:
                fully_analyzed_count += 1
            if outcome["failed"]:
                failed_tickers.append({"ticker": ticker, "reason": outcome["reason"]})
            if outcome["row"] is not None:
                rows.append(outcome["row"])
            processed_count += 1
            snapshot = {
                "screened": processed_count,
                "total": total,
                "qualified": len(rows),
                "fast_filtered": fast_filtered_count,
                "fully_analyzed": fully_analyzed_count,
                "failed_count": len(failed_tickers),
            }
        if on_ticker_processed:
            on_ticker_processed(ticker, snapshot)

    stopped = False
    if parallel:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process, t): t for t in tickers}
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
    else:
        for ticker in tickers:
            if stop_check is not None and stop_check():
                stopped = True
                break
            _process(ticker)

    results = pd.DataFrame(rows) if rows else pd.DataFrame()
    results.attrs["stopped"] = stopped
    results.attrs["source_ticker_count"] = total
    results.attrs["scanned_count"] = total
    results.attrs["qualified_count"] = len(rows)
    results.attrs["fast_filtered_count"] = fast_filtered_count
    results.attrs["fully_analyzed_count"] = fully_analyzed_count
    results.attrs["failed_count"] = len(failed_tickers)
    results.attrs["failed_tickers"] = failed_tickers
    return results


def filter_by_upside(
    results: pd.DataFrame,
    horizon: str,
    min_upside_pct: float = DEFAULT_MIN_UPSIDE_PCT,
    max_results: int | None = None,
) -> pd.DataFrame:
    """Filter ``scan_profit_opportunities`` output to rows meeting
    ``min_upside_pct``, sorted by projected upside (descending)."""
    if results is None or results.empty or "Projected Upside %" not in results.columns:
        return pd.DataFrame()
    filtered = results[results["Projected Upside %"] >= float(min_upside_pct)]
    filtered = filtered.sort_values("Projected Upside %", ascending=False).reset_index(drop=True)
    if max_results is not None:
        filtered = filtered.head(max_results)
    return filtered.reset_index(drop=True)


@cache_data(ttl=1800)
def estimate_target_date(
    ticker: str,
    target_price: float,
    horizon: str,
    rsi: float | None = None,
) -> tuple[str, str]:
    """Return ``(estimated_date_range, confidence)`` for when ``ticker`` may
    reach ``target_price``, using an ARIMA forecast crossing (falling back to
    a linear trend, then a fixed heuristic) for medium/long-term horizons.

    ``horizon`` is a canonical key (``"short_term"``, ``"medium_term"`` or
    ``"long_term"``).
    """
    settings = HORIZON_SETTINGS[horizon]

    def _format_date_range(center_date: date, confidence: str) -> str:
        span = settings["date_windows"][confidence]
        start = center_date - timedelta(days=span)
        end = center_date + timedelta(days=span)
        return f"~{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"

    today = date.today()
    if horizon == "short_term":
        days = 21
        if rsi is not None:
            if rsi < 50:
                days = 28
            elif rsi > 60:
                days = 14
        est = today + timedelta(days=days)
        return _format_date_range(est, "Medium"), "Medium"

    fallback_days = 90 if horizon == "medium_term" else 365
    max_days = 180 if horizon == "medium_term" else 720

    try:
        data = get_stock_data(ticker, period="2y", interval="1d")
    except Exception:
        data = pd.DataFrame()

    if data.empty or "Close" not in data:
        est = today + timedelta(days=fallback_days)
        return _format_date_range(est, "Low"), "Low"

    close = data["Close"].dropna().astype(float)
    if close.empty:
        est = today + timedelta(days=fallback_days)
        return _format_date_range(est, "Low"), "Low"

    arima_day: int | None = None
    arima_hit = False
    trend_day: int | None = None

    if ARIMA_AVAILABLE and len(close) >= 60:
        try:
            arima_forecast = ARIMA(close.tail(252), order=(5, 1, 0)).fit().forecast(steps=max_days)
            crossing_idx = np.where(arima_forecast.values.astype(float) >= float(target_price))[0]
            arima_hit = len(crossing_idx) > 0
            if arima_hit:
                arima_day = int(crossing_idx[0]) + 1
        except Exception:
            arima_hit = False
            arima_day = None

    if len(close) >= 30:
        try:
            series = close.tail(200).reset_index(drop=True)
            x = np.arange(len(series)).reshape(-1, 1)
            if SKLEARN_AVAILABLE:
                model = LinearRegression()
                model.fit(x, series.values)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
            else:
                slope, intercept = np.polyfit(np.arange(len(series)), series.values, 1)
            if slope > 0:
                day_x = int(round((float(target_price) - intercept) / slope))
                if day_x > len(series):
                    trend_day = day_x - len(series)
        except Exception:
            trend_day = None

    if arima_day is not None:
        est = today + timedelta(days=min(max_days, max(1, arima_day)))
        confidence = "Medium"
        return _format_date_range(est, confidence), confidence

    if trend_day is not None:
        est = today + timedelta(days=min(max_days, max(1, trend_day)))
        confidence = "Medium" if arima_hit else "Low"
        return _format_date_range(est, confidence), confidence

    est = today + timedelta(days=fallback_days)
    confidence = "Medium" if arima_hit else "Low"
    return _format_date_range(est, confidence), confidence


def add_estimated_dates(rows: list[dict], horizon: str) -> list[dict]:
    """Return a copy of ``rows`` with ``Est. Target Date``/``Confidence``
    populated via :func:`estimate_target_date` (overwriting ``Confidence``
    with the date-estimation confidence, matching the Streamlit page's
    historical behaviour)."""
    enriched = []
    for row in rows:
        row = dict(row)
        est_date, confidence = estimate_target_date(row["Ticker"], row.get("Target Price", 0), horizon, row.get("_rsi"))
        row["Est. Target Date"] = est_date
        row["Confidence"] = confidence
        enriched.append(row)
    return enriched
