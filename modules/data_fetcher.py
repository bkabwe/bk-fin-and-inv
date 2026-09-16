from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd
import requests

from modules.logger import get_logger
from modules.polygon_client import (
    PolygonNotConfiguredError,
    build_info_adapter,
    get_news_polygon,
    get_stock_data_polygon,
    is_polygon_configured,
    list_active_ticker_details,
    list_active_tickers,
)
from modules.validators import sanitize_ticker

logger = get_logger(__name__)

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    import threading as _threading
    import time as _time

    def cache_data(ttl: int | None = None):  # type: ignore[misc]
        """TTL-aware, stampede-safe cache fallback for non-Streamlit deployments."""

        def decorator(func):
            _cache: dict = {}
            _inflight: dict = {}
            _lock = _threading.Lock()

            def wrapper(*args, **kwargs):
                key = (args, tuple(sorted(kwargs.items())))
                while True:
                    now = _time.monotonic()
                    with _lock:
                        entry = _cache.get(key)
                        if entry is not None:
                            value, ts = entry
                            if ttl is None or (now - ts) < ttl:
                                return value
                        event = _inflight.get(key)
                        if event is None:
                            ev = _threading.Event()
                            _inflight[key] = ev
                            break
                    event.wait(timeout=300)

                try:
                    result = func(*args, **kwargs)
                    with _lock:
                        _cache[key] = (result, _time.monotonic())
                    return result
                finally:
                    with _lock:
                        ev = _inflight.pop(key, None)
                    if ev is not None:
                        ev.set()

            def clear() -> None:
                """Mirror `st.cache_data`'s `.clear()` so callers (including
                tests) can reset this cache the same way regardless of
                whether Streamlit is installed."""
                with _lock:
                    _cache.clear()
                    _inflight.clear()

            wrapper.clear = clear
            return wrapper

        return decorator


@cache_data(ttl=3600)
def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker = sanitize_ticker(ticker)
    except ValueError as exc:
        logger.warning("Invalid ticker in get_stock_data: %s", exc)
        return pd.DataFrame()

    try:
        data = get_stock_data_polygon(ticker, period=period, interval=interval)
    except PolygonNotConfiguredError as exc:
        logger.warning("Polygon not configured for get_stock_data(%s): %s", ticker, exc)
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Failed to fetch stock data for %s: %s", ticker, exc)
        return pd.DataFrame()

    return data.dropna(how="all") if isinstance(data, pd.DataFrame) else pd.DataFrame()


@cache_data(ttl=3600)
def get_stock_info(ticker: str) -> dict[str, Any]:
    try:
        ticker = sanitize_ticker(ticker)
    except ValueError as exc:
        logger.warning("Invalid ticker in get_stock_info: %s", exc)
        return {}

    try:
        info = build_info_adapter(ticker)
    except PolygonNotConfiguredError as exc:
        logger.warning("Polygon not configured for get_stock_info(%s): %s", ticker, exc)
        return {}
    except Exception as exc:
        logger.error("Failed to fetch stock info for %s: %s", ticker, exc)
        return {}

    return info if isinstance(info, dict) else {}


@cache_data(ttl=900)
def get_news(ticker: str) -> list[dict[str, Any]]:
    try:
        ticker = sanitize_ticker(ticker)
    except ValueError as exc:
        logger.warning("Invalid ticker in get_news: %s", exc)
        return []

    try:
        return get_news_polygon(ticker) or []
    except PolygonNotConfiguredError as exc:
        logger.warning("Polygon not configured for get_news(%s): %s", ticker, exc)
        return []
    except Exception as exc:
        logger.warning("Failed to fetch Polygon news for %s: %s", ticker, exc)
        return []


_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _normalize_tickers(values: list[Any], limit: int | None = None) -> list[str]:
    cleaned = []
    for value in values:
        ticker = str(value).strip().upper().replace(".", "-")
        if ticker and ticker not in {"NAN", "NONE", "SYMBOL"} and ticker.isascii():
            cleaned.append(ticker)
    unique = sorted(set(cleaned))
    return unique[:limit] if limit is not None else unique


@cache_data(ttl=86400)
def get_all_active_ticker_details() -> list[dict[str, Any]]:
    if not is_polygon_configured():
        raise RuntimeError("POLYGON_API_KEY is not configured. Set it to fetch active tickers.")
    try:
        tickers = list_active_ticker_details()
    except PolygonNotConfiguredError as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch active tickers from Polygon: {exc}") from exc
    if not tickers:
        raise RuntimeError("Polygon returned no active tickers")
    return tickers


@cache_data(ttl=86400)
def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 constituents via Wikipedia table (index-membership source).

    KNOWN LIMITATION (survivorship bias): this returns *current* index
    membership only. Both LightGBM training and walk-forward backtests that
    sample from this universe therefore only ever see tickers still in the
    index today, which can inflate apparent historical accuracy relative to
    what a point-in-time historical membership list would show. Treat
    reported backtest win rates with that caveat in mind; a true fix would
    require sourcing a point-in-time historical membership list.
    """
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=_DEFAULT_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for table in tables:
            cols = {str(c).strip().lower(): c for c in table.columns}
            sym_col = cols.get("symbol") or cols.get("ticker")
            if sym_col:
                tickers = _normalize_tickers(table[sym_col].astype(str).tolist())
                if len(tickers) > 100:
                    return tickers
    except Exception as exc:
        logger.error("Failed to fetch S&P 500 tickers from Wikipedia: %s", exc)
    raise RuntimeError("Failed to fetch S&P 500 tickers")


@cache_data(ttl=86400)
def get_nasdaq_tickers() -> list[str]:
    if not is_polygon_configured():
        raise RuntimeError("POLYGON_API_KEY is not configured. Set it to fetch NASDAQ tickers.")
    try:
        tickers = list_active_tickers(primary_exchange="XNAS", otc=False)
    except PolygonNotConfiguredError as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch NASDAQ tickers from Polygon: {exc}") from exc
    if not tickers:
        raise RuntimeError("Polygon returned no NASDAQ tickers")
    return tickers


@cache_data(ttl=86400)
def get_nyseamerican_tickers() -> list[str]:
    if not is_polygon_configured():
        raise RuntimeError("POLYGON_API_KEY is not configured. Set it to fetch NYSE American tickers.")
    try:
        tickers = list_active_tickers(primary_exchange="XASE", otc=False)
    except PolygonNotConfiguredError as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch NYSE American tickers from Polygon: {exc}") from exc
    if not tickers:
        raise RuntimeError("Polygon returned no NYSE American tickers")
    return tickers


@cache_data(ttl=86400)
def get_otc_tickers() -> list[str]:
    if not is_polygon_configured():
        raise RuntimeError("POLYGON_API_KEY is not configured. Set it to fetch OTC tickers.")
    try:
        tickers = list_active_tickers(otc=True)
    except PolygonNotConfiguredError as exc:
        raise RuntimeError(str(exc)) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch OTC tickers from Polygon: {exc}") from exc
    if not tickers:
        raise RuntimeError("Polygon returned no OTC tickers")
    return tickers
