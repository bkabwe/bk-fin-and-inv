from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from modules.env import load_environment
from modules.logger import get_logger
from modules.sec_edgar_client import get_fundamentals_info_adapter
from modules.validators import sanitize_ticker

load_environment()

logger = get_logger(__name__)

POLYGON_API_BASE_URL = "https://api.polygon.io"
POLYGON_API_KEY_ENV = "POLYGON_API_KEY"
MAX_LOOKBACK_YEARS = 5
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_BASE_SECONDS = 1.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}


class PolygonNotConfiguredError(RuntimeError):
    """Raised when Polygon access is requested without POLYGON_API_KEY configured."""


try:
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



def _sanitize_polygon_symbol(ticker: str) -> str:
    value = str(ticker or "").strip().upper()
    if value.startswith("I:"):
        suffix = sanitize_ticker(value[2:])
        return f"I:{suffix}"
    return sanitize_ticker(value)


def is_polygon_configured() -> bool:
    return bool(os.getenv(POLYGON_API_KEY_ENV, "").strip())


def _api_key() -> str:
    key = os.getenv(POLYGON_API_KEY_ENV, "").strip()
    if not key:
        raise PolygonNotConfiguredError(f"{POLYGON_API_KEY_ENV} is not configured")
    return key


def _status_code_from_exception(exc: Exception) -> int | None:
    response = None
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
    elif isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
    if response is None:
        return None
    return int(response.status_code) if response.status_code is not None else None


def _retry_after_delay_from_exception(exc: Exception) -> float | None:
    response = None
    if isinstance(exc, requests.exceptions.HTTPError):
        response = exc.response
    elif isinstance(exc, requests.exceptions.RequestException):
        response = getattr(exc, "response", None)
    if response is None:
        return None
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        delay = float(retry_after)
        return delay if delay > 0 else None
    except Exception:
        return None


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    status_code = _status_code_from_exception(exc)
    if status_code is None:
        return False
    return status_code in _RETRYABLE_STATUS_CODES or status_code >= 500


def _fetch_with_retry(fetch_fn, max_retries: int = REQUEST_MAX_RETRIES, base_delay: float = REQUEST_BACKOFF_BASE_SECONDS):
    for attempt in range(max_retries):
        try:
            return fetch_fn()
        except Exception as exc:
            status_code = _status_code_from_exception(exc)
            retryable = _is_retryable_exception(exc)
            if not retryable:
                if status_code in _NON_RETRYABLE_STATUS_CODES:
                    logger.warning(
                        "Polygon request failed with non-retryable status %s (entitlement/permissions or client issue) — not retrying",
                        status_code,
                    )
                raise
            if attempt >= max_retries - 1:
                raise
            retry_after_delay = _retry_after_delay_from_exception(exc) if status_code == 429 else None
            delay = retry_after_delay if retry_after_delay is not None else (base_delay * (2**attempt))
            logger.warning(
                "Polygon fetch attempt %s/%s failed: %s. Retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    return None


def _request_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    params["apiKey"] = _api_key()

    def _fetch():
        response = requests.get(f"{POLYGON_API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            status = str(payload.get("status") or "").upper()
            if status in {"ERROR", "NOT_AUTHORIZED", "UNAUTHORIZED", "FORBIDDEN"}:
                raise RuntimeError(payload.get("error") or payload.get("message") or payload.get("status") or "Polygon API error")
        return payload

    return _fetch_with_retry(_fetch) or {}


def _parse_period_to_days(period: str) -> int:
    value = str(period or "1y").strip().lower()
    if value == "max":
        raise ValueError(f"period='max' is not supported on Polygon Starter. Use <= {MAX_LOOKBACK_YEARS}y.")
    mapping = {
        "1d": 1,
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 182,
        "1y": 365,
        "2y": 730,
        "3y": 1095,
        "5y": 1825,
    }
    if value in mapping:
        return mapping[value]
    try:
        if value.endswith("y"):
            parsed = int(value[:-1]) * 365
            if parsed > (MAX_LOOKBACK_YEARS * 365):
                raise ValueError(f"period exceeds Polygon Starter lookback ({MAX_LOOKBACK_YEARS}y): {value}")
            return parsed
        if value.endswith("mo"):
            parsed = int(value[:-2]) * 30
            if parsed > (MAX_LOOKBACK_YEARS * 365):
                raise ValueError(f"period exceeds Polygon Starter lookback ({MAX_LOOKBACK_YEARS}y): {value}")
            return parsed
        if value.endswith("d"):
            parsed = int(value[:-1])
            if parsed > (MAX_LOOKBACK_YEARS * 365):
                raise ValueError(f"period exceeds Polygon Starter lookback ({MAX_LOOKBACK_YEARS}y): {value}")
            return parsed
    except ValueError:
        raise
    except Exception:
        pass
    raise ValueError(f"Unsupported period value: {period}")


def _interval_to_polygon(interval: str) -> tuple[int, str]:
    value = str(interval or "1d").strip().lower()
    mapping = {
        "1d": (1, "day"),
        "1wk": (1, "week"),
        "1mo": (1, "month"),
        "1h": (1, "hour"),
        "30m": (30, "minute"),
        "15m": (15, "minute"),
        "5m": (5, "minute"),
        "1m": (1, "minute"),
    }
    return mapping.get(value, (1, "day"))


def _lookback_floor(today: date | None = None) -> date:
    now = today or date.today()
    return now - timedelta(days=MAX_LOOKBACK_YEARS * 365)


def _clamp_from_date(from_date: date, to_date: date) -> date:
    floor = _lookback_floor(to_date)
    return max(from_date, floor)


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def _coerce_aggregate_value(value: Any) -> float:
    numeric = pd.to_numeric(value, errors="coerce")
    return float(numeric) if pd.notna(numeric) else float("nan")


@cache_data(ttl=3600)
def get_aggregates_range(
    ticker: str,
    *,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    adjusted: bool = True,
) -> pd.DataFrame:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    payload = _request_json(
        f"/v2/aggs/ticker/{clean_ticker}/range/{int(multiplier)}/{timespan}/{from_date}/{to_date}",
        {
            "adjusted": str(bool(adjusted)).lower(),
            "sort": "asc",
            "limit": 50000,
        },
    )
    results = payload.get("results") or []
    if not results:
        return _empty_ohlcv()

    rows = []
    for item in results:
        rows.append(
            {
                "Date": pd.to_datetime(item.get("t"), unit="ms", utc=True, errors="coerce"),
                "Open": _coerce_aggregate_value(item.get("o")),
                "High": _coerce_aggregate_value(item.get("h")),
                "Low": _coerce_aggregate_value(item.get("l")),
                "Close": _coerce_aggregate_value(item.get("c")),
                "Volume": _coerce_aggregate_value(item.get("v")),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")


@cache_data(ttl=3600)
def get_stock_data_polygon(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    to_date = date.today()
    days = _parse_period_to_days(period)
    from_date = _clamp_from_date(to_date - timedelta(days=max(days, 1)), to_date)
    multiplier, timespan = _interval_to_polygon(interval)
    return get_aggregates_range(
        clean_ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        adjusted=True,
    )


@cache_data(ttl=900)
def get_previous_close(ticker: str) -> float | None:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    payload = _request_json(f"/v2/aggs/ticker/{clean_ticker}/prev", {"adjusted": "true"})
    results = payload.get("results") or []
    if not results:
        return None
    try:
        return float(results[0].get("c"))
    except Exception:
        return None


@cache_data(ttl=3600)
def get_ticker_overview(ticker: str) -> dict[str, Any]:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    payload = _request_json(f"/v3/reference/tickers/{clean_ticker}")
    return payload.get("results") or {}


@cache_data(ttl=3600)
def get_reference_dividends(ticker: str, limit: int = 20) -> list[dict[str, Any]]:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    payload = _request_json("/v3/reference/dividends", {"ticker": clean_ticker, "limit": int(limit), "order": "desc", "sort": "ex_dividend_date"})
    values = payload.get("results") or []
    return values if isinstance(values, list) else []


@cache_data(ttl=3600)
def get_reference_splits(ticker: str, execution_date_gte: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    total_limit = int(max(1, limit))
    params: dict[str, Any] = {
        "ticker": clean_ticker,
        "limit": int(min(total_limit, 1000)),
        "order": "asc",
        "sort": "execution_date",
    }
    if execution_date_gte:
        params["execution_date.gte"] = execution_date_gte

    results: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        call_params = dict(params)
        if cursor:
            call_params["cursor"] = cursor
        payload = _request_json("/v3/reference/splits", call_params)
        page_values = payload.get("results") or []
        if isinstance(page_values, list) and page_values:
            results.extend(page_values)
            if len(results) >= total_limit:
                return results[:total_limit]
        next_url = payload.get("next_url")
        if not next_url:
            break
        cursor = None
        if "cursor=" in str(next_url):
            cursor = str(next_url).split("cursor=", 1)[-1].split("&", 1)[0]
        if not cursor:
            break
    return results[:total_limit]


@cache_data(ttl=900)
def get_news_polygon(ticker: str, limit: int = 20) -> list[dict[str, Any]]:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    payload = _request_json("/v2/reference/news", {"ticker": clean_ticker, "limit": int(limit), "order": "desc", "sort": "published_utc"})
    values = payload.get("results") or []
    return values if isinstance(values, list) else []


@cache_data(ttl=3600)
def get_indicator_series(
    indicator: str,
    ticker: str,
    *,
    window: int | None = None,
    timespan: str = "day",
    series_type: str = "close",
) -> pd.DataFrame:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    params: dict[str, Any] = {
        "timespan": timespan,
        "series_type": series_type,
        "adjusted": "true",
        "order": "asc",
        "limit": 5000,
    }
    if window is not None:
        params["window"] = int(window)

    payload = _request_json(f"/v1/indicators/{indicator}/{clean_ticker}", params)
    results = ((payload.get("results") or {}).get("values")) or []
    if not results:
        return pd.DataFrame()

    rows = []
    for item in results:
        timestamp = item.get("timestamp")
        value = item.get("value")
        if timestamp is None or value is None:
            continue
        dt = pd.to_datetime(timestamp, unit="ms", utc=True, errors="coerce")
        if pd.isna(dt):
            dt = pd.to_datetime(timestamp, utc=True, errors="coerce")
        if pd.isna(dt):
            continue
        rows.append({"Date": dt, "value": float(value)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df


@cache_data(ttl=86400)
def list_active_ticker_details(
    *,
    primary_exchange: str | None = None,
    market: str = "stocks",
    otc: bool | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "market": market,
        "active": "true",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
    }
    if primary_exchange:
        params["primary_exchange"] = primary_exchange
    if otc is not None:
        params["otc"] = str(bool(otc)).lower()

    tickers: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor: str | None = None

    while True:
        call_params = dict(params)
        if cursor:
            call_params["cursor"] = cursor
        payload = _request_json("/v3/reference/tickers", call_params)
        rows = payload.get("results") or []
        if not rows:
            break

        for row in rows:
            symbol = str(row.get("ticker") or "").strip().upper()
            if not symbol:
                continue
            if otc is True and "OTC" not in str(row.get("primary_exchange") or "").upper():
                continue
            if otc is False and "OTC" in str(row.get("primary_exchange") or "").upper():
                continue
            if symbol in seen:
                continue
            seen.add(symbol)
            tickers.append(
                {
                    "ticker": symbol,
                    "primary_exchange": row.get("primary_exchange"),
                    "type": row.get("type"),
                    "market": row.get("market"),
                    "locale": row.get("locale"),
                    "name": row.get("name"),
                    "active": row.get("active"),
                    "currency_name": row.get("currency_name"),
                }
            )
            if limit and len(tickers) >= limit:
                return tickers[:limit]

        next_url = payload.get("next_url")
        if not next_url:
            break
        cursor = None
        if "cursor=" in str(next_url):
            cursor = str(next_url).split("cursor=", 1)[-1].split("&", 1)[0]
        if not cursor:
            break

    return tickers[:limit] if limit else tickers


@cache_data(ttl=86400)
def list_active_tickers(
    *,
    primary_exchange: str | None = None,
    market: str = "stocks",
    otc: bool | None = None,
    limit: int | None = None,
) -> list[str]:
    rows = list_active_ticker_details(
        primary_exchange=primary_exchange,
        market=market,
        otc=otc,
        limit=limit,
    )
    return [str(row.get("ticker") or "").strip().upper() for row in rows if str(row.get("ticker") or "").strip()]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


@cache_data(ttl=1800)
def build_info_adapter(ticker: str) -> dict[str, Any]:
    """Return a legacy info-schema-compatible dict populated from Polygon endpoints.

    Field mapping notes:
    - trailingPE/trailingEps/debtToEquity/returnOnEquity/growth metrics: computed from SEC EDGAR filings
      via modules.sec_edgar_client and mapped into Yahoo-compatible field names
    - forwardPE/forwardEps/analyst fields: intentionally unavailable from SEC filings and left None
    - currentPrice: previous-day close from /v2/aggs/ticker/{ticker}/prev (Starter plan constraint)
    """
    clean_ticker = _sanitize_polygon_symbol(ticker)
    overview = get_ticker_overview(clean_ticker)
    current_price = get_previous_close(clean_ticker)
    sec_fundamentals = get_fundamentals_info_adapter(clean_ticker, current_price=current_price)

    trailing_year = date.today() - timedelta(days=365)
    high_52w = _safe_float(overview.get("fifty_two_week_high") or overview.get("52_week_high"))
    low_52w = _safe_float(overview.get("fifty_two_week_low") or overview.get("52_week_low"))

    dividend_yield = None
    if current_price and current_price > 0:
        dividends = get_reference_dividends(clean_ticker, limit=40)
        trailing_divs = 0.0
        for event in dividends:
            ex_date = str(event.get("ex_dividend_date") or "")
            amount = _safe_float(event.get("cash_amount"))
            if not ex_date or amount is None:
                continue
            try:
                if datetime.fromisoformat(ex_date).date() >= trailing_year:
                    trailing_divs += amount
            except Exception:
                continue
        if trailing_divs > 0:
            dividend_yield = trailing_divs / current_price

    info = {
        "symbol": clean_ticker,
        "shortName": overview.get("name") or clean_ticker,
        "longName": overview.get("name") or clean_ticker,
        "sector": overview.get("sic_description") or overview.get("sector"),
        "marketCap": _safe_float(overview.get("market_cap")),
        "exchange": overview.get("primary_exchange"),
        "fullExchangeName": overview.get("primary_exchange"),
        "currentPrice": current_price,
        "regularMarketPrice": current_price,
        "targetMeanPrice": None,
        "recommendationKey": None,
        "trailingPE": sec_fundamentals.get("trailingPE"),
        "forwardPE": sec_fundamentals.get("forwardPE"),
        "trailingEps": sec_fundamentals.get("trailingEps"),
        "forwardEps": sec_fundamentals.get("forwardEps"),
        "earningsGrowth": sec_fundamentals.get("earningsGrowth"),
        "revenueGrowth": sec_fundamentals.get("revenueGrowth"),
        "debtToEquity": sec_fundamentals.get("debtToEquity"),
        "returnOnEquity": sec_fundamentals.get("returnOnEquity"),
        "fiftyTwoWeekHigh": high_52w,
        "fiftyTwoWeekLow": low_52w,
        "dividendYield": dividend_yield,
    }
    return info
