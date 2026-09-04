from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from modules.logger import get_logger
from modules.validators import sanitize_ticker

logger = get_logger(__name__)

POLYGON_API_BASE_URL = "https://api.polygon.io"
POLYGON_API_KEY_ENV = "POLYGON_API_KEY"
MAX_LOOKBACK_YEARS = 5
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_BASE_SECONDS = 1.0


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


def _fetch_with_retry(fetch_fn, max_retries: int = REQUEST_MAX_RETRIES, base_delay: float = REQUEST_BACKOFF_BASE_SECONDS):
    for attempt in range(max_retries):
        try:
            return fetch_fn()
        except Exception as exc:
            if attempt >= max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
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
        if isinstance(payload, dict) and payload.get("status") == "ERROR":
            raise RuntimeError(payload.get("error") or payload.get("message") or "Polygon API error")
        return payload

    return _fetch_with_retry(_fetch) or {}


def _parse_period_to_days(period: str) -> int:
    value = str(period or "1y").strip().lower()
    if value == "max":
        return MAX_LOOKBACK_YEARS * 365
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
        "10y": 3650,
    }
    if value in mapping:
        return mapping[value]
    try:
        if value.endswith("y"):
            return int(value[:-1]) * 365
        if value.endswith("mo"):
            return int(value[:-2]) * 30
        if value.endswith("d"):
            return int(value[:-1])
    except Exception:
        pass
    return 365


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
                "Open": float(item.get("o", 0) or 0),
                "High": float(item.get("h", 0) or 0),
                "Low": float(item.get("l", 0) or 0),
                "Close": float(item.get("c", 0) or 0),
                "Volume": float(item.get("v", 0) or 0),
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


def _extract_latest_record(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("results", "data"):
        values = payload.get(key)
        if isinstance(values, list) and values:
            return values[0] or {}
        if isinstance(values, dict):
            return values
    return payload if isinstance(payload, dict) else {}


@cache_data(ttl=3600)
def get_financial_ratios(ticker: str) -> dict[str, Any]:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    payload = _request_json("/stocks/financials/v1/ratios", {"ticker": clean_ticker, "limit": 1})
    return _extract_latest_record(payload)


@cache_data(ttl=3600)
def get_income_statements(ticker: str, limit: int = 6) -> list[dict[str, Any]]:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    payload = _request_json("/stocks/financials/v1/income-statements", {"ticker": clean_ticker, "limit": int(limit)})
    values = payload.get("results") or payload.get("data") or []
    return values if isinstance(values, list) else []


@cache_data(ttl=3600)
def get_reference_dividends(ticker: str, limit: int = 20) -> list[dict[str, Any]]:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    payload = _request_json("/v3/reference/dividends", {"ticker": clean_ticker, "limit": int(limit), "order": "desc", "sort": "ex_dividend_date"})
    values = payload.get("results") or []
    return values if isinstance(values, list) else []


@cache_data(ttl=3600)
def get_reference_splits(ticker: str, execution_date_gte: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    clean_ticker = _sanitize_polygon_symbol(ticker)
    params: dict[str, Any] = {
        "ticker": clean_ticker,
        "limit": int(limit),
        "order": "asc",
        "sort": "execution_date",
    }
    if execution_date_gte:
        params["execution_date.gte"] = execution_date_gte
    payload = _request_json("/v3/reference/splits", params)
    values = payload.get("results") or []
    return values if isinstance(values, list) else []


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
def list_active_tickers(
    *,
    primary_exchange: str | None = None,
    market: str = "stocks",
    otc: bool | None = None,
    limit: int | None = None,
) -> list[str]:
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
        params["locale"] = "us"

    tickers: list[str] = []
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
            tickers.append(symbol)
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


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _extract_ratio(record: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict):
            for nested in ("value", "amount", "ratio"):
                if nested in value:
                    parsed = _safe_float(value.get(nested))
                    if parsed is not None:
                        return parsed
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    return None


def _compute_growth_from_income(income_rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    if len(income_rows) < 2:
        return None, None

    def _metric(record: dict[str, Any], keys: list[str]) -> float | None:
        return _extract_ratio(record, keys)

    current = income_rows[0]
    prior = income_rows[1]

    rev_cur = _metric(current, ["revenue", "total_revenue", "revenues"])
    rev_prev = _metric(prior, ["revenue", "total_revenue", "revenues"])
    eps_cur = _metric(current, ["diluted_eps", "basic_eps", "eps", "earnings_per_share"])
    eps_prev = _metric(prior, ["diluted_eps", "basic_eps", "eps", "earnings_per_share"])

    revenue_growth = None
    earnings_growth = None
    if rev_cur is not None and rev_prev is not None and abs(rev_prev) > 1e-9:
        revenue_growth = (rev_cur - rev_prev) / abs(rev_prev)
    if eps_cur is not None and eps_prev is not None and abs(eps_prev) > 1e-9:
        earnings_growth = (eps_cur - eps_prev) / abs(eps_prev)

    return revenue_growth, earnings_growth


@cache_data(ttl=1800)
def build_info_adapter(ticker: str) -> dict[str, Any]:
    """Return a legacy info-schema-compatible dict populated from Polygon endpoints.

    Field mapping notes:
    - trailingPE/forwardPE: sourced from ratios endpoint (closest available P/E fields)
    - trailingEps/forwardEps: sourced from ratios + income statements EPS fields
    - debtToEquity: debt/equity ratio from Polygon converted to Yahoo-like percent scale
    - returnOnEquity: ROE ratio from Polygon (decimal)
    - earningsGrowth/revenueGrowth: derived from latest two income statements when not explicit
    - currentPrice: previous-day close from /v2/aggs/ticker/{ticker}/prev (Starter plan constraint)
    """
    clean_ticker = _sanitize_polygon_symbol(ticker)
    overview = get_ticker_overview(clean_ticker)
    ratios = get_financial_ratios(clean_ticker)
    income_rows = get_income_statements(clean_ticker)
    current_price = get_previous_close(clean_ticker)

    revenue_growth, earnings_growth = _compute_growth_from_income(income_rows)

    trailing_pe = _extract_ratio(ratios, ["price_to_earnings_ratio", "pe_ratio", "trailing_pe", "price_earnings"])
    forward_pe = _extract_ratio(ratios, ["forward_pe_ratio", "forward_pe"])
    trailing_eps = _extract_ratio(ratios, ["earnings_per_share", "eps", "basic_eps", "diluted_eps"])
    forward_eps = _extract_ratio(ratios, ["forward_eps", "projected_eps", "estimated_eps"])
    debt_to_equity_raw = _extract_ratio(ratios, ["debt_to_equity_ratio", "debt_equity_ratio", "debt_to_equity"])
    roe = _extract_ratio(ratios, ["return_on_equity", "return_on_equity_ratio", "roe"])

    if earnings_growth is None:
        earnings_growth = _extract_ratio(ratios, ["earnings_growth", "net_income_growth", "eps_growth"])
    if revenue_growth is None:
        revenue_growth = _extract_ratio(ratios, ["revenue_growth", "sales_growth"])

    trailing_year = date.today() - timedelta(days=365)
    try:
        last_year = get_stock_data_polygon(clean_ticker, period="1y", interval="1d")
    except Exception:
        last_year = _empty_ohlcv()
    high_52w = float(last_year["High"].max()) if not last_year.empty and "High" in last_year else None
    low_52w = float(last_year["Low"].min()) if not last_year.empty and "Low" in last_year else None

    dividend_yield = _extract_ratio(ratios, ["dividend_yield", "dividend_yield_ttm"])
    if dividend_yield is None and current_price and current_price > 0:
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

    # Legacy debtToEquity consumers expect a percent-like scale (e.g., 150 == 1.5x)
    debt_to_equity = debt_to_equity_raw * 100.0 if debt_to_equity_raw is not None and debt_to_equity_raw < 20 else debt_to_equity_raw

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
        "trailingPE": trailing_pe,
        "forwardPE": forward_pe,
        "trailingEps": trailing_eps,
        "forwardEps": forward_eps,
        "earningsGrowth": earnings_growth,
        "revenueGrowth": revenue_growth,
        "debtToEquity": debt_to_equity,
        "returnOnEquity": roe,
        "fiftyTwoWeekHigh": high_52w,
        "fiftyTwoWeekLow": low_52w,
        "dividendYield": dividend_yield,
    }
    return info
