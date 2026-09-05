from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests

from modules.env import load_environment
from modules.logger import get_logger

load_environment()

logger = get_logger(__name__)

FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
FRED_API_KEY_ENV = "FRED_API_KEY"
FRED_VIX_SERIES_ID = "VIXCLS"
FRED_10Y_TREASURY_SERIES_ID = "DGS10"
FRED_CPI_SERIES_ID = "CPIAUCSL"
FRED_FED_FUNDS_SERIES_ID = "FEDFUNDS"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_BASE_SECONDS = 1.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}


class FredNotConfiguredError(RuntimeError):
    """Raised when FRED access is requested without FRED_API_KEY configured."""


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


def is_fred_configured() -> bool:
    return bool(os.getenv(FRED_API_KEY_ENV, "").strip())


def _api_key() -> str:
    key = os.getenv(FRED_API_KEY_ENV, "").strip()
    if not key:
        raise FredNotConfiguredError(f"{FRED_API_KEY_ENV} is not configured")
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
                    logger.warning("FRED request failed with non-retryable status %s — not retrying", status_code)
                raise
            if attempt >= max_retries - 1:
                raise
            retry_after_delay = _retry_after_delay_from_exception(exc) if status_code == 429 else None
            delay = retry_after_delay if retry_after_delay is not None else (base_delay * (2**attempt))
            logger.warning(
                "FRED fetch attempt %s/%s failed: %s. Retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    return None


def _request_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(params or {})
    params["api_key"] = _api_key()
    params["file_type"] = "json"

    def _fetch():
        response = requests.get(f"{FRED_API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error_code"):
            raise RuntimeError(payload.get("error_message") or payload.get("error_code") or "FRED API error")
        return payload if isinstance(payload, dict) else {}

    return _fetch_with_retry(_fetch) or {}


def _normalize_date(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


@cache_data(ttl=21600)
def get_vix_observations(start_date: date | datetime | str, end_date: date | datetime | str) -> pd.DataFrame:
    payload = _request_json(
        "/series/observations",
        {
            "series_id": FRED_VIX_SERIES_ID,
            "observation_start": _normalize_date(start_date),
            "observation_end": _normalize_date(end_date),
            "sort_order": "asc",
        },
    )
    observations = payload.get("observations") or []
    if not isinstance(observations, list) or not observations:
        return pd.DataFrame(columns=["Close"])

    rows = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or "").strip()
        if not value or value == ".":
            continue
        try:
            close = float(value)
        except Exception:
            continue
        observed_at = pd.to_datetime(item.get("date"), utc=True, errors="coerce")
        if pd.isna(observed_at):
            continue
        rows.append({"Date": observed_at, "Close": close})

    if not rows:
        return pd.DataFrame(columns=["Close"])

    df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df[["Close"]].dropna(how="all")


@cache_data(ttl=21600)
def get_series_observations(series_id: str, start_date: date | datetime | str, end_date: date | datetime | str) -> pd.DataFrame:
    payload = _request_json(
        "/series/observations",
        {
            "series_id": str(series_id).strip(),
            "observation_start": _normalize_date(start_date),
            "observation_end": _normalize_date(end_date),
            "sort_order": "asc",
        },
    )
    observations = payload.get("observations") or []
    if not isinstance(observations, list) or not observations:
        return pd.DataFrame(columns=["value"])

    rows: list[dict[str, Any]] = []
    for item in observations:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or "").strip()
        if not value or value == ".":
            continue
        try:
            numeric_value = float(value)
        except Exception:
            continue
        observed_at = pd.to_datetime(item.get("date"), utc=True, errors="coerce")
        if pd.isna(observed_at):
            continue
        rows.append({"Date": observed_at, "value": numeric_value})

    if not rows:
        return pd.DataFrame(columns=["value"])

    df = pd.DataFrame(rows).dropna(subset=["Date"]).set_index("Date").sort_index()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    return df[["value"]].dropna(how="all")


@cache_data(ttl=21600)
def get_macro_feature_table(start_date: date | datetime | str, end_date: date | datetime | str) -> pd.DataFrame:
    series_map = {
        "dgs10": FRED_10Y_TREASURY_SERIES_ID,
        "cpiaucsl": FRED_CPI_SERIES_ID,
        "fedfunds": FRED_FED_FUNDS_SERIES_ID,
    }

    frames: list[pd.DataFrame] = []
    for prefix, series_id in series_map.items():
        try:
            series_df = get_series_observations(series_id, start_date, end_date).rename(columns={"value": f"{prefix}_level"})
        except Exception as exc:
            logger.warning("FRED series %s unavailable: %s", series_id, exc)
            series_df = pd.DataFrame(columns=[f"{prefix}_level"])
        frames.append(series_df)

    if not frames:
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1).sort_index()
    daily_index = pd.date_range(
        start=pd.to_datetime(_normalize_date(start_date), errors="coerce"),
        end=pd.to_datetime(_normalize_date(end_date), errors="coerce"),
        freq="D",
    )
    if len(daily_index) > 0:
        macro = macro.reindex(daily_index).ffill()
    for prefix in series_map:
        level_col = f"{prefix}_level"
        if level_col not in macro.columns:
            macro[level_col] = pd.NA
        for window in (5, 30):
            delta_col = f"{prefix}_delta_{window}d"
            pct_col = f"{prefix}_pct_change_{window}d"
            macro[delta_col] = macro[level_col] - macro[level_col].shift(window)
            macro[pct_col] = macro[level_col].pct_change(periods=window)
    return macro
