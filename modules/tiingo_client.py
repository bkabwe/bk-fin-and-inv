from __future__ import annotations

import os
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from modules.data_fetcher import _fetch_with_retry, cache_data
from modules.logger import get_logger
from modules.scoring_engine import analyze_stock, get_price_projections
from modules.validators import sanitize_ticker

logger = get_logger(__name__)

TIINGO_API_BASE_URL = "https://api.tiingo.com/tiingo/daily"
TIINGO_API_KEY_ENV = "TIINGO_API_KEY"
TIINGO_REVALIDATION_DEFAULT_TOP_N = 50
TIINGO_REVALIDATION_MAX_TOP_N = 100
TIINGO_DISCREPANCY_THRESHOLD_PCT = 2.0
TIINGO_REQUEST_TIMEOUT_SECONDS = 20

# Tiingo's documented free tier currently advertises 50 requests/hour and
# 1,000 requests/day, so keep the top-N batch modest and throttle requests in
# process to reduce avoidable 429s during optional revalidation.
TIINGO_FREE_TIER_MAX_REQUESTS_PER_HOUR = 45
TIINGO_FREE_TIER_WINDOW_SECONDS = 3600

_rate_limit_lock = threading.Lock()
_request_times: deque[float] = deque()


class TiingoUnavailableError(RuntimeError):
    """Raised when Tiingo revalidation cannot run in the current environment."""


def is_tiingo_configured() -> bool:
    return bool(os.getenv(TIINGO_API_KEY_ENV, "").strip())


def clamp_revalidation_top_n(value: int | None) -> int:
    try:
        parsed = int(value or TIINGO_REVALIDATION_DEFAULT_TOP_N)
    except Exception:
        parsed = TIINGO_REVALIDATION_DEFAULT_TOP_N
    return max(1, min(parsed, TIINGO_REVALIDATION_MAX_TOP_N))


def _get_api_key() -> str:
    api_key = os.getenv(TIINGO_API_KEY_ENV, "").strip()
    if not api_key:
        raise TiingoUnavailableError(f"{TIINGO_API_KEY_ENV} is not configured")
    return api_key


def _period_start(period: str) -> str:
    mapping = {
        "6mo": 183,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }
    days = mapping.get(str(period or "2y").lower(), 730)
    return (datetime.now(UTC) - timedelta(days=days)).date().isoformat()


def _throttle_request_rate() -> None:
    while True:
        sleep_for = 0.0
        with _rate_limit_lock:
            now = time.monotonic()
            while _request_times and (now - _request_times[0]) >= TIINGO_FREE_TIER_WINDOW_SECONDS:
                _request_times.popleft()
            if len(_request_times) < TIINGO_FREE_TIER_MAX_REQUESTS_PER_HOUR:
                _request_times.append(now)
                return
            sleep_for = max(0.5, TIINGO_FREE_TIER_WINDOW_SECONDS - (now - _request_times[0]) + 0.1)
        time.sleep(sleep_for)


def _normalize_price_frame(payload: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(payload)
    if frame.empty or "date" not in frame:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None)
    frame = frame.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    column_map = {
        "Open": "adjOpen",
        "High": "adjHigh",
        "Low": "adjLow",
        "Close": "adjClose",
        "Volume": "adjVolume",
    }
    fallback_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    normalized = pd.DataFrame(index=frame["date"])
    for output_col, source_col in column_map.items():
        if source_col in frame:
            normalized[output_col] = pd.to_numeric(frame[source_col], errors="coerce")
        elif fallback_map[output_col] in frame:
            normalized[output_col] = pd.to_numeric(frame[fallback_map[output_col]], errors="coerce")
        else:
            normalized[output_col] = pd.NA
    normalized.index.name = "Date"
    return normalized.dropna(how="all")


@cache_data(ttl=3600)
def get_tiingo_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    ticker = sanitize_ticker(ticker)
    api_key = _get_api_key()
    start_date = _period_start(period)
    end_date = datetime.now(UTC).date().isoformat()

    def _fetch() -> pd.DataFrame:
        _throttle_request_rate()
        response = requests.get(
            f"{TIINGO_API_BASE_URL}/{ticker}/prices",
            params={
                "startDate": start_date,
                "endDate": end_date,
                "resampleFreq": "daily",
                "token": api_key,
            },
            timeout=TIINGO_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code in {401, 403}:
            raise TiingoUnavailableError("Tiingo rejected the configured API key")
        if response.status_code == 429:
            raise RuntimeError("Tiingo rate limit exceeded")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise ValueError("Empty Tiingo price response")
        data = _normalize_price_frame(payload)
        if data.empty:
            raise ValueError("No usable Tiingo price data returned")
        return data

    try:
        return _fetch_with_retry(_fetch, max_retries=3, base_delay=2.0)
    except TiingoUnavailableError:
        raise
    except Exception as exc:
        logger.warning("Tiingo fetch failed for %s: %s", ticker, exc)
        raise


def _slice_to_period(data: pd.DataFrame, period: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    latest = pd.to_datetime(data.index.max())
    days = {"6mo": 183, "1y": 365, "2y": 730}.get(str(period or "1y").lower(), 365)
    start = latest - pd.Timedelta(days=days)
    sliced = data.loc[data.index >= start].copy()
    return sliced if not sliced.empty else data.copy()


def _pct_delta(updated: Any, original: Any) -> float | None:
    try:
        original_value = float(original)
        updated_value = float(updated)
        if original_value == 0:
            return None
        return round(((updated_value - original_value) / original_value) * 100.0, 2)
    except Exception:
        return None


def _verification_fields(
    *,
    current_price: Any,
    verified_price: Any,
    target_price: Any,
    verified_target: Any,
    threshold_pct: float,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    price_delta = _pct_delta(verified_price, current_price)
    target_delta = _pct_delta(verified_target, target_price)
    flagged = price_delta is not None and abs(price_delta) > threshold_pct
    return {
        "Verification": "verified - discrepancy flagged" if flagged else "verified",
        "Data Source Agreement": f"price diff {price_delta:.2f}%" if flagged and price_delta is not None else "agree",
        "Price Difference %": price_delta,
        "Target Difference %": target_delta,
        **(extra_fields or {}),
    }


def _unavailable_fields(message: str, field_names: list[str]) -> dict[str, Any]:
    payload = {name: None for name in field_names}
    payload["Verification"] = "unavailable"
    payload["Data Source Agreement"] = message
    payload["Price Difference %"] = None
    payload["Target Difference %"] = None
    return payload


def revalidate_profit_rows(
    rows: list[dict[str, Any]],
    *,
    horizon_key: str,
    top_n: int = TIINGO_REVALIDATION_DEFAULT_TOP_N,
    discrepancy_threshold_pct: float = TIINGO_DISCREPANCY_THRESHOLD_PCT,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    verified_fields = ["Verified Current Price", "Verified Target Price", "Verified Projected Upside %"]
    limit = min(clamp_revalidation_top_n(top_n), len(rows))
    updated_rows = [dict(row) for row in rows]

    if not is_tiingo_configured():
        unavailable = _unavailable_fields(f"{TIINGO_API_KEY_ENV} missing", verified_fields)
        for idx, row in enumerate(updated_rows):
            if idx < limit:
                row.update(unavailable)
            else:
                row.setdefault("Verification", "not requested")
        return updated_rows

    target_key = f"{horizon_key}_target"
    upside_key = f"{horizon_key}_upside"

    for idx, row in enumerate(updated_rows):
        if idx >= limit:
            row.setdefault("Verification", "not requested")
            continue
        ticker = row.get("Ticker") or row.get("ticker")
        try:
            tiingo_data = get_tiingo_stock_data(str(ticker), period="2y")
            projections = get_price_projections(str(ticker), data_override=tiingo_data)
            verified_price = projections.get("current_price")
            verified_target = projections.get(target_key)
            verified_upside = projections.get(upside_key)
            row.update(
                _verification_fields(
                    current_price=row.get("Current Price"),
                    verified_price=verified_price,
                    target_price=row.get("Target Price"),
                    verified_target=verified_target,
                    threshold_pct=discrepancy_threshold_pct,
                    extra_fields={
                        "Verified Current Price": round(float(verified_price), 2) if verified_price is not None else None,
                        "Verified Target Price": round(float(verified_target), 2) if verified_target is not None else None,
                        "Verified Projected Upside %": round(float(verified_upside), 2) if verified_upside is not None else None,
                    },
                )
            )
        except Exception as exc:
            row.update(_unavailable_fields(str(exc), verified_fields))
    return updated_rows


def revalidate_screener_rows(
    rows: list[dict[str, Any]],
    *,
    top_n: int = TIINGO_REVALIDATION_DEFAULT_TOP_N,
    discrepancy_threshold_pct: float = TIINGO_DISCREPANCY_THRESHOLD_PCT,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    verified_fields = [
        "Verified Score",
        "Verified Current Price",
        "Verified Entry Price",
        "Verified Target Price",
        "Verified Stop Loss",
    ]
    limit = min(clamp_revalidation_top_n(top_n), len(rows))
    updated_rows = [dict(row) for row in rows]

    if not is_tiingo_configured():
        unavailable = _unavailable_fields(f"{TIINGO_API_KEY_ENV} missing", verified_fields)
        for idx, row in enumerate(updated_rows):
            if idx < limit:
                row.update(unavailable)
            else:
                row.setdefault("Verification", "not requested")
        return updated_rows

    for idx, row in enumerate(updated_rows):
        if idx >= limit:
            row.setdefault("Verification", "not requested")
            continue
        ticker = row.get("Ticker") or row.get("ticker")
        try:
            tiingo_2y = get_tiingo_stock_data(str(ticker), period="2y")
            analysis = analyze_stock(
                str(ticker),
                data_override=_slice_to_period(tiingo_2y, "1y"),
                projection_data_override=tiingo_2y,
            )
            verified_price = analysis.get("current_price")
            verified_target = analysis.get("target_price")
            row.update(
                _verification_fields(
                    current_price=row.get("Current Price"),
                    verified_price=verified_price,
                    target_price=row.get("Target Price"),
                    verified_target=verified_target,
                    threshold_pct=discrepancy_threshold_pct,
                    extra_fields={
                        "Verified Score": analysis.get("score"),
                        "Verified Current Price": round(float(verified_price), 2) if verified_price is not None else None,
                        "Verified Entry Price": analysis.get("entry_price"),
                        "Verified Target Price": round(float(verified_target), 2) if verified_target is not None else None,
                        "Verified Stop Loss": analysis.get("stop_loss"),
                    },
                )
            )
        except Exception as exc:
            row.update(_unavailable_fields(str(exc), verified_fields))
    return updated_rows
