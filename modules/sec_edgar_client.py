from __future__ import annotations

import os
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any

import requests

from modules.env import load_environment
from modules.logger import get_logger
from modules.validators import sanitize_ticker

load_environment()

logger = get_logger(__name__)

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_EDGAR_CONTACT_EMAIL_ENV = "SEC_EDGAR_CONTACT_EMAIL"
SEC_EDGAR_DEFAULT_CONTACT_EMAIL = "set-this-to-your-email@example.com"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_MAX_RETRIES = 3
REQUEST_BACKOFF_BASE_SECONDS = 1.0
SEC_MAX_REQUESTS_PER_SECOND = 10.0
SEC_MIN_REQUEST_INTERVAL_SECONDS = 1.0 / SEC_MAX_REQUESTS_PER_SECOND

_RATE_LIMIT_LOCK = threading.Lock()
_LAST_REQUEST_TS = 0.0
_USER_AGENT_WARNING_LOCK = threading.Lock()
_USER_AGENT_WARNING_EMITTED = False

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}

_CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "shares_outstanding": ["WeightedAverageNumberOfDilutedSharesOutstanding", "CommonStockSharesOutstanding"],
}

_CONCEPT_UNITS: dict[str, list[str]] = {
    "revenue": ["USD"],
    "net_income": ["USD"],
    "assets": ["USD"],
    "liabilities": ["USD"],
    "equity": ["USD"],
    "eps": ["USD/shares", "USD / shares", "USD"],
    "shares_outstanding": ["shares"],
}

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


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


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
    value = _safe_float(retry_after)
    if value is not None and value > 0:
        return float(value)
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
                    logger.warning("SEC EDGAR request failed with non-retryable status %s — not retrying", status_code)
                raise
            if attempt >= max_retries - 1:
                raise
            retry_after_delay = _retry_after_delay_from_exception(exc) if status_code == 429 else None
            delay = retry_after_delay if retry_after_delay is not None else (base_delay * (2**attempt))
            logger.warning(
                "SEC EDGAR fetch attempt %s/%s failed: %s. Retrying in %.1fs",
                attempt + 1,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    return None


def _sec_user_agent() -> str:
    global _USER_AGENT_WARNING_EMITTED
    contact_email = os.getenv(SEC_EDGAR_CONTACT_EMAIL_ENV, "").strip() or SEC_EDGAR_DEFAULT_CONTACT_EMAIL
    if contact_email == SEC_EDGAR_DEFAULT_CONTACT_EMAIL:
        with _USER_AGENT_WARNING_LOCK:
            if not _USER_AGENT_WARNING_EMITTED:
                _USER_AGENT_WARNING_EMITTED = True
                logger.warning(
                    "%s is not set. Using default placeholder in SEC User-Agent; set a real contact email for SEC guidance compliance.",
                    SEC_EDGAR_CONTACT_EMAIL_ENV,
                )
    return f"BK-Stock-Analyzer/1.0 (contact: {contact_email})"


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": _sec_user_agent(),
        "Accept": "application/json",
    }


def _throttle_request_rate() -> None:
    global _LAST_REQUEST_TS
    with _RATE_LIMIT_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_REQUEST_TS
        if elapsed < SEC_MIN_REQUEST_INTERVAL_SECONDS:
            time.sleep(SEC_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        _LAST_REQUEST_TS = time.monotonic()


def _request_json(url: str) -> dict[str, Any]:
    def _fetch():
        _throttle_request_rate()
        response = requests.get(url, headers=_sec_headers(), timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    return _fetch_with_retry(_fetch) or {}


def _normalize_lookup_ticker(ticker: str) -> str:
    clean = sanitize_ticker(ticker)
    return clean.replace(".", "-")


@cache_data(ttl=86400)
def _get_ticker_to_cik_mapping() -> dict[str, int]:
    payload = _request_json(SEC_TICKER_MAP_URL)
    mapping: dict[str, int] = {}
    rows = payload.values() if isinstance(payload, dict) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik_value = row.get("cik_str")
        if not ticker:
            continue
        try:
            cik = int(cik_value)
        except Exception:
            continue
        mapping[ticker] = cik
        mapping[ticker.replace(".", "-")] = cik
        mapping[ticker.replace("-", ".")] = cik
    return mapping


def get_cik_for_ticker(ticker: str) -> int | None:
    lookup = _normalize_lookup_ticker(ticker)
    mapping = _get_ticker_to_cik_mapping()
    cik = mapping.get(lookup)
    if cik is None:
        alt = lookup.replace("-", ".")
        cik = mapping.get(alt)
    if cik is None:
        logger.warning("No SEC CIK mapping found for ticker %s", lookup)
    return cik


@cache_data(ttl=3600)
def get_company_facts(ticker: str) -> dict[str, Any]:
    cik = get_cik_for_ticker(ticker)
    if cik is None:
        return {}
    cik_padded = f"{int(cik):010d}"
    return _request_json(SEC_COMPANY_FACTS_URL.format(cik=cik_padded))


def _parse_fact_end_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except Exception:
        return None


def _collect_tag_entries(company_facts: dict[str, Any], tag: str, preferred_units: list[str]) -> list[dict[str, Any]]:
    facts = ((company_facts.get("facts") or {}).get("us-gaap") or {})
    concept = facts.get(tag)
    if not isinstance(concept, dict):
        return []
    units = concept.get("units") or {}
    if not isinstance(units, dict) or not units:
        return []

    ordered_units = []
    seen_units = set()
    for unit in preferred_units + list(units.keys()):
        if unit in units and unit not in seen_units:
            ordered_units.append(unit)
            seen_units.add(unit)

    rows: list[dict[str, Any]] = []
    for unit in ordered_units:
        values = units.get(unit) or []
        if not isinstance(values, list):
            continue
        for row in values:
            if not isinstance(row, dict):
                continue
            val = _safe_float(row.get("val"))
            end_date = _parse_fact_end_date(row.get("end"))
            if val is None or end_date is None:
                continue
            rows.append(
                {
                    "tag": tag,
                    "unit": unit,
                    "value": val,
                    "form": str(row.get("form") or "").upper(),
                    "end": end_date,
                    "filed": _parse_fact_end_date(row.get("filed")) or end_date,
                }
            )
    return rows


def _select_preferred_latest(entries: list[dict[str, Any]], older_than: date | None = None) -> dict[str, Any] | None:
    if not entries:
        return None
    filtered = [entry for entry in entries if older_than is None or entry["end"] <= older_than]
    if not filtered:
        return None
    for preferred_form in ("10-K", "10-Q"):
        form_matches = [entry for entry in filtered if entry.get("form") == preferred_form]
        if form_matches:
            return max(form_matches, key=lambda item: (item["end"], item["filed"]))
    return max(filtered, key=lambda item: (item["end"], item["filed"]))


def _extract_concept(company_facts: dict[str, Any], concept_name: str) -> dict[str, Any]:
    tags = _CONCEPT_TAGS.get(concept_name, [])
    units = _CONCEPT_UNITS.get(concept_name, [])
    for tag in tags:
        entries = _collect_tag_entries(company_facts, tag, units)
        current = _select_preferred_latest(entries)
        if current is None:
            continue
        prior_target_date = current["end"] - timedelta(days=300)
        prior = _select_preferred_latest(entries, older_than=prior_target_date)
        if prior is None:
            prior = _select_preferred_latest(entries, older_than=current["end"] - timedelta(days=1))
        return {
            "tag": tag,
            "current": current["value"],
            "current_end": current["end"],
            "current_form": current["form"],
            "prior": prior["value"] if prior else None,
            "prior_end": prior["end"] if prior else None,
            "unit": current["unit"],
        }
    return {"tag": None, "current": None, "current_end": None, "current_form": None, "prior": None, "prior_end": None, "unit": None}


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-12:
        return None
    return numerator / denominator


def build_fundamentals_info_adapter(company_facts: dict[str, Any], current_price: float | None = None) -> dict[str, Any]:
    revenue = _extract_concept(company_facts, "revenue")
    net_income = _extract_concept(company_facts, "net_income")
    liabilities = _extract_concept(company_facts, "liabilities")
    equity = _extract_concept(company_facts, "equity")
    eps = _extract_concept(company_facts, "eps")
    shares_outstanding = _extract_concept(company_facts, "shares_outstanding")

    trailing_eps = _safe_float(eps.get("current"))
    trailing_pe = _safe_div(_safe_float(current_price), trailing_eps)
    roe = _safe_div(_safe_float(net_income.get("current")), _safe_float(equity.get("current")))
    debt_to_equity_ratio = _safe_div(_safe_float(liabilities.get("current")), _safe_float(equity.get("current")))
    debt_to_equity = (debt_to_equity_ratio * 100.0) if debt_to_equity_ratio is not None else None

    revenue_growth = _safe_div(
        (_safe_float(revenue.get("current")) or 0.0) - (_safe_float(revenue.get("prior")) or 0.0),
        _safe_float(revenue.get("prior")),
    )
    earnings_growth = _safe_div(
        (_safe_float(net_income.get("current")) or 0.0) - (_safe_float(net_income.get("prior")) or 0.0),
        _safe_float(net_income.get("prior")),
    )

    return {
        "trailingPE": trailing_pe,
        "forwardPE": None,
        "trailingEps": trailing_eps,
        "forwardEps": None,
        "earningsGrowth": earnings_growth,
        "revenueGrowth": revenue_growth,
        "debtToEquity": debt_to_equity,
        "returnOnEquity": roe,
        "sharesOutstanding": _safe_float(shares_outstanding.get("current")),
    }


@cache_data(ttl=1800)
def get_fundamentals_info_adapter(ticker: str, current_price: float | None = None) -> dict[str, Any]:
    company_facts = get_company_facts(ticker)
    if not company_facts:
        return {
            "trailingPE": None,
            "forwardPE": None,
            "trailingEps": None,
            "forwardEps": None,
            "earningsGrowth": None,
            "revenueGrowth": None,
            "debtToEquity": None,
            "returnOnEquity": None,
            "sharesOutstanding": None,
        }
    return build_fundamentals_info_adapter(company_facts, current_price=current_price)
