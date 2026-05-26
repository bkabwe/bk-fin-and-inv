from __future__ import annotations

import time
from functools import lru_cache
from io import StringIO
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from modules.logger import get_logger
from modules.validators import sanitize_ticker

logger = get_logger(__name__)

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    def cache_data(ttl: int | None = None):
        def decorator(func):
            return lru_cache(maxsize=128)(func)

        return decorator


def _fetch_with_retry(fetch_fn, max_retries: int = 3, base_delay: float = 1.0):
    """Retry a fetch function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            result = fetch_fn()
            if result is not None:
                return result
        except Exception as exc:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning("Fetch attempt %s/%s failed: %s. Retrying in %.1fs", attempt + 1, max_retries, exc, delay)
                time.sleep(delay)
            else:
                raise
    return None


@cache_data(ttl=3600)
def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker = sanitize_ticker(ticker)
    except ValueError as exc:
        logger.warning("Invalid ticker in get_stock_data: %s", exc)
        return pd.DataFrame()

    def _fetch():
        data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if data is None or data.empty:
            raise ValueError("Empty price response (possible temporary rate limit)")
        return data

    try:
        data = _fetch_with_retry(_fetch)
    except Exception as exc:
        logger.error("Failed to fetch stock data for %s after retries: %s", ticker, exc)
        return pd.DataFrame()

    return data.dropna(how="all") if isinstance(data, pd.DataFrame) else pd.DataFrame()


@cache_data(ttl=3600)
def get_stock_info(ticker: str) -> dict[str, Any]:
    try:
        ticker = sanitize_ticker(ticker)
    except ValueError as exc:
        logger.warning("Invalid ticker in get_stock_info: %s", exc)
        return {}

    def _fetch():
        info = yf.Ticker(ticker).info or {}
        if not info:
            raise ValueError("Empty info response (possible temporary rate limit)")
        return info

    try:
        info = _fetch_with_retry(_fetch)
    except Exception as exc:
        logger.error("Failed to fetch stock info for %s after retries: %s", ticker, exc)
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
        return yf.Ticker(ticker).news or []
    except Exception:
        return []


def _safe_read_html(html: str) -> list[pd.DataFrame]:
    try:
        return pd.read_html(StringIO(html))
    except Exception:
        return []


_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
_REQUEST_HEADERS = _DEFAULT_HEADERS
_REQUEST_TIMEOUT = 15


def _normalize_tickers(values: list[Any], limit: int | None = None) -> list[str]:
    cleaned = []
    for value in values:
        ticker = str(value).strip().upper().replace(".", "-")
        if ticker and ticker not in {"NAN", "NONE", "SYMBOL"} and ticker.isascii():
            cleaned.append(ticker)
    unique = sorted(set(cleaned))
    return unique[:limit] if limit is not None else unique


def _clean_ticker(value: Any) -> str | None:
    ticker = str(value or "").strip().upper().replace(".", "-")
    if not ticker or any(ch.isspace() for ch in ticker):
        return None
    if ticker in {"NAN", "NONE", "SYMBOL"}:
        return None
    try:
        return sanitize_ticker(ticker)
    except ValueError:
        return None


def _extract_tickers_from_table(html: str) -> list[str]:
    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        tables = []
    for table in tables:
        columns = {str(col).strip().lower(): col for col in table.columns}
        symbol_col = columns.get("symbol") or columns.get("ticker")
        if symbol_col is None:
            continue
        tickers = [_clean_ticker(x) for x in table[symbol_col].tolist()]
        return [x for x in tickers if x]
    return []


def _extract_tickers_from_rows(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    rows = soup.select("table tbody tr")
    tickers = []
    for row in rows:
        cell = row.find("td")
        if not cell:
            continue
        ticker = _clean_ticker(cell.get_text(strip=True))
        if ticker:
            tickers.append(ticker)
    return tickers


def _extract_tickers_from_tables(tables: list[pd.DataFrame], limit: int | None = None) -> list[str]:
    for table in tables:
        cols = {str(col).strip().lower(): col for col in table.columns}
        candidate = next(
            (cols[key] for key in cols if key in {"symbol", "ticker", "ticker symbol"}),
            None,
        )
        if candidate is not None:
            tickers = _normalize_tickers(table[candidate].tolist(), limit=limit)
            if tickers:
                return tickers
    return []


def _scrape_stockanalysis_paged(url: str, max_pages: int = 100, limit: int | None = None) -> list[str]:
    """Scrape stockanalysis.com lists using ?page=N pagination (page 1 = base URL, page 2+ = ?page=N)."""
    tickers: list[str] = []
    seen: set[str] = set()
    session = requests.Session()
    empty_streak = 0
    for page in range(1, max_pages + 1):
        page_url = url if page == 1 else f"{url}?page={page}"
        try:
            response = session.get(page_url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT)
            response.raise_for_status()
        except Exception:
            empty_streak += 1
            if empty_streak >= 3:
                break
            continue
        page_tickers = _extract_tickers_from_table(response.text) or _extract_tickers_from_rows(response.text)
        if not page_tickers:
            empty_streak += 1
            if empty_streak >= 3:
                break
            continue
        empty_streak = 0
        before = len(tickers)
        for symbol in page_tickers:
            if symbol in seen:
                continue
            seen.add(symbol)
            tickers.append(symbol)
            if limit and len(tickers) >= limit:
                return tickers[:limit]
        if len(tickers) == before:
            break
    return tickers[:limit] if limit else tickers


@cache_data(ttl=86400)
def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from stockanalysis.com."""
    url = "https://stockanalysis.com/list/sp-500-stocks/"
    try:
        tickers = _scrape_stockanalysis_paged(url, max_pages=6)
        if len(tickers) > 100:
            logger.info("Fetched %s S&P 500 tickers from stockanalysis.com", len(tickers))
            return tickers
    except Exception as exc:
        logger.error("S&P 500 stockanalysis scrape failed: %s", exc)

    # Fallback: Wikipedia
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=_DEFAULT_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for table in tables:
            cols = {str(c).strip().lower(): c for c in table.columns}
            sym_col = cols.get("symbol") or cols.get("ticker")
            if sym_col:
                tickers = _normalize_tickers(table[sym_col].astype(str).tolist())
                if len(tickers) > 100:
                    logger.info("Fetched %s S&P 500 tickers from Wikipedia fallback", len(tickers))
                    return tickers
    except Exception as exc:
        logger.error("S&P 500 Wikipedia fallback failed: %s", exc)

    raise RuntimeError("Failed to fetch S&P 500 tickers from stockanalysis.com or Wikipedia.")


@cache_data(ttl=86400)
def get_nasdaq_tickers() -> list[str]:
    """Fetch full NASDAQ stock list from stockanalysis.com (3000+ stocks, ~500 per page, ?page=N pagination)."""
    url = "https://stockanalysis.com/list/nasdaq-stocks/"
    try:
        tickers = _scrape_stockanalysis_paged(url, max_pages=100)
        if len(tickers) > 100:
            logger.info("Fetched %s full NASDAQ tickers from stockanalysis.com", len(tickers))
            return tickers
    except Exception as exc:
        logger.error("Full NASDAQ scrape failed: %s", exc)
    raise RuntimeError("Failed to fetch full NASDAQ tickers from stockanalysis.com.")


@cache_data(ttl=86400)
def get_nyseamerican_tickers() -> list[str]:
    """Fetch NYSE American stock list from stockanalysis.com (paginated with ?page=N)."""
    url = "https://stockanalysis.com/list/nyseamerican-stocks/"
    try:
        tickers = _scrape_stockanalysis_paged(url, max_pages=20)
        if tickers:
            logger.info("Fetched %s NYSE American tickers from stockanalysis.com", len(tickers))
            return tickers
    except Exception as exc:
        logger.error("NYSE American scrape failed: %s", exc)
    raise RuntimeError("Failed to fetch NYSE American tickers from stockanalysis.com.")


@cache_data(ttl=86400)
def get_otc_tickers() -> list[str]:
    """Fetch full OTC stock list from stockanalysis.com (21+ pages of ~500 each, ?page=N pagination)."""
    url = "https://stockanalysis.com/list/otc-stocks/"
    try:
        tickers = _scrape_stockanalysis_paged(url, max_pages=50)
        if tickers:
            logger.info("Fetched %s OTC tickers from stockanalysis.com", len(tickers))
            return tickers
    except Exception as exc:
        logger.error("OTC scrape failed: %s", exc)
    raise RuntimeError("Failed to fetch OTC tickers from stockanalysis.com.")
