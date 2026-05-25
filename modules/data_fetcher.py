from __future__ import annotations

import re
import time
from functools import lru_cache
from io import StringIO
from typing import Any
from urllib.parse import urljoin

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


@cache_data(ttl=86400)
def get_sp500_tickers() -> list[str]:
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = sorted(set(table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()))
        logger.info("Fetched %s S&P 500 tickers", len(tickers))
        return tickers
    except Exception:
        raise RuntimeError("Failed to fetch S&P 500 tickers. Check internet connection.")


_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
# Alias kept for internal helpers
_REQUEST_HEADERS = _DEFAULT_HEADERS


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


def _scrape_stockanalysis_tickers(url: str, max_pages: int = 1, limit: int | None = None) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    session = requests.Session()
    for page in range(1, max_pages + 1):
        page_url = url if page == 1 else f"{url}?p={page}"
        response = session.get(page_url, headers=_REQUEST_HEADERS, timeout=20)
        response.raise_for_status()
        page_tickers = _extract_tickers_from_table(response.text) or _extract_tickers_from_rows(response.text)
        if not page_tickers:
            break
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


def _extract_chartmill_tickers(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    found: set[str] = set()

    for table in soup.select("table"):
        for row in table.select("tbody tr"):
            for cell in row.find_all("td")[:3]:
                ticker = _clean_ticker(cell.get_text(strip=True))
                if ticker:
                    found.add(ticker)
            for anchor in row.find_all("a", href=True):
                ticker = _clean_ticker(anchor.get_text(strip=True))
                if ticker:
                    found.add(ticker)
                match = re.search(r"/stock/(?:quote|fundamental)/([A-Za-z0-9.\-]+)", anchor["href"])
                if match:
                    ticker = _clean_ticker(match.group(1))
                    if ticker:
                        found.add(ticker)

    for value in re.findall(r'"symbol"\s*:\s*"([A-Za-z0-9.\-]+)"', html):
        ticker = _clean_ticker(value)
        if ticker:
            found.add(ticker)

    return sorted(found)


def _collect_chartmill_pagination_urls(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []
    seen: set[str] = set()
    for anchor in soup.select("a[href]"):
        href = anchor.get("href") or ""
        if "russell-2000" not in href:
            continue
        if not re.search(r"[?&](?:page|p)=\d+", href):
            continue
        full_url = urljoin(base_url, href)
        if full_url in seen:
            continue
        seen.add(full_url)
        urls.append(full_url)
    return urls


def _scrape_chartmill_russell2000(base_url: str, max_pages: int = 60) -> list[str]:
    session = requests.Session()
    seen: set[str] = set()
    tickers: list[str] = []

    first = session.get(base_url, headers=_REQUEST_HEADERS, timeout=20)
    first.raise_for_status()
    first_page_tickers = _extract_chartmill_tickers(first.text)
    for ticker in first_page_tickers:
        if ticker not in seen:
            seen.add(ticker)
            tickers.append(ticker)

    page_urls = _collect_chartmill_pagination_urls(base_url, first.text)
    if not page_urls:
        page_urls = [f"{base_url}?page={page}" for page in range(2, max_pages + 1)]

    empty_streak = 0
    for page_url in page_urls[: max_pages - 1]:
        response = session.get(page_url, headers=_REQUEST_HEADERS, timeout=20)
        if response.status_code >= 400:
            continue
        page_tickers = _extract_chartmill_tickers(response.text)
        if not page_tickers:
            empty_streak += 1
            if empty_streak >= 3:
                break
            continue

        before = len(tickers)
        for ticker in page_tickers:
            if ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)

        if len(tickers) == before:
            empty_streak += 1
            if empty_streak >= 3:
                break
        else:
            empty_streak = 0

    return tickers


@cache_data(ttl=86400)
def get_nasdaq100_tickers() -> list[str]:
    url = "https://stockanalysis.com/list/nasdaq-100-stocks/"
    try:
        tickers = _scrape_stockanalysis_tickers(url, max_pages=2)
        if tickers:
            logger.info("Fetched %s NASDAQ 100 tickers", len(tickers))
            return tickers
    except Exception as exc:
        logger.error("NASDAQ 100 scrape failed: %s", exc)
    raise RuntimeError("Failed to fetch NASDAQ 100 tickers.")


@cache_data(ttl=86400)
def get_russell2000_tickers() -> list[str]:
    url = "https://www.chartmill.com/stock/markets/usa/index/russell-2000"
    try:
        tickers = _scrape_chartmill_russell2000(url, max_pages=60)
        if tickers:
            logger.info("Fetched %s Russell 2000 tickers from Chartmill", len(tickers))
            return tickers
    except Exception as exc:
        logger.error("Russell 2000 scrape failed: %s", exc)
    raise RuntimeError("Failed to fetch Russell 2000 tickers from chartmill.com.")


@cache_data(ttl=86400)
def get_otc_tickers() -> list[str]:
    url = "https://stockanalysis.com/list/otc-stocks/"
    try:
        tickers = _scrape_stockanalysis_tickers(url, max_pages=50, limit=500)
        if tickers:
            logger.info("Fetched %s OTC tickers", len(tickers))
            return tickers
    except Exception as exc:
        logger.error("OTC scrape failed: %s", exc)
    raise RuntimeError("Failed to fetch OTC tickers.")
