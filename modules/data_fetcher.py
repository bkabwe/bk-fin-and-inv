from __future__ import annotations

from functools import lru_cache
from io import StringIO
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

try:
    import streamlit as st

    cache_data = st.cache_data
except Exception:  # pragma: no cover
    def cache_data(ttl: int | None = None):
        def decorator(func):
            return lru_cache(maxsize=128)(func)

        return decorator


@cache_data(ttl=3600)
def get_stock_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    ticker = ticker.upper().strip()
    if not ticker:
        return pd.DataFrame()
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        return data.dropna(how="all") if data is not None and not data.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@cache_data(ttl=3600)
def get_stock_info(ticker: str) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    if not ticker:
        return {}
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@cache_data(ttl=900)
def get_news(ticker: str) -> list[dict[str, Any]]:
    ticker = ticker.upper().strip()
    if not ticker:
        return []
    try:
        return yf.Ticker(ticker).news or []
    except Exception:
        return []


@cache_data(ttl=86400)
def get_sp500_tickers() -> list[str]:
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return sorted(set(table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()))
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
    return ticker


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


@cache_data(ttl=86400)
def get_nasdaq100_tickers() -> list[str]:
    url = "https://stockanalysis.com/list/nasdaq-100-stocks/"
    try:
        tickers = _scrape_stockanalysis_tickers(url, max_pages=2)
        if tickers:
            return tickers
    except Exception:
        pass
    raise RuntimeError("Failed to fetch NASDAQ 100 tickers.")


@cache_data(ttl=86400)
def get_russell2000_tickers() -> list[str]:
    url = "https://stockanalysis.com/list/russell-2000-stocks/"
    try:
        tickers = _scrape_stockanalysis_tickers(url, max_pages=30)
        if tickers:
            return tickers
    except Exception:
        pass
    raise RuntimeError("Failed to fetch Russell 2000 tickers.")


@cache_data(ttl=86400)
def get_otc_tickers() -> list[str]:
    url = "https://stockanalysis.com/list/otc-stocks/"
    try:
        tickers = _scrape_stockanalysis_tickers(url, max_pages=50, limit=500)
        if tickers:
            return tickers
    except Exception:
        pass
    raise RuntimeError("Failed to fetch OTC tickers.")
