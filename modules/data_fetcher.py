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
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "XOM", "V"]


_NASDAQ100_FALLBACK = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "GOOG",
    "TSLA",
    "AVGO",
    "COST",
    "NFLX",
    "AMD",
    "ADBE",
    "CSCO",
    "INTC",
    "INTU",
    "QCOM",
    "AMGN",
    "TXN",
    "PEP",
]
_RUSSELL2000_FALLBACK = ["SMCI", "CROX", "FSLY", "UPWK", "PLUG", "RUN", "RIOT", "MARA", "RKT", "SOFI", "RKLB", "LMND"]
_OTC_FALLBACK = ["NLST", "RHHBY", "NSRGY", "BUDFF", "TCEHY", "NTDOY", "SFTBY", "BAMXF", "BYDDF", "PPRUY"]
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _clean_ticker(value: Any) -> str | None:
    ticker = str(value or "").strip().upper().replace(".", "-")
    if not ticker or any(ch.isspace() for ch in ticker):
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


def _scrape_stockanalysis_tickers(url: str, max_pages: int = 1, limit: int | None = None) -> list[str]:
    tickers: list[str] = []
    seen = set()
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
    try:
        tickers = _scrape_stockanalysis_tickers("https://stockanalysis.com/list/nasdaq-100-stocks/", max_pages=2)
        return tickers or _NASDAQ100_FALLBACK
    except Exception:
        return _NASDAQ100_FALLBACK


@cache_data(ttl=86400)
def get_russell2000_tickers() -> list[str]:
    try:
        tickers = _scrape_stockanalysis_tickers("https://stockanalysis.com/list/russell-2000-stocks/", max_pages=30)
        return tickers or _RUSSELL2000_FALLBACK
    except Exception:
        return _RUSSELL2000_FALLBACK


@cache_data(ttl=86400)
def get_otc_tickers() -> list[str]:
    try:
        tickers = _scrape_stockanalysis_tickers("https://stockanalysis.com/list/otc-stocks/", max_pages=50, limit=500)
        return tickers or _OTC_FALLBACK
    except Exception:
        return _OTC_FALLBACK
