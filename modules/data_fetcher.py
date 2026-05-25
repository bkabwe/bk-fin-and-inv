from __future__ import annotations

from functools import lru_cache
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


def _normalize_tickers(values: list[Any], limit: int | None = None) -> list[str]:
    cleaned = []
    for value in values:
        ticker = str(value).strip().upper().replace(".", "-")
        if ticker and ticker not in {"NAN", "NONE", "SYMBOL"} and ticker.isascii():
            cleaned.append(ticker)
    unique = sorted(set(cleaned))
    return unique[:limit] if limit is not None else unique


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


@cache_data(ttl=86400)
def get_nasdaq100_tickers() -> list[str]:
    url = "https://stockanalysis.com/list/nasdaq-100-stocks/"
    try:
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        tickers = _extract_tickers_from_tables(tables)
        if tickers:
            return tickers
    except Exception:
        pass
    try:
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            symbols = []
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                symbols.append(cells[1].get_text(strip=True))
            tickers = _normalize_tickers(symbols)
            if tickers:
                return tickers
    except Exception:
        pass
    raise RuntimeError("Failed to fetch NASDAQ 100 tickers.")


@cache_data(ttl=86400)
def get_russell2000_tickers() -> list[str]:
    url = "https://www.chartmill.com/stock/markets/usa/index/russell-2000"
    try:
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        tickers = _extract_tickers_from_tables(tables)
        if tickers:
            return tickers
    except Exception:
        pass
    try:
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        symbols = [cell.get_text(strip=True) for cell in soup.select("table td:nth-child(1)")]
        tickers = _normalize_tickers(symbols)
        if tickers:
            return tickers
    except Exception:
        pass
    raise RuntimeError("Failed to fetch Russell 2000 tickers from chartmill.com.")


@cache_data(ttl=86400)
def get_otc_tickers() -> list[str]:
    url = "https://stockanalysis.com/list/otc-stocks/"
    try:
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        tickers = _extract_tickers_from_tables(tables, limit=500)
        if tickers:
            return tickers
    except Exception:
        pass
    try:
        response = requests.get(url, headers=_DEFAULT_HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        table = soup.find("table")
        if table:
            symbols = []
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    symbols.append(cells[1].get_text(strip=True))
            tickers = _normalize_tickers(symbols, limit=500)
            if tickers:
                return tickers
    except Exception:
        pass
    raise RuntimeError("Failed to fetch OTC tickers.")
