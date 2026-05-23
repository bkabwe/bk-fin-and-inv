from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
import yfinance as yf

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


def get_nasdaq100_tickers() -> list[str]:
    return [
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


def get_russell2000_sample() -> list[str]:
    return ["SMCI", "CROX", "FSLY", "UPWK", "PLUG", "RUN", "RIOT", "MARA", "RKT", "SOFI", "RKLB", "LMND"]


def get_otc_sample() -> list[str]:
    return ["NLST", "RHHBY", "NSRGY", "BUDFF", "TCEHY", "NTDOY", "SFTBY", "BAMXF", "BYDDF", "PPRUY"]
