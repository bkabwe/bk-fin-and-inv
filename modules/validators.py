from __future__ import annotations

import re

TICKER_PATTERN = re.compile(r"^[A-Z0-9\-\.]{1,10}$")
MAX_TICKER_LENGTH = 10
MAX_CUSTOM_TICKERS = 200


def sanitize_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()[:MAX_TICKER_LENGTH]
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    if not TICKER_PATTERN.match(ticker):
        raise ValueError(
            f"Invalid ticker symbol: '{ticker}'. Only alphanumeric characters, hyphens, and dots are allowed."
        )
    return ticker


def sanitize_ticker_list(tickers: list[str]) -> list[str]:
    result = []
    for ticker in tickers[:MAX_CUSTOM_TICKERS]:
        try:
            result.append(sanitize_ticker(ticker))
        except ValueError:
            pass
    return result
