from __future__ import annotations

from fastapi import APIRouter

from modules.sentiment_analysis import analyze_sentiment

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.get("/market")
def market_sentiment() -> dict:
    tickers = ["SPY", "QQQ", "DIA", "IWM"]
    return {ticker: analyze_sentiment(ticker) for ticker in tickers}


@router.get("/{ticker}")
def ticker_sentiment(ticker: str) -> dict:
    return analyze_sentiment(ticker)
