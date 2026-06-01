from __future__ import annotations

from fastapi import APIRouter

from modules.data_fetcher import get_stock_data
from modules.scoring_engine import analyze_stock

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/{ticker}")
def stock_analysis(ticker: str, period: str = "1y") -> dict:
    return analyze_stock(ticker, period=period)


@router.get("/{ticker}/price")
def stock_price(ticker: str, period: str = "1y", interval: str = "1d") -> list[dict]:
    df = get_stock_data(ticker=ticker, period=period, interval=interval)
    if df.empty:
        return []
    frame = df.reset_index()
    frame.columns = [str(c) for c in frame.columns]
    return frame.to_dict(orient="records")
