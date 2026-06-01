from __future__ import annotations

from fastapi import APIRouter, HTTPException

from modules.portfolio import add_holding, add_to_watchlist, get_watchlist, remove_from_watchlist
from modules.scoring_engine import analyze_stock

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


@router.get("")
def list_watchlist() -> list[dict]:
    rows = []
    for ticker in get_watchlist():
        try:
            analysis = analyze_stock(ticker)
            rows.append(
                {
                    "Ticker": ticker,
                    "Price": analysis.get("current_price"),
                    "Score": analysis.get("score"),
                    "Recommendation": analysis.get("recommendation"),
                    "Time Horizon": analysis.get("time_horizon"),
                }
            )
        except Exception:
            rows.append({"Ticker": ticker, "Price": None, "Score": None, "Recommendation": "Unavailable"})
    return rows


@router.post("/{ticker}")
def create_watchlist(ticker: str) -> dict:
    try:
        add_to_watchlist(ticker)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to add ticker to watchlist") from exc
    return {"status": "ok"}


@router.delete("/{ticker}")
def delete_watchlist(ticker: str) -> dict:
    try:
        remove_from_watchlist(ticker)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to remove ticker from watchlist") from exc
    return {"status": "ok"}


@router.post("/{ticker}/portfolio")
def move_to_portfolio(ticker: str) -> dict:
    try:
        add_holding(ticker=ticker, shares=0, avg_cost=0, date_purchased="", notes="Moved from watchlist")
        remove_from_watchlist(ticker)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to move ticker to portfolio") from exc
    return {"status": "ok"}
