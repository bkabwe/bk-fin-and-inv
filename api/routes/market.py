from __future__ import annotations

from fastapi import APIRouter

from modules.data_fetcher import get_stock_data
from modules.notifications import get_notifications, mark_all_read

router = APIRouter(tags=["market"])


@router.get("/market/overview")
def market_overview() -> dict:
    items = {}
    for ticker in ["SPY", "QQQ", "DIA", "IWM"]:
        df = get_stock_data(ticker, period="5d", interval="1d")
        if df.empty or len(df) < 2:
            items[ticker] = {"price": None, "daily_pct": None}
            continue
        close = df["Close"].astype(float)
        prev = float(close.iloc[-2])
        last = float(close.iloc[-1])
        pct = ((last - prev) / prev * 100) if prev else 0.0
        items[ticker] = {"price": round(last, 2), "daily_pct": round(pct, 2)}
    return items


@router.get("/notifications")
def notifications() -> list[dict]:
    return get_notifications()


@router.post("/notifications/read")
def notifications_read() -> dict:
    mark_all_read()
    return {"status": "ok"}
