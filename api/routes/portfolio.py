from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from api.deps import redis_client
from api.schemas.models import HoldingRequest, JobResponse, ProgressResponse
from api.worker import celery_app
from modules.portfolio import add_holding, get_portfolio, remove_holding, update_holding
from modules.scoring_engine import analyze_stock, get_price_projections

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("")
def list_portfolio() -> list[dict]:
    return get_portfolio()


@router.post("/holding")
def create_holding(payload: HoldingRequest) -> dict:
    add_holding(payload.ticker, payload.shares, payload.avg_cost, payload.date_purchased, payload.notes)
    return {"status": "ok"}


@router.put("/holding/{ticker}")
def edit_holding(ticker: str, payload: HoldingRequest) -> dict:
    update_holding(ticker, payload.shares, payload.avg_cost)
    return {"status": "ok"}


@router.delete("/holding/{ticker}")
def delete_holding(ticker: str) -> dict:
    remove_holding(ticker)
    return {"status": "ok"}


@router.post("/analyze/start", response_model=JobResponse)
def start_portfolio_analysis() -> JobResponse:
    job_id = str(uuid4())
    redis_client.set(
        f"job:{job_id}",
        json.dumps(
            {
                "status": "running",
                "screened": 0,
                "total": len(get_portfolio()),
                "current_ticker": None,
                "results": [],
                "qualified": 0,
                "stop_requested": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        ex=3600,
    )
    analyze_portfolio_task.delay(job_id)
    return JobResponse(job_id=job_id, status="started")


@router.get("/analyze/{job_id}/progress", response_model=ProgressResponse)
def portfolio_progress(job_id: str) -> ProgressResponse:
    raw = redis_client.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    state = json.loads(raw)
    return ProgressResponse(
        job_id=job_id,
        status=state.get("status", "error"),
        screened=int(state.get("screened", 0)),
        total=int(state.get("total", 0)),
        current_ticker=state.get("current_ticker"),
        results=state.get("results", []),
        qualified=int(state.get("qualified", 0)),
    )


@celery_app.task
def analyze_portfolio_task(job_id: str):
    holdings = get_portfolio()
    total = len(holdings)
    rows: list[dict] = []

    for i, holding in enumerate(holdings, start=1):
        ticker = holding.get("ticker", "")
        state = json.loads(redis_client.get(f"job:{job_id}") or "{}")
        state["screened"] = i
        state["total"] = total
        state["current_ticker"] = ticker
        state["qualified"] = len(rows)
        redis_client.set(f"job:{job_id}", json.dumps(state), ex=3600)

        try:
            analysis = analyze_stock(ticker, avg_cost=float(holding.get("avg_cost", 0) or 0))
            projections = analysis.get("projections") or get_price_projections(ticker)
            shares = float(holding.get("shares", 0) or 0)
            avg = float(holding.get("avg_cost", 0) or 0)
            current = float(analysis.get("current_price") or 0)
            value = shares * current
            basis = shares * avg
            pnl = value - basis
            rows.append(
                {
                    "Ticker": ticker,
                    "Shares": shares,
                    "Avg Cost": avg,
                    "Current Price": round(current, 2),
                    "Current Value": round(value, 2),
                    "Cost Basis": round(basis, 2),
                    "P&L $": round(pnl, 2),
                    "P&L %": round((pnl / basis * 100), 2) if basis else 0,
                    "Score": analysis.get("score", 0),
                    "Recommendation": analysis.get("recommendation", ""),
                    "Projection": projections,
                }
            )
        except Exception as exc:
            rows.append({"Ticker": ticker, "Error": str(exc)})

    redis_client.set(
        f"job:{job_id}",
        json.dumps(
            {
                "status": "complete",
                "screened": total,
                "total": total,
                "current_ticker": None,
                "results": rows,
                "qualified": len(rows),
            }
        ),
        ex=3600,
    )
