from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from api.deps import redis_client
from api.schemas.models import JobResponse, ProfitRequest, ProgressResponse
from api.worker import celery_app

router = APIRouter(prefix="/profit", tags=["profit"])


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def _load(job_id: str) -> dict | None:
    raw = redis_client.get(_key(job_id))
    if not raw:
        return None
    return json.loads(raw)


def _save(job_id: str, payload: dict) -> None:
    redis_client.set(_key(job_id), json.dumps(payload), ex=3600)


@router.post("/start", response_model=JobResponse)
def start_profit(payload: ProfitRequest) -> JobResponse:
    job_id = str(uuid4())
    _save(
        job_id,
        {
            "status": "running",
            "screened": 0,
            "total": 0,
            "current_ticker": None,
            "results": [],
            "qualified": 0,
            "stop_requested": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    run_profit_task.delay(job_id, payload.model_dump())
    return JobResponse(job_id=job_id, status="started")


@router.get("/{job_id}/progress", response_model=ProgressResponse)
def profit_progress(job_id: str) -> ProgressResponse:
    state = _load(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    return ProgressResponse(
        job_id=job_id,
        status=state.get("status", "error"),
        screened=int(state.get("screened", 0)),
        total=int(state.get("total", 0)),
        current_ticker=state.get("current_ticker"),
        results=state.get("results", []),
        qualified=int(state.get("qualified", 0)),
    )


@router.post("/{job_id}/stop")
def stop_profit(job_id: str) -> dict:
    state = _load(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    state["stop_requested"] = True
    _save(job_id, state)
    return {"status": "stop_requested"}


@celery_app.task
def run_profit_task(job_id: str, params: dict):
    from api.deps import redis_client
    from modules.data_fetcher import get_nasdaq_tickers, get_nyseamerican_tickers, get_otc_tickers, get_sp500_tickers
    from modules.scoring_engine import analyze_stock

    universe_map = {
        "S&P 500": get_sp500_tickers,
        "NASDAQ": get_nasdaq_tickers,
        "NYSE American": get_nyseamerican_tickers,
        "OTC": get_otc_tickers,
    }

    horizons = params.get("horizon", "Short-Term (1–4 weeks)")
    min_upside_pct = float(params.get("min_upside_pct", 15.0))
    max_results = int(params.get("max_results", 25))

    selected = params.get("universes", ["S&P 500", "NASDAQ"])
    all_tickers: list[str] = []
    for name in selected:
        fetcher = universe_map.get(name)
        if not fetcher:
            continue
        try:
            all_tickers.extend(fetcher())
        except Exception:
            continue

    tickers = sorted(set(all_tickers))
    total = len(tickers)

    def _load_state() -> dict:
        return json.loads(redis_client.get(f"job:{job_id}") or "{}")

    def _save_state(payload: dict) -> None:
        redis_client.set(f"job:{job_id}", json.dumps(payload), ex=3600)

    state = _load_state()
    state["total"] = total
    _save_state(state)

    if "short" in horizons.lower():
        key = "short_term"
    elif "medium" in horizons.lower():
        key = "medium_term"
    else:
        key = "long_term"

    rows: list[dict] = []
    for i, ticker in enumerate(tickers, start=1):
        current = _load_state()
        if current.get("stop_requested"):
            current["status"] = "stopped"
            current["screened"] = i - 1
            current["current_ticker"] = None
            current["qualified"] = len(rows)
            current["results"] = sorted(rows, key=lambda x: x.get("Projected Upside %", 0), reverse=True)[:max_results]
            _save_state(current)
            return

        current["screened"] = i
        current["current_ticker"] = ticker
        current["qualified"] = len(rows)
        _save_state(current)

        try:
            result = analyze_stock(ticker)
            projections = result.get("projections") or {}
            upside = projections.get(f"{key}_upside")
            target = projections.get(f"{key}_target")
            if upside is None:
                continue
            upside = float(upside)
            if upside < min_upside_pct:
                continue

            confidence = projections.get("data_quality", "Limited")
            rows.append(
                {
                    "Ticker": ticker,
                    "Company": result.get("company", ticker),
                    "Index": ", ".join(selected),
                    "Score": result.get("score", 0),
                    "Current Price": result.get("current_price"),
                    "Target Price": target,
                    "Projected Upside %": round(upside, 2),
                    "Est. Target Date": None,
                    "Confidence": confidence,
                }
            )
        except Exception:
            continue

        if i % 50 == 0:
            current = _load_state()
            current["results"] = sorted(rows, key=lambda x: x.get("Projected Upside %", 0), reverse=True)[:max_results]
            current["qualified"] = len(rows)
            _save_state(current)

    final = sorted(rows, key=lambda x: x.get("Projected Upside %", 0), reverse=True)[:max_results]
    _save_state(
        {
            "status": "complete",
            "screened": total,
            "total": total,
            "current_ticker": None,
            "results": final,
            "qualified": len(rows),
            "stop_requested": False,
            "created_at": state.get("created_at"),
        }
    )
