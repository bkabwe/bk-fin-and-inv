from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from api.deps import redis_client
from api.schemas.models import JobResponse, ProgressResponse, ScreenerRequest
from api.worker import celery_app

router = APIRouter(prefix="/screener", tags=["screener"])


def _key(job_id: str) -> str:
    return f"job:{job_id}"


def _load_state(job_id: str) -> dict | None:
    raw = redis_client.get(_key(job_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _save_state(job_id: str, state: dict) -> None:
    redis_client.set(_key(job_id), json.dumps(state), ex=3600)


@router.post("/start", response_model=JobResponse)
def start_screener(payload: ScreenerRequest) -> JobResponse:
    job_id = str(uuid4())
    state = {
        "status": "running",
        "screened": 0,
        "total": 0,
        "current_ticker": None,
        "results": [],
        "qualified": 0,
        "stop_requested": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_state(job_id, state)
    run_screener_task.delay(job_id, payload.model_dump())
    return JobResponse(job_id=job_id, status="started")


@router.get("/{job_id}/progress", response_model=ProgressResponse)
def screener_progress(job_id: str) -> ProgressResponse:
    state = _load_state(job_id)
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
def stop_screener(job_id: str) -> dict:
    state = _load_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    state["stop_requested"] = True
    _save_state(job_id, state)
    return {"status": "stop_requested"}


@celery_app.task
def run_screener_task(job_id: str, params: dict):
    from api.deps import redis_client
    from modules.data_fetcher import get_nasdaq_tickers, get_nyseamerican_tickers, get_otc_tickers, get_sp500_tickers
    from modules.scoring_engine import analyze_stock
    from modules.validators import sanitize_ticker

    universe_map = {
        "sp500": get_sp500_tickers,
        "nasdaq": get_nasdaq_tickers,
        "nyseamerican": get_nyseamerican_tickers,
        "otc": get_otc_tickers,
    }

    custom = params.get("custom_tickers", [])
    if params.get("universe") == "custom" and custom:
        tickers = []
        for ticker in custom:
            if not ticker:
                continue
            try:
                tickers.append(sanitize_ticker(ticker))
            except ValueError:
                continue
    else:
        fetcher = universe_map.get(params.get("universe"), get_sp500_tickers)
        try:
            tickers = fetcher()
        except Exception:
            tickers = []

    total = len(tickers)
    min_score = params.get("min_score", 60)
    max_results = params.get("max_results", 25)

    def _save(state):
        redis_client.set(f"job:{job_id}", json.dumps(state), ex=3600)

    state = json.loads(redis_client.get(f"job:{job_id}") or "{}")
    state["total"] = total
    _save(state)

    rows = []
    for i, ticker in enumerate(tickers, 1):
        current = json.loads(redis_client.get(f"job:{job_id}") or "{}")
        if current.get("stop_requested"):
            current["status"] = "stopped"
            current["screened"] = i - 1
            current["results"] = sorted(rows, key=lambda x: x.get("Score", 0), reverse=True)[:max_results]
            current["qualified"] = len(rows)
            _save(current)
            return

        current["screened"] = i
        current["current_ticker"] = ticker
        current["qualified"] = len(rows)
        _save(current)

        try:
            result = analyze_stock(ticker)
            score = result.get("score", 0)
            if score >= min_score:
                rows.append(
                    {
                        "Ticker": ticker,
                        "Company": result.get("company", ticker),
                        "Score": score,
                        "Recommendation": result.get("recommendation", ""),
                        "Time Horizon": result.get("time_horizon", ""),
                        "Current Price": result.get("current_price"),
                        "Entry Price": result.get("entry_price"),
                        "Target Price": result.get("target_price"),
                        "Stop Loss": result.get("stop_loss"),
                    }
                )
        except Exception:
            pass

    final = sorted(rows, key=lambda x: x.get("Score", 0), reverse=True)[:max_results]
    _save(
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
