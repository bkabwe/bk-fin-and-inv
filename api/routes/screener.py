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
        "fast_filtered": 0,
        "fully_analyzed": 0,
        "failed_count": 0,
        "failed_tickers": [],
        "revalidation_status": "not_requested",
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
        fast_filtered=int(state.get("fast_filtered", 0)),
        fully_analyzed=int(state.get("fully_analyzed", 0)),
        failed_count=int(state.get("failed_count", 0)),
        failed_tickers=state.get("failed_tickers", []),
        revalidation_status=state.get("revalidation_status", "not_requested"),
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
    from modules.screener import run_screener

    def _save(state: dict) -> None:
        redis_client.set(f"job:{job_id}", json.dumps(state), ex=3600)

    def _load() -> dict:
        return json.loads(redis_client.get(f"job:{job_id}") or "{}")

    initial_state = _load()
    created_at = initial_state.get("created_at")

    def _stop_requested() -> bool:
        return bool(_load().get("stop_requested"))

    def _on_ticker_processed(ticker: str, counts: dict) -> None:
        _save(
            {
                "status": "running",
                "screened": counts["screened"],
                "total": counts["total"],
                "current_ticker": ticker,
                "qualified": counts["qualified"],
                "fast_filtered": counts["fast_filtered"],
                "fully_analyzed": counts["fully_analyzed"],
                "failed_count": counts["failed_count"],
                "results": [],
                "stop_requested": _stop_requested(),
                "revalidation_status": "not_requested",
                "created_at": created_at,
            }
        )

    custom_tickers = params.get("custom_tickers", [])
    universe = params.get("universe", "sp500")

    def _on_total_known(total: int) -> None:
        state = _load()
        state["total"] = total
        _save(state)

    try:
        results = run_screener(
            universe=universe,
            min_score=params.get("min_score", 60),
            max_results=params.get("max_results", 25),
            custom_tickers=custom_tickers if universe == "custom" else None,
            use_fast_screen=params.get("use_fast_screen", True),
            fast_screen_margin=params.get("fast_screen_margin", 15),
            on_ticker_processed=_on_ticker_processed,
            stop_check=_stop_requested,
            on_total_known=_on_total_known,
        )
    except RuntimeError:
        results = None

    if results is None:
        rows: list[dict] = []
        total = 0
        stopped = False
        qualified_count = fast_filtered_count = fully_analyzed_count = failed_count = 0
        failed_tickers: list[dict] = []
    else:
        rows = results.to_dict(orient="records")
        total = int(results.attrs.get("source_ticker_count", 0))
        stopped = bool(results.attrs.get("stopped", False))
        qualified_count = int(results.attrs.get("qualified_count", 0))
        fast_filtered_count = int(results.attrs.get("fast_filtered_count", 0))
        fully_analyzed_count = int(results.attrs.get("fully_analyzed_count", 0))
        failed_count = int(results.attrs.get("failed_count", 0))
        failed_tickers = results.attrs.get("failed_tickers", [])

    _save(
        {
            "status": "stopped" if stopped else "complete",
            "screened": total,
            "total": total,
            "current_ticker": None,
            "results": rows,
            "qualified": qualified_count,
            "fast_filtered": fast_filtered_count,
            "fully_analyzed": fully_analyzed_count,
            "failed_count": failed_count,
            "failed_tickers": failed_tickers,
            "revalidation_status": "not_requested",
            "stop_requested": stopped,
            "created_at": created_at,
        }
    )
