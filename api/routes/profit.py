from __future__ import annotations

import contextlib
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
            "fast_filtered": 0,
            "fully_analyzed": 0,
            "revalidation_status": "not_requested",
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
        fast_filtered=int(state.get("fast_filtered", 0)),
        fully_analyzed=int(state.get("fully_analyzed", 0)),
        failed_count=int(state.get("failed_count", 0)),
        failed_tickers=state.get("failed_tickers", []),
        revalidation_status=state.get("revalidation_status", "not_requested"),
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
    from modules.prediction_tracker import SUBSCORE_ROW_FIELDS, record_predictions_from_scan
    from modules.profit_opportunities import (
        add_estimated_dates,
        collect_universe_tickers,
        filter_by_upside,
        scan_profit_opportunities,
    )

    horizon = str(params.get("horizon", "Short-Term (1–4 weeks)"))
    min_upside_pct = float(params.get("min_upside_pct", 15.0))
    max_results = int(params.get("max_results", 25))
    scan_mode = str(params.get("scan_mode", "fast"))
    use_fast_screen = bool(params.get("use_fast_screen", True))
    fast_screen_margin = int(params.get("fast_screen_margin", 15))

    selected = params.get("universes", ["S&P 500", "NASDAQ"])
    horizon_map = {
        "Short-Term (1–4 weeks)": "short_term",
        "Medium-Term (1–6 months)": "medium_term",
        "Long-Term (6+ months)": "long_term",
    }
    key = horizon_map.get(horizon, "short_term")

    tickers, source_labels = collect_universe_tickers(selected)

    def _save_state(state: dict) -> None:
        redis_client.set(f"job:{job_id}", json.dumps(state), ex=3600)

    def _load_state() -> dict:
        return json.loads(redis_client.get(f"job:{job_id}") or "{}")

    initial_state = _load_state()
    created_at = initial_state.get("created_at")

    def _stop_requested() -> bool:
        return bool(_load_state().get("stop_requested"))

    def _on_total_known(total: int) -> None:
        state = _load_state()
        state["total"] = total
        _save_state(state)

    def _on_ticker_processed(ticker: str, counts: dict) -> None:
        _save_state(
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

    scanned = scan_profit_opportunities(
        tickers,
        key,
        use_fast_screen=use_fast_screen,
        fast_screen_margin=fast_screen_margin,
        parallel=(scan_mode == "fast"),
        source_labels=source_labels,
        on_ticker_processed=_on_ticker_processed,
        stop_check=_stop_requested,
        on_total_known=_on_total_known,
    )

    filtered = filter_by_upside(scanned, key, min_upside_pct=min_upside_pct, max_results=max_results)
    final_rows = add_estimated_dates(filtered.to_dict("records"), key) if not filtered.empty else []

    # Record predictions for the track-record feature. Must happen before the
    # internal _rsi/_*_score fields below are stripped, since
    # record_predictions_from_scan reads the sub-score fields to populate
    # sub_scores for compute_subscore_correlation_stats.
    with contextlib.suppress(Exception):  # Never let tracking failures break the task
        record_predictions_from_scan(final_rows, horizon=key, source="profit_opportunities")

    for row in final_rows:
        row.pop("_rsi", None)
        row.pop("_lightgbm_backtested", None)
        for field in SUBSCORE_ROW_FIELDS:
            row.pop(field, None)

    stopped = bool(scanned.attrs.get("stopped", False))
    total = int(scanned.attrs.get("source_ticker_count", len(tickers)))
    _save_state(
        {
            "status": "stopped" if stopped else "complete",
            "screened": total,
            "total": total,
            "current_ticker": None,
            "results": final_rows,
            "qualified": int(scanned.attrs.get("qualified_count", 0)),
            "fast_filtered": int(scanned.attrs.get("fast_filtered_count", 0)),
            "fully_analyzed": int(scanned.attrs.get("fully_analyzed_count", 0)),
            "failed_count": int(scanned.attrs.get("failed_count", 0)),
            "failed_tickers": scanned.attrs.get("failed_tickers", []),
            "revalidation_status": "not_requested",
            "stop_requested": stopped,
            "created_at": created_at,
        }
    )
