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
    import concurrent.futures
    import threading

    from api.deps import redis_client
    from modules.data_fetcher import get_nasdaq_tickers, get_nyseamerican_tickers, get_otc_tickers, get_sp500_tickers
    from modules.scoring_engine import analyze_stock, fast_screen_score
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
    use_fast_screen = params.get("use_fast_screen", True)
    fast_screen_margin = params.get("fast_screen_margin", 15)
    fast_screen_threshold = min_score - fast_screen_margin

    def _save(state):
        redis_client.set(f"job:{job_id}", json.dumps(state), ex=3600)

    state = json.loads(redis_client.get(f"job:{job_id}") or "{}")
    state["total"] = total
    _save(state)

    rows: list[dict] = []
    failed_tickers: list[dict] = []
    fast_filtered = 0
    fully_analyzed = 0
    processed = 0
    _stop_requested = [False]  # mutable container so _process can check it under lock
    _lock = threading.Lock()

    def _process(ticker: str) -> None:
        nonlocal fast_filtered, fully_analyzed, processed
        try:
            # Check stop flag via in-memory lock (avoids a Redis round-trip per ticker)
            with _lock:
                if _stop_requested[0]:
                    return
            if use_fast_screen:
                fast_score, err = fast_screen_score(ticker)
                if err is not None:
                    with _lock:
                        failed_tickers.append({"ticker": ticker, "reason": f"fast-screen error: {err}"})
                    return
                if fast_score < fast_screen_threshold:
                    with _lock:
                        fast_filtered += 1
                    return
            result = analyze_stock(ticker)
            score = result.get("score", 0)
            with _lock:
                fully_analyzed += 1
            if score >= min_score:
                with _lock:
                    rows.append(
                        {
                            "Ticker": ticker,
                            "Company": result.get("company", ticker),
                            "Score": score,
                            "Recommendation": result.get("recommendation", ""),
                            "Time Horizon": result.get("time_horizon", ""),
                            "Sector Trend": str(result.get("sector_trend") or "unknown").replace("_", " ").title(),
                            "Market Cap Tier": result.get("market_cap_tier") or "unknown",
                            "Long-Term Stage": result.get("longterm_stage") or "",
                            "Current Price": result.get("current_price"),
                            "Entry Price": result.get("entry_price"),
                            "Target Price": result.get("target_price"),
                            "Stop Loss": result.get("stop_loss"),
                        }
                    )
        except Exception as exc:
            with _lock:
                failed_tickers.append({"ticker": ticker, "reason": str(exc) or type(exc).__name__})
        finally:
            # Build the progress snapshot from in-memory state under the lock to
            # avoid a non-atomic Redis read→write that could overwrite newer data.
            with _lock:
                processed += 1
                snap = {
                    "status": "running",
                    "screened": processed,
                    "total": total,
                    "current_ticker": ticker,
                    "qualified": len(rows),
                    "fast_filtered": fast_filtered,
                    "fully_analyzed": fully_analyzed,
                    "failed_count": len(failed_tickers),
                    "results": [],
                    "stop_requested": _stop_requested[0],
                    "revalidation_status": "not_requested",
                    "created_at": state.get("created_at"),
                }
            _save(snap)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_process, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            snap = json.loads(redis_client.get(f"job:{job_id}") or "{}")
            if snap.get("stop_requested"):
                with _lock:
                    _stop_requested[0] = True
                executor.shutdown(wait=False, cancel_futures=True)
                with _lock:
                    final_rows = sorted(rows, key=lambda x: x.get("Score", 0), reverse=True)[:max_results]
                    _save({
                        "status": "stopped",
                        "screened": processed,
                        "total": total,
                        "current_ticker": None,
                        "results": final_rows,
                        "qualified": len(rows),
                        "fast_filtered": fast_filtered,
                        "fully_analyzed": fully_analyzed,
                        "failed_count": len(failed_tickers),
                        "failed_tickers": failed_tickers,
                        "revalidation_status": "not_requested",
                        "stop_requested": True,
                        "created_at": state.get("created_at"),
                    })
                return
            try:
                future.result()
            except Exception:
                pass

    with _lock:
        ranked_rows = sorted(rows, key=lambda x: x.get("Score", 0), reverse=True)
        qualified_count = len(rows)
        fast_filtered_count = fast_filtered
        fully_analyzed_count = fully_analyzed
        failed_count = len(failed_tickers)
        failed_snapshot = list(failed_tickers)
    revalidation_status = "not_requested"
    final_rows = ranked_rows[:max_results]
    with _lock:
        _save(
            {
                "status": "complete",
                "screened": total,
                "total": total,
                "current_ticker": None,
                "results": final_rows,
                "qualified": qualified_count,
                "fast_filtered": fast_filtered_count,
                "fully_analyzed": fully_analyzed_count,
                "failed_count": failed_count,
                "failed_tickers": failed_snapshot,
                "revalidation_status": revalidation_status,
                "stop_requested": False,
                "created_at": state.get("created_at"),
            }
        )
