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
    import concurrent.futures
    import threading

    from api.deps import redis_client
    from modules.data_fetcher import get_nasdaq_tickers, get_nyseamerican_tickers, get_otc_tickers, get_sp500_tickers
    from modules.prediction_tracker import record_predictions_from_scan
    from modules.scoring_engine import analyze_stock, fast_screen_score
    from modules.tiingo_client import revalidate_profit_rows, summarize_revalidation

    universe_map = {
        "S&P 500": get_sp500_tickers,
        "NASDAQ": get_nasdaq_tickers,
        "NYSE American": get_nyseamerican_tickers,
        "OTC": get_otc_tickers,
    }

    horizon = str(params.get("horizon", "Short-Term (1–4 weeks)"))
    min_upside_pct = float(params.get("min_upside_pct", 15.0))
    max_results = int(params.get("max_results", 25))
    scan_mode = str(params.get("scan_mode", "fast"))
    use_fast_screen = bool(params.get("use_fast_screen", True))
    fast_screen_margin = int(params.get("fast_screen_margin", 15))
    revalidate_with_tiingo = bool(params.get("revalidate_with_tiingo", False))
    revalidate_top_n = int(params.get("revalidate_top_n", 50))
    # Conservative proxy threshold matching the Streamlit page logic.
    _FAST_SCREEN_BASE = 30  # base proxy cutoff (out of 100, matching Streamlit page)
    _FAST_SCREEN_PROXY_THRESHOLD = _FAST_SCREEN_BASE - fast_screen_margin  # e.g. 15 with default margin

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

    horizon_map = {
        "Short-Term (1–4 weeks)": "short_term",
        "Medium-Term (1–6 months)": "medium_term",
        "Long-Term (6+ months)": "long_term",
    }
    key = horizon_map.get(horizon, "short_term")

    rows: list[dict] = []
    failed_tickers: list[dict] = []

    if scan_mode == "fast":
        # ------------------------------------------------------------------
        # FAST MODE: parallel thread pool + optional fast-screen pre-filter
        # ------------------------------------------------------------------
        fast_filtered = 0
        fully_analyzed = 0
        processed = 0
        _stop_requested = [False]
        _lock = threading.Lock()

        def _process(ticker: str) -> None:
            nonlocal fast_filtered, fully_analyzed, processed
            try:
                with _lock:
                    if _stop_requested[0]:
                        return
                if use_fast_screen:
                    fast_score, err = fast_screen_score(ticker)
                    if err is not None:
                        with _lock:
                            failed_tickers.append({"ticker": ticker, "reason": f"fast-screen error: {err}"})
                        return
                    if fast_score < _FAST_SCREEN_PROXY_THRESHOLD:
                        with _lock:
                            fast_filtered += 1
                        return

                result = analyze_stock(ticker)
                with _lock:
                    fully_analyzed += 1
                projections = result.get("projections") or {}
                upside = projections.get(f"{key}_upside")
                target = projections.get(f"{key}_target")
                if upside is None:
                    return
                upside = float(upside)
                if upside < min_upside_pct:
                    return
                confidence = projections.get("data_quality", "Limited")
                with _lock:
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
            except Exception as exc:
                with _lock:
                    failed_tickers.append({"ticker": ticker, "reason": str(exc) or type(exc).__name__})
            finally:
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
                        "revalidation_status": "pending" if revalidate_with_tiingo else "not_requested",
                        "created_at": state.get("created_at"),
                    }
                _save_state(snap)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(_process, t): t for t in tickers}
            for future in concurrent.futures.as_completed(futures):
                snap = json.loads(redis_client.get(f"job:{job_id}") or "{}")
                if snap.get("stop_requested"):
                    with _lock:
                        _stop_requested[0] = True
                    # Cancel pending futures and wait for in-flight workers to
                    # finish before reading shared counters/rows to avoid a race.
                    executor.shutdown(wait=True, cancel_futures=True)
                    with _lock:
                        final_rows = sorted(rows, key=lambda x: x.get("Projected Upside %", 0), reverse=True)[:max_results]
                        _save_state({
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
                    record_predictions_from_scan(final_rows, horizon=key, source="profit_opportunities")
                    return
                try:
                    future.result()
                except Exception:
                    pass

        with _lock:
            ranked_rows = sorted(rows, key=lambda x: x.get("Projected Upside %", 0), reverse=True)
            qualified_count = len(rows)
            fast_filtered_count = fast_filtered
            fully_analyzed_count = fully_analyzed
            failed_count = len(failed_tickers)
            failed_snapshot = list(failed_tickers)
        if revalidate_with_tiingo:
            state = _load_state()
            state["revalidation_status"] = "running"
            _save_state(state)
            ranked_rows = revalidate_profit_rows(ranked_rows, horizon_key=key, top_n=revalidate_top_n)
            revalidation_status = str(summarize_revalidation(ranked_rows).get("status", "unavailable"))
        else:
            revalidation_status = "not_requested"
        final_rows = ranked_rows[:max_results]
        _save_state({
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
        })

    else:
        # ------------------------------------------------------------------
        # THOROUGH MODE: original sequential loop (unchanged behaviour)
        # ------------------------------------------------------------------
        for i, ticker in enumerate(tickers, start=1):
            current = _load_state()
            if current.get("stop_requested"):
                current["status"] = "stopped"
                current["screened"] = i - 1
                current["current_ticker"] = None
                current["qualified"] = len(rows)
                current["failed_count"] = len(failed_tickers)
                current["failed_tickers"] = failed_tickers
                current["results"] = sorted(rows, key=lambda x: x.get("Projected Upside %", 0), reverse=True)[:max_results]
                current["revalidation_status"] = "not_requested"
                _save_state(current)
                return

            current["screened"] = i
            current["current_ticker"] = ticker
            current["qualified"] = len(rows)
            current["revalidation_status"] = "pending" if revalidate_with_tiingo else "not_requested"
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
            except Exception as exc:
                failed_tickers.append({"ticker": ticker, "reason": str(exc) or type(exc).__name__})

            if i % 50 == 0:
                current = _load_state()
                current["results"] = sorted(rows, key=lambda x: x.get("Projected Upside %", 0), reverse=True)[:max_results]
                current["qualified"] = len(rows)
                current["failed_count"] = len(failed_tickers)
                current["revalidation_status"] = "pending" if revalidate_with_tiingo else "not_requested"
                _save_state(current)

        ranked_rows = sorted(rows, key=lambda x: x.get("Projected Upside %", 0), reverse=True)

        if revalidate_with_tiingo:
            current = _load_state()
            current["revalidation_status"] = "running"
            _save_state(current)
            ranked_rows = revalidate_profit_rows(ranked_rows, horizon_key=key, top_n=revalidate_top_n)
            revalidation_status = str(summarize_revalidation(ranked_rows).get("status", "unavailable"))
        else:
            revalidation_status = "not_requested"
        final_rows = ranked_rows[:max_results]
        final_state = _load_state()
        _save_state(
            {
                "status": "complete",
                "screened": total,
                "total": total,
                "current_ticker": None,
                "results": final_rows,
                "qualified": len(rows),
                "fast_filtered": 0,
                "fully_analyzed": total - len(failed_tickers),
                "failed_count": len(failed_tickers),
                "failed_tickers": failed_tickers,
                "revalidation_status": revalidation_status,
                "stop_requested": False,
                "created_at": final_state.get("created_at"),
            }
        )

    # Record predictions for the track-record feature.
    # Use final_rows already computed in each branch (avoids a Redis re-fetch).
    try:
        record_predictions_from_scan(final_rows, horizon=key, source="profit_opportunities")
    except Exception:
        pass  # Never let tracking failures break the task
