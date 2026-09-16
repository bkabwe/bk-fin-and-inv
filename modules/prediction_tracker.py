"""Prediction Tracker — persist price projections and resolve them after their target date.

Uses the same atomic-JSON-write pattern as modules/portfolio.py.
Storage: data/predictions.json (excluded from git via .gitignore data/*.json).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from modules.data_fetcher import get_stock_data
from modules.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"

# Horizon → midpoint days used for target_date calculation.
# Align with estimate_target_date fallback_days in pages/6_Profit_Opportunities.py.
HORIZON_DAYS: dict[str, int] = {
    "short_term": 21,
    "medium_term": 90,
    "long_term": 365,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: Any) -> None:
    """Write JSON atomically using temp file + rename to prevent corruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, indent=2, default=str)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _load_all() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PREDICTIONS_FILE.exists():
        return []
    try:
        return json.loads(PREDICTIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_all(records: list[dict]) -> None:
    _atomic_write(PREDICTIONS_FILE, records)


# ---------------------------------------------------------------------------
# Public API — recording
# ---------------------------------------------------------------------------

def _dedupe_key(ticker: str, horizon: str, scan_date: str) -> str:
    """Natural dedupe key: ticker + horizon + ISO scan date (day precision)."""
    return f"{ticker.upper()}|{horizon}|{scan_date[:10]}"


def record_prediction(
    *,
    ticker: str,
    company: str,
    horizon: str,  # "short_term" | "medium_term" | "long_term"
    current_price: float,
    target_price: float,
    target_low: float,
    target_high: float,
    projected_upside_pct: float,
    score: int | float,
    data_quality: str,
    models_used: str,
    source: str,  # "profit_opportunities" | "stock_analysis" | "screener"
) -> str | None:
    """Persist a prediction if no unresolved record exists for (ticker, horizon) today.

    Returns the prediction id if recorded, or None if it was deduped.
    """
    today = date.today().isoformat()
    records = _load_all()

    # Dedupe: skip if there's already a pending prediction for the same ticker+horizon today.
    pending = {
        _dedupe_key(r["ticker"], r["horizon"], r["scan_date"])
        for r in records
        if r.get("status") == "pending"
    }
    dk = _dedupe_key(ticker, horizon, today)
    if dk in pending:
        logger.debug("Prediction already recorded for %s (dedupe_key=%s)", ticker, dk)
        return None

    horizon_days = HORIZON_DAYS.get(horizon, 90)
    target_date = (date.today() + timedelta(days=horizon_days)).isoformat()

    rec: dict = {
        "id": str(uuid4()),
        "ticker": ticker.upper(),
        "company": company or ticker.upper(),
        "horizon": horizon,
        "scan_date": today,
        "target_date": target_date,
        "current_price_at_scan": round(float(current_price), 4),
        "target_price": round(float(target_price), 4),
        "target_low": round(float(target_low), 4),
        "target_high": round(float(target_high), 4),
        "projected_upside_pct": round(float(projected_upside_pct), 4),
        "score": score,
        "data_quality": data_quality,
        "models_used": models_used,
        "source": source,
        "status": "pending",
        # resolved fields — filled in later
        "actual_price_at_target_date": None,
        "actual_return_pct": None,
        "hit_target": None,
        "hit_band": None,
        "resolved_at": None,
    }
    records.append(rec)
    _save_all(records)
    logger.info("Recorded prediction %s for %s (%s) — target %s", rec["id"], ticker, horizon, target_date)
    return rec["id"]


def record_predictions_from_scan(
    display_rows: list[dict],
    horizon: str,
    source: str,
) -> int:
    """Convenience wrapper: record multiple predictions from a scan result list.

    Each item in display_rows is expected to have at minimum:
        ticker, company, score, current_price (or "Current Price"),
        target_price (or "Target Price"), upside (or "Projected Upside %"),
        and optionally target_low / target_high, data_quality, basis.

    Returns the number of newly recorded predictions.
    """
    count = 0
    for row in display_rows:
        ticker = str(row.get("ticker") or row.get("Ticker") or "").strip().upper()
        if not ticker:
            continue
        try:
            current_price = float(row.get("current_price") or row.get("Current Price") or 0)
            target_price = float(row.get("target_price") or row.get("Target Price") or 0)
            upside = float(row.get("projected_upside_pct") or row.get("Projected Upside %") or 0)
            if current_price <= 0 or target_price <= 0:
                continue
            # low/high — gracefully fall back to ±5 % of target if missing
            target_low = float(row.get("target_low") or target_price * 0.95)
            target_high = float(row.get("target_high") or target_price * 1.05)
            pid = record_prediction(
                ticker=ticker,
                company=str(row.get("company") or row.get("Company") or ticker),
                horizon=horizon,
                current_price=current_price,
                target_price=target_price,
                target_low=target_low,
                target_high=target_high,
                projected_upside_pct=upside,
                score=int(row.get("score") or row.get("Score") or 0),
                data_quality=str(row.get("data_quality") or row.get("Confidence") or "Limited"),
                models_used=str(row.get("basis") or row.get("Basis") or row.get("models_used") or ""),
                source=source,
            )
            if pid:
                count += 1
        except Exception as exc:
            logger.warning("Failed to record prediction for %s: %s", ticker, exc)
    return count


# ---------------------------------------------------------------------------
# Public API — excursion checks
# ---------------------------------------------------------------------------

def compute_max_price_since_scan(
    ticker: str,
    scan_date: str | date,
    end_date: str | date | None = None,
) -> float | None:
    try:
        start = scan_date if isinstance(scan_date, date) else date.fromisoformat(str(scan_date)[:10])
    except ValueError:
        return None

    try:
        stop = date.today() if end_date is None else (end_date if isinstance(end_date, date) else date.fromisoformat(str(end_date)[:10]))
    except ValueError:
        return None

    if stop < start:
        return None

    days_back = max(30, (stop - start).days + 10)
    df = get_stock_data(str(ticker or "").strip().upper(), period=f"{days_back}d", interval="1d")
    if df.empty:
        return None

    price_col = "High" if "High" in df.columns else "Close" if "Close" in df.columns else None
    if price_col is None:
        return None

    prices = df[price_col].astype(float)
    candidates = [
        float(price)
        for idx, price in zip(prices.index, prices.values)
        if start <= (idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx)[:10])) <= stop
        and price == price
    ]
    return round(max(candidates), 4) if candidates else None


# ---------------------------------------------------------------------------
# Public API — resolution
# ---------------------------------------------------------------------------

def resolve_pending_predictions() -> dict[str, int]:
    """Check all pending predictions whose target_date has passed and resolve them.

    Returns a summary dict: {"resolved": n, "no_data": n, "still_pending": n}.
    """
    records = _load_all()
    today = date.today()
    resolved_count = 0
    no_data_count = 0
    still_pending_count = 0

    for rec in records:
        if rec.get("status") != "pending":
            continue
        target_date_str = rec.get("target_date", "")
        try:
            target_date = date.fromisoformat(target_date_str)
        except ValueError:
            continue
        if target_date > today:
            still_pending_count += 1
            continue

        # Fetch historical price on/near the target date (use split-adjusted data).
        ticker = rec["ticker"]
        try:
            # Fetch enough history to cover the target date.
            days_back = max(30, (today - target_date).days + 10)
            period = f"{days_back}d"
            df = get_stock_data(ticker, period=period, interval="1d")
            if df.empty or "Close" not in df:
                raise ValueError("empty data")

            close = df["Close"].astype(float)
            # Normalize the index to date objects for comparison.
            index_dates = [
                d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])
                for d in close.index
            ]
            # Find the closest available date on or after target_date.
            candidates = [(d, p) for d, p in zip(index_dates, close.values) if d >= target_date]
            if not candidates:
                # Fall back to the most recent available date.
                candidates = list(zip(index_dates, close.values))
            if not candidates:
                raise ValueError("no price candidates")

            # Pick the date closest to target_date.
            actual_date, actual_price = min(candidates, key=lambda x: abs((x[0] - target_date).days))
            actual_price = float(actual_price)

            current_price = float(rec.get("current_price_at_scan") or 0)
            if current_price <= 0:
                raise ValueError("no current_price_at_scan")

            actual_return_pct = round(((actual_price - current_price) / current_price) * 100, 4)
            target_price = float(rec.get("target_price") or 0)
            target_low = float(rec.get("target_low") or target_price * 0.95)
            target_high = float(rec.get("target_high") or target_price * 1.05)

            hit_target = actual_price >= target_price if target_price > 0 else False
            hit_band = (target_low <= actual_price <= target_high) if (target_low > 0 and target_high > 0) else False

            rec["actual_price_at_target_date"] = round(actual_price, 4)
            rec["actual_return_pct"] = actual_return_pct
            rec["hit_target"] = hit_target
            rec["hit_band"] = hit_band
            rec["status"] = "resolved"
            rec["resolved_at"] = datetime.now(timezone.utc).isoformat()
            resolved_count += 1
            logger.info(
                "Resolved prediction %s for %s: actual=%.2f, hit_target=%s, hit_band=%s",
                rec["id"],
                ticker,
                actual_price,
                hit_target,
                hit_band,
            )
        except Exception as exc:
            logger.warning("Could not resolve prediction %s for %s: %s", rec.get("id"), ticker, exc)
            # Only mark unresolved_no_data if the target date is well past (7+ days),
            # to avoid permanently marking transient network failures.
            days_past = (today - target_date).days
            if days_past >= 7:
                rec["status"] = "unresolved_no_data"
                rec["resolved_at"] = datetime.now(timezone.utc).isoformat()
                no_data_count += 1
            else:
                still_pending_count += 1

    _save_all(records)
    return {
        "resolved": resolved_count,
        "no_data": no_data_count,
        "still_pending": still_pending_count,
    }


# ---------------------------------------------------------------------------
# Public API — reporting
# ---------------------------------------------------------------------------

def get_all_predictions() -> list[dict]:
    return _load_all()


def get_summary_stats() -> dict:
    """Compute hit-rate and accuracy stats across all resolved predictions."""
    records = _load_all()
    resolved = [r for r in records if r.get("status") == "resolved"]
    pending = [r for r in records if r.get("status") == "pending"]

    if not resolved:
        return {
            "total_predictions": len(records),
            "total_resolved": 0,
            "total_pending": len(pending),
            "overall_hit_rate_target": None,
            "overall_hit_rate_band": None,
            "mean_absolute_error_pct": None,
            "median_absolute_error_pct": None,
            "by_horizon": {},
            "by_source": {},
            "by_confidence": {},
        }

    def _stats(subset: list[dict]) -> dict:
        if not subset:
            return {"count": 0, "hit_rate_target": None, "hit_rate_band": None, "mae_pct": None}
        hits_t = [r for r in subset if r.get("hit_target")]
        hits_b = [r for r in subset if r.get("hit_band")]
        errors = [
            abs(
                (r["actual_return_pct"] if r.get("actual_return_pct") is not None else 0)
                - (r["projected_upside_pct"] if r.get("projected_upside_pct") is not None else 0)
            )
            for r in subset
        ]
        return {
            "count": len(subset),
            "hit_rate_target": round(len(hits_t) / len(subset) * 100, 1),
            "hit_rate_band": round(len(hits_b) / len(subset) * 100, 1),
            "mae_pct": round(sum(errors) / len(errors), 2) if errors else None,
        }

    horizons = {h: [r for r in resolved if r.get("horizon") == h] for h in HORIZON_DAYS}
    sources = {}
    for r in resolved:
        s = r.get("source", "unknown")
        sources.setdefault(s, []).append(r)
    confidences = {}
    for r in resolved:
        c = r.get("data_quality", "Unknown")
        confidences.setdefault(c, []).append(r)

    errors_all = [
        abs(
            (r["actual_return_pct"] if r.get("actual_return_pct") is not None else 0)
            - (r["projected_upside_pct"] if r.get("projected_upside_pct") is not None else 0)
        )
        for r in resolved
    ]
    sorted_errors = sorted(errors_all)
    n = len(sorted_errors)
    median_err = (
        sorted_errors[n // 2] if n % 2 == 1
        else (sorted_errors[n // 2 - 1] + sorted_errors[n // 2]) / 2
        if n > 0 else None
    )

    return {
        "total_predictions": len(records),
        "total_resolved": len(resolved),
        "total_pending": len(pending),
        "overall_hit_rate_target": round(
            len([r for r in resolved if r.get("hit_target")]) / len(resolved) * 100, 1
        ),
        "overall_hit_rate_band": round(
            len([r for r in resolved if r.get("hit_band")]) / len(resolved) * 100, 1
        ),
        "mean_absolute_error_pct": round(sum(errors_all) / n, 2) if n else None,
        "median_absolute_error_pct": round(median_err, 2) if median_err is not None else None,
        "by_horizon": {h: _stats(v) for h, v in horizons.items()},
        "by_source": {s: _stats(v) for s, v in sources.items()},
        "by_confidence": {c: _stats(v) for c, v in confidences.items()},
    }
