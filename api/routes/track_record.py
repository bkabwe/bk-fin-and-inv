"""Track Record API — summary stats and resolved predictions list."""
from __future__ import annotations

from fastapi import APIRouter

from modules.prediction_tracker import get_all_predictions, get_summary_stats

router = APIRouter(prefix="/track-record", tags=["track_record"])


@router.get("/summary")
def track_record_summary() -> dict:
    """Return summary hit-rate statistics across all resolved predictions."""
    return get_summary_stats()


@router.get("/predictions")
def track_record_predictions(status: str | None = None) -> list[dict]:
    """Return all predictions, optionally filtered by status (pending/resolved/unresolved_no_data)."""
    records = get_all_predictions()
    if status:
        records = [r for r in records if r.get("status") == status]
    return records
