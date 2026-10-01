"""Prediction Tracker — persist price projections and resolve them after their target date.

Storage: data/predictions.db (SQLite). Unlike other data/*.json files, this
database is explicitly un-ignored in .gitignore and committed back by CI (see
the scan-email-* / grading-report-* workflows) so history survives across
ephemeral workflow filesystems.

Prior to this module's SQLite migration, every call (including a single new
prediction, or checking a handful of pending target dates) loaded and
re-serialized the entire growing `data/predictions.json` file. SQLite lets
``record_prediction`` and ``resolve_pending_predictions`` issue targeted
INSERT/UPDATE statements against only the rows they touch, and lets the
read-only validation stats (``compute_score_validation_stats``,
``compute_upside_validation_stats``, ``compute_subscore_correlation_stats``)
select only the (resolved, or sub-score-bearing) rows they actually need
instead of materializing every historical record in Python -- so these all
stay cheap as prediction history keeps accumulating week over week.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from modules.logger import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PREDICTIONS_DB_FILE = DATA_DIR / "predictions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    company TEXT,
    horizon TEXT NOT NULL,
    scan_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    current_price_at_scan REAL,
    target_price REAL,
    target_low REAL,
    target_high REAL,
    projected_upside_pct REAL,
    score REAL,
    data_quality TEXT,
    models_used TEXT,
    source TEXT,
    sub_scores_json TEXT,
    status TEXT NOT NULL,
    actual_price_at_target_date REAL,
    actual_return_pct REAL,
    hit_target INTEGER,
    hit_band INTEGER,
    resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(status);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_horizon_scandate
    ON predictions(ticker, horizon, scan_date);
CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON predictions(target_date);
"""

# Columns in storage/row order -- keep in sync with _SCHEMA and _row_to_record/_record_to_row.
_COLUMNS: tuple[str, ...] = (
    "id",
    "ticker",
    "company",
    "horizon",
    "scan_date",
    "target_date",
    "current_price_at_scan",
    "target_price",
    "target_low",
    "target_high",
    "projected_upside_pct",
    "score",
    "data_quality",
    "models_used",
    "source",
    "sub_scores_json",
    "status",
    "actual_price_at_target_date",
    "actual_return_pct",
    "hit_target",
    "hit_band",
    "resolved_at",
)

# Horizon → midpoint days used for target_date calculation.
# Align with estimate_target_date fallback_days in pages/6_Profit_Opportunities.py.
HORIZON_DAYS: dict[str, int] = {
    "short_term": 21,
    "medium_term": 90,
    "long_term": 365,
}

# Maps the internal row-dict keys populated by
# modules.profit_opportunities.analyze_ticker_for_horizon (prefixed with "_"
# like _rsi, stripped before display/API response) to the field names used
# in a prediction record's "sub_scores" dict. These are the five technical
# sub-scores flagged in the architecture review as suspected of being highly
# correlated (see modules/scoring_engine.py's analyze_stock); recording them
# here is what makes compute_subscore_correlation_stats() below usable.
SUBSCORE_ROW_FIELDS: dict[str, str] = {
    "_trend_score": "trend",
    "_momentum_score": "momentum",
    "_rs_score": "relative_strength",
    "_breakout_score": "breakout",
    "_volume_quality_score": "volume_quality",
}


# ---------------------------------------------------------------------------
# Internal helpers — SQLite connection + record <-> row conversion
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """Open a connection to data/predictions.db, creating the schema if needed.

    Short-lived connection per call (mirrors the old open/read/close-per-call
    JSON pattern) rather than a module-level singleton, so tests that patch
    ``PREDICTIONS_DB_FILE`` to a fresh temp path don't need any extra teardown.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(PREDICTIONS_DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def _row_to_record(row: sqlite3.Row) -> dict:
    rec = {col: row[col] for col in _COLUMNS}
    rec["sub_scores"] = json.loads(rec.pop("sub_scores_json")) if rec.get("sub_scores_json") else None
    rec["hit_target"] = bool(rec["hit_target"]) if rec["hit_target"] is not None else None
    rec["hit_band"] = bool(rec["hit_band"]) if rec["hit_band"] is not None else None
    return rec


def _record_to_row(rec: dict) -> dict:
    row = {col: rec.get(col) for col in _COLUMNS if col != "sub_scores_json"}
    sub_scores = rec.get("sub_scores")
    row["sub_scores_json"] = json.dumps(sub_scores) if sub_scores else None
    if row.get("hit_target") is not None:
        row["hit_target"] = int(bool(row["hit_target"]))
    if row.get("hit_band") is not None:
        row["hit_band"] = int(bool(row["hit_band"]))
    return row


def _load_all() -> list[dict]:
    """Load every prediction record. Prefer ``_load_by_status``/targeted
    queries where only a subset of rows is actually needed."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY rowid").fetchall()
    return [_row_to_record(row) for row in rows]


def _load_by_status(status: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE status = ? ORDER BY rowid", (status,)
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def _load_with_subscores() -> list[dict]:
    """Rows (pending or resolved) with a recorded sub_scores blob -- used by
    compute_subscore_correlation_stats, which needs the full scan history,
    not just resolved outcomes."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE sub_scores_json IS NOT NULL ORDER BY rowid"
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def _load_pending_dedupe_keys() -> set[str]:
    """Light-weight variant of the dedupe check in record_prediction: only
    pulls the 3 columns needed to build dedupe keys from pending rows,
    instead of loading every historical (mostly resolved) record."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ticker, horizon, scan_date FROM predictions WHERE status = 'pending'"
        ).fetchall()
    return {_dedupe_key(row["ticker"], row["horizon"], row["scan_date"]) for row in rows}


def _insert_record(rec: dict) -> None:
    row = _record_to_row(rec)
    columns = ", ".join(_COLUMNS)
    placeholders = ", ".join(f":{col}" for col in _COLUMNS)
    with _connect() as conn:
        conn.execute(f"INSERT INTO predictions ({columns}) VALUES ({placeholders})", row)
        conn.commit()


def _update_record(rec: dict) -> None:
    row = _record_to_row(rec)
    assignments = ", ".join(f"{col} = :{col}" for col in _COLUMNS if col != "id")
    with _connect() as conn:
        conn.execute(f"UPDATE predictions SET {assignments} WHERE id = :id", row)
        conn.commit()


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
    sub_scores: dict[str, float] | None = None,
) -> str | None:
    """Persist a prediction if no unresolved record exists for (ticker, horizon) today.

    ``sub_scores`` optionally records the technical sub-score breakdown (see
    SUBSCORE_ROW_FIELDS) at scan time, enabling compute_subscore_correlation_stats.

    Returns the prediction id if recorded, or None if it was deduped.
    """
    today = date.today().isoformat()

    # Dedupe: skip if there's already a pending prediction for the same ticker+horizon today.
    # Only pending rows are loaded (not the whole, ever-growing resolved history).
    pending = _load_pending_dedupe_keys()
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
        "sub_scores": dict(sub_scores) if sub_scores else None,
        "status": "pending",
        # resolved fields — filled in later
        "actual_price_at_target_date": None,
        "actual_return_pct": None,
        "hit_target": None,
        "hit_band": None,
        "resolved_at": None,
    }
    _insert_record(rec)
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
            sub_scores = {
                record_key: float(row[row_key])
                for row_key, record_key in SUBSCORE_ROW_FIELDS.items()
                if row.get(row_key) is not None
            }
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
                sub_scores=sub_scores or None,
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
    # Imported locally (rather than at module level) so tests can patch
    # ``modules.data_fetcher.get_stock_data`` directly.
    from modules.data_fetcher import get_stock_data
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
        for idx, price in zip(prices.index, prices.values, strict=False)
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
    # Imported locally (rather than at module level) so tests can patch
    # ``modules.data_fetcher.get_stock_data`` directly.
    from modules.data_fetcher import get_stock_data

    # Only pending rows are loaded -- resolved/unresolved_no_data history (which
    # only grows and never changes again) is never re-read or re-written here.
    records = _load_by_status("pending")
    today = date.today()
    resolved_count = 0
    no_data_count = 0
    still_pending_count = 0

    for rec in records:
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
            candidates = [(d, p) for d, p in zip(index_dates, close.values, strict=False) if d >= target_date]
            if not candidates:
                # Fall back to the most recent available date.
                candidates = list(zip(index_dates, close.values, strict=False))
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
            _update_record(rec)
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
                _update_record(rec)
                no_data_count += 1
            else:
                still_pending_count += 1

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


# ---------------------------------------------------------------------------
# Public API — score-vs-outcome validation
# ---------------------------------------------------------------------------

def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient. Returns None if undefined (no variance)."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=False))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    if denom <= 1e-12:
        return None
    return cov / denom


def _score_ic_stats(subset: list[dict], min_samples: int) -> dict:
    """Information coefficient + top/bottom-third precision for one subset."""
    n = len(subset)
    if n < min_samples:
        return {
            "count": n,
            "information_coefficient": None,
            "precision_at_top_third": None,
            "precision_at_bottom_third": None,
        }

    scores = [float(r["score"]) for r in subset]
    returns = [float(r["actual_return_pct"]) for r in subset]
    ic = _pearson_correlation(scores, returns)

    ordered = sorted(subset, key=lambda r: float(r["score"]), reverse=True)
    third = max(1, n // 3)
    top, bottom = ordered[:third], ordered[-third:]

    def _hit_rate(rows: list[dict]) -> float | None:
        return round(sum(1 for r in rows if r.get("hit_target")) / len(rows) * 100, 1) if rows else None

    return {
        "count": n,
        "information_coefficient": round(ic, 4) if ic is not None else None,
        "precision_at_top_third": _hit_rate(top),
        "precision_at_bottom_third": _hit_rate(bottom),
    }


def compute_score_validation_stats(min_samples: int = 5) -> dict:
    """Correlate recorded composite scores (analyze_stock's 0-100 score) against
    realized forward returns, once predictions are resolved.

    This closes the validation loop flagged in the architecture review: the
    composite score is recorded at scan time (see record_prediction) but was
    never checked against what actually happened afterward. Reports:
      - information_coefficient: Pearson correlation between the recorded
        score and the realized return_pct (positive => higher scores tend to
        precede better outcomes).
      - precision_at_top_third / precision_at_bottom_third: the price-target
        hit rate for the highest- and lowest-scored thirds, so a skewed
        distribution doesn't hide a genuinely useful (or useless) score.

    Returns {"overall": {...}, "by_horizon": {horizon: {...}}}. Any bucket
    with fewer than `min_samples` resolved predictions returns None values
    rather than a misleadingly precise statistic from a tiny sample.
    """
    # Only resolved rows are loaded -- pending predictions never contribute
    # to these stats, so there's no need to load/filter them in Python.
    resolved_rows = _load_by_status("resolved")
    resolved = [
        r for r in resolved_rows if r.get("score") is not None and r.get("actual_return_pct") is not None
    ]

    by_horizon = {h: _score_ic_stats([r for r in resolved if r.get("horizon") == h], min_samples) for h in HORIZON_DAYS}
    return {
        "overall": _score_ic_stats(resolved, min_samples),
        "by_horizon": by_horizon,
    }


def _upside_calibration_stats(subset: list[dict], min_samples: int) -> dict:
    """Information coefficient + bias stats for one subset, testing whether
    the projected_upside_pct forecast itself is systematically over/under-
    optimistic relative to realized returns -- not just whether the
    composite Score ranks outcomes correctly (see _score_ic_stats above)."""
    n = len(subset)
    if n < min_samples:
        return {
            "count": n,
            "information_coefficient": None,
            "mean_bias_pct": None,
            "median_bias_pct": None,
            "overoptimism_rate": None,
        }

    projected = [float(r["projected_upside_pct"]) for r in subset]
    returns = [float(r["actual_return_pct"]) for r in subset]
    ic = _pearson_correlation(projected, returns)

    # bias = actual - projected; negative => the realized return fell short
    # of the projected upside (over-optimistic forecast), positive => the
    # forecast was conservative relative to what actually happened.
    biases = sorted(actual - proj for proj, actual in zip(projected, returns, strict=True))
    mean_bias = sum(biases) / n
    mid = n // 2
    median_bias = biases[mid] if n % 2 == 1 else (biases[mid - 1] + biases[mid]) / 2
    overoptimism_rate = round(sum(1 for b in biases if b < 0) / n * 100, 1)

    return {
        "count": n,
        "information_coefficient": round(ic, 4) if ic is not None else None,
        "mean_bias_pct": round(mean_bias, 2),
        "median_bias_pct": round(median_bias, 2),
        "overoptimism_rate": overoptimism_rate,
    }


def compute_upside_validation_stats(min_samples: int = 5) -> dict:
    """Correlate recorded projected_upside_pct forecasts against realized
    forward returns, once predictions are resolved.

    Complements compute_score_validation_stats, which only tests the
    composite 0-100 Score's rank-ordering power against outcomes -- this
    tests the upside forecast number itself (the raw "+N% upside" figures
    surfaced in the profit-opportunities report, e.g. an aggressive
    long-horizon projection) for systematic bias, not just directional
    correlation. Reports:
      - information_coefficient: Pearson correlation between projected
        upside and realized return (positive => bigger projected upside
        tends to precede bigger realized returns).
      - mean_bias_pct / median_bias_pct: average/median (actual - projected)
        gap; a large negative value flags systematic over-optimism.
      - overoptimism_rate: % of resolved predictions where the realized
        return fell short of the projected upside.

    Returns {"overall": {...}, "by_horizon": {horizon: {...}}}. Any bucket
    with fewer than `min_samples` resolved predictions returns None values
    rather than a misleadingly precise statistic from a tiny sample.
    """
    # Only resolved rows are loaded -- see compute_score_validation_stats above.
    resolved_rows = _load_by_status("resolved")
    resolved = [
        r for r in resolved_rows if r.get("projected_upside_pct") is not None and r.get("actual_return_pct") is not None
    ]

    by_horizon = {
        h: _upside_calibration_stats([r for r in resolved if r.get("horizon") == h], min_samples) for h in HORIZON_DAYS
    }
    return {
        "overall": _upside_calibration_stats(resolved, min_samples),
        "by_horizon": by_horizon,
    }


def compute_subscore_correlation_stats(min_samples: int = 10) -> dict:
    """Pairwise Pearson correlation among the 5 technical sub-scores recorded
    at scan time (see SUBSCORE_ROW_FIELDS): trend, momentum, relative_strength,
    breakout, and volume_quality.

    This is the empirical check flagged in the architecture review and in
    modules/scoring_engine.py's analyze_stock docstring comment: these
    sub-scores were suspected of mostly re-deriving the same underlying
    "uptrend + volume confirmation" signal, over-crediting it several times
    within the 0-50 technical total, but that was never measured against
    real scan history.

    Unlike compute_score_validation_stats above, this does **not** require
    resolved outcomes -- it correlates the sub-scores against each other
    across every recorded scan (pending or resolved), so it becomes usable
    as soon as enough scan history exists, without waiting weeks/months for
    target dates to resolve.

    Returns {"pairs": {"breakout/momentum": {"correlation": float | None,
    "count": int}, ...}, "total_records_with_subscores": int}. A pair with
    fewer than `min_samples` records with both fields populated reports
    correlation=None rather than a misleading statistic from a tiny sample.
    High positive correlations (e.g. > 0.6-0.7) between a pair would support
    down-weighting or consolidating them; this function only measures the
    correlation; deciding what to do about it is a follow-up.
    """
    # Only rows with a recorded sub_scores blob are loaded (pending or
    # resolved alike -- see the docstring above), skipping every record from
    # before sub-score recording existed and any scan source that doesn't
    # populate it.
    records = _load_with_subscores()
    subscore_rows = [r["sub_scores"] for r in records if isinstance(r.get("sub_scores"), dict)]

    field_names = sorted(set(SUBSCORE_ROW_FIELDS.values()))
    pairs: dict[str, dict] = {}
    for i, field_a in enumerate(field_names):
        for field_b in field_names[i + 1 :]:
            xs: list[float] = []
            ys: list[float] = []
            for row in subscore_rows:
                a, b = row.get(field_a), row.get(field_b)
                if a is not None and b is not None:
                    xs.append(float(a))
                    ys.append(float(b))
            correlation = _pearson_correlation(xs, ys) if len(xs) >= min_samples else None
            pairs[f"{field_a}/{field_b}"] = {
                "correlation": round(correlation, 4) if correlation is not None else None,
                "count": len(xs),
            }

    return {"pairs": pairs, "total_records_with_subscores": len(subscore_rows)}
