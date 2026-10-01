"""SQLite-backed storage for experimental backtest-comparison runs.

Callers are expected to pass ``git_commit`` explicitly when they want a run
associated with a specific repository revision; this module intentionally does
not shell out to auto-detect it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from modules.logger import get_logger

logger = get_logger(__name__)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
EXPERIMENTS_DB = DATA_DIR / "experiments.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiment_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    git_commit TEXT,
    recorded_at TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    sample_tickers_json TEXT
)
"""


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(EXPERIMENTS_DB)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(_CREATE_TABLE_SQL)
    connection.commit()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _deserialize_row(row: sqlite3.Row) -> dict[str, Any]:
    row_id = int(row["id"])
    try:
        config = json.loads(row["config_json"])
        metrics = json.loads(row["metrics_json"])
        sample_tickers = json.loads(row["sample_tickers_json"]) if row["sample_tickers_json"] is not None else None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover - exercised in tests
        raise ValueError(f"invalid JSON payloads for row {row_id}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"config_json for row {row_id} did not decode to a dict")
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics_json for row {row_id} did not decode to a dict")
    if sample_tickers is not None and not isinstance(sample_tickers, list):
        raise ValueError(f"sample_tickers_json for row {row_id} did not decode to a list")

    return {
        "id": row_id,
        "run_name": str(row["run_name"]),
        "horizon": int(row["horizon"]),
        "git_commit": row["git_commit"],
        "recorded_at": str(row["recorded_at"]),
        "config": config,
        "metrics": metrics,
        "sample_tickers": sample_tickers,
    }


def _shared_numeric_model_metric_deltas(previous: dict[str, Any], latest: dict[str, Any]) -> dict[str, float]:
    previous_models = previous.get("models")
    latest_models = latest.get("models")
    if not isinstance(previous_models, dict) or not isinstance(latest_models, dict):
        return {}

    deltas: dict[str, float] = {}
    for model_name in sorted(set(previous_models) & set(latest_models)):
        previous_model_metrics = previous_models.get(model_name)
        latest_model_metrics = latest_models.get(model_name)
        if not isinstance(previous_model_metrics, dict) or not isinstance(latest_model_metrics, dict):
            continue

        for metric_name in sorted(set(previous_model_metrics) & set(latest_model_metrics)):
            previous_value = previous_model_metrics.get(metric_name)
            latest_value = latest_model_metrics.get(metric_name)
            if isinstance(previous_value, bool) or isinstance(latest_value, bool):
                continue
            if not isinstance(previous_value, (int, float)) or not isinstance(latest_value, (int, float)):
                continue
            deltas[f"models.{model_name}.{metric_name}"] = float(latest_value) - float(previous_value)
    return deltas


def record_experiment_run(
    *,
    run_name: str,
    horizon: int,
    git_commit: str | None = None,
    config: dict,
    metrics: dict,
    sample_tickers: list[str] | None = None,
) -> int:
    """Insert one experiment-run record and return its SQLite row id."""
    recorded_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _connect() as connection:
        _ensure_schema(connection)
        cursor = connection.execute(
            """
            INSERT INTO experiment_runs (
                run_name,
                horizon,
                git_commit,
                recorded_at,
                config_json,
                metrics_json,
                sample_tickers_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(run_name),
                int(horizon),
                git_commit,
                recorded_at,
                _json_dumps(config),
                _json_dumps(metrics),
                _json_dumps(list(sample_tickers)) if sample_tickers is not None else None,
            ),
        )
        connection.commit()
        row_id = int(cursor.lastrowid)

    logger.info("Recorded experiment run %s (#%s) for horizon %s", run_name, row_id, horizon)
    return row_id


def list_experiment_runs(
    run_name: str | None = None,
    horizon: int | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent experiment runs, most recent first, with JSON fields deserialized."""
    clauses: list[str] = []
    params: list[Any] = []
    if run_name is not None:
        clauses.append("run_name = ?")
        params.append(str(run_name))
    if horizon is not None:
        clauses.append("horizon = ?")
        params.append(int(horizon))

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = (
        "SELECT id, run_name, horizon, git_commit, recorded_at, config_json, metrics_json, sample_tickers_json "
        "FROM experiment_runs "
        f"{where_clause} "
        "ORDER BY id DESC LIMIT ?"
    )
    params.append(max(1, int(limit)))

    try:
        with _connect() as connection:
            _ensure_schema(connection)
            rows = connection.execute(query, params).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Failed to list experiment runs: %s", exc)
        return []

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            parsed_rows.append(_deserialize_row(row))
        except ValueError as exc:
            logger.warning("Skipping malformed experiment run row %s: %s", row["id"], exc)
    return parsed_rows


def compare_latest_two_runs(run_name: str, horizon: int) -> dict[str, Any] | None:
    """Compare the two most recent valid runs for a run-name/horizon pair."""
    runs = list_experiment_runs(run_name=run_name, horizon=horizon, limit=50)
    if len(runs) < 2:
        return None

    latest = runs[0]
    previous = runs[1]
    metric_deltas = _shared_numeric_model_metric_deltas(previous["metrics"], latest["metrics"])
    return {
        "previous": previous,
        "latest": latest,
        "metric_deltas": metric_deltas,
    }
