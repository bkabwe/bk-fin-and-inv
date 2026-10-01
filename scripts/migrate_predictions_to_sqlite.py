"""One-time migration: backfill data/predictions.json into data/predictions.db.

modules.prediction_tracker now stores predictions in SQLite
(data/predictions.db) instead of a flat JSON file, so that recording a new
prediction or resolving a handful of pending ones no longer requires
loading and rewriting the entire, ever-growing prediction history (see
modules/prediction_tracker.py's module docstring for the full rationale).

This script is idempotent: it inserts each JSON record into the database
with ``INSERT OR IGNORE`` keyed on the record's existing ``id``, so running
it more than once (e.g. if a workflow re-runs) is harmless. It does not
delete data/predictions.json; once this script runs on the production
history and the new .db file is confirmed working, the .json file can be
removed and its .gitignore carve-out retired in a follow-up change.

Usage:
    python scripts/migrate_predictions_to_sqlite.py
    python scripts/migrate_predictions_to_sqlite.py --json-file path/to/predictions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import prediction_tracker  # noqa: E402
from modules.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def migrate(json_file: Path) -> dict[str, int]:
    """Backfill every record in ``json_file`` into data/predictions.db.

    Returns {"migrated": n, "skipped_existing": n, "total": n}.
    """
    if not json_file.exists():
        logger.info("No JSON file found at %s; nothing to migrate.", json_file)
        return {"migrated": 0, "skipped_existing": 0, "total": 0}

    records = json.loads(json_file.read_text(encoding="utf-8"))
    existing_ids = {rec["id"] for rec in prediction_tracker.get_all_predictions()}

    migrated = 0
    skipped = 0
    for rec in records:
        if rec.get("id") in existing_ids:
            skipped += 1
            continue
        prediction_tracker._insert_record(dict(rec))  # noqa: SLF001 -- intentional one-time backfill
        migrated += 1

    return {"migrated": migrated, "skipped_existing": skipped, "total": len(records)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-file",
        default=str(prediction_tracker.DATA_DIR / "predictions.json"),
        help="Path to the legacy predictions.json file to backfill from.",
    )
    args = parser.parse_args()

    result = migrate(Path(args.json_file))
    print(
        f"Migrated {result['migrated']} record(s), skipped {result['skipped_existing']} "
        f"already present, out of {result['total']} total JSON record(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
