from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from modules.validators import sanitize_ticker

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NOTIFICATIONS_FILE = DATA_DIR / "notifications.json"
MAX_NOTIFICATIONS = 100


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not NOTIFICATIONS_FILE.exists():
        _atomic_write(NOTIFICATIONS_FILE, [])


def _atomic_write(path: Path, data) -> None:
    """Write JSON atomically using temp file + rename to prevent corruption."""
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=dir_path,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _load() -> list[dict]:
    _ensure()
    try:
        return json.loads(NOTIFICATIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(items: list[dict]):
    _ensure()
    _atomic_write(NOTIFICATIONS_FILE, items)


def add_notification(title: str, message: str, ticker: str, type: str = "info") -> dict:
    items = _load()
    try:
        safe_ticker = sanitize_ticker(ticker)
    except ValueError:
        safe_ticker = "UNKNOWN"
    note = {
        "title": title,
        "message": message,
        "ticker": safe_ticker,
        "type": type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "read": False,
    }
    items.insert(0, note)
    items = items[:MAX_NOTIFICATIONS]
    _save(items)
    return note


def get_notifications() -> list[dict]:
    return _load()


def get_unread_count() -> int:
    return sum(1 for n in _load() if not n.get("read"))


def mark_all_read():
    items = _load()
    for note in items:
        note["read"] = True
    _save(items)


def clear_notifications():
    _save([])
