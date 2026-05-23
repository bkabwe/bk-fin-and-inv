from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NOTIFICATIONS_FILE = DATA_DIR / "notifications.json"


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not NOTIFICATIONS_FILE.exists():
        NOTIFICATIONS_FILE.write_text("[]", encoding="utf-8")


def _load() -> list[dict]:
    _ensure()
    try:
        return json.loads(NOTIFICATIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(items: list[dict]):
    _ensure()
    NOTIFICATIONS_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")


def add_notification(title: str, message: str, ticker: str, type: str = "info") -> dict:
    items = _load()
    note = {
        "title": title,
        "message": message,
        "ticker": ticker.upper(),
        "type": type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "read": False,
    }
    items.insert(0, note)
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
