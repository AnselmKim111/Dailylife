"""SQLite-backed schedule store for the Dailylife bot."""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, List, Optional, Tuple

DB_PATH = os.environ.get("DAILYLIFE_DB_PATH", "/data/dailylife.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    when_utc TEXT NOT NULL,         -- ISO 8601 UTC
    notes TEXT,
    remind_lead_minutes INTEGER,    -- null = no reminder
    reminded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_events_chat_when ON events(chat_id, when_utc);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(chat_id, key)
);

CREATE TABLE IF NOT EXISTS recurring_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    cron_kst TEXT NOT NULL,          -- "HH:MM" (daily) for now
    prompt TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_run_utc TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""

_lock = threading.Lock()


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as c:
        c.executescript(SCHEMA)


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    with _lock:
        conn = sqlite3.connect(DB_PATH, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()


def add_event(
    chat_id: int,
    title: str,
    when_utc: datetime,
    notes: Optional[str] = None,
    remind_lead_minutes: Optional[int] = None,
) -> int:
    when_iso = when_utc.astimezone(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO events (chat_id, title, when_utc, notes, remind_lead_minutes) VALUES (?,?,?,?,?)",
            (chat_id, title, when_iso, notes, remind_lead_minutes),
        )
        return cur.lastrowid


def list_events(
    chat_id: int,
    from_utc: Optional[datetime] = None,
    to_utc: Optional[datetime] = None,
) -> List[sqlite3.Row]:
    q = "SELECT * FROM events WHERE chat_id=?"
    params: List = [chat_id]
    if from_utc is not None:
        q += " AND when_utc >= ?"
        params.append(from_utc.astimezone(timezone.utc).isoformat())
    if to_utc is not None:
        q += " AND when_utc <= ?"
        params.append(to_utc.astimezone(timezone.utc).isoformat())
    q += " ORDER BY when_utc ASC"
    with _conn() as c:
        return list(c.execute(q, params))


def get_event(event_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        row = c.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return row


def update_event(
    event_id: int,
    title: Optional[str] = None,
    when_utc: Optional[datetime] = None,
    notes: Optional[str] = None,
    remind_lead_minutes: Optional[int] = None,
) -> bool:
    fields: List[str] = []
    params: List = []
    if title is not None:
        fields.append("title=?")
        params.append(title)
    if when_utc is not None:
        fields.append("when_utc=?")
        params.append(when_utc.astimezone(timezone.utc).isoformat())
        # Re-arm reminder when time changes.
        fields.append("reminded=0")
    if notes is not None:
        fields.append("notes=?")
        params.append(notes)
    if remind_lead_minutes is not None:
        fields.append("remind_lead_minutes=?")
        params.append(remind_lead_minutes)
        fields.append("reminded=0")
    if not fields:
        return False
    params.append(event_id)
    with _conn() as c:
        cur = c.execute(f"UPDATE events SET {', '.join(fields)} WHERE id=?", params)
        return cur.rowcount > 0


def delete_event(event_id: int, chat_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM events WHERE id=? AND chat_id=?", (event_id, chat_id))
        return cur.rowcount > 0


def mark_reminded(event_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE events SET reminded=1 WHERE id=?", (event_id,))


def pending_reminders() -> List[sqlite3.Row]:
    """Events whose reminder has not yet fired and whose time is in the future."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        return list(
            c.execute(
                "SELECT * FROM events WHERE reminded=0 AND remind_lead_minutes IS NOT NULL AND when_utc > ?",
                (now,),
            )
        )


# ---------------- facts ----------------


def remember_fact(chat_id: int, key: str, value: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO facts (chat_id, key, value, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(chat_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (chat_id, key.strip(), value.strip(), now),
        )


def forget_fact(chat_id: int, key: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM facts WHERE chat_id=? AND key=?", (chat_id, key.strip()))
        return cur.rowcount > 0


def list_facts(chat_id: int) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(
            c.execute(
                "SELECT key, value, updated_at FROM facts WHERE chat_id=? ORDER BY key",
                (chat_id,),
            )
        )


# ---------------- recurring tasks ----------------


def add_recurring_task(chat_id: int, cron_kst: str, prompt: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO recurring_tasks (chat_id, cron_kst, prompt) VALUES (?,?,?)",
            (chat_id, cron_kst.strip(), prompt.strip()),
        )
        return cur.lastrowid


def list_recurring_tasks(chat_id: Optional[int] = None) -> List[sqlite3.Row]:
    with _conn() as c:
        if chat_id is None:
            return list(c.execute("SELECT * FROM recurring_tasks WHERE enabled=1 ORDER BY id"))
        return list(
            c.execute(
                "SELECT * FROM recurring_tasks WHERE chat_id=? ORDER BY id", (chat_id,)
            )
        )


def get_recurring_task(task_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute("SELECT * FROM recurring_tasks WHERE id=?", (task_id,)).fetchone()


def delete_recurring_task(task_id: int, chat_id: int) -> bool:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM recurring_tasks WHERE id=? AND chat_id=?", (task_id, chat_id)
        )
        return cur.rowcount > 0


def mark_recurring_run(task_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("UPDATE recurring_tasks SET last_run_utc=? WHERE id=?", (now, task_id))
