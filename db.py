"""SQLite-backed schedule store for the Dailylife bot."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Tuple
from zoneinfo import ZoneInfo

DB_PATH = os.environ.get("DAILYLIFE_DB_PATH", "/data/dailylife.db")
USER_TZ_NAME = os.environ.get("USER_TZ", "Asia/Seoul")
_USER_TZ = ZoneInfo(USER_TZ_NAME)
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

CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd REAL,
    kind TEXT,                  -- 'chat' | 'vision' | 'transcribe'
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage(created_at);

CREATE TABLE IF NOT EXISTS deleted_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    kind TEXT NOT NULL,           -- 'event' | 'goal' | 'recurring' | 'fact' | 'note'
    payload_json TEXT NOT NULL,   -- full row at time of delete (for restore)
    callback_token TEXT NOT NULL UNIQUE,  -- short token for callback_data
    deleted_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    restored INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_audit_token ON deleted_audit(callback_token);

CREATE TABLE IF NOT EXISTS goals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    why TEXT,
    target_date_local TEXT,                       -- "YYYY-MM-DD" in user's TZ; nullable
    horizon TEXT NOT NULL DEFAULT 'long',         -- 'short' | 'medium' | 'long'
    status TEXT NOT NULL DEFAULT 'open',          -- 'open' | 'done' | 'paused' | 'dropped'
    sub_tasks_json TEXT NOT NULL DEFAULT '[]',    -- [{text, done}]
    watch_query TEXT,                              -- optional periodic search query
    last_reviewed_utc TEXT,
    last_action_utc TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_goals_chat_status ON goals(chat_id, status);

CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    tags TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_notes_chat ON notes(chat_id, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts
    USING fts5(content, tags, content='notes', content_rowid='id', tokenize='trigram');

CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, content, tags) VALUES (new.id, new.content, COALESCE(new.tags, ''));
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, content, tags) VALUES('delete', old.id, old.content, COALESCE(old.tags, ''));
    INSERT INTO notes_fts(rowid, content, tags) VALUES (new.id, new.content, COALESCE(new.tags, ''));
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, content, tags) VALUES('delete', old.id, old.content, COALESCE(old.tags, ''));
END;

CREATE TABLE IF NOT EXISTS chat_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    role TEXT NOT NULL,         -- 'user' | 'assistant'
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_chat_log_chat ON chat_log(chat_id, created_at DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS chat_log_fts
    USING fts5(content, content='chat_log', content_rowid='id', tokenize='trigram');

CREATE TRIGGER IF NOT EXISTS chat_log_ai AFTER INSERT ON chat_log BEGIN
    INSERT INTO chat_log_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chat_log_ad AFTER DELETE ON chat_log BEGIN
    INSERT INTO chat_log_fts(chat_log_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    role TEXT,
    notes TEXT,
    important_dates_json TEXT NOT NULL DEFAULT '[]',
    last_contact_utc TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_people_chat ON people(chat_id);

CREATE TABLE IF NOT EXISTS oauth_tokens (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    provider TEXT NOT NULL,           -- 'google'
    access_token TEXT NOT NULL,
    refresh_token TEXT,
    expires_at_utc TEXT,              -- ISO 8601 UTC; NULL = no known expiry
    scopes TEXT,                       -- space-separated
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(chat_id, provider)
);
"""

_lock = threading.Lock()

# Idempotent ALTER TABLE migrations — each is `(table, column, definition)`.
# Applied at init_db. SQLite doesn't have ADD COLUMN IF NOT EXISTS so we read
# pragma_table_info first.
_MIGRATIONS: List[Tuple[str, str, str]] = [
    ("events", "gcal_event_id", "TEXT"),
    ("events", "gcal_sync_state", "TEXT"),
    # 'missed' marks reminders whose scheduled fire was past at bot startup
    # (bot was offline). Surfaced via /diag and folded into daily imminent push.
    ("events", "missed", "INTEGER NOT NULL DEFAULT 0"),
    ("goals", "watch_frequency_days", "INTEGER NOT NULL DEFAULT 7"),
    ("goals", "last_watch_run_utc", "TEXT"),
]


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as c:
        c.executescript(SCHEMA)
        _run_migrations(c)


def _run_migrations(c: sqlite3.Connection) -> None:
    for table, col, defn in _MIGRATIONS:
        cols = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")


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
    gcal_event_id: Optional[str] = None,
    gcal_sync_state: Optional[str] = None,
) -> int:
    when_iso = when_utc.astimezone(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO events (chat_id, title, when_utc, notes, remind_lead_minutes, "
            "gcal_event_id, gcal_sync_state) VALUES (?,?,?,?,?,?,?)",
            (chat_id, title, when_iso, notes, remind_lead_minutes,
             gcal_event_id, gcal_sync_state),
        )
        return cur.lastrowid


def set_event_gcal(event_id: int, gcal_event_id: Optional[str], gcal_sync_state: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE events SET gcal_event_id=?, gcal_sync_state=? WHERE id=?",
            (gcal_event_id, gcal_sync_state, event_id),
        )


def find_event_by_gcal_id(chat_id: int, gcal_event_id: str) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM events WHERE chat_id=? AND gcal_event_id=?",
            (chat_id, gcal_event_id),
        ).fetchone()


def list_pending_gcal_sync() -> List[sqlite3.Row]:
    """Events whose local insert succeeded but GCal create failed; for retry cron."""
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM events WHERE gcal_sync_state='pending' AND when_utc > strftime('%Y-%m-%dT%H:%M:%fZ','now')"
        ))


def mark_missed_if_past(event_id: int) -> bool:
    """If a reminder time is past AND not reminded AND not missed, flag it."""
    with _conn() as c:
        row = c.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None or row["reminded"] or row["missed"]:
            return False
        c.execute("UPDATE events SET missed=1 WHERE id=?", (event_id,))
        return True


def missed_reminders(chat_id: int, hours: int = 48) -> List[sqlite3.Row]:
    """Reminders marked missed in the last N hours (for daily imminent push + /diag)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM events WHERE chat_id=? AND missed=1 AND created_at>=? "
            "ORDER BY when_utc DESC",
            (chat_id, cutoff),
        ))


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


def set_recurring_enabled(task_id: int, chat_id: int, enabled: bool) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE recurring_tasks SET enabled=? WHERE id=? AND chat_id=?",
            (1 if enabled else 0, task_id, chat_id),
        )
        return cur.rowcount > 0


# ---------------- usage / cost ----------------


def log_usage(
    chat_id: Optional[int],
    model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    kind: str,
) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO usage (chat_id, model, prompt_tokens, completion_tokens, cost_usd, kind) "
            "VALUES (?,?,?,?,?,?)",
            (chat_id, model, prompt_tokens, completion_tokens, cost_usd, kind),
        )


def usage_summary(chat_id: int) -> Dict:
    """Return today/this-month totals + last 7-day daily series.

    Boundaries are computed in the user's local TZ but compared against UTC
    timestamps in the DB — convert the local boundary to UTC ISO."""
    now_local = datetime.now(_USER_TZ)
    today_local_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    month_local_start = today_local_start.replace(day=1)
    today_iso = today_local_start.astimezone(timezone.utc).isoformat()
    month_start_iso = month_local_start.astimezone(timezone.utc).isoformat()
    with _conn() as c:
        today_row = c.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost, "
            "  COALESCE(SUM(prompt_tokens),0) AS pt, COALESCE(SUM(completion_tokens),0) AS ct "
            "FROM usage WHERE chat_id=? AND created_at >= ?",
            (chat_id, today_iso),
        ).fetchone()
        month_row = c.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost, "
            "  COALESCE(SUM(prompt_tokens),0) AS pt, COALESCE(SUM(completion_tokens),0) AS ct "
            "FROM usage WHERE chat_id=? AND created_at >= ?",
            (chat_id, month_start_iso),
        ).fetchone()
        by_model = c.execute(
            "SELECT model, COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost "
            "FROM usage WHERE chat_id=? AND created_at >= ? GROUP BY model ORDER BY cost DESC",
            (chat_id, month_start_iso),
        ).fetchall()
    return {
        "today": dict(today_row),
        "month": dict(month_row),
        "by_model_month": [dict(r) for r in by_model],
    }


# ---------------- delete audit (for undo) ----------------


def push_deleted_audit(chat_id: int, kind: str, payload: Dict, token: str) -> int:
    payload_json = _json.dumps(payload, ensure_ascii=False)
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO deleted_audit (chat_id, kind, payload_json, callback_token) VALUES (?,?,?,?)",
            (chat_id, kind, payload_json, token),
        )
        return cur.lastrowid


def get_deleted_audit(token: str) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM deleted_audit WHERE callback_token=? AND restored=0", (token,)
        ).fetchone()


def mark_audit_restored(audit_id: int) -> None:
    with _conn() as c:
        c.execute("UPDATE deleted_audit SET restored=1 WHERE id=?", (audit_id,))


def cleanup_deleted_audit(max_age_days: int = 30) -> int:
    """Sweep restored or stale rows. Returns deleted count."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM deleted_audit WHERE restored=1 OR deleted_at < ?",
            (cutoff,),
        )
        return cur.rowcount


# ---------------- goals ----------------


import json as _json


def add_goal(
    chat_id: int,
    title: str,
    why: Optional[str] = None,
    target_date_local: Optional[str] = None,
    horizon: str = "long",
    sub_tasks: Optional[List[str]] = None,
    watch_query: Optional[str] = None,
) -> int:
    sub_tasks_json = _json.dumps(
        [{"text": t, "done": False} for t in (sub_tasks or [])], ensure_ascii=False
    )
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO goals (chat_id, title, why, target_date_local, horizon, sub_tasks_json, watch_query) "
            "VALUES (?,?,?,?,?,?,?)",
            (chat_id, title.strip(), why, target_date_local, horizon, sub_tasks_json, watch_query),
        )
        return cur.lastrowid


def list_goals(chat_id: int, status: Optional[str] = "open") -> List[sqlite3.Row]:
    with _conn() as c:
        if status:
            return list(
                c.execute(
                    "SELECT * FROM goals WHERE chat_id=? AND status=? "
                    "ORDER BY (target_date_local IS NULL), target_date_local ASC, id",
                    (chat_id, status),
                )
            )
        return list(
            c.execute("SELECT * FROM goals WHERE chat_id=? ORDER BY id", (chat_id,))
        )


def get_goal(goal_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute("SELECT * FROM goals WHERE id=?", (goal_id,)).fetchone()


def update_goal(
    goal_id: int,
    chat_id: int,
    *,
    title: Optional[str] = None,
    why: Optional[str] = None,
    target_date_local: Optional[str] = None,
    horizon: Optional[str] = None,
    status: Optional[str] = None,
    watch_query: Optional[str] = None,
) -> bool:
    fields: List[str] = []
    params: List = []
    for k, v in (
        ("title", title),
        ("why", why),
        ("target_date_local", target_date_local),
        ("horizon", horizon),
        ("status", status),
        ("watch_query", watch_query),
    ):
        if v is not None:
            fields.append(f"{k}=?")
            params.append(v)
    if not fields:
        return False
    params.extend([goal_id, chat_id])
    with _conn() as c:
        cur = c.execute(
            f"UPDATE goals SET {', '.join(fields)} WHERE id=? AND chat_id=?", params
        )
        return cur.rowcount > 0


def add_goal_subtask(goal_id: int, chat_id: int, text: str) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT sub_tasks_json FROM goals WHERE id=? AND chat_id=?", (goal_id, chat_id)
        ).fetchone()
        if row is None:
            return False
        items = _json.loads(row["sub_tasks_json"] or "[]")
        items.append({"text": text.strip(), "done": False})
        c.execute(
            "UPDATE goals SET sub_tasks_json=?, last_action_utc=? WHERE id=?",
            (
                _json.dumps(items, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
                goal_id,
            ),
        )
        return True


def complete_goal_subtask(goal_id: int, chat_id: int, sub_index: int) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT sub_tasks_json FROM goals WHERE id=? AND chat_id=?", (goal_id, chat_id)
        ).fetchone()
        if row is None:
            return False
        items = _json.loads(row["sub_tasks_json"] or "[]")
        if not (0 <= sub_index < len(items)):
            return False
        items[sub_index]["done"] = True
        c.execute(
            "UPDATE goals SET sub_tasks_json=?, last_action_utc=? WHERE id=?",
            (
                _json.dumps(items, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
                goal_id,
            ),
        )
        return True


def mark_goal_reviewed(goal_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("UPDATE goals SET last_reviewed_utc=? WHERE id=?", (now, goal_id))


def mark_goal_watch_run(goal_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("UPDATE goals SET last_watch_run_utc=? WHERE id=?", (now, goal_id))


def goal_watch_due(row) -> bool:
    """True if this goal has a watch_query AND last_watch_run_utc is older than
    watch_frequency_days, OR has never been watched."""
    if not row["watch_query"]:
        return False
    freq = row["watch_frequency_days"] or 7
    last = row["last_watch_run_utc"]
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - last_dt) >= timedelta(days=freq)


def goals_due_within(chat_id: int, days: int) -> List[sqlite3.Row]:
    """Open goals with target_date_local within today..today+days (inclusive).

    'today' is computed in the user's TZ since target_date_local is also stored
    in the user's TZ as YYYY-MM-DD."""
    today = datetime.now(_USER_TZ).date()
    upper = today.fromordinal(today.toordinal() + max(0, days))
    with _conn() as c:
        return list(
            c.execute(
                "SELECT * FROM goals WHERE chat_id=? AND status='open' "
                "AND target_date_local IS NOT NULL "
                "AND target_date_local >= ? AND target_date_local <= ? "
                "ORDER BY target_date_local",
                (chat_id, today.isoformat(), upper.isoformat()),
            )
        )


def all_chat_ids_with_goals() -> List[int]:
    with _conn() as c:
        return [r[0] for r in c.execute("SELECT DISTINCT chat_id FROM goals WHERE status='open'")]


# ---------------- notes ----------------


def add_note(chat_id: int, content: str, tags: Optional[str] = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO notes (chat_id, content, tags) VALUES (?,?,?)",
            (chat_id, content.strip(), (tags or "").strip() or None),
        )
        return cur.lastrowid


def list_notes(chat_id: int, limit: int = 20) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(
            c.execute(
                "SELECT * FROM notes WHERE chat_id=? ORDER BY created_at DESC LIMIT ?",
                (chat_id, limit),
            )
        )


def delete_note(note_id: int, chat_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM notes WHERE id=? AND chat_id=?", (note_id, chat_id))
        return cur.rowcount > 0


def search_notes(chat_id: int, query: str, limit: int = 5) -> List[Dict]:
    """Full-text search via FTS5 trigram + per-token LIKE fallback.

    Korean queries often have many 2-char tokens that don't trigger trigram FTS.
    We always also try a per-token OR LIKE so '점심 만난 사람' matches a note
    containing '점심에 김철수 만남'."""
    q = _fts_safe(query)
    seen: Dict[int, Dict] = {}
    with _conn() as c:
        if q:
            for r in c.execute(
                "SELECT n.id, n.content, n.tags, n.created_at, "
                "  snippet(notes_fts, 0, '«', '»', '…', 12) AS snippet "
                "FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
                "WHERE notes_fts MATCH ? AND n.chat_id=? "
                "ORDER BY rank LIMIT ?",
                (q, chat_id, limit),
            ).fetchall():
                seen[r["id"]] = dict(r)
        if len(seen) < limit:
            for r in _like_fallback(c, "notes", chat_id, query, limit, "tags"):
                if r["id"] not in seen:
                    seen[r["id"]] = r
                    if len(seen) >= limit:
                        break
    return list(seen.values())[:limit]


# ---------------- chat log (episodic memory) ----------------


def log_chat(chat_id: int, role: str, content: str) -> None:
    if not content.strip():
        return
    with _conn() as c:
        c.execute("INSERT INTO chat_log (chat_id, role, content) VALUES (?,?,?)",
                  (chat_id, role, content.strip()))


def search_chat_log(chat_id: int, query: str, limit: int = 5) -> List[Dict]:
    q = _fts_safe(query)
    seen: Dict[int, Dict] = {}
    with _conn() as c:
        if q:
            for r in c.execute(
                "SELECT cl.id, cl.role, cl.content, cl.created_at, "
                "  snippet(chat_log_fts, 0, '«', '»', '…', 14) AS snippet "
                "FROM chat_log_fts JOIN chat_log cl ON cl.id = chat_log_fts.rowid "
                "WHERE chat_log_fts MATCH ? AND cl.chat_id=? "
                "ORDER BY rank LIMIT ?",
                (q, chat_id, limit),
            ).fetchall():
                d = dict(r)
                seen[d["id"]] = d
        if len(seen) < limit:
            for r in _like_fallback(c, "chat_log", chat_id, query, limit, role_col=True):
                if r["id"] not in seen:
                    seen[r["id"]] = r
                    if len(seen) >= limit:
                        break
    return list(seen.values())[:limit]


def _fts_safe(query: str) -> str:
    """Build an FTS5 MATCH clause for the trigram tokenizer.

    Trigram FTS5 needs each MATCH term to be ≥3 chars. We ALSO accept 2-char
    tokens by stitching them onto adjacent tokens — so '점심 만난' becomes
    '점심만난' (5 chars, indexed via the trigram windows '점심만'/'심만난'/'만난').
    Tokens that can't be combined are dropped (caller has LIKE fallback)."""
    raw = [t.replace(chr(34), "") for t in re.split(r"\s+", query.strip()) if t]
    if not raw:
        return ""
    out: List[str] = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        if len(tok) >= 3:
            out.append(tok)
            i += 1
            continue
        # Stitch with next short token if available, else with previous.
        if i + 1 < len(raw) and len(raw[i + 1]) < 3:
            out.append(tok + raw[i + 1])
            i += 2
        elif i + 1 < len(raw):
            out.append(tok + raw[i + 1])
            i += 2
        elif out:
            out[-1] = out[-1] + tok
            i += 1
        else:
            i += 1
    out = [t for t in out if len(t) >= 3]
    if not out:
        return ""
    return " OR ".join(f'"{t}"' for t in out)


def _like_fallback(
    c: sqlite3.Connection,
    table: str,
    chat_id: int,
    query: str,
    limit: int,
    extra_field: Optional[str] = None,
    role_col: bool = False,
) -> List[Dict]:
    """Per-token OR LIKE search — works for any short Korean token the trigram FTS
    can't index. Tokens of len 1 are also kept (cheap and harmless on a small DB)."""
    tokens = [t for t in re.split(r"\s+", query.strip()) if t]
    if not tokens:
        return []
    parts: List[str] = []
    params: List = [chat_id]
    for t in tokens:
        parts.append("content LIKE ?")
        params.append(f"%{t}%")
        if extra_field:
            parts.append(f"{extra_field} LIKE ?")
            params.append(f"%{t}%")
    where = " OR ".join(parts)
    if role_col:
        sql = (
            f"SELECT id, role, content, created_at, content AS snippet FROM {table} "
            f"WHERE chat_id=? AND ({where}) ORDER BY created_at DESC LIMIT ?"
        )
    else:
        sql = (
            f"SELECT id, content, tags, created_at, content AS snippet FROM {table} "
            f"WHERE chat_id=? AND ({where}) ORDER BY created_at DESC LIMIT ?"
        )
    params.append(limit)
    return [dict(r) for r in c.execute(sql, params).fetchall()]


# ---------------- oauth tokens (Google Calendar etc.) ----------------


def save_oauth_token(
    chat_id: int,
    provider: str,
    access_token: str,
    refresh_token: Optional[str],
    expires_at_utc: Optional[str],
    scopes: Optional[str],
) -> None:
    """Upsert tokens. We KEEP the existing refresh_token if Google omits one
    on a refresh response (which they often do)."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        existing = c.execute(
            "SELECT refresh_token FROM oauth_tokens WHERE chat_id=? AND provider=?",
            (chat_id, provider),
        ).fetchone()
        if existing and not refresh_token:
            refresh_token = existing["refresh_token"]
        c.execute(
            "INSERT INTO oauth_tokens (chat_id, provider, access_token, refresh_token, "
            "expires_at_utc, scopes, updated_at) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id, provider) DO UPDATE SET "
            "  access_token=excluded.access_token, "
            "  refresh_token=excluded.refresh_token, "
            "  expires_at_utc=excluded.expires_at_utc, "
            "  scopes=excluded.scopes, "
            "  updated_at=excluded.updated_at",
            (chat_id, provider, access_token, refresh_token, expires_at_utc, scopes, now),
        )


def get_oauth_token(chat_id: int, provider: str) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM oauth_tokens WHERE chat_id=? AND provider=?",
            (chat_id, provider),
        ).fetchone()


def delete_oauth_token(chat_id: int, provider: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM oauth_tokens WHERE chat_id=? AND provider=?", (chat_id, provider)
        )
        return cur.rowcount > 0


def all_chat_ids_with_google_oauth() -> List[int]:
    with _conn() as c:
        return [r[0] for r in c.execute(
            "SELECT DISTINCT chat_id FROM oauth_tokens WHERE provider='google'"
        )]


# ---------------- people / relationships ----------------


def add_person(
    chat_id: int,
    name: str,
    aliases: Optional[List[str]] = None,
    role: Optional[str] = None,
    notes: Optional[str] = None,
    important_dates: Optional[List[Dict]] = None,
) -> int:
    """Insert a new person. Updates if (chat_id, name) already exists — merge aliases."""
    with _conn() as c:
        existing = c.execute(
            "SELECT id, aliases_json, important_dates_json FROM people WHERE chat_id=? AND name=?",
            (chat_id, name.strip()),
        ).fetchone()
        if existing:
            cur_aliases = set(_json.loads(existing["aliases_json"] or "[]"))
            cur_aliases.update(aliases or [])
            cur_dates = _json.loads(existing["important_dates_json"] or "[]")
            new_dates = important_dates or []
            seen_labels = {(d.get("label"), d.get("date_local")) for d in cur_dates}
            for d in new_dates:
                if (d.get("label"), d.get("date_local")) not in seen_labels:
                    cur_dates.append(d)
            c.execute(
                "UPDATE people SET aliases_json=?, role=COALESCE(?, role), "
                "notes=COALESCE(?, notes), important_dates_json=? WHERE id=?",
                (_json.dumps(sorted(cur_aliases), ensure_ascii=False), role, notes,
                 _json.dumps(cur_dates, ensure_ascii=False), existing["id"]),
            )
            return existing["id"]
        cur = c.execute(
            "INSERT INTO people (chat_id, name, aliases_json, role, notes, important_dates_json) "
            "VALUES (?,?,?,?,?,?)",
            (
                chat_id, name.strip(),
                _json.dumps(aliases or [], ensure_ascii=False),
                role, notes,
                _json.dumps(important_dates or [], ensure_ascii=False),
            ),
        )
        return cur.lastrowid


def update_person(
    person_id: int,
    chat_id: int,
    *,
    name: Optional[str] = None,
    role: Optional[str] = None,
    notes: Optional[str] = None,
    add_aliases: Optional[List[str]] = None,
    important_dates: Optional[List[Dict]] = None,
) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM people WHERE id=? AND chat_id=?", (person_id, chat_id)
        ).fetchone()
        if row is None:
            return False
        new_aliases = set(_json.loads(row["aliases_json"] or "[]"))
        if add_aliases:
            new_aliases.update(add_aliases)
        cur_dates = _json.loads(row["important_dates_json"] or "[]")
        if important_dates:
            seen = {(d.get("label"), d.get("date_local")) for d in cur_dates}
            for d in important_dates:
                if (d.get("label"), d.get("date_local")) not in seen:
                    cur_dates.append(d)
        c.execute(
            "UPDATE people SET name=COALESCE(?, name), role=COALESCE(?, role), "
            "notes=COALESCE(?, notes), aliases_json=?, important_dates_json=? "
            "WHERE id=? AND chat_id=?",
            (name, role, notes,
             _json.dumps(sorted(new_aliases), ensure_ascii=False),
             _json.dumps(cur_dates, ensure_ascii=False),
             person_id, chat_id),
        )
        return True


def list_people(chat_id: int, role: Optional[str] = None) -> List[sqlite3.Row]:
    with _conn() as c:
        if role:
            return list(c.execute(
                "SELECT * FROM people WHERE chat_id=? AND role=? ORDER BY name",
                (chat_id, role)))
        return list(c.execute(
            "SELECT * FROM people WHERE chat_id=? ORDER BY name", (chat_id,)))


def get_person(person_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute("SELECT * FROM people WHERE id=?", (person_id,)).fetchone()


def delete_person(person_id: int, chat_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM people WHERE id=? AND chat_id=?",
                        (person_id, chat_id))
        return cur.rowcount > 0


def mark_contact(person_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("UPDATE people SET last_contact_utc=? WHERE id=?", (now, person_id))


def find_people_in_text(chat_id: int, text: str) -> List[sqlite3.Row]:
    """Return people whose name or any alias substring-matches in `text`.

    Case-insensitive Latin; raw substring match for Korean. Returns full rows."""
    people = list_people(chat_id)
    if not people:
        return []
    lowered = text.lower()
    hits = []
    for p in people:
        names: List[str] = [p["name"]]
        try:
            names.extend(_json.loads(p["aliases_json"] or "[]"))
        except Exception:
            pass
        for n in names:
            if not n:
                continue
            n_strip = n.strip()
            if not n_strip:
                continue
            # Korean substring direct; Latin lowercased
            if n_strip in text or n_strip.lower() in lowered:
                hits.append(p)
                break
    return hits
