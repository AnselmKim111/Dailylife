"""SQLite-backed schedule store for the Dailylife bot."""

from __future__ import annotations

import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

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
    """Return today/this-month totals + last 7-day daily series."""
    today_local = datetime.now(timezone.utc).astimezone().date()
    month_start = today_local.replace(day=1)
    today_iso = today_local.isoformat()
    month_start_iso = month_start.isoformat()
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


def goals_due_within(chat_id: int, days: int) -> List[sqlite3.Row]:
    """Open goals with target_date_local within today..today+days (inclusive)."""
    today = datetime.now(timezone.utc).astimezone().date()
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
    """Full-text search via FTS5 with LIKE fallback for short queries (trigram needs >=3 chars)."""
    q = _fts_safe(query)
    with _conn() as c:
        if q:
            rows = c.execute(
                "SELECT n.id, n.content, n.tags, n.created_at, "
                "  snippet(notes_fts, 0, '«', '»', '…', 12) AS snippet "
                "FROM notes_fts JOIN notes n ON n.id = notes_fts.rowid "
                "WHERE notes_fts MATCH ? AND n.chat_id=? "
                "ORDER BY rank LIMIT ?",
                (q, chat_id, limit),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        # LIKE fallback for short / non-trigrammable queries
        rows = c.execute(
            "SELECT id, content, tags, created_at, content AS snippet FROM notes "
            "WHERE chat_id=? AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
            (chat_id, f"%{query.strip()}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------- chat log (episodic memory) ----------------


def log_chat(chat_id: int, role: str, content: str) -> None:
    if not content.strip():
        return
    with _conn() as c:
        c.execute("INSERT INTO chat_log (chat_id, role, content) VALUES (?,?,?)",
                  (chat_id, role, content.strip()))


def search_chat_log(chat_id: int, query: str, limit: int = 5) -> List[Dict]:
    q = _fts_safe(query)
    with _conn() as c:
        if q:
            rows = c.execute(
                "SELECT cl.id, cl.role, cl.content, cl.created_at, "
                "  snippet(chat_log_fts, 0, '«', '»', '…', 14) AS snippet "
                "FROM chat_log_fts JOIN chat_log cl ON cl.id = chat_log_fts.rowid "
                "WHERE chat_log_fts MATCH ? AND cl.chat_id=? "
                "ORDER BY rank LIMIT ?",
                (q, chat_id, limit),
            ).fetchall()
            if rows:
                return [dict(r) for r in rows]
        rows = c.execute(
            "SELECT id, role, content, created_at, content AS snippet FROM chat_log "
            "WHERE chat_id=? AND content LIKE ? ORDER BY created_at DESC LIMIT ?",
            (chat_id, f"%{query.strip()}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]


def _fts_safe(query: str) -> str:
    """Build an FTS5 MATCH clause. Drops tokens <3 chars (trigram tokenizer requires >=3).
    Returns '' if nothing usable — caller should fall back to LIKE."""
    tokens = [t for t in re.split(r"\s+", query.strip()) if len(t) >= 3]
    if not tokens:
        return ""
    return " ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)
