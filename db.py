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

CREATE TABLE IF NOT EXISTS expenses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    amount_won INTEGER NOT NULL,
    category TEXT,
    merchant TEXT,
    when_local TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_expenses_chat_when ON expenses(chat_id, when_local DESC);

CREATE TABLE IF NOT EXISTS habits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    habit_key TEXT NOT NULL,
    duration_min INTEGER,
    notes TEXT,
    when_local TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_habits_chat_when ON habits(chat_id, when_local DESC);

CREATE TABLE IF NOT EXISTS daily_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    date_local TEXT NOT NULL,
    briefing_sent INTEGER NOT NULL DEFAULT 0,
    reflection_prompted INTEGER NOT NULL DEFAULT 0,
    reflection_response TEXT,
    UNIQUE(chat_id, date_local)
);
CREATE INDEX IF NOT EXISTS idx_daily_state_chat ON daily_state(chat_id, date_local DESC);

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

CREATE TABLE IF NOT EXISTS processed_gmail_msg_ids (
    chat_id INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    processed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    action TEXT NOT NULL,             -- 'offered' | 'added' | 'skipped' | 'low_conf'
    PRIMARY KEY (chat_id, message_id)
);

CREATE TABLE IF NOT EXISTS goal_milestones_sent (
    chat_id INTEGER NOT NULL,
    goal_id INTEGER NOT NULL,
    milestone INTEGER NOT NULL,       -- 30 | 14 | 3 | 1
    sent_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (goal_id, milestone)
);

CREATE TABLE IF NOT EXISTS auto_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    rule_kind TEXT NOT NULL,           -- 'gmail_auto_add_event' | 'gcal_auto_rsvp'
    condition_json TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_auto_rules_chat ON auto_rules(chat_id, enabled);

CREATE TABLE IF NOT EXISTS agent_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    action_kind TEXT NOT NULL,         -- 'gmail_send'|'gcal_rsvp'|'event_auto_add'|...
    summary TEXT NOT NULL,             -- single-line human-readable
    payload_json TEXT NOT NULL,        -- full execution detail
    reversible_json TEXT,              -- {kind, args} or NULL if non-undoable
    status TEXT NOT NULL DEFAULT 'executed',  -- executed|reversed|digested
    digested_at TEXT,
    executed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_actions_chat ON agent_actions(chat_id, executed_at DESC);

CREATE TABLE IF NOT EXISTS persona_doc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    version INTEGER NOT NULL,
    content_md TEXT NOT NULL,
    sources_json TEXT NOT NULL DEFAULT '{}',
    generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(chat_id, version)
);
CREATE INDEX IF NOT EXISTS idx_persona_chat ON persona_doc(chat_id, version DESC);

CREATE TABLE IF NOT EXISTS habit_streaks (
    chat_id INTEGER NOT NULL,
    habit_key TEXT NOT NULL,
    current_streak INTEGER NOT NULL DEFAULT 0,
    best_streak INTEGER NOT NULL DEFAULT 0,
    last_log_date TEXT,                -- YYYY-MM-DD KST
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (chat_id, habit_key)
);

-- v5 tables --

CREATE TABLE IF NOT EXISTS missions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    goal_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',  -- running|paused|done|failed|cancelled
    current_hop INTEGER NOT NULL DEFAULT 0,
    max_hops INTEGER NOT NULL DEFAULT 80,
    cost_usd_running REAL NOT NULL DEFAULT 0,
    history_json TEXT NOT NULL DEFAULT '[]',  -- the agent message history for resumable runs
    result_md TEXT,                            -- final markdown when status=done
    last_checkpoint_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_missions_chat_status ON missions(chat_id, status);

CREATE TABLE IF NOT EXISTS nudge_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    nudge_kind TEXT NOT NULL,        -- 'midday_checkin'|'briefing'|'mail_card'|'goal_milestone'|...
    nudge_id TEXT,                    -- optional row id from the source (e.g. event id)
    sent_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    reaction TEXT,                    -- engaged|dismissed|toggled_off|no_response|undo
    reaction_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_nudge_outcomes_chat ON nudge_outcomes(chat_id, sent_at DESC);

CREATE TABLE IF NOT EXISTS system_prompt_overrides (
    chat_id INTEGER PRIMARY KEY,
    override_md TEXT NOT NULL,
    generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    from_kind TEXT NOT NULL,
    from_id INTEGER NOT NULL,
    to_kind TEXT NOT NULL,
    to_id INTEGER NOT NULL,
    relation_kind TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',  -- manual|auto_extract|llm_inferred
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(chat_id, from_kind, from_id, to_kind, to_id, relation_kind)
);
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(chat_id, from_kind, from_id);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(chat_id, to_kind, to_id);

CREATE TABLE IF NOT EXISTS macros (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    prompt TEXT NOT NULL,           -- the natural-language recipe
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(chat_id, name)
);
CREATE INDEX IF NOT EXISTS idx_macros_chat ON macros(chat_id);

CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    topic TEXT NOT NULL,
    cron_kst TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_run_utc TEXT,
    last_digest_md TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(chat_id, name)
);
CREATE INDEX IF NOT EXISTS idx_subs_chat ON subscriptions(chat_id, enabled);

CREATE TABLE IF NOT EXISTS travel_periods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    destination TEXT NOT NULL,         -- '도쿄' '부산' etc.
    start_date_local TEXT NOT NULL,    -- YYYY-MM-DD
    end_date_local TEXT NOT NULL,
    detected_from_event_id INTEGER,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_travel_chat_dates ON travel_periods(chat_id, start_date_local, end_date_local);

CREATE TABLE IF NOT EXISTS habit_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    start_date_local TEXT NOT NULL,
    end_date_local TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',   -- active|completed|abandoned
    result_md TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_experiments_chat_status ON habit_experiments(chat_id, status);

CREATE TABLE IF NOT EXISTS error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    level TEXT NOT NULL,
    source TEXT NOT NULL,         -- logger name
    message TEXT NOT NULL,
    traceback TEXT,
    chat_id INTEGER,
    context_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_error_log_ts ON error_log(ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_error_log_source ON error_log(source, ts_utc DESC);

-- v11 tables --

CREATE TABLE IF NOT EXISTS vector_store (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    entity_kind TEXT NOT NULL,        -- 'chat_log'|'note'|'event'|'reflection'|'goal'
    entity_id INTEGER NOT NULL,
    content_text TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,     -- float32 array, 1536 dims = 6144 bytes
    dimensions INTEGER NOT NULL DEFAULT 1536,
    generated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(chat_id, entity_kind, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_vector_chat_kind ON vector_store(chat_id, entity_kind);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    options_json TEXT NOT NULL DEFAULT '[]',
    factors_json TEXT NOT NULL DEFAULT '{}',
    questions_asked INTEGER NOT NULL DEFAULT 0,
    simulation_md TEXT,
    recommendation TEXT,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending|decided|abandoned
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_decisions_chat ON decisions(chat_id, status);

CREATE TABLE IF NOT EXISTS watch_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    condition_md TEXT NOT NULL,         -- 자연어 조건
    check_kind TEXT NOT NULL,           -- 'web_scrape'|'web_search'|'gmail_query'|'fetch_url'
    check_args_json TEXT NOT NULL DEFAULT '{}',
    check_interval_min INTEGER NOT NULL DEFAULT 60,
    last_check_utc TEXT,
    last_state_json TEXT,
    triggered_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',  -- active|triggered|cancelled
    action_md TEXT,                      -- 발사 시 실행할 action (선택)
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_watches_chat_status ON watch_tasks(chat_id, status);

CREATE TABLE IF NOT EXISTS web_credentials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    site_label TEXT NOT NULL,
    username TEXT NOT NULL,
    password_enc BLOB NOT NULL,         -- AES-GCM encrypted with vault_passphrase
    nonce BLOB NOT NULL,
    last_used_utc TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(chat_id, site_label)
);

CREATE TABLE IF NOT EXISTS voice_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    to_number TEXT NOT NULL,
    purpose TEXT NOT NULL,
    twilio_call_sid TEXT,
    status TEXT NOT NULL DEFAULT 'queued',  -- queued|active|done|failed|cancelled
    transcript_md TEXT,
    summary_md TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    duration_sec INTEGER,
    started_at TEXT,
    ended_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_calls_chat ON voice_calls(chat_id, status);
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
    # Optional location string (free-form address or place name) — feeds leave-by.
    ("events", "location", "TEXT"),
    # APScheduler job id of the active leave-by trigger, if any.
    ("events", "leave_by_job_id", "TEXT"),
    ("goals", "watch_frequency_days", "INTEGER NOT NULL DEFAULT 7"),
    ("goals", "last_watch_run_utc", "TEXT"),
    # JSON list of person_ids whose important_date falls on this date_local.
    # Populated by morning briefing; consumed by 09:00 birthday-solo cron.
    ("daily_state", "birthdays_today_json", "TEXT"),
    ("daily_state", "midday_checkin_sent", "INTEGER NOT NULL DEFAULT 0"),
    # v4 additions
    ("daily_state", "mood_sentiment", "TEXT"),  # 'positive'|'neutral'|'negative'
    ("daily_state", "learning_question_asked", "INTEGER NOT NULL DEFAULT 0"),
    ("daily_state", "learning_question_key", "TEXT"),  # what fact/person we asked about
    # v8: live mission UI — the Telegram message_id we keep editing for progress.
    ("missions", "progress_message_id", "INTEGER"),
    # v12 W8: person 모국어 (ko/en/ja/zh, default ko) — 메일 자동 통역에 사용
    ("people", "preferred_language", "TEXT NOT NULL DEFAULT 'ko'"),
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
    location: Optional[str] = None,
) -> int:
    when_iso = when_utc.astimezone(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO events (chat_id, title, when_utc, notes, remind_lead_minutes, "
            "gcal_event_id, gcal_sync_state, location) VALUES (?,?,?,?,?,?,?,?)",
            (chat_id, title, when_iso, notes, remind_lead_minutes,
             gcal_event_id, gcal_sync_state, location),
        )
        return cur.lastrowid


def set_event_leave_by_job(event_id: int, job_id: Optional[str]) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE events SET leave_by_job_id=? WHERE id=?",
            (job_id, event_id),
        )


def events_with_location_in_window(chat_id: int, hours: int = 48) -> List[sqlite3.Row]:
    """Future events within the next N hours that have a non-empty location.
    Used by the daily leave-by recompute cron."""
    now_iso = datetime.now(timezone.utc).isoformat()
    end_iso = (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM events WHERE chat_id=? AND when_utc BETWEEN ? AND ? "
            "AND location IS NOT NULL AND location <> '' ORDER BY when_utc ASC",
            (chat_id, now_iso, end_iso),
        ))


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
    preferred_language: Optional[str] = None,
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
                "notes=COALESCE(?, notes), important_dates_json=?, "
                "preferred_language=COALESCE(?, preferred_language) WHERE id=?",
                (_json.dumps(sorted(cur_aliases), ensure_ascii=False), role, notes,
                 _json.dumps(cur_dates, ensure_ascii=False), preferred_language, existing["id"]),
            )
            return existing["id"]
        cur = c.execute(
            "INSERT INTO people (chat_id, name, aliases_json, role, notes, important_dates_json, "
            "preferred_language) VALUES (?,?,?,?,?,?,?)",
            (
                chat_id, name.strip(),
                _json.dumps(aliases or [], ensure_ascii=False),
                role, notes,
                _json.dumps(important_dates or [], ensure_ascii=False),
                preferred_language or "ko",
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


# ---------------- daily_state (briefing + reflection) ----------------


def get_daily_state(chat_id: int, date_local: str) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM daily_state WHERE chat_id=? AND date_local=?",
            (chat_id, date_local),
        ).fetchone()


def mark_briefing_sent(chat_id: int, date_local: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO daily_state (chat_id, date_local, briefing_sent) VALUES (?,?,1) "
            "ON CONFLICT(chat_id, date_local) DO UPDATE SET briefing_sent=1",
            (chat_id, date_local),
        )


def mark_reflection_prompted(chat_id: int, date_local: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO daily_state (chat_id, date_local, reflection_prompted) VALUES (?,?,1) "
            "ON CONFLICT(chat_id, date_local) DO UPDATE SET reflection_prompted=1",
            (chat_id, date_local),
        )


def save_reflection_response(chat_id: int, date_local: str, text: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO daily_state (chat_id, date_local, reflection_response) VALUES (?,?,?) "
            "ON CONFLICT(chat_id, date_local) DO UPDATE SET reflection_response=excluded.reflection_response",
            (chat_id, date_local, text.strip()),
        )


def recent_reflections(chat_id: int, days: int = 7) -> List[sqlite3.Row]:
    with _conn() as c:
        cutoff_date = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
        return list(c.execute(
            "SELECT * FROM daily_state WHERE chat_id=? AND date_local >= ? "
            "AND reflection_response IS NOT NULL ORDER BY date_local DESC",
            (chat_id, cutoff_date),
        ))


# ---------------- expenses + habits ----------------


def log_expense(
    chat_id: int,
    amount_won: int,
    category: Optional[str] = None,
    merchant: Optional[str] = None,
    when_local: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    when = when_local or datetime.now(_USER_TZ).strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO expenses (chat_id, amount_won, category, merchant, when_local, notes) "
            "VALUES (?,?,?,?,?,?)",
            (chat_id, int(amount_won), category, merchant, when, notes),
        )
        return cur.lastrowid


def list_expenses(chat_id: int, days: int = 30) -> List[sqlite3.Row]:
    cutoff = (datetime.now(_USER_TZ) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM expenses WHERE chat_id=? AND when_local>=? ORDER BY when_local DESC",
            (chat_id, cutoff),
        ))


def summarize_expenses(chat_id: int, days: int = 30) -> Dict:
    rows = list_expenses(chat_id, days=days)
    total = sum(r["amount_won"] for r in rows)
    by_cat: Dict[str, int] = {}
    for r in rows:
        cat = r["category"] or "기타"
        by_cat[cat] = by_cat.get(cat, 0) + r["amount_won"]
    return {
        "days": days, "count": len(rows), "total_won": total,
        "by_category": sorted(by_cat.items(), key=lambda x: -x[1]),
    }


def delete_expense(expense_id: int, chat_id: int) -> Optional[Dict]:
    with _conn() as c:
        row = c.execute("SELECT * FROM expenses WHERE id=? AND chat_id=?",
                         (expense_id, chat_id)).fetchone()
        if row is None:
            return None
        payload = dict(row)
        c.execute("DELETE FROM expenses WHERE id=? AND chat_id=?", (expense_id, chat_id))
        return payload


def log_habit(
    chat_id: int,
    habit_key: str,
    duration_min: Optional[int] = None,
    notes: Optional[str] = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO habits (chat_id, habit_key, duration_min, notes) VALUES (?,?,?,?)",
            (chat_id, habit_key.strip(), duration_min, notes),
        )
        return cur.lastrowid


def summarize_habits(chat_id: int, days: int = 7) -> Dict:
    cutoff = (datetime.now(_USER_TZ) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as c:
        rows = list(c.execute(
            "SELECT habit_key, COUNT(*) AS n, COALESCE(SUM(duration_min),0) AS mins "
            "FROM habits WHERE chat_id=? AND when_local>=? GROUP BY habit_key ORDER BY n DESC",
            (chat_id, cutoff),
        ))
    return {"days": days, "by_habit": [dict(r) for r in rows]}


# ---------------- pre-emptive helpers (Gmail dedup, milestone dedup, daily state) ----------------


def is_gmail_processed(chat_id: int, message_id: str) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM processed_gmail_msg_ids WHERE chat_id=? AND message_id=?",
            (chat_id, message_id),
        ).fetchone()
        return row is not None


def mark_gmail_processed(chat_id: int, message_id: str, action: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO processed_gmail_msg_ids (chat_id, message_id, action) VALUES (?,?,?) "
            "ON CONFLICT(chat_id, message_id) DO UPDATE SET action=excluded.action, "
            "processed_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            (chat_id, message_id, action),
        )


def milestone_already_sent(goal_id: int, milestone: int) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM goal_milestones_sent WHERE goal_id=? AND milestone=?",
            (goal_id, milestone),
        ).fetchone()
        return row is not None


def record_milestone_sent(chat_id: int, goal_id: int, milestone: int) -> None:
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO goal_milestones_sent (chat_id, goal_id, milestone) VALUES (?,?,?)",
            (chat_id, goal_id, milestone),
        )


def mark_birthdays_today(chat_id: int, date_local: str, person_ids: List[int]) -> None:
    payload = _json.dumps(person_ids)
    with _conn() as c:
        c.execute(
            "INSERT INTO daily_state (chat_id, date_local, birthdays_today_json) VALUES (?,?,?) "
            "ON CONFLICT(chat_id, date_local) DO UPDATE SET birthdays_today_json=excluded.birthdays_today_json",
            (chat_id, date_local, payload),
        )


def mark_midday_checkin_sent(chat_id: int, date_local: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO daily_state (chat_id, date_local, midday_checkin_sent) VALUES (?,?,1) "
            "ON CONFLICT(chat_id, date_local) DO UPDATE SET midday_checkin_sent=1",
            (chat_id, date_local),
        )


def set_mood_sentiment(chat_id: int, date_local: str, sentiment: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO daily_state (chat_id, date_local, mood_sentiment) VALUES (?,?,?) "
            "ON CONFLICT(chat_id, date_local) DO UPDATE SET mood_sentiment=excluded.mood_sentiment",
            (chat_id, date_local, sentiment),
        )


def mark_learning_question_asked(chat_id: int, date_local: str, key: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO daily_state (chat_id, date_local, learning_question_asked, learning_question_key) "
            "VALUES (?,?,1,?) ON CONFLICT(chat_id, date_local) DO UPDATE SET "
            "learning_question_asked=1, learning_question_key=excluded.learning_question_key",
            (chat_id, date_local, key),
        )


def recent_mood_stats(chat_id: int, days: int = 30) -> Dict:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    with _conn() as c:
        rows = list(c.execute(
            "SELECT mood_sentiment, COUNT(*) AS n FROM daily_state "
            "WHERE chat_id=? AND date_local>=? AND mood_sentiment IS NOT NULL "
            "GROUP BY mood_sentiment",
            (chat_id, cutoff),
        ))
    return {r["mood_sentiment"]: r["n"] for r in rows}


# ---------------- auto_rules (sent-consent rule engine) ----------------


def add_auto_rule(chat_id: int, rule_kind: str, condition: Dict) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO auto_rules (chat_id, rule_kind, condition_json) VALUES (?,?,?)",
            (chat_id, rule_kind, _json.dumps(condition, ensure_ascii=False)),
        )
        return cur.lastrowid


def list_auto_rules(chat_id: int, only_enabled: bool = True) -> List[sqlite3.Row]:
    with _conn() as c:
        if only_enabled:
            return list(c.execute(
                "SELECT * FROM auto_rules WHERE chat_id=? AND enabled=1 ORDER BY id",
                (chat_id,)))
        return list(c.execute(
            "SELECT * FROM auto_rules WHERE chat_id=? ORDER BY id", (chat_id,)))


def disable_auto_rule(chat_id: int, rule_id: int) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE auto_rules SET enabled=0 WHERE chat_id=? AND id=?",
            (chat_id, rule_id))
        return cur.rowcount > 0


def delete_auto_rule(chat_id: int, rule_id: int) -> bool:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM auto_rules WHERE chat_id=? AND id=?", (chat_id, rule_id))
        return cur.rowcount > 0


# ---------------- agent_actions (audit log for autonomous actions) ----------------


def log_agent_action(
    chat_id: int,
    action_kind: str,
    summary: str,
    payload: Dict,
    reversible: Optional[Dict] = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO agent_actions (chat_id, action_kind, summary, payload_json, reversible_json) "
            "VALUES (?,?,?,?,?)",
            (chat_id, action_kind, summary,
             _json.dumps(payload, ensure_ascii=False),
             _json.dumps(reversible, ensure_ascii=False) if reversible else None),
        )
        return cur.lastrowid


def list_undigested_actions(chat_id: int) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM agent_actions WHERE chat_id=? AND status='executed' "
            "AND digested_at IS NULL ORDER BY executed_at ASC",
            (chat_id,)))


def mark_actions_digested(chat_id: int, ids: List[int]) -> None:
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    with _conn() as c:
        c.execute(
            f"UPDATE agent_actions SET digested_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
            f"WHERE chat_id=? AND id IN ({placeholders})",
            (chat_id, *ids))


def get_agent_action(chat_id: int, action_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM agent_actions WHERE chat_id=? AND id=?",
            (chat_id, action_id)).fetchone()


def mark_action_reversed(chat_id: int, action_id: int) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE agent_actions SET status='reversed' WHERE chat_id=? AND id=?",
            (chat_id, action_id))


# ---------------- persona_doc ----------------


def get_latest_persona(chat_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM persona_doc WHERE chat_id=? ORDER BY version DESC LIMIT 1",
            (chat_id,)).fetchone()


def save_persona(chat_id: int, content_md: str, sources: Dict) -> int:
    cur_version = 0
    latest = get_latest_persona(chat_id)
    if latest:
        cur_version = latest["version"]
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO persona_doc (chat_id, version, content_md, sources_json) "
            "VALUES (?,?,?,?)",
            (chat_id, cur_version + 1, content_md,
             _json.dumps(sources, ensure_ascii=False)))
        return cur.lastrowid


def cleanup_old_personas(keep_recent: int = 8) -> int:
    """Drop persona_doc rows older than `keep_recent` versions per chat."""
    with _conn() as c:
        deleted = 0
        for r in c.execute("SELECT DISTINCT chat_id FROM persona_doc"):
            cid = r["chat_id"]
            rows = list(c.execute(
                "SELECT id FROM persona_doc WHERE chat_id=? ORDER BY version DESC",
                (cid,)))
            stale = [row["id"] for row in rows[keep_recent:]]
            if stale:
                placeholders = ",".join("?" * len(stale))
                c.execute(f"DELETE FROM persona_doc WHERE id IN ({placeholders})", stale)
                deleted += len(stale)
        return deleted


# ---------------- habit_streaks ----------------


def upsert_habit_streak(
    chat_id: int, habit_key: str,
    current_streak: int, last_log_date: str,
) -> None:
    with _conn() as c:
        row = c.execute(
            "SELECT best_streak FROM habit_streaks WHERE chat_id=? AND habit_key=?",
            (chat_id, habit_key)).fetchone()
        best = max(current_streak, row["best_streak"] if row else 0)
        c.execute(
            "INSERT INTO habit_streaks (chat_id, habit_key, current_streak, best_streak, last_log_date, updated_at) "
            "VALUES (?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now')) "
            "ON CONFLICT(chat_id, habit_key) DO UPDATE SET "
            "current_streak=excluded.current_streak, best_streak=excluded.best_streak, "
            "last_log_date=excluded.last_log_date, updated_at=excluded.updated_at",
            (chat_id, habit_key, current_streak, best, last_log_date))


def get_habit_streak(chat_id: int, habit_key: str) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM habit_streaks WHERE chat_id=? AND habit_key=?",
            (chat_id, habit_key)).fetchone()


def list_habit_streaks(chat_id: int) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM habit_streaks WHERE chat_id=? ORDER BY current_streak DESC",
            (chat_id,)))


def habit_logs_by_date(chat_id: int, habit_key: str, days: int = 60) -> List[str]:
    """Return distinct date_local (YYYY-MM-DD) strings where this habit was logged."""
    cutoff = (datetime.now(_USER_TZ) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    with _conn() as c:
        rows = list(c.execute(
            "SELECT DISTINCT substr(when_local, 1, 10) AS d FROM habits "
            "WHERE chat_id=? AND habit_key=? AND when_local>=? ORDER BY d",
            (chat_id, habit_key, cutoff)))
    return [r["d"] for r in rows]


def all_habit_keys(chat_id: int) -> List[str]:
    with _conn() as c:
        rows = list(c.execute(
            "SELECT DISTINCT habit_key FROM habits WHERE chat_id=?", (chat_id,)))
    return [r["habit_key"] for r in rows]


# ---------------- v5: missions ----------------


def add_mission(chat_id: int, title: str, goal_text: str, max_hops: int = 80) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO missions (chat_id, title, goal_text, max_hops) VALUES (?,?,?,?)",
            (chat_id, title, goal_text, max_hops))
        return cur.lastrowid


def get_mission(mission_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()


def list_running_missions() -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM missions WHERE status='running' ORDER BY updated_at ASC"))


def list_missions(chat_id: int, status: Optional[str] = None) -> List[sqlite3.Row]:
    with _conn() as c:
        if status:
            return list(c.execute(
                "SELECT * FROM missions WHERE chat_id=? AND status=? "
                "ORDER BY created_at DESC", (chat_id, status)))
        return list(c.execute(
            "SELECT * FROM missions WHERE chat_id=? ORDER BY created_at DESC",
            (chat_id,)))


def update_mission(mission_id: int, **fields) -> bool:
    if not fields:
        return False
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    cols = ", ".join(f"{k}=?" for k in fields)
    params = list(fields.values()) + [mission_id]
    with _conn() as c:
        cur = c.execute(f"UPDATE missions SET {cols} WHERE id=?", params)
        return cur.rowcount > 0


def cancel_mission(chat_id: int, mission_id: int) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE missions SET status='cancelled', "
            "updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
            "WHERE id=? AND chat_id=? AND status='running'",
            (mission_id, chat_id))
        return cur.rowcount > 0


# ---------------- v5: nudge_outcomes ----------------


def record_nudge(chat_id: int, nudge_kind: str, nudge_id: Optional[str] = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO nudge_outcomes (chat_id, nudge_kind, nudge_id) VALUES (?,?,?)",
            (chat_id, nudge_kind, nudge_id))
        return cur.lastrowid


def react_to_recent_nudge(
    chat_id: int, nudge_kind: Optional[str], reaction: str,
    *, within_hours: int = 24,
) -> int:
    """Update the most recent pending nudge (reaction NULL) of this kind to `reaction`.
    Returns number of rows updated (0 or 1)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=within_hours)).isoformat()
    q = ("UPDATE nudge_outcomes SET reaction=?, "
         "reaction_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
         "WHERE id=(SELECT id FROM nudge_outcomes "
         "          WHERE chat_id=? AND reaction IS NULL AND sent_at>=? "
         + ("AND nudge_kind=? " if nudge_kind else "")
         + "ORDER BY sent_at DESC LIMIT 1)")
    params: List = [reaction, chat_id, cutoff]
    if nudge_kind:
        params.append(nudge_kind)
    with _conn() as c:
        cur = c.execute(q, params)
        return cur.rowcount


def mark_stale_nudges_no_response(hours: int = 24) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _conn() as c:
        cur = c.execute(
            "UPDATE nudge_outcomes SET reaction='no_response', "
            "reaction_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
            "WHERE reaction IS NULL AND sent_at<?", (cutoff,))
        return cur.rowcount


def nudge_stats_by_kind(chat_id: int, days: int = 28) -> List[Dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _conn() as c:
        rows = list(c.execute(
            "SELECT nudge_kind, "
            "  COUNT(*) AS sent, "
            "  SUM(CASE WHEN reaction='engaged' THEN 1 ELSE 0 END) AS engaged, "
            "  SUM(CASE WHEN reaction='dismissed' THEN 1 ELSE 0 END) AS dismissed, "
            "  SUM(CASE WHEN reaction='no_response' THEN 1 ELSE 0 END) AS no_response, "
            "  SUM(CASE WHEN reaction='toggled_off' THEN 1 ELSE 0 END) AS toggled_off "
            "FROM nudge_outcomes WHERE chat_id=? AND sent_at>=? "
            "GROUP BY nudge_kind ORDER BY sent DESC",
            (chat_id, cutoff)))
    return [dict(r) for r in rows]


# ---------------- v5: system_prompt_overrides ----------------


def get_prompt_override(chat_id: int) -> Optional[str]:
    with _conn() as c:
        row = c.execute(
            "SELECT override_md FROM system_prompt_overrides WHERE chat_id=?",
            (chat_id,)).fetchone()
        return row["override_md"] if row else None


def set_prompt_override(chat_id: int, override_md: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO system_prompt_overrides (chat_id, override_md) VALUES (?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET override_md=excluded.override_md, "
            "generated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            (chat_id, override_md))


def clear_prompt_override(chat_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM system_prompt_overrides WHERE chat_id=?",
                         (chat_id,))
        return cur.rowcount > 0


# ---------------- v5: relations (knowledge graph) ----------------


def add_relation(
    chat_id: int, from_kind: str, from_id: int, to_kind: str, to_id: int,
    relation_kind: str, source: str = "manual",
) -> Optional[int]:
    with _conn() as c:
        try:
            cur = c.execute(
                "INSERT INTO relations (chat_id, from_kind, from_id, to_kind, to_id, "
                "relation_kind, source) VALUES (?,?,?,?,?,?,?)",
                (chat_id, from_kind, from_id, to_kind, to_id, relation_kind, source))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None  # already exists


def neighbors(chat_id: int, kind: str, entity_id: int) -> List[sqlite3.Row]:
    """All outgoing + incoming edges for a node."""
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM relations WHERE chat_id=? AND "
            "((from_kind=? AND from_id=?) OR (to_kind=? AND to_id=?))",
            (chat_id, kind, entity_id, kind, entity_id)))


def graph_bfs(
    chat_id: int, start_kind: str, start_id: int,
    max_depth: int = 2, relation_filter: Optional[List[str]] = None,
) -> Dict:
    """Breadth-first traversal returning {nodes, edges}. Nodes are
    (kind, id) tuples; edges are full relation rows."""
    seen = {(start_kind, start_id)}
    frontier = [(start_kind, start_id)]
    edges = []
    for _depth in range(max_depth):
        next_frontier = []
        for kind, eid in frontier:
            for e in neighbors(chat_id, kind, eid):
                if relation_filter and e["relation_kind"] not in relation_filter:
                    continue
                edges.append(dict(e))
                pair = (e["to_kind"], e["to_id"]) if (e["from_kind"], e["from_id"]) == (kind, eid) \
                       else (e["from_kind"], e["from_id"])
                if pair not in seen:
                    seen.add(pair)
                    next_frontier.append(pair)
        frontier = next_frontier
    return {
        "nodes": [{"kind": k, "id": i} for k, i in seen],
        "edges": edges,
    }


# ---------------- v5: error_log ----------------


def log_error(
    level: str, source: str, message: str, traceback: Optional[str] = None,
    chat_id: Optional[int] = None, context: Optional[Dict] = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO error_log (level, source, message, traceback, chat_id, context_json) "
            "VALUES (?,?,?,?,?,?)",
            (level, source, message[:2000], (traceback or "")[:6000],
             chat_id, _json.dumps(context or {}, ensure_ascii=False)))
        return cur.lastrowid


def recent_errors(hours: int = 24, limit: int = 50) -> List[sqlite3.Row]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM error_log WHERE ts_utc>=? ORDER BY ts_utc DESC LIMIT ?",
            (cutoff, limit)))


def error_counts_by_source(hours: int = 24) -> List[Dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _conn() as c:
        rows = list(c.execute(
            "SELECT source, COUNT(*) AS n FROM error_log "
            "WHERE ts_utc>=? GROUP BY source ORDER BY n DESC LIMIT 10", (cutoff,)))
    return [dict(r) for r in rows]


def add_macro(chat_id: int, name: str, prompt: str) -> Optional[int]:
    with _conn() as c:
        try:
            cur = c.execute(
                "INSERT INTO macros (chat_id, name, prompt) VALUES (?,?,?)",
                (chat_id, name.strip(), prompt.strip()))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None  # duplicate name for this chat


def get_macro(chat_id: int, name: str) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM macros WHERE chat_id=? AND name=?",
            (chat_id, name.strip())).fetchone()


def list_macros(chat_id: int) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM macros WHERE chat_id=? ORDER BY name", (chat_id,)))


def delete_macro(chat_id: int, name: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM macros WHERE chat_id=? AND name=?",
            (chat_id, name.strip()))
        return cur.rowcount > 0


# ---------------- v8: subscriptions ----------------


def add_subscription(chat_id: int, name: str, topic: str, cron_kst: str) -> Optional[int]:
    with _conn() as c:
        try:
            cur = c.execute(
                "INSERT INTO subscriptions (chat_id, name, topic, cron_kst) VALUES (?,?,?,?)",
                (chat_id, name.strip(), topic.strip(), cron_kst.strip()))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None


def list_subscriptions(chat_id: int, only_enabled: bool = False) -> List[sqlite3.Row]:
    with _conn() as c:
        if only_enabled:
            return list(c.execute(
                "SELECT * FROM subscriptions WHERE chat_id=? AND enabled=1 ORDER BY name",
                (chat_id,)))
        return list(c.execute(
            "SELECT * FROM subscriptions WHERE chat_id=? ORDER BY name",
            (chat_id,)))


def get_subscription(sub_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute("SELECT * FROM subscriptions WHERE id=?", (sub_id,)).fetchone()


def all_enabled_subscriptions() -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute("SELECT * FROM subscriptions WHERE enabled=1"))


def update_subscription_run(sub_id: int, digest_md: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE subscriptions SET last_run_utc=strftime('%Y-%m-%dT%H:%M:%fZ','now'), "
            "last_digest_md=? WHERE id=?",
            (digest_md[:4000], sub_id))


def disable_subscription(chat_id: int, name: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE subscriptions SET enabled=0 WHERE chat_id=? AND name=?",
            (chat_id, name.strip()))
        return cur.rowcount > 0


def delete_subscription(chat_id: int, name: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM subscriptions WHERE chat_id=? AND name=?",
            (chat_id, name.strip()))
        return cur.rowcount > 0


# ---------------- v9: travel periods ----------------


def add_travel_period(
    chat_id: int, destination: str,
    start_date_local: str, end_date_local: str,
    detected_from_event_id: Optional[int] = None,
    notes: Optional[str] = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO travel_periods (chat_id, destination, start_date_local, "
            "end_date_local, detected_from_event_id, notes) VALUES (?,?,?,?,?,?)",
            (chat_id, destination.strip(), start_date_local, end_date_local,
             detected_from_event_id, notes))
        return cur.lastrowid


def active_travel_for(chat_id: int, today_iso: str) -> Optional[sqlite3.Row]:
    """Return the travel_period covering today_iso (if any)."""
    with _conn() as c:
        return c.execute(
            "SELECT * FROM travel_periods WHERE chat_id=? "
            "AND start_date_local <= ? AND end_date_local >= ? "
            "ORDER BY start_date_local DESC LIMIT 1",
            (chat_id, today_iso, today_iso)).fetchone()


def list_travel_periods(chat_id: int) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM travel_periods WHERE chat_id=? "
            "ORDER BY start_date_local DESC LIMIT 30", (chat_id,)))


def delete_travel_period(chat_id: int, travel_id: int) -> bool:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM travel_periods WHERE chat_id=? AND id=?",
            (chat_id, travel_id))
        return cur.rowcount > 0


# ---------------- v9: habit_experiments ----------------


def add_habit_experiment(
    chat_id: int, title: str, hypothesis: str,
    start_date_local: str, end_date_local: str,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO habit_experiments (chat_id, title, hypothesis, "
            "start_date_local, end_date_local) VALUES (?,?,?,?,?)",
            (chat_id, title.strip(), hypothesis.strip(),
             start_date_local, end_date_local))
        return cur.lastrowid


def list_habit_experiments(chat_id: int, status: Optional[str] = None) -> List[sqlite3.Row]:
    with _conn() as c:
        if status:
            return list(c.execute(
                "SELECT * FROM habit_experiments WHERE chat_id=? AND status=? "
                "ORDER BY created_at DESC", (chat_id, status)))
        return list(c.execute(
            "SELECT * FROM habit_experiments WHERE chat_id=? "
            "ORDER BY created_at DESC", (chat_id,)))


def get_experiment(exp_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM habit_experiments WHERE id=?", (exp_id,)).fetchone()


def complete_experiment(chat_id: int, exp_id: int, result_md: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE habit_experiments SET status='completed', result_md=? "
            "WHERE id=? AND chat_id=? AND status='active'",
            (result_md[:2000], exp_id, chat_id))
        return cur.rowcount > 0


def all_active_experiments() -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM habit_experiments WHERE status='active' "
            "AND end_date_local <= date('now','+1 day')"))


# ---------------- v11: vector_store (lifelog RAG) ----------------


def upsert_embedding(
    chat_id: int, entity_kind: str, entity_id: int,
    content_text: str, embedding_blob: bytes, dimensions: int = 1536,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO vector_store (chat_id, entity_kind, entity_id, content_text, "
            "embedding_blob, dimensions) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(chat_id, entity_kind, entity_id) DO UPDATE SET "
            "content_text=excluded.content_text, embedding_blob=excluded.embedding_blob, "
            "generated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')",
            (chat_id, entity_kind, entity_id, content_text[:4000],
             embedding_blob, dimensions))
        return cur.lastrowid


def list_all_embeddings(chat_id: int) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT id, entity_kind, entity_id, content_text, embedding_blob, dimensions "
            "FROM vector_store WHERE chat_id=?", (chat_id,)))


def already_embedded(chat_id: int, entity_kind: str, entity_id: int) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM vector_store WHERE chat_id=? AND entity_kind=? AND entity_id=?",
            (chat_id, entity_kind, entity_id)).fetchone()
        return row is not None


def yesterdays_new_entities(chat_id: int) -> Dict[str, List[Dict]]:
    """어제(KST) 새로 추가된 chat_log/notes/events/reflections — lifelog cron이 사용."""
    cutoff_start = (datetime.now(timezone.utc) - timedelta(hours=36)).isoformat()
    cutoff_end = datetime.now(timezone.utc).isoformat()
    out: Dict[str, List[Dict]] = {"chat_log": [], "note": [], "event": [], "reflection": []}
    with _conn() as c:
        for r in c.execute(
            "SELECT id, content FROM chat_log WHERE chat_id=? AND role='user' "
            "AND created_at BETWEEN ? AND ? AND length(content) > 20",
            (chat_id, cutoff_start, cutoff_end)):
            out["chat_log"].append({"id": r["id"], "content": r["content"]})
        for r in c.execute(
            "SELECT id, content FROM notes WHERE chat_id=? AND created_at BETWEEN ? AND ?",
            (chat_id, cutoff_start, cutoff_end)):
            out["note"].append({"id": r["id"], "content": r["content"]})
        for r in c.execute(
            "SELECT id, title, notes FROM events WHERE chat_id=? AND created_at BETWEEN ? AND ?",
            (chat_id, cutoff_start, cutoff_end)):
            txt = f"{r['title']} {r['notes'] or ''}".strip()
            out["event"].append({"id": r["id"], "content": txt})
        for r in c.execute(
            "SELECT id, reflection_response, date_local FROM daily_state WHERE chat_id=? "
            "AND reflection_response IS NOT NULL AND length(reflection_response) > 10 "
            "AND date_local >= date(?,'-2 day')",
            (chat_id, cutoff_end)):
            out["reflection"].append(
                {"id": r["id"], "content": f"{r['date_local']}: {r['reflection_response']}"})
    return out


def all_chat_ids_active(days: int = 30) -> List[int]:
    """최근 N일 이내 활동한 chat_ids — cron iteration용."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _conn() as c:
        return [r["chat_id"] for r in c.execute(
            "SELECT DISTINCT chat_id FROM chat_log WHERE created_at>=?", (cutoff,))]


# ---------------- v11: decisions ----------------


def create_decision(chat_id: int, title: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO decisions (chat_id, title) VALUES (?,?)",
            (chat_id, title.strip()))
        return cur.lastrowid


def get_decision(decision_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()


def list_decisions(chat_id: int, status: Optional[str] = None) -> List[sqlite3.Row]:
    with _conn() as c:
        if status:
            return list(c.execute(
                "SELECT * FROM decisions WHERE chat_id=? AND status=? "
                "ORDER BY created_at DESC", (chat_id, status)))
        return list(c.execute(
            "SELECT * FROM decisions WHERE chat_id=? ORDER BY created_at DESC LIMIT 20",
            (chat_id,)))


def get_pending_decision(chat_id: int) -> Optional[sqlite3.Row]:
    """현재 진행 중인 결정 1개 (가장 최근, status=pending)."""
    with _conn() as c:
        return c.execute(
            "SELECT * FROM decisions WHERE chat_id=? AND status='pending' "
            "ORDER BY id DESC LIMIT 1", (chat_id,)).fetchone()


def update_decision(decision_id: int, **fields) -> bool:
    if not fields:
        return False
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        cur = c.execute(
            f"UPDATE decisions SET {cols} WHERE id=?",
            list(fields.values()) + [decision_id])
        return cur.rowcount > 0


# ---------------- v11: watch_tasks ----------------


def add_watch(
    chat_id: int, label: str, condition_md: str,
    check_kind: str, check_args: Dict, check_interval_min: int = 60,
    action_md: Optional[str] = None,
) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO watch_tasks (chat_id, label, condition_md, check_kind, "
            "check_args_json, check_interval_min, action_md) VALUES (?,?,?,?,?,?,?)",
            (chat_id, label.strip(), condition_md.strip(), check_kind,
             _json.dumps(check_args, ensure_ascii=False),
             max(30, check_interval_min), action_md))
        return cur.lastrowid


def list_watches(chat_id: int, status: Optional[str] = None) -> List[sqlite3.Row]:
    with _conn() as c:
        if status:
            return list(c.execute(
                "SELECT * FROM watch_tasks WHERE chat_id=? AND status=? "
                "ORDER BY id DESC", (chat_id, status)))
        return list(c.execute(
            "SELECT * FROM watch_tasks WHERE chat_id=? ORDER BY id DESC LIMIT 20",
            (chat_id,)))


def get_watch(watch_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM watch_tasks WHERE id=?", (watch_id,)).fetchone()


def cancel_watch(chat_id: int, watch_id: int) -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE watch_tasks SET status='cancelled' WHERE chat_id=? AND id=?",
            (chat_id, watch_id))
        return cur.rowcount > 0


def watches_due(now_utc: Optional[datetime] = None) -> List[sqlite3.Row]:
    """status=active + 마지막 체크 이후 interval 지남."""
    now_utc = now_utc or datetime.now(timezone.utc)
    with _conn() as c:
        rows = list(c.execute(
            "SELECT * FROM watch_tasks WHERE status='active' ORDER BY last_check_utc ASC NULLS FIRST"))
    out = []
    for r in rows:
        if r["last_check_utc"]:
            try:
                last = datetime.fromisoformat(r["last_check_utc"])
                if (now_utc - last).total_seconds() < r["check_interval_min"] * 60:
                    continue
            except Exception:
                pass
        out.append(r)
    return out


def update_watch_check(watch_id: int, state_json: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE watch_tasks SET last_check_utc=strftime('%Y-%m-%dT%H:%M:%fZ','now'), "
            "last_state_json=? WHERE id=?", (state_json, watch_id))


def trigger_watch(watch_id: int) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE watch_tasks SET status='triggered', "
            "triggered_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?",
            (watch_id,))


# ---------------- v11: web_credentials (encrypted vault) ----------------


def add_web_credential(
    chat_id: int, site_label: str, username: str,
    password_enc: bytes, nonce: bytes,
) -> int:
    with _conn() as c:
        try:
            cur = c.execute(
                "INSERT INTO web_credentials (chat_id, site_label, username, password_enc, nonce) "
                "VALUES (?,?,?,?,?)",
                (chat_id, site_label.strip().lower(), username.strip(), password_enc, nonce))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            # upsert
            c.execute(
                "UPDATE web_credentials SET username=?, password_enc=?, nonce=?, "
                "created_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                "WHERE chat_id=? AND site_label=?",
                (username.strip(), password_enc, nonce, chat_id, site_label.strip().lower()))
            return c.execute(
                "SELECT id FROM web_credentials WHERE chat_id=? AND site_label=?",
                (chat_id, site_label.strip().lower())).fetchone()["id"]


def get_web_credential(chat_id: int, site_label: str) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM web_credentials WHERE chat_id=? AND site_label=?",
            (chat_id, site_label.strip().lower())).fetchone()


def list_web_credentials(chat_id: int) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT id, site_label, username, last_used_utc FROM web_credentials "
            "WHERE chat_id=? ORDER BY site_label", (chat_id,)))


def delete_web_credential(chat_id: int, site_label: str) -> bool:
    with _conn() as c:
        cur = c.execute(
            "DELETE FROM web_credentials WHERE chat_id=? AND site_label=?",
            (chat_id, site_label.strip().lower()))
        return cur.rowcount > 0


def touch_web_credential(chat_id: int, site_label: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE web_credentials SET last_used_utc=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
            "WHERE chat_id=? AND site_label=?", (chat_id, site_label.strip().lower()))


# ---------------- v11: voice_calls ----------------


def create_voice_call(chat_id: int, to_number: str, purpose: str) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO voice_calls (chat_id, to_number, purpose) VALUES (?,?,?)",
            (chat_id, to_number.strip(), purpose.strip()))
        return cur.lastrowid


def update_voice_call(call_id: int, **fields) -> bool:
    if not fields:
        return False
    cols = ", ".join(f"{k}=?" for k in fields)
    with _conn() as c:
        cur = c.execute(
            f"UPDATE voice_calls SET {cols} WHERE id=?",
            list(fields.values()) + [call_id])
        return cur.rowcount > 0


def get_voice_call(call_id: int) -> Optional[sqlite3.Row]:
    with _conn() as c:
        return c.execute(
            "SELECT * FROM voice_calls WHERE id=?", (call_id,)).fetchone()


def list_voice_calls(chat_id: int) -> List[sqlite3.Row]:
    with _conn() as c:
        return list(c.execute(
            "SELECT * FROM voice_calls WHERE chat_id=? ORDER BY created_at DESC LIMIT 20",
            (chat_id,)))


def voice_call_cost_today(chat_id: int) -> float:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    with _conn() as c:
        r = c.execute(
            "SELECT COALESCE(SUM(cost_usd),0) AS s FROM voice_calls "
            "WHERE chat_id=? AND created_at>=?", (chat_id, cutoff)).fetchone()
        return float(r["s"] or 0.0)


def find_duplicate_people(chat_id: int) -> List[Dict]:
    """Return candidate duplicate pairs based on name OR overlapping aliases."""
    people = list_people(chat_id)
    dups = []
    by_name: Dict[str, List[Dict]] = {}
    for p in people:
        nm = (p["name"] or "").strip()
        if not nm:
            continue
        by_name.setdefault(nm, []).append(dict(p))
    for nm, rows in by_name.items():
        if len(rows) > 1:
            dups.append({"name": nm, "ids": [r["id"] for r in rows]})
    return dups


def merge_people(chat_id: int, keep_id: int, drop_ids: List[int]) -> Dict:
    """Move important_dates, aliases, last_contact_utc from drop_ids into
    keep_id, then delete the dropped rows. Also rewrites relations table."""
    if not drop_ids:
        return {"moved_dates": 0, "moved_aliases": 0, "rewired_relations": 0}
    moved_dates = moved_aliases = rewired = 0
    with _conn() as c:
        keep = c.execute(
            "SELECT * FROM people WHERE id=? AND chat_id=?",
            (keep_id, chat_id)).fetchone()
        if not keep:
            return {"error": "keep_id not found"}
        keep_dates = _json.loads(keep["important_dates_json"] or "[]")
        keep_aliases = set(_json.loads(keep["aliases_json"] or "[]"))
        last_contact = keep["last_contact_utc"]
        for did in drop_ids:
            d = c.execute(
                "SELECT * FROM people WHERE id=? AND chat_id=?",
                (did, chat_id)).fetchone()
            if not d:
                continue
            for ed in _json.loads(d["important_dates_json"] or "[]"):
                if ed not in keep_dates:
                    keep_dates.append(ed); moved_dates += 1
            for a in _json.loads(d["aliases_json"] or "[]"):
                if a not in keep_aliases:
                    keep_aliases.add(a); moved_aliases += 1
            if d["last_contact_utc"] and (not last_contact or d["last_contact_utc"] > last_contact):
                last_contact = d["last_contact_utc"]
            # Rewrite relations
            cur = c.execute(
                "UPDATE relations SET from_id=? WHERE chat_id=? AND from_kind='person' AND from_id=?",
                (keep_id, chat_id, did))
            rewired += cur.rowcount
            cur = c.execute(
                "UPDATE relations SET to_id=? WHERE chat_id=? AND to_kind='person' AND to_id=?",
                (keep_id, chat_id, did))
            rewired += cur.rowcount
            c.execute("DELETE FROM people WHERE id=? AND chat_id=?", (did, chat_id))
        c.execute(
            "UPDATE people SET important_dates_json=?, aliases_json=?, last_contact_utc=? "
            "WHERE id=? AND chat_id=?",
            (_json.dumps(keep_dates, ensure_ascii=False),
             _json.dumps(sorted(keep_aliases), ensure_ascii=False),
             last_contact, keep_id, chat_id))
    return {"moved_dates": moved_dates, "moved_aliases": moved_aliases,
            "rewired_relations": rewired, "dropped": len(drop_ids)}


def find_duplicate_facts(chat_id: int) -> List[Dict]:
    """Return facts with identical (case-folded) key. The most-recent wins
    when consolidated."""
    rows = list_facts(chat_id)
    by_key: Dict[str, List[Dict]] = {}
    for r in rows:
        k = (r["key"] or "").strip().lower()
        if not k:
            continue
        by_key.setdefault(k, []).append(dict(r))
    return [{"key": k, "ids": sorted([r["id"] for r in v])}
             for k, v in by_key.items() if len(v) > 1]


def cleanup_orphan_relations(chat_id: int) -> int:
    """Drop relations whose from_id or to_id no longer exists in the source table."""
    dropped = 0
    with _conn() as c:
        # for each (kind, id) referenced, check existence
        rows = list(c.execute(
            "SELECT id, from_kind, from_id, to_kind, to_id FROM relations WHERE chat_id=?",
            (chat_id,)))
        for r in rows:
            for kind, eid in (("from", r["from_id"]), ("to", r["to_id"])):
                tk = r[f"{kind}_kind"]
                table = {"person": "people", "event": "events", "goal": "goals",
                          "note": "notes", "expense": "expenses",
                          "mission": "missions"}.get(tk)
                if not table:
                    continue
                exists = c.execute(
                    f"SELECT 1 FROM {table} WHERE id=? AND chat_id=?",
                    (eid, chat_id)).fetchone()
                if not exists:
                    c.execute("DELETE FROM relations WHERE id=?", (r["id"],))
                    dropped += 1
                    break
    return dropped


def cleanup_old_errors(days: int = 30) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _conn() as c:
        cur = c.execute("DELETE FROM error_log WHERE ts_utc<?", (cutoff,))
        return cur.rowcount


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

