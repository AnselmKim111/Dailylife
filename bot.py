"""Dailylife — Telegram personal assistant powered by an OpenRouter LLM with tool-use."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import secrets

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import db
import external
import gcal
import gmail as gmail_mod
import browser as browser_mod
import korean_calendar
import lifelog
import lunar
import voice_call
import oauth_server
import routines
import scheduler
import transcribe
import translator
import experts
import crm
import finance
import writing
import trip_planner
import negotiator
import weather as weather_mod
from llm import TOOLS, USER_TZ, chat_completion, parse_tool_calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class _DBErrorHandler(logging.Handler):
    """v5: persist ERROR-level (and above) log records to error_log table for
    queryable observability via /metrics. Best-effort — never raise to caller."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR:
            return
        try:
            import db as _db
            tb = None
            if record.exc_info:
                import traceback as _tb
                tb = "".join(_tb.format_exception(*record.exc_info))
            _db.log_error(
                level=record.levelname,
                source=record.name,
                message=record.getMessage(),
                traceback=tb,
            )
        except Exception:
            pass  # never let logging crash the bot


logging.getLogger().addHandler(_DBErrorHandler())
logger = logging.getLogger("dailylife")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "16"))
TZ = ZoneInfo(USER_TZ)

SYSTEM_PROMPT_TEMPLATE = (
    "You are the user's chief of staff — not a cheerleader, not a coach, not a "
    "tracker. Job #1: ABSORB DECISIONS so the user makes fewer.\n\n"
    "TONE RULES (override anything below if conflict):\n"
    "1. Brevity is respect. Match user message length. 5 단어면 5 단어로 답.\n"
    "2. Decisions: 추천 1개 + 이유 1줄. 옵션 나열 금지. 사용자가 명시적으로 "
    "'대안 알려줘' 했을 때만 1개 alternative + trade-off 1줄.\n"
    "3. Never score the user. 점수·%·streak·'잘하고 있어'·'이번 주 어땠어' 금지. "
    "🔥·💪·✅ emoji는 performance mark로 쓰지 마. 데이터는 조용히 추적, 사용자는 "
    "결과만 받음.\n"
    "4. 사용자 기분·컨디션 질문 금지. 사용자가 먼저 말하지 않으면 묻지 않음.\n"
    "5. 모르면 '모름' — guess 금지. 추측이면 '추측이지만' 접두.\n"
    "6. Persona (위에) = source of truth. 사용자의 평소 패턴을 자연스럽게 회상 "
    "('너 평소 화/목 운동이지', '그거 경서랑 약속한 거 아니야?'). DB가 아니라 "
    "사람처럼 기억해. Persona에 없는 디테일은 만들지 마.\n"
    "7. 한국어 기본. 사용자가 다른 언어로 쓰면 그 언어로.\n\n"
    "All datetimes the user mentions are in {tz} timezone. Convert relative times "
    "('내일 3시', 'in 2 hours', '다음 주 월요일') against current_time below.\n"
    "current_time: {now} ({tz})\n\n"
    "Tool routing:\n"
    "- Stable identity facts (집 주소, 자대 위치, 가족 이름, 선호도) → remember_fact.\n"
    "- Free-form short notes the user wants saved (점심 약속 메모, 책 추천, 선물 후보) → save_note.\n"
    "- 'What did I say about X' / '지난주에 X 얘기 어땠지' style recall → search_memory.\n"
    "- Time-bound appointment with a clock time → add_event AND, if user has connected Google Calendar, also gcal_create_event for cross-device sync.\n"
    "- Daily recurring briefing ('매일 X시에 …') → add_recurring_task.\n"
    "- Live data (영업시간, 길찾기, 운항정보, 시간표) → use web_search / kakao_local_search / "
    "  fetch_url / kakao_directions_drive — don't guess.\n"
    "- Google Calendar specifically (구글 캘린더 / Google Calendar 키워드) → use gcal_* tools.\n"
    "- Gmail / 이메일 / 메일 키워드 (메일 왔어, 항공권 확인 메일, 어제 받은 메일 등) → use gmail_* tools.\n"
    "  If a single message clearly contains a date/time/place (flight, dinner reservation), call gmail_get_message to read the body, then add_event (+ gcal_create_event if linked).\n"
    "- A named person introduced or referenced by the user (약혼녀 경서, 동기 관현, 동료 김철수 등): "
    "if new, call add_person; if known, use the 'Mentioned people in this turn' context block "
    "the system injects, and call recall_person / log_contact_with as needed. "
    "Birthdays/anniversaries → important_dates with recurring_yearly=true. "
    "양가 어른 생신 등 음력 날짜는 is_lunar=true로 저장하면 매년 양력 자동 변환.\n"
    "- Document/image attachments arrive as text starting with '[pdf 첨부 · …]' or "
    "'[image 첨부 · …]' or '[사진 첨부 · …]'. The header includes classified=<kind> hint: "
    "use it as a strong prior. receipt → log_expense, business_card → add_person, "
    "event/poster → add_event (+ gcal_create_event), document_text → save_note. "
    "Always extract the concrete details (date, place, name, amount) and call the right save tool — "
    "don't just acknowledge the upload.\n"
    "- Forwarded messages arrive with '[forwarded from <sender>]' header — treat as "
    "third-party content to summarize/extract/file (save_note + people if a new name appears), "
    "not as the user's own speech. If a long contract/약관/이력서 PDF text is in there and the "
    "user asks to 'read carefully' or 'check risks', call analyze_document.\n"
    "- Spending mention with a price ('스벅 6500원') → log_expense. Habit mention "
    "('운동 1시간', '책 30분') → log_habit. Both have inline-undo if mis-categorized.\n"
    "- Pre-emptive nudges (메일→일정 자동 카드, 출발 알림(leave-by), 골 D-30/14/3/1, 점심 안부 체크인) "
    "are armed by per-chat crons. If the user says '시끄러워' / '꺼' / '알림 줄여' point them to "
    "/nudges for toggles instead of arguing.\n"
    "- Cross-entity recall (X 관련된 거 다, X에 대해 정리) → cross_recall FIRST instead of "
    "firing list_events/search_memory separately.\n"
    "- 모호한 회상 ('그 영화 뭐였더라', '작년 그 카페', '그때 들었던 노래') → lifelog_search "
    "(semantic). cross_recall은 정확한 이름·키워드 매칭만.\n"
    "- Outbound actions on the user's behalf are enabled: gmail_send_email / gmail_reply_to / "
    "gcal_rsvp work. Only fire them when the user explicitly asks ('메일 보내줘', "
    "'회의 수락해줘') OR an auto_rule matched (which the runner enforces, you don't gate). "
    "Otherwise PROPOSE the draft and ask.\n"
    "Known facts about this user:\n{facts_block}"
)

# Per-chat in-memory short-term history (raw tool turns retained).
chat_history: Dict[int, List[Dict]] = {}
# Per-chat last-activity timestamps (UTC) for TTL sweeping idle chats.
_chat_last_active: Dict[int, datetime] = {}
# Idle chats older than this are dropped from in-memory state on next sweep.
CHAT_IDLE_TTL = timedelta(hours=24)
# How long to keep a compacted-summary marker in place.
COMPACT_THRESHOLD = 40
COMPACT_KEEP_RECENT = 20
# token -> (kind, chat_id) — short-lived in-memory map of pending undo opportunities.
_pending_undos: Dict[str, tuple] = {}


# ---------------- helpers ----------------


def _history(chat_id: int) -> List[Dict]:
    _chat_last_active[chat_id] = datetime.now(timezone.utc)
    return chat_history.setdefault(chat_id, [])


def _sweep_idle_chats() -> int:
    """Drop in-memory state for chats idle longer than CHAT_IDLE_TTL.
    Persistent data (events, facts, …) is untouched — only the LLM short-term
    history. Called opportunistically per message."""
    now = datetime.now(timezone.utc)
    stale = [cid for cid, last in _chat_last_active.items() if (now - last) > CHAT_IDLE_TTL]
    for cid in stale:
        chat_history.pop(cid, None)
        _chat_last_active.pop(cid, None)
    if stale:
        logger.info("swept %d idle chat histories", len(stale))
    return len(stale)


async def _compact_history_if_needed(chat_id: int) -> None:
    """When in-memory history grows past COMPACT_THRESHOLD turns, ask the model
    to summarize everything except the last COMPACT_KEEP_RECENT turns into a
    single 'compacted:' system note. Saves cost + improves recall."""
    h = chat_history.get(chat_id, [])
    if len(h) <= COMPACT_THRESHOLD:
        return
    old, keep = h[:-COMPACT_KEEP_RECENT], h[-COMPACT_KEEP_RECENT:]
    # Skip if the first kept message is already a compaction marker (avoid recursive growth)
    if old and isinstance(old[0].get("content"), str) and old[0].get("content", "").startswith("compacted:"):
        # Re-compact: merge previous compaction text + everything that came after into a new summary.
        pass
    # Build a tiny conversation excerpt for the summarizer
    excerpt_lines: List[str] = []
    for m in old[-60:]:  # cap at 60 messages of input
        role = m.get("role", "")
        if role == "tool":
            excerpt_lines.append(f"[tool {m.get('name')} result]")
            continue
        text = (m.get("content") or "")[:400]
        if text:
            excerpt_lines.append(f"{role}: {text}")
    if not excerpt_lines:
        return
    prompt = (
        "Summarize the following Dailylife bot conversation into one compact "
        "Korean paragraph (≤400 chars). Preserve: any commitments, named "
        "people/places, decisions, ongoing topics. Drop pleasantries.\n\n"
        + "\n".join(excerpt_lines)
    )
    try:
        data = await chat_completion(
            [{"role": "system", "content": "You compress long conversations into compact memory notes."},
             {"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="compaction",
        )
        summary = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("history compaction failed; leaving history intact")
        return
    if not summary:
        return
    marker = {"role": "system", "content": f"compacted: {summary[:600]}"}
    chat_history[chat_id] = [marker, *keep]
    logger.info("compacted history for chat %s: %d → %d turns",
                chat_id, len(h), len(chat_history[chat_id]))


def _trim_history(chat_id: int) -> None:
    """Hard cap as a final safety net even if compaction didn't fire."""
    h = chat_history.get(chat_id, [])
    if len(h) > HISTORY_LIMIT * 2:
        chat_history[chat_id] = h[-HISTORY_LIMIT * 2 :]


def _facts_block(chat_id: int) -> str:
    rows = db.list_facts(chat_id)
    if not rows:
        return "  (none yet)"
    return "\n".join(f"  - {r['key']}: {r['value']}" for r in rows)


def _solar_date_for(date_entry: Dict, this_year: int) -> Optional[Tuple[int, int, int]]:
    """Resolve an important_date entry to (Y,M,D) in the solar calendar for
    a given solar `this_year`. Honors `is_lunar=True` → convert via lunar.py."""
    raw = (date_entry.get("date_local") or "").strip()
    if not raw:
        return None
    try:
        y, m, d = (int(x) for x in raw.split("-"))
    except Exception:
        return None
    if date_entry.get("is_lunar"):
        sd = lunar.lunar_to_solar_recurring(m, d, this_year)
        if sd is None:
            return None
        return (sd.year, sd.month, sd.day)
    return (y, m, d)


def _date_matches_today(date_entry: Dict, today: "datetime.date") -> bool:
    """True if the entry's recurring (or one-shot) date lands on `today`."""
    resolved = _solar_date_for(date_entry, today.year)
    if not resolved:
        return False
    _, m, d = resolved
    if date_entry.get("recurring_yearly") or date_entry.get("is_lunar"):
        return (today.month, today.day) == (m, d)
    # one-shot: full date must match
    return (resolved[0], m, d) == (today.year, today.month, today.day)


def _days_until_recurring(date_entry: Dict, today: "datetime.date") -> Optional[int]:
    """For a recurring/lunar entry, days until the next occurrence (≥0).
    For a one-shot date, days until that exact date (can be negative)."""
    import datetime as _dt
    resolved = _solar_date_for(date_entry, today.year)
    if not resolved:
        return None
    sd = _dt.date(*resolved)
    if date_entry.get("recurring_yearly") or date_entry.get("is_lunar"):
        if sd < today:
            # roll to next year
            next_year = today.year + 1
            r2 = _solar_date_for(date_entry, next_year)
            if not r2:
                return None
            sd = _dt.date(*r2)
        return (sd - today).days
    return (sd - today).days


def _facts_block_deprecated_placeholder(_):  # safety: nothing
    pass


def _people_context_for_text(chat_id: int, text: str) -> str:
    """Build a one-block summary of every known person mentioned in `text`.
    Empty string if nothing matched. Also marks contact (last_contact_utc)."""
    hits = db.find_people_in_text(chat_id, text)
    if not hits:
        return ""
    now = datetime.now(timezone.utc)
    lines = []
    for p in hits:
        bits = [p["name"]]
        if p["role"]:
            bits.append(p["role"])
        if p["last_contact_utc"]:
            try:
                lc = datetime.fromisoformat(p["last_contact_utc"])
                days = (now - lc).days
                bits.append(f"last contact {days}d ago")
            except Exception:
                pass
        dates = json.loads(p["important_dates_json"] or "[]")
        today = datetime.now(TZ).date()
        for d in dates:
            try:
                days_left = _days_until_recurring(d, today)
                if days_left is None:
                    continue
                bits.append(f"{d['label']} D-{days_left}")
            except Exception:
                pass
        lines.append("  - " + " · ".join(bits))
        # mark contact
        try:
            db.mark_contact(p["id"])
        except Exception:
            pass
    return "Mentioned people in this turn:\n" + "\n".join(lines)


def _system_message(chat_id: int, recent_user_text: str = "") -> Dict:
    now_local = datetime.now(TZ).strftime("%Y-%m-%d %H:%M (%a)")
    base = SYSTEM_PROMPT_TEMPLATE.format(
        tz=USER_TZ, now=now_local, facts_block=_facts_block(chat_id)
    )
    # v4: inject the latest persona doc as the most up-to-date model of the user.
    persona = db.get_latest_persona(chat_id)
    if persona:
        base = (
            f"About this user (persona v{persona['version']}, refreshed "
            f"{persona['generated_at'][:10]}):\n{persona['content_md']}\n\n"
            + base
        )
    # v5: self-improving overrides (tone, no-fly topics, etc.) — bot writes
    # these to itself based on engagement signal.
    override = db.get_prompt_override(chat_id)
    if override:
        base = (
            f"Bot self-tuning overrides (from observed engagement, "
            "treat as supplementary not contradictory):\n"
            f"{override}\n\n"
            + base
        )
    # v9: travel mode — when user is currently on a trip, prefix the prompt so
    # weather/leave-by/persona answers anchor to the destination, not home.
    today_iso = datetime.now(TZ).date().isoformat()
    travel = db.active_travel_for(chat_id, today_iso)
    if travel:
        base = (
            f"🛫 TRAVEL MODE — User is currently in {travel['destination']} "
            f"({travel['start_date_local']} → {travel['end_date_local']}). "
            "Anchor weather/transport/leave-by to the destination unless asked "
            "otherwise. Tone slightly more 'travelling' (less routine nudges).\n\n"
            + base
        )
    if recent_user_text:
        people_ctx = _people_context_for_text(chat_id, recent_user_text)
        if people_ctx:
            base = base + "\n\n" + people_ctx
    # v12 W7: active expert session — 도메인 persona 시스템 프롬프트 최상단 주입
    expert = db.active_expert_session(chat_id)
    if expert:
        base = (
            f"🎓 EXPERT MODE [{expert['domain']}] — session #{expert['id']}.\n"
            f"{expert['persona_md']}\n\n"
            + base
        )
    # v12 W4: 활성 writing session 컨텍스트
    writing = db.current_writing(chat_id)
    if writing:
        base = (
            f"✍️ WRITING MODE — session #{writing['id']}, "
            f"section {writing['current_section']}/{(writing['outline_md'] or '').count(chr(10))+1}. "
            f"Purpose: {writing['purpose']}. Audience: {writing['audience'] or '미지정'}.\n"
            "사용자 톤 학습 후 multi-pass 협업. 사용자가 redline 주면 그대로 반영, "
            "다음 섹션 자동 진행 X — 사용자 OK 기다림.\n\n"
            + base
        )
    return {"role": "system", "content": base}


def _parse_local_iso(s: str) -> datetime:
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.fromisoformat(s.rstrip("Z"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.astimezone(timezone.utc)


def _format_event_row(row) -> str:
    when_local = datetime.fromisoformat(row["when_utc"]).astimezone(TZ)
    base = f"#{row['id']} · {when_local.strftime('%Y-%m-%d %H:%M')} · {row['title']}"
    if row["notes"]:
        base += f"\n     ↳ {row['notes']}"
    return base


# ---------------- merged schedule view (local + Google Calendar) ----------------


async def _merge_schedule(
    chat_id: int,
    from_utc: datetime,
    to_utc: Optional[datetime] = None,
) -> List[Dict]:
    """Return a unified, time-sorted list of events from the local DB and Google
    Calendar (if connected). Items with the same `gcal_event_id` are collapsed.
    Items within ±10 min and >0.8 title similarity are deduped opportunistically."""
    import difflib

    local_rows = db.list_events(chat_id, from_utc, to_utc)
    local_items: List[Dict] = []
    for r in local_rows:
        local_items.append({
            "source": "local",
            "id": r["id"],
            "gcal_id": r["gcal_event_id"],
            "title": r["title"],
            "when_utc": datetime.fromisoformat(r["when_utc"]),
            "notes": r["notes"],
        })

    gcal_items: List[Dict] = []
    try:
        if db.get_oauth_token(chat_id, "google"):
            ge = await gcal.list_events(
                chat_id,
                time_min_iso=from_utc.astimezone(TZ).isoformat(),
                time_max_iso=to_utc.astimezone(TZ).isoformat() if to_utc else None,
                max_results=100,
            )
            for e in ge:
                start = e.get("start")
                if not start:
                    continue
                try:
                    # GCal returns RFC3339 with offset; normalize
                    dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                except ValueError:
                    continue
                gcal_items.append({
                    "source": "gcal",
                    "id": None,
                    "gcal_id": e.get("id"),
                    "title": e.get("summary") or "(제목 없음)",
                    "when_utc": dt.astimezone(timezone.utc),
                    "notes": e.get("description"),
                    "location": e.get("location"),
                })
    except Exception:
        logger.exception("gcal fetch in _merge_schedule failed; showing local only")

    # First pass — collapse by gcal_id linkage
    by_gid: Dict[str, Dict] = {}
    others: List[Dict] = []
    for item in [*local_items, *gcal_items]:
        gid = item.get("gcal_id")
        if gid and gid in by_gid:
            by_gid[gid]["source"] = "both"
            # Prefer local id for action buttons; keep gcal extras
            existing = by_gid[gid]
            if existing.get("id") is None and item.get("id") is not None:
                existing["id"] = item["id"]
            existing.setdefault("location", item.get("location"))
            continue
        if gid:
            by_gid[gid] = dict(item)
        else:
            others.append(item)

    # Second pass — fuzzy dedup of "others" against by_gid (same time window, similar title)
    final = list(by_gid.values()) + others
    final.sort(key=lambda x: x["when_utc"])
    deduped: List[Dict] = []
    for item in final:
        merged = False
        for existing in deduped:
            if abs((item["when_utc"] - existing["when_utc"]).total_seconds()) <= 600:
                ratio = difflib.SequenceMatcher(
                    None, item["title"], existing["title"]
                ).ratio()
                if ratio >= 0.8:
                    existing["source"] = "both"
                    if existing.get("id") is None and item.get("id") is not None:
                        existing["id"] = item["id"]
                    if existing.get("gcal_id") is None and item.get("gcal_id") is not None:
                        existing["gcal_id"] = item["gcal_id"]
                    merged = True
                    break
        if not merged:
            deduped.append(item)
    return deduped


_SOURCE_ICONS = {"local": "📍", "gcal": "🟦", "both": "✅"}


def _format_merged_event(item: Dict) -> str:
    when_local = item["when_utc"].astimezone(TZ)
    icon = _SOURCE_ICONS.get(item["source"], "•")
    head = f"{icon} {when_local.strftime('%m-%d %H:%M')} · {item['title']}"
    if item.get("id") is not None:
        head += f"  (#{item['id']})"
    if item.get("notes"):
        head += f"\n     ↳ {item['notes'][:80]}"
    return head


# ---------------- sync schedule tool handlers ----------------


def tool_add_event(chat_id: int, args: Dict) -> Dict:
    title = (args.get("title") or "").strip()
    when_iso = args.get("when_iso")
    if not title or not when_iso:
        return {"ok": False, "error": "title and when_iso are required"}
    try:
        when_utc = _parse_local_iso(when_iso)
    except ValueError as e:
        return {"ok": False, "error": f"invalid when_iso: {e}"}
    notes = args.get("notes")
    location = (args.get("location") or "").strip() or None
    lead = args.get("remind_lead_minutes")
    lead = 30 if lead is None else int(lead)
    lead = lead if lead > 0 else None
    # Conflict pre-check against local DB (and best-effort GCal via merge view later).
    conflicts = _check_local_conflicts(chat_id, when_utc, lookahead_minutes=120)
    eid = db.add_event(chat_id, title, when_utc, notes, lead, location=location)
    row = db.get_event(eid)
    armed = scheduler.schedule_for(row) if row else False
    # Best-effort leave-by scheduling — fire-and-forget background task.
    leave_by_queued = False
    if row and location and not _toggle_off_local(chat_id, "leave_by_enabled"):
        try:
            asyncio.create_task(_try_schedule_leave_by(chat_id, row))
            leave_by_queued = True
        except Exception:
            logger.exception("leave-by enqueue failed for event %s", eid)
    # v9: travel-mode auto-detect on new event.
    if row and not _toggle_off_local(chat_id, "travel_autodetect_enabled"):
        try:
            asyncio.create_task(_maybe_detect_travel(chat_id, row))
        except Exception:
            logger.exception("travel autodetect enqueue failed")
    return {
        "ok": True,
        "event_id": eid,
        "when_local": when_utc.astimezone(TZ).isoformat(),
        "remind_lead_minutes": lead,
        "reminder_armed": armed,
        "location": location,
        "leave_by_queued": leave_by_queued,
        "conflicts": conflicts,
    }


def _check_local_conflicts(chat_id: int, when_utc: datetime, lookahead_minutes: int = 120) -> List[Dict]:
    """Look up local events within ±lookahead_minutes window. Returns a small
    list the LLM can mention to the user before adding (or to dedupe)."""
    lo = when_utc - timedelta(minutes=lookahead_minutes)
    hi = when_utc + timedelta(minutes=lookahead_minutes)
    rows = db.list_events(chat_id, lo, hi)
    out = []
    for r in rows[:5]:
        out.append({
            "id": r["id"],
            "title": r["title"],
            "when_local": datetime.fromisoformat(r["when_utc"]).astimezone(TZ).isoformat(),
        })
    return out


def tool_list_events(chat_id: int, args: Dict) -> Dict:
    from_iso = args.get("from_iso")
    to_iso = args.get("to_iso")
    from_utc = _parse_local_iso(from_iso) if from_iso else datetime.now(timezone.utc)
    to_utc = _parse_local_iso(to_iso) if to_iso else None
    rows = db.list_events(chat_id, from_utc, to_utc)
    return {
        "ok": True,
        "events": [
            {
                "id": r["id"],
                "title": r["title"],
                "when_local": datetime.fromisoformat(r["when_utc"]).astimezone(TZ).isoformat(),
                "notes": r["notes"],
                "remind_lead_minutes": r["remind_lead_minutes"],
            }
            for r in rows
        ],
    }


def tool_update_event(chat_id: int, args: Dict) -> Dict:
    eid = args.get("event_id")
    if not eid:
        return {"ok": False, "error": "event_id required"}
    row = db.get_event(int(eid))
    if row is None or row["chat_id"] != chat_id:
        return {"ok": False, "error": "event not found"}
    when_utc = _parse_local_iso(args["when_iso"]) if args.get("when_iso") else None
    ok = db.update_event(
        int(eid),
        title=args.get("title"),
        when_utc=when_utc,
        notes=args.get("notes"),
        remind_lead_minutes=args.get("remind_lead_minutes"),
    )
    if ok:
        new = db.get_event(int(eid))
        scheduler.schedule_for(new)
    return {"ok": ok, "event_id": int(eid)}


def tool_delete_event(chat_id: int, args: Dict) -> Dict:
    eid = args.get("event_id")
    if not eid:
        return {"ok": False, "error": "event_id required"}
    row = db.get_event(int(eid))
    if not row or row["chat_id"] != chat_id:
        return {"ok": False, "error": "event not found"}
    payload = dict(row)
    ok = db.delete_event(int(eid), chat_id)
    if ok:
        token = secrets.token_urlsafe(8)
        db.push_deleted_audit(chat_id, "event", payload, token)
        _pending_undos[token] = ("event", chat_id)
    return {"ok": ok, "event_id": int(eid),
            "undo_token": token if ok else None,
            "undo_label": f"이벤트 #{eid} '{payload.get('title')}' 삭제 — 취소"}


# ---------------- facts ----------------


def tool_remember_fact(chat_id: int, args: Dict) -> Dict:
    key = (args.get("key") or "").strip()
    value = (args.get("value") or "").strip()
    if not key or not value:
        return {"ok": False, "error": "key and value required"}
    db.remember_fact(chat_id, key, value)
    return {"ok": True, "key": key, "value": value}


def tool_forget_fact(chat_id: int, args: Dict) -> Dict:
    key = (args.get("key") or "").strip()
    if not key:
        return {"ok": False, "error": "key required"}
    # Capture pre-delete value for undo.
    rows = [r for r in db.list_facts(chat_id) if r["key"] == key]
    payload = dict(rows[0]) if rows else None
    ok = db.forget_fact(chat_id, key)
    token = None
    if ok and payload:
        token = secrets.token_urlsafe(8)
        db.push_deleted_audit(chat_id, "fact", payload, token)
        _pending_undos[token] = ("fact", chat_id)
    return {"ok": ok, "key": key, "undo_token": token,
            "undo_label": f"기억 '{key}' 삭제 — 취소"}


# ---------------- recurring tasks ----------------


def tool_add_recurring_task(chat_id: int, args: Dict) -> Dict:
    cron = (args.get("cron_kst") or "").strip()
    prompt = (args.get("prompt") or "").strip()
    if not cron or not prompt:
        return {"ok": False, "error": "cron_kst and prompt required"}
    try:
        hh, mm = cron.split(":")
        int(hh), int(mm)
    except (ValueError, AttributeError):
        return {"ok": False, "error": "cron_kst must be HH:MM"}
    tid = db.add_recurring_task(chat_id, cron, prompt)
    row = db.get_recurring_task(tid)
    armed = scheduler.schedule_recurring(row) if row else False
    return {"ok": True, "task_id": tid, "cron_kst": cron, "armed": armed}


def tool_list_recurring_tasks(chat_id: int, args: Dict) -> Dict:
    rows = db.list_recurring_tasks(chat_id)
    return {
        "ok": True,
        "tasks": [
            {
                "id": r["id"],
                "cron_kst": r["cron_kst"],
                "prompt": r["prompt"],
                "enabled": bool(r["enabled"]),
                "last_run_utc": r["last_run_utc"],
            }
            for r in rows
        ],
    }


def tool_delete_recurring_task(chat_id: int, args: Dict) -> Dict:
    tid = args.get("task_id")
    if not tid:
        return {"ok": False, "error": "task_id required"}
    row = db.get_recurring_task(int(tid))
    if not row or row["chat_id"] != chat_id:
        return {"ok": False, "error": "task not found"}
    payload = dict(row)
    ok = db.delete_recurring_task(int(tid), chat_id)
    token = None
    if ok:
        scheduler.cancel_recurring(int(tid))
        token = secrets.token_urlsafe(8)
        db.push_deleted_audit(chat_id, "recurring", payload, token)
        _pending_undos[token] = ("recurring", chat_id)
    return {"ok": ok, "task_id": int(tid), "undo_token": token,
            "undo_label": f"정기작업 #{tid} 삭제 — 취소"}


def tool_enable_recurring_task(chat_id: int, args: Dict) -> Dict:
    tid = args.get("task_id")
    if not tid:
        return {"ok": False, "error": "task_id required"}
    ok = db.set_recurring_enabled(int(tid), chat_id, True)
    if ok:
        row = db.get_recurring_task(int(tid))
        if row:
            scheduler.schedule_recurring(row)
    return {"ok": ok, "task_id": int(tid)}


def tool_disable_recurring_task(chat_id: int, args: Dict) -> Dict:
    tid = args.get("task_id")
    if not tid:
        return {"ok": False, "error": "task_id required"}
    ok = db.set_recurring_enabled(int(tid), chat_id, False)
    if ok:
        scheduler.cancel_recurring(int(tid))
    return {"ok": ok, "task_id": int(tid)}


# ---------------- goals (long-horizon, proactive) ----------------


def tool_add_goal(chat_id: int, args: Dict) -> Dict:
    title = (args.get("title") or "").strip()
    if not title:
        return {"ok": False, "error": "title required"}
    target = args.get("target_date_local")
    sub_tasks = args.get("sub_tasks") or []
    gid = db.add_goal(
        chat_id,
        title,
        why=args.get("why"),
        target_date_local=target,
        horizon=args.get("horizon", "long"),
        sub_tasks=sub_tasks,
        watch_query=args.get("watch_query"),
    )
    scheduler.ensure_proactive_for(chat_id)
    scheduler.ensure_daily_rhythm_for(chat_id)
    # v4: long-horizon goals with no sub_tasks → auto-decompose in background.
    auto_queued = False
    if (target and not sub_tasks
            and not _toggle_off_local(chat_id, "auto_decompose_enabled")):
        try:
            target_date = datetime.fromisoformat(target).date()
            days_ahead = (target_date - datetime.now(TZ).date()).days
            if days_ahead > 28:
                asyncio.create_task(_try_auto_decompose_goal(chat_id, gid, title,
                                                              args.get("why"), days_ahead))
                auto_queued = True
        except Exception:
            logger.exception("auto_decompose enqueue failed for goal %s", gid)
    return {"ok": True, "goal_id": gid, "auto_decompose_queued": auto_queued}


async def _try_auto_decompose_goal(
    chat_id: int, goal_id: int, title: str,
    why: Optional[str], days_ahead: int,
) -> None:
    """Background micro-LLM: propose 4-6 weekly micro-actions, append as sub_tasks."""
    prompt = (
        "사용자의 장기 골을 주간 단위 micro-action 4-6개로 분해해. "
        "각각 30자 이내. JSON 배열만 출력 (예: [\"운동 주 3회\", \"항공권 알리미 등록\"]).\n\n"
        f"골 제목: {title}\n"
        f"이유: {why or '(없음)'}\n"
        f"target: D-{days_ahead}\n"
    )
    try:
        data = await chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="auto_decompose", max_tokens=300,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("auto_decompose LLM call failed")
        return
    import re as _re
    blob = _re.sub(r"^\s*```(?:json)?\s*", "", content, flags=_re.IGNORECASE)
    blob = _re.sub(r"\s*```\s*$", "", blob)
    m = _re.search(r"\[[\s\S]*\]", blob)
    if not m:
        return
    try:
        tasks = json.loads(m.group(0))
    except Exception:
        return
    if not isinstance(tasks, list):
        return
    cleaned = [str(t).strip()[:60] for t in tasks if str(t).strip()]
    cleaned = cleaned[:6]
    if not cleaned:
        return
    for text in cleaned:
        db.add_goal_subtask(goal_id, chat_id, text)
    if _app and _app.bot:
        try:
            await _app.bot.send_message(
                chat_id=chat_id,
                text=f"🎯 골 #{goal_id} 자동 분해 — {len(cleaned)}개 sub_task 추가\n  • "
                     + "\n  • ".join(cleaned)
                     + "\n\n수정·삭제는 /goals 에서 가능. 자동 분해 끄려면 `/nudges off decompose`.",
            )
            db.log_agent_action(
                chat_id, "auto_decompose",
                summary=f"골 #{goal_id} 분해 {len(cleaned)}개",
                payload={"goal_id": goal_id, "sub_tasks": cleaned},
            )
        except Exception:
            logger.exception("auto_decompose notify failed")


def tool_list_goals(chat_id: int, args: Dict) -> Dict:
    status = args.get("status") or "open"
    if status == "all":
        rows = db.list_goals(chat_id, status=None)
    else:
        rows = db.list_goals(chat_id, status=status)
    return {
        "ok": True,
        "goals": [
            {
                "id": r["id"],
                "title": r["title"],
                "why": r["why"],
                "target_date_local": r["target_date_local"],
                "horizon": r["horizon"],
                "status": r["status"],
                "sub_tasks": _parse_sub_tasks(r["sub_tasks_json"]),
                "watch_query": r["watch_query"],
                "last_reviewed_utc": r["last_reviewed_utc"],
            }
            for r in rows
        ],
    }


def tool_update_goal(chat_id: int, args: Dict) -> Dict:
    gid = args.get("goal_id")
    if not gid:
        return {"ok": False, "error": "goal_id required"}
    ok = db.update_goal(
        int(gid),
        chat_id,
        title=args.get("title"),
        why=args.get("why"),
        target_date_local=args.get("target_date_local"),
        horizon=args.get("horizon"),
        status=args.get("status"),
        watch_query=args.get("watch_query"),
    )
    return {"ok": ok, "goal_id": int(gid)}


def tool_complete_goal(chat_id: int, args: Dict) -> Dict:
    gid = args.get("goal_id")
    if not gid:
        return {"ok": False, "error": "goal_id required"}
    return {"ok": db.update_goal(int(gid), chat_id, status="done"), "goal_id": int(gid)}


def tool_add_goal_subtask(chat_id: int, args: Dict) -> Dict:
    gid = args.get("goal_id")
    text = (args.get("text") or "").strip()
    if not gid or not text:
        return {"ok": False, "error": "goal_id + text required"}
    return {"ok": db.add_goal_subtask(int(gid), chat_id, text), "goal_id": int(gid)}


def tool_complete_goal_subtask(chat_id: int, args: Dict) -> Dict:
    gid = args.get("goal_id")
    idx = args.get("sub_index")
    if not gid or idx is None:
        return {"ok": False, "error": "goal_id + sub_index required"}
    return {
        "ok": db.complete_goal_subtask(int(gid), chat_id, int(idx)),
        "goal_id": int(gid),
        "sub_index": int(idx),
    }


def _parse_sub_tasks(raw: Optional[str]) -> list:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


# ---------------- expenses + habits ----------------


def tool_log_expense(chat_id: int, args: Dict) -> Dict:
    amt = args.get("amount_won")
    if amt is None:
        return {"ok": False, "error": "amount_won required"}
    eid = db.log_expense(
        chat_id,
        amount_won=int(amt),
        category=args.get("category"),
        merchant=args.get("merchant"),
        when_local=args.get("when_local"),
        notes=args.get("notes"),
    )
    return {"ok": True, "expense_id": eid}


def tool_summarize_expenses(chat_id: int, args: Dict) -> Dict:
    days = int(args.get("days", 30))
    return {"ok": True, **db.summarize_expenses(chat_id, days=days)}


def tool_log_habit(chat_id: int, args: Dict) -> Dict:
    key = (args.get("habit_key") or "").strip()
    if not key:
        return {"ok": False, "error": "habit_key required"}
    hid = db.log_habit(
        chat_id, habit_key=key,
        duration_min=int(args["duration_min"]) if args.get("duration_min") is not None else None,
        notes=args.get("notes"),
    )
    # Bump streak now (don't wait for nightly cron) so the LLM can comment.
    today_local = datetime.now(TZ).date()
    today_iso = today_local.isoformat()
    yesterday_iso = (today_local - timedelta(days=1)).isoformat()
    prev = db.get_habit_streak(chat_id, key)
    new_streak = 1
    if prev and prev["last_log_date"] == today_iso:
        new_streak = prev["current_streak"]  # second log same day — no change
    elif prev and prev["last_log_date"] == yesterday_iso:
        new_streak = prev["current_streak"] + 1
    db.upsert_habit_streak(chat_id, key, new_streak, today_iso)
    streak_row = db.get_habit_streak(chat_id, key)
    return {
        "ok": True, "habit_id": hid,
        "current_streak": streak_row["current_streak"] if streak_row else new_streak,
        "best_streak": streak_row["best_streak"] if streak_row else new_streak,
    }


def tool_summarize_habits(chat_id: int, args: Dict) -> Dict:
    days = int(args.get("days", 7))
    return {"ok": True, **db.summarize_habits(chat_id, days=days)}


# ---------------- Korean-life helpers + wedding timeline ----------------


def tool_find_duplicates(chat_id: int, args: Dict) -> Dict:
    return {
        "ok": True,
        "duplicate_people": db.find_duplicate_people(chat_id),
        "duplicate_facts": db.find_duplicate_facts(chat_id),
    }


def tool_merge_people(chat_id: int, args: Dict) -> Dict:
    keep = int(args.get("keep_id") or 0)
    drops = [int(x) for x in (args.get("drop_ids") or [])]
    if not keep or not drops:
        return {"ok": False, "error": "keep_id + non-empty drop_ids required"}
    res = db.merge_people(chat_id, keep, drops)
    return {"ok": "error" not in res, **res}


# (SYNC_HANDLERS.update for find_duplicates/merge_people lives later, after the
# SYNC_HANDLERS dict is defined.)


# ---------------- v9: inbox zero — daily Gmail triage ----------------


INBOX_TRIAGE_PROMPT = (
    "다음 Gmail 메시지 한 통을 4 카테고리로 분류해 1줄 JSON: "
    '{{"category": "important"|"draft_reply"|"archive"|"noise", '
    '"summary": "<≤80자 한국어 요약>"}}. '
    "important: 사용자가 봐야 할 메일 (직장·계약·중요한 사람). "
    "draft_reply: 답장이 필요한 메일 — 짧은 답장 초안. "
    "archive: 알림·뉴스레터·확인 (자동 archive). "
    "noise: 광고·스팸 (자동 archive).\n\n"
    "보낸 사람: {sender}\n제목: {subject}\n본문:\n{body}"
)


async def run_inbox_triage(chat_id: int) -> None:
    """Daily 06:30 — fetch all unread in last 24h, classify each, archive
    auto categories, send digest of important + draft suggestions."""
    if _toggle_off_local(chat_id, "inbox_triage_enabled"):
        return
    if not db.get_oauth_token(chat_id, "google"):
        return
    try:
        ids = await gmail_mod.list_messages(
            chat_id, query="is:unread newer_than:1d", max_results=30)
    except Exception:
        logger.exception("inbox_triage list failed")
        return
    if not ids:
        return
    important: List[Dict] = []
    drafts: List[Dict] = []
    archived = 0
    noise = 0
    for mid in ids[:25]:  # cap LLM calls
        try:
            msg = await gmail_mod.get_message(chat_id, mid, body_max_chars=2000)
        except Exception:
            continue
        body = (msg.get("body") or msg.get("snippet") or "")[:1500]
        sender = msg.get("from") or ""
        subject = msg.get("subject") or ""
        prompt = INBOX_TRIAGE_PROMPT.format(
            sender=sender, subject=subject, body=body)
        try:
            data = await chat_completion(
                [{"role": "user", "content": prompt}],
                tools=None, chat_id=chat_id, kind="classify_content",
                max_tokens=160,
            )
            content = (data["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            logger.exception("triage LLM failed for %s", mid)
            continue
        import re as _re
        m = _re.search(r"\{[\s\S]*\}", content)
        if not m:
            continue
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            continue
        cat = parsed.get("category", "")
        summary = (parsed.get("summary") or "")[:120]
        entry = {"id": mid, "subject": subject, "from": sender,
                  "summary": summary}
        if cat == "important":
            important.append(entry)
        elif cat == "draft_reply":
            drafts.append(entry)
        elif cat in ("archive", "noise"):
            try:
                await gmail_mod.archive(chat_id, mid)
                if cat == "noise":
                    noise += 1
                else:
                    archived += 1
            except Exception:
                logger.debug("archive failed for %s", mid)
    if _app is None or _app.bot is None:
        return
    lines = ["📬 Inbox Zero"]
    if important:
        lines.append(f"⭐ 중요 {len(important)}건 — 봐줘:")
        for e in important[:6]:
            lines.append(f"  • {e['subject'][:60]}")
            lines.append(f"      ↳ {e['summary']}")
    if drafts:
        lines.append(f"✍️ 답장 후보 {len(drafts)}건:")
        for e in drafts[:5]:
            lines.append(f"  • {e['subject'][:60]} ({e['from'][:30]})")
    lines.append("")
    lines.append(f"🗃 자동 아카이브: {archived}건 · 🗑 스팸: {noise}건")
    if archived + noise > 0:
        db.log_agent_action(
            chat_id, "inbox_triage",
            summary=f"Inbox triage: arch={archived} noise={noise} imp={len(important)} drft={len(drafts)}",
            payload={"archived": archived, "noise": noise,
                     "important_count": len(important), "drafts_count": len(drafts)},
        )
    try:
        await _app.bot.send_message(chat_id=chat_id, text="\n".join(lines))
        db.record_nudge(chat_id, "inbox_triage")
    except Exception:
        logger.exception("inbox_triage send failed")


# ---------------- v9: habit experiments ----------------


# v11: lifelog 매일 04:30 — 어제 entity embed


async def run_lifelog_index(chat_id: int) -> None:
    if _toggle_off_local(chat_id, "lifelog_indexing_enabled"):
        return
    try:
        counts = await lifelog.index_yesterday(chat_id)
        total = sum(counts.values())
        if total:
            logger.info("lifelog chat=%s indexed=%s", chat_id, counts)
    except Exception:
        logger.exception("lifelog index runner failed chat=%s", chat_id)


# v12 W5: 매주 일요일 09:00 — 모든 사람 relationship_pulse 계산
async def run_relationship_pulse(chat_id: int) -> None:
    if not _toggle_on_local(chat_id, "crm_pulse_enabled"):
        return
    try:
        out = crm.compute_pulse_for_week(chat_id)
        logger.info("relationship_pulse chat=%s people=%d", chat_id, len(out))
    except Exception:
        logger.exception("relationship_pulse failed chat=%s", chat_id)


# v12 W2: 매일 06:00 — 신규 구독 후보 + 가격 변동 감지 (briefing 통합용)
async def run_finance_scan(chat_id: int) -> None:
    if not _toggle_on_local(chat_id, "finance_scan_enabled"):
        return
    try:
        cands = await finance.detect_new_subscriptions(chat_id)
        if cands and _app and _app.bot:
            lines = ["💳 새 구독 후보"]
            for c in cands:
                lines.append(f"  • {c['merchant']}: ₩{c['amount_won']:,} ({c['occurrences']}회)")
            lines.append("등록: 봇한테 '<이름> 구독 등록' / 무시: skip")
            await _app.bot.send_message(chat_id=chat_id, text="\n".join(lines))
    except Exception:
        logger.exception("finance scan failed chat=%s", chat_id)


# v11: watch_tasks pump (10분) — 별도 wave 3에서 검사 로직 구현
async def run_watch_pump() -> None:
    """매 10분 — active watch 중 due 항목 1-3개 체크."""
    rows = db.watches_due()
    if not rows:
        return
    for w in rows[:3]:
        try:
            await _check_watch(w)
        except Exception:
            logger.exception("watch check failed id=%s", w["id"])


WATCH_CHECK_PROMPT = (
    "사용자가 등록한 watch task의 조건이 충족됐는지 판단.\n"
    "JSON 한 줄: {{\"matched\": true|false, \"evidence\": \"<≤120자 한국어 근거>\", \"new_state\": \"<≤200자 현재 상태 요약>\"}}\n\n"
    "조건: {condition}\n"
    "직전 상태: {last_state}\n"
    "이번 체크 결과:\n{check_result}"
)


async def _check_watch(watch_row) -> None:
    """단일 watch — check_kind에 따라 도구 실행 후 LLM이 조건 매칭 판단."""
    chat_id = watch_row["chat_id"]
    try:
        args = json.loads(watch_row["check_args_json"] or "{}")
    except Exception:
        args = {}
    # 1) Run the check tool
    check_kind = watch_row["check_kind"]
    result_text = ""
    try:
        if check_kind == "fetch_url":
            r = await tool_fetch_url(chat_id, args)
            result_text = json.dumps(r, ensure_ascii=False)[:3000]
        elif check_kind == "web_search":
            r = await tool_web_search(chat_id, args)
            result_text = json.dumps(r, ensure_ascii=False)[:3000]
        elif check_kind == "gmail_query":
            r = await tool_gmail_search(chat_id, args)
            result_text = json.dumps(r, ensure_ascii=False)[:3000]
        elif check_kind == "web_scrape":
            # Wave 2에서 추가; 없으면 fetch_url로 폴백
            try:
                import browser
                r = await browser.scrape_url(args.get("url", ""),
                                              wait_for_selector=args.get("wait_for_selector"))
                result_text = (r or "")[:3000]
            except Exception:
                r = await tool_fetch_url(chat_id, {"url": args.get("url", "")})
                result_text = json.dumps(r, ensure_ascii=False)[:3000]
        else:
            result_text = f"(unsupported check_kind={check_kind})"
    except Exception as e:
        logger.exception("watch %s check %s failed", watch_row["id"], check_kind)
        result_text = f"(check error: {e})"
    # 2) LLM judges match
    prompt = WATCH_CHECK_PROMPT.format(
        condition=watch_row["condition_md"],
        last_state=(watch_row["last_state_json"] or "(first check)")[:600],
        check_result=result_text,
    )
    try:
        data = await chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="classify_content", max_tokens=240,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("watch judge LLM failed id=%s", watch_row["id"])
        return
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if not m:
        return
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return
    new_state = (parsed.get("new_state") or "")[:300]
    db.update_watch_check(watch_row["id"], json.dumps(parsed, ensure_ascii=False))
    if parsed.get("matched") is True:
        db.trigger_watch(watch_row["id"])
        if _app and _app.bot:
            txt = (f"🛎 watch #{watch_row['id']} 발사 — {watch_row['label']}\n"
                   f"  • {parsed.get('evidence','')}\n"
                   f"  • 다음 발사 안 함 (`/watch_cancel`로 정리, 재등록은 `/watch`)")
            try:
                await _app.bot.send_message(chat_id=chat_id, text=txt)
            except Exception:
                logger.exception("watch trigger send failed")
            db.log_agent_action(
                chat_id, "watch_trigger",
                summary=f"watch {watch_row['label']} 발사",
                payload={"watch_id": watch_row["id"], "evidence": parsed.get("evidence")},
            )


def tool_lifelog_search(chat_id: int, args: Dict) -> Dict:
    """동기 시그니처지만 내부적으로 비동기 호출이 필요 — async wrapper 사용."""
    q = (args.get("query") or "").strip()
    if not q:
        return {"ok": False, "error": "query required"}
    k = max(1, min(int(args.get("k") or 8), 20))
    # Sync wrapper around async — run on event loop if possible
    import asyncio as _asyncio
    try:
        loop = _asyncio.get_event_loop()
        if loop.is_running():
            # called from within run_agent → wrap as task and wait
            fut = _asyncio.ensure_future(lifelog.search(chat_id, q, k=k))
            # Can't synchronously await — but tool handlers in run_agent are
            # awaited if async. So we need to be async. Convert below.
            return {"ok": False, "error": "use async tool_lifelog_search_async"}
        results = loop.run_until_complete(lifelog.search(chat_id, q, k=k))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "results": results}


async def tool_lifelog_search_async(chat_id: int, args: Dict) -> Dict:
    q = (args.get("query") or "").strip()
    if not q:
        return {"ok": False, "error": "query required"}
    k = max(1, min(int(args.get("k") or 8), 20))
    try:
        results = await lifelog.search(chat_id, q, k=k)
    except Exception as e:
        logger.exception("lifelog_search failed")
        return {"ok": False, "error": str(e)}
    return {"ok": True, "results": results}


# ---------------- v11: voice call (Twilio) ----------------


async def cmd_call(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/call <번호> <목적>` 봇이 한국어로 outbound call. 봇이 instruction
    말하고 → 응답 녹음 → 종료 → transcript + 요약 자동 보고."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        rows = db.list_voice_calls(chat_id)
        if not rows:
            await update.message.reply_text(
                "/call 02-555-1234 강남 라멘집 오늘 8시 4명 예약\n"
                "(Twilio + PUBLIC_BASE_URL 환경변수 필요)"
            )
            return
        lines = ["📞 통화 기록"]
        for r in rows[:10]:
            mark = {"queued": "⏳", "active": "📞", "done": "✓",
                    "failed": "✗", "cancelled": "×"}.get(r["status"], "·")
            lines.append(f"  {mark} #{r['id']} {r['to_number']} — {r['purpose'][:40]}")
        await update.message.reply_text("\n".join(lines))
        return
    if not voice_call.is_configured():
        await update.message.reply_text(
            "⚠ Twilio 미설정. Railway env에 TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / "
            "TWILIO_NUMBER / PUBLIC_BASE_URL 등록 필요.")
        return
    if len(args) < 2:
        await update.message.reply_text("/call <번호> <한 줄 instruction>")
        return
    # 비용 가드
    try:
        cap = float(_fact_value_local(chat_id, "call_daily_cost_limit_usd") or "5.0")
    except ValueError:
        cap = 5.0
    spent = db.voice_call_cost_today(chat_id)
    if spent >= cap:
        await update.message.reply_text(
            f"⚠ 일 통화 비용 한도 도달 (${spent:.4f}/${cap:.2f}). "
            "`remember_fact call_daily_cost_limit_usd 10` 로 올리거나 내일.")
        return
    to_number = args[0]
    purpose = " ".join(args[1:]).strip()
    call_id = db.create_voice_call(chat_id, to_number, purpose)
    res = await voice_call.place_call(to_number, purpose, call_id)
    if not res.get("ok"):
        db.update_voice_call(call_id, status="failed")
        await update.message.reply_text(f"⚠ 통화 실패: {res.get('error')}")
        return
    db.update_voice_call(call_id, twilio_call_sid=res["twilio_call_sid"])
    await update.message.reply_text(
        f"📞 #{call_id} 통화 시작 → {to_number}\n"
        f"  목적: {purpose[:80]}\n"
        f"  종료 후 transcript + 요약 자동 보고")


async def cmd_cancel_call(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("/cancel_call <id>")
        return
    try:
        cid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("id는 숫자")
        return
    row = db.get_voice_call(cid)
    if not row or row["chat_id"] != chat_id:
        await update.message.reply_text("없는 통화")
        return
    if row["twilio_call_sid"]:
        ok = await voice_call.cancel_call(row["twilio_call_sid"])
        db.update_voice_call(cid, status="cancelled")
        await update.message.reply_text("⏹ 중단" if ok else "⚠ 이미 종료됨")
    else:
        await update.message.reply_text("Twilio SID 없음")


async def tool_place_phone_call(chat_id: int, args: Dict) -> Dict:
    to_number = (args.get("to_number") or "").strip()
    purpose = (args.get("purpose") or "").strip()
    if not to_number or not purpose:
        return {"ok": False, "error": "to_number + purpose required"}
    if not voice_call.is_configured():
        return {"ok": False, "error": "twilio not configured"}
    try:
        cap = float(_fact_value_local(chat_id, "call_daily_cost_limit_usd") or "5.0")
    except ValueError:
        cap = 5.0
    if db.voice_call_cost_today(chat_id) >= cap:
        return {"ok": False, "error": "daily call cost cap reached"}
    call_id = db.create_voice_call(chat_id, to_number, purpose)
    res = await voice_call.place_call(to_number, purpose, call_id)
    if not res.get("ok"):
        db.update_voice_call(call_id, status="failed")
        return {"ok": False, "error": res.get("error")}
    db.update_voice_call(call_id, twilio_call_sid=res["twilio_call_sid"])
    return {"ok": True, "call_id": call_id, "twilio_sid": res["twilio_call_sid"]}


# ---------------- v11: browser automation + vault ----------------


def _get_vault_passphrase(chat_id: int) -> Optional[str]:
    return _fact_value_local(chat_id, "vault_passphrase")


async def cmd_vault(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/vault add <site> <user> <pass>` · `/vault list` · `/vault remove <site>`.

    비밀번호는 fact `vault_passphrase`로 AES-GCM 암호화. 사용자가
    /facts에 passphrase 먼저 등록 필요."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    pw = _get_vault_passphrase(chat_id)
    if not args:
        rows = db.list_web_credentials(chat_id)
        if not rows:
            txt = ("🗝 vault 비어 있음.\n"
                   "1) `remember_fact vault_passphrase 길고복잡한문자열`\n"
                   "2) /vault add catchtable my@mail mypassword")
        else:
            lines = ["🗝 등록 사이트"]
            for r in rows[:20]:
                lines.append(f"  • {r['site_label']} ({r['username']})")
            txt = "\n".join(lines)
        await update.message.reply_text(txt)
        return
    op = args[0].lower()
    if op == "add" and len(args) >= 4:
        if not pw:
            await update.message.reply_text(
                "⚠ vault_passphrase fact 없음 — `remember_fact vault_passphrase <문구>` 먼저")
            return
        site, user = args[1].lower(), args[2]
        password = " ".join(args[3:])
        enc = browser_mod.encrypt_password(password, pw)
        if not enc:
            await update.message.reply_text("⚠ 암호화 라이브러리 누락")
            return
        wid = db.add_web_credential(
            chat_id, site, user, enc["ciphertext"], enc["nonce"])
        await update.message.reply_text(f"🗝 #{wid} {site} 저장 ({user}) — 비밀번호 암호화됨")
        # 사용자가 비밀번호를 chat에 노출했으니 그 메시지 삭제 권장
        try:
            await update.message.delete()
        except Exception:
            pass
        return
    if op == "remove" and len(args) >= 2:
        ok = db.delete_web_credential(chat_id, args[1])
        await update.message.reply_text("🗑 제거" if ok else "없음")
        return
    await update.message.reply_text("사용: /vault | /vault add <site> <user> <pass> | /vault remove <site>")


async def tool_web_scrape(chat_id: int, args: Dict) -> Dict:
    url = (args.get("url") or "").strip()
    if not url:
        return {"ok": False, "error": "url required"}
    res = await browser_mod.scrape_url(
        url, wait_for_selector=args.get("wait_for_selector"),
        max_chars=int(args.get("max_chars") or 6000),
    )
    return res


async def tool_web_screenshot(chat_id: int, args: Dict) -> Dict:
    url = (args.get("url") or "").strip()
    if not url:
        return {"ok": False, "error": "url required"}
    res = await browser_mod.screenshot_url(url, full_page=bool(args.get("full_page", True)))
    if res.get("ok") and _app and _app.bot:
        try:
            await _app.bot.send_photo(
                chat_id=chat_id, photo=res["png"],
                caption=(args.get("purpose") or url)[:120],
            )
        except Exception:
            logger.exception("screenshot send failed")
    res.pop("png", None)  # don't leak bytes to LLM
    return res


# ---------------- v11: watch_tasks (장기 trigger) ----------------


WATCH_PARSE_PROMPT = (
    "사용자의 자연어 watch 요청을 다음 JSON으로 파싱:\n"
    '{{"label": "<짧은 한국어 라벨 ≤30자>",\n'
    ' "condition_md": "<원본 조건 정리>",\n'
    ' "check_kind": "fetch_url"|"web_search"|"gmail_query",\n'
    ' "check_args": {{... kind에 맞는 인자}},\n'
    ' "check_interval_min": <int, 최소 30>}}\n\n'
    "check_args 예시:\n"
    '  fetch_url: {{"url": "https://..."}}\n'
    '  web_search: {{"query": "DMC파크뷰자이 매물"}}\n'
    '  gmail_query: {{"query": "from:no-reply@delta.com"}}\n\n'
    "원본 자연어: {raw}"
)


async def cmd_watch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/watch <자연어 조건>` 등록 · `/watches` 목록 · `/watch_cancel <id>`."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        rows = db.list_watches(chat_id)
        if not rows:
            await update.message.reply_text(
                "/watch DMC파크뷰자이 매매가 11억 이하 매주 — 자연어로 등록"
            )
            return
        lines = ["🛎 watches"]
        for r in rows[:10]:
            mark = {"active": "•", "triggered": "✓", "cancelled": "×"}.get(r["status"], "·")
            lines.append(f"  {mark} #{r['id']} {r['label']} ({r['check_kind']}/{r['check_interval_min']}분)")
        await update.message.reply_text("\n".join(lines))
        return
    raw = " ".join(args).strip()
    await update.message.reply_text("🛎 파싱 중…")
    try:
        data = await chat_completion(
            [{"role": "user", "content": WATCH_PARSE_PROMPT.format(raw=raw)}],
            tools=None, chat_id=chat_id, kind="classify_content", max_tokens=400,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        await update.message.reply_text(f"⚠ {e}")
        return
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if not m:
        await update.message.reply_text(f"⚠ 파싱 실패\n{content[:200]}")
        return
    try:
        p = json.loads(m.group(0))
    except Exception:
        await update.message.reply_text(f"⚠ JSON 오류")
        return
    # active watch cap 10
    active = db.list_watches(chat_id, status="active")
    if len(active) >= 10:
        await update.message.reply_text("⚠ 활성 watch 10개 초과 — 먼저 정리")
        return
    wid = db.add_watch(
        chat_id,
        label=p.get("label", raw[:30]),
        condition_md=p.get("condition_md", raw),
        check_kind=p.get("check_kind", "web_search"),
        check_args=p.get("check_args") or {"query": raw},
        check_interval_min=int(p.get("check_interval_min") or 60),
    )
    await update.message.reply_text(
        f"🛎 watch #{wid} 등록 — {p.get('label', raw[:30])} ({p.get('check_kind')}/{p.get('check_interval_min', 60)}분)"
    )


async def cmd_watch_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("/watch_cancel <id>")
        return
    try:
        wid = int(context.args[0])
    except ValueError:
        await update.message.reply_text("id는 숫자")
        return
    ok = db.cancel_watch(chat_id, wid)
    await update.message.reply_text(f"🗑 #{wid} 정리" if ok else "없음")


def tool_add_watch(chat_id: int, args: Dict) -> Dict:
    """agent도 호출 가능 — 사용자 자연어 요청을 받았을 때."""
    label = (args.get("label") or "").strip()
    cond = (args.get("condition_md") or label).strip()
    if not label or not cond:
        return {"ok": False, "error": "label + condition_md required"}
    active = db.list_watches(chat_id, status="active")
    if len(active) >= 10:
        return {"ok": False, "error": "active watch cap (10)"}
    wid = db.add_watch(
        chat_id, label=label, condition_md=cond,
        check_kind=(args.get("check_kind") or "web_search"),
        check_args=args.get("check_args") or {"query": label},
        check_interval_min=int(args.get("check_interval_min") or 60),
    )
    return {"ok": True, "watch_id": wid}


def tool_list_watches(chat_id: int, args: Dict) -> Dict:
    rows = db.list_watches(chat_id, status=args.get("status"))
    return {"ok": True, "watches": [
        {"id": r["id"], "label": r["label"], "status": r["status"],
         "check_kind": r["check_kind"], "interval_min": r["check_interval_min"],
         "last_check_utc": r["last_check_utc"]}
        for r in rows]}


# ---------------- v11: decision simulation ----------------


DECIDE_KICKOFF_PROMPT = (
    "사용자가 큰 결정을 봇에게 가져왔다. 옵션과 핵심 factor를 분리하고 "
    "사용자에게 추가로 필요한 정보 3-5개를 짧은 질문 리스트로 받아내자.\n"
    'JSON: {{"options": ["A","B",...], "factors_needed": ["월세 차이","통근 시간",...], '
    '"clarify_questions": ["월세 차이 얼마?", "통근 거리?", ...]}}\n\n'
    "결정 주제: {title}"
)


DECIDE_SIMULATE_PROMPT = (
    "사용자가 결정해야 하는 옵션과 factor가 모였다. 각 옵션을 6/12/24개월 후 "
    "시나리오로 시뮬레이션해 추천 1개 + 이유 + 위험 + 대안 1개. 한국어 마크다운, "
    "≤600자.\n"
    "주제: {title}\n"
    "옵션: {options}\n"
    "factors: {factors}\n"
    "persona 요약: {persona}"
)


async def cmd_decide(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/decide <상황>` 결정 시작 · `/decide list` 목록 · `/decide <id>` 결과 보기."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        rows = db.list_decisions(chat_id)
        if not rows:
            await update.message.reply_text(
                "/decide 강남 vs 합정 이사  ← 결정 시작\n"
                "/decide <id>  ← 결과 보기"
            )
            return
        lines = ["🎯 최근 결정"]
        for r in rows[:10]:
            emoji = {"pending": "⏳", "decided": "✓", "abandoned": "×"}.get(r["status"], "·")
            lines.append(f"  {emoji} #{r['id']} {r['title']}")
        await update.message.reply_text("\n".join(lines))
        return
    if args[0].lower() == "list":
        rows = db.list_decisions(chat_id)
        for r in rows[:15]:
            await update.message.reply_text(f"#{r['id']} [{r['status']}] {r['title']}")
        return
    if len(args) == 2 and args[0].isdigit() and args[1].lower() == "go":
        did = int(args[0])
        d = db.get_decision(did)
        if not d or d["chat_id"] != chat_id:
            await update.message.reply_text("없는 결정")
            return
        await update.message.reply_text("🎯 시뮬 중…")
        sim = await _run_simulation(chat_id, did)
        for i in range(0, len(sim), 4000):
            await update.message.reply_text(sim[i:i + 4000])
        return
    if len(args) == 1 and args[0].isdigit():
        d = db.get_decision(int(args[0]))
        if not d or d["chat_id"] != chat_id:
            await update.message.reply_text("없는 결정")
            return
        text = f"#{d['id']} [{d['status']}] {d['title']}"
        if d["simulation_md"]:
            text += "\n\n" + d["simulation_md"][:3500]
        await update.message.reply_text(text)
        return
    # 새 결정 시작
    title = " ".join(args).strip()
    did = db.create_decision(chat_id, title)
    await update.message.reply_text(f"🎯 결정 #{did} 시작 — '{title}'\n분석 중…")
    try:
        data = await chat_completion(
            [{"role": "user", "content": DECIDE_KICKOFF_PROMPT.format(title=title)}],
            tools=None, chat_id=chat_id, kind="ask", max_tokens=400,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        await update.message.reply_text(f"⚠ {e}")
        return
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if not m:
        await update.message.reply_text(f"⚠ 분석 실패\n{content[:300]}")
        return
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        await update.message.reply_text(f"⚠ JSON 파싱 실패\n{content[:300]}")
        return
    options = parsed.get("options") or []
    factors = parsed.get("factors_needed") or []
    questions = parsed.get("clarify_questions") or []
    db.update_decision(
        did,
        options_json=json.dumps(options, ensure_ascii=False),
        factors_json=json.dumps({"needed": factors, "answers": {}}, ensure_ascii=False),
        questions_asked=len(questions),
    )
    msg = f"옵션: {' vs '.join(options) if options else '(미정)'}\n\n"
    if questions:
        msg += "다음 정보 필요 — 메시지로 그냥 답하면 돼:\n"
        for q in questions[:5]:
            msg += f"  • {q}\n"
        msg += f"\n답 다 모이면 `/decide {did} go` 로 시뮬레이션."
    await update.message.reply_text(msg)


async def _maybe_finalize_decision(chat_id: int) -> Optional[Dict]:
    """대화 중에 pending decision이 있고 사용자가 'go' / 'simulate' 같은 신호
    주면 시뮬레이션 실행. 호출자는 메시지 응답 결정."""
    pending = db.get_pending_decision(chat_id)
    if not pending:
        return None
    return {"id": pending["id"], "title": pending["title"]}


async def _run_simulation(chat_id: int, decision_id: int) -> str:
    """factors_json + persona로 LLM이 시뮬레이션 작성."""
    d = db.get_decision(decision_id)
    if not d:
        return "없는 결정"
    persona = db.get_latest_persona(chat_id)
    persona_md = (persona["content_md"][:600] if persona else "(없음)")
    try:
        data = await chat_completion(
            [{"role": "user", "content": DECIDE_SIMULATE_PROMPT.format(
                title=d["title"],
                options=d["options_json"],
                factors=d["factors_json"],
                persona=persona_md,
            )}],
            tools=None, chat_id=chat_id, kind="ask", max_tokens=1200,
        )
        sim = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        return f"⚠ 시뮬 실패: {e}"
    db.update_decision(
        decision_id, simulation_md=sim, status="decided",
    )
    return sim


async def run_experiment_followup(exp_id: int) -> None:
    """End-of-experiment evaluator: summarize habit log delta, mark complete."""
    exp = db.get_experiment(exp_id)
    if not exp or exp["status"] != "active":
        return
    chat_id = exp["chat_id"]
    # crude evaluation: how many habit logs in the experiment window?
    with db._conn() as c:
        rows = list(c.execute(
            "SELECT habit_key, COUNT(*) AS n FROM habits "
            "WHERE chat_id=? AND substr(when_local,1,10)>=? AND substr(when_local,1,10)<=? "
            "GROUP BY habit_key",
            (chat_id, exp["start_date_local"], exp["end_date_local"])))
    counts = {r["habit_key"]: r["n"] for r in rows}
    summary_md = (
        f"## 실험 결과 — {exp['title']}\n\n"
        f"**가설**: {exp['hypothesis']}\n\n"
        f"**기간**: {exp['start_date_local']} ~ {exp['end_date_local']}\n\n"
        f"**habit 로그 수**:\n"
        + ("\n".join(f"  - {k}: {v}회" for k, v in counts.items())
           or "  - (없음)")
        + "\n\n계속할지 결정해줘 — `/experiment keep` 또는 `/experiment drop`."
    )
    db.complete_experiment(chat_id, exp_id, summary_md)
    if _app and _app.bot:
        try:
            await _app.bot.send_message(
                chat_id=chat_id,
                text=f"🧪 실험 종료 #{exp_id}\n\n{summary_md}",
            )
        except Exception:
            logger.exception("experiment followup send failed")


async def cmd_experiment(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/experiment` 목록 · `/experiment new <title> <hypothesis>` · `/experiment <id>`."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        rows = db.list_habit_experiments(chat_id)
        if not rows:
            await update.message.reply_text(
                "🧪 실험 없음.\n"
                "사용: /experiment new 일찍자기 \"3일 22시 취침 → 컨디션 좋아진다\"\n"
                "      /experiment <id>"
            )
            return
        lines = ["🧪 실험 기록"]
        for r in rows[:15]:
            emoji = "🟢" if r["status"] == "active" else "✅" if r["status"] == "completed" else "·"
            lines.append(f"  {emoji} #{r['id']} {r['title']} ({r['start_date_local']}~{r['end_date_local']})")
        await update.message.reply_text("\n".join(lines))
        return
    if args[0].lower() == "new" and len(args) >= 3:
        title = args[1]
        hyp = " ".join(args[2:])
        # default 7-day window
        today = datetime.now(TZ).date()
        end = today + timedelta(days=7)
        eid = db.add_habit_experiment(
            chat_id, title, hyp, today.isoformat(), end.isoformat())
        scheduler.schedule_experiment_followup(eid, end.isoformat())
        await update.message.reply_text(
            f"🧪 실험 #{eid} 시작 — {title}\n"
            f"  • 가설: {hyp}\n"
            f"  • 기간: {today} ~ {end}\n"
            f"  • 종료일 21:00에 결과 자동 정리"
        )
        return
    if args[0].isdigit():
        row = db.get_experiment(int(args[0]))
        if not row or row["chat_id"] != chat_id:
            await update.message.reply_text("없는 실험")
            return
        text = (
            f"🧪 실험 #{row['id']} [{row['status']}]\n"
            f"• {row['title']}\n"
            f"• 가설: {row['hypothesis']}\n"
            f"• 기간: {row['start_date_local']} ~ {row['end_date_local']}"
        )
        if row["result_md"]:
            text += "\n\n" + row["result_md"][:2000]
        await update.message.reply_text(text)
        return
    await update.message.reply_text(
        "/experiment | /experiment new <title> <hypothesis> | /experiment <id>")


SUBSCRIPTION_SYSTEM = (
    "다음은 사용자가 정기 구독한 토픽이야. 최신 정보를 web_search/fetch_url로 조사해 "
    "한국어 마크다운 ≤500자 digest로 만들어. 핵심 수치·날짜·링크 inline 인용. "
    "변동 없으면 한 줄 'no change since {last_run}'.\n\n토픽: {topic}\n"
    "지난 실행: {last_run_utc}\n지난 digest:\n{last_md}"
)


async def run_subscription(sub_id: int) -> None:
    """Cron-fired: run a tiny research mission for a subscription topic."""
    row = db.get_subscription(sub_id)
    if not row or not row["enabled"]:
        return
    chat_id = row["chat_id"]
    last_md = (row["last_digest_md"] or "(첫 실행)")[:1500]
    prompt = SUBSCRIPTION_SYSTEM.format(
        topic=row["topic"],
        last_run_utc=row["last_run_utc"] or "—",
        last_md=last_md,
    )
    try:
        digest = await run_agent(chat_id, prompt, history=[], max_hops=6, kind="ask")
    except Exception:
        logger.exception("subscription %s run failed", sub_id)
        return
    db.update_subscription_run(sub_id, digest)
    if _app and _app.bot and digest.strip():
        try:
            text = f"📰 {row['name']}\n\n{digest[:3500]}"
            await _app.bot.send_message(chat_id=chat_id, text=text)
        except Exception:
            logger.exception("subscription send failed")


async def cmd_subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/subscribe <name> <HH:MM | mon HH:MM> <topic>` 정기 토픽 구독."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        rows = db.list_subscriptions(chat_id)
        if not rows:
            await update.message.reply_text(
                "📰 구독 없음.\n"
                "사용: /subscribe 환율 09:00 USD/KRW 환율 + 한 줄 분석\n"
                "       /subscribe 항공권 mon 08:00 부산↔도쿄 특가\n"
                "       /unsubscribe <name>"
            )
            return
        lines = ["📰 구독 목록"]
        for r in rows:
            status = "✓" if r["enabled"] else "✗"
            lines.append(f"  {status} {r['name']} @{r['cron_kst']} — {r['topic'][:50]}")
            if r["last_run_utc"]:
                lines.append(f"      마지막 실행 {r['last_run_utc'][:16]}")
        await update.message.reply_text("\n".join(lines))
        return
    if len(args) < 3:
        await update.message.reply_text("사용: /subscribe <name> <HH:MM | mon HH:MM> <topic>")
        return
    name = args[0]
    # Look for "mon HH:MM" form
    if len(args) >= 4 and args[1].lower() in {"mon","tue","wed","thu","fri","sat","sun"} \
            and ":" in args[2]:
        cron = f"{args[1].lower()} {args[2]}"
        topic = " ".join(args[3:])
    elif ":" in args[1]:
        cron = args[1]
        topic = " ".join(args[2:])
    else:
        await update.message.reply_text("cron 형식 오류 (예: 09:00 또는 mon 09:00).")
        return
    sid = db.add_subscription(chat_id, name, topic, cron)
    if not sid:
        await update.message.reply_text(f"⚠ '{name}' 이미 있음 — /unsubscribe 먼저")
        return
    row = db.get_subscription(sid)
    if row:
        scheduler.schedule_subscription(row)
    await update.message.reply_text(f"📰 구독 #{sid} '{name}' 생성 — {cron} KST")


async def cmd_unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("사용: /unsubscribe <name>")
        return
    name = context.args[0]
    rows = db.list_subscriptions(chat_id)
    target = next((r for r in rows if r["name"] == name), None)
    if not target:
        await update.message.reply_text("없음.")
        return
    scheduler.cancel_subscription_job(target["id"])
    db.delete_subscription(chat_id, name)
    await update.message.reply_text(f"🗑 '{name}' 구독 해지")


async def cmd_quiet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/quiet` 상태 · `/quiet 22:30 07:30` 변경 · `/quiet off` 끄기."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        start = _fact_value_local(chat_id, "quiet_hours_start") or "23:00"
        end = _fact_value_local(chat_id, "quiet_hours_end") or "07:00"
        disabled = _toggle_off_local(chat_id, "quiet_hours_enabled")
        now_silent = "🌙 지금 조용 모드" if _is_quiet_now(chat_id) else "🔔 지금 알림 가능"
        await update.message.reply_text(
            f"{'🔕 OFF' if disabled else f'🌙 조용 시간 {start}–{end}'}\n"
            f"{now_silent}\n\n"
            "사용:\n  /quiet 22:30 07:30  — 시간 변경\n"
            "  /quiet off            — 항시 알림\n"
            "  /quiet on             — 다시 켜기"
        )
        return
    if args[0].lower() == "off":
        db.remember_fact(chat_id, "quiet_hours_enabled", "false")
        await update.message.reply_text("🔔 조용 시간 OFF — 24h 알림 가능")
        return
    if args[0].lower() == "on":
        db.forget_fact(chat_id, "quiet_hours_enabled")
        await update.message.reply_text("🌙 조용 시간 ON")
        return
    if len(args) >= 2:
        try:
            for s in (args[0], args[1]):
                hh, mm = s.split(":")
                int(hh); int(mm)
        except Exception:
            await update.message.reply_text("HH:MM 형식 오류")
            return
        db.remember_fact(chat_id, "quiet_hours_start", args[0])
        db.remember_fact(chat_id, "quiet_hours_end", args[1])
        db.forget_fact(chat_id, "quiet_hours_enabled")
        await update.message.reply_text(f"🌙 조용 시간 {args[0]}–{args[1]} KST")
        return
    await update.message.reply_text("/quiet | /quiet HH:MM HH:MM | /quiet off")


async def cmd_macro(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/macro` 목록 · `/macro save <name> <recipe>` 저장 · `/macro <name>` 실행 ·
    `/macro delete <name>` 삭제."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        rows = db.list_macros(chat_id)
        if not rows:
            await update.message.reply_text(
                "🎬 매크로 없음.\n"
                "저장: /macro save 출근체크 오늘 일정 + 날씨 + 미세먼지 요약해줘\n"
                "실행: /macro 출근체크"
            )
            return
        lines = ["🎬 저장된 매크로"]
        for r in rows:
            lines.append(f"  • {r['name']} — {r['prompt'][:60]}...")
        await update.message.reply_text("\n".join(lines))
        return
    op = args[0].lower()
    if op == "save" and len(args) >= 3:
        name = args[1]
        prompt = " ".join(args[2:])
        mid = db.add_macro(chat_id, name, prompt)
        if mid:
            await update.message.reply_text(f"💾 매크로 '{name}' 저장")
        else:
            await update.message.reply_text(f"⚠ '{name}' 이미 있음 — 먼저 delete")
        return
    if op == "delete" and len(args) >= 2:
        ok = db.delete_macro(chat_id, args[1])
        await update.message.reply_text("🗑 삭제됨" if ok else "없음")
        return
    # /macro <name> — execute
    name = args[0]
    row = db.get_macro(chat_id, name)
    if not row:
        await update.message.reply_text(f"⚠ 매크로 '{name}' 없음")
        return
    await update.message.reply_text(f"▶️ 매크로 '{name}' 실행 중…")
    try:
        reply = await run_agent(chat_id, row["prompt"], history=[], max_hops=8, kind="chat")
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i:i + 4000])
    except Exception as e:
        await update.message.reply_text(f"⚠ {e}")


async def cmd_cleanup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/cleanup` 메모리 정리: 중복 사람·facts 보고 + 고아 relations 자동 삭제."""
    chat_id = update.effective_chat.id
    dup_people = db.find_duplicate_people(chat_id)
    dup_facts = db.find_duplicate_facts(chat_id)
    orphan = db.cleanup_orphan_relations(chat_id)
    lines = ["🧹 메모리 정리"]
    if dup_people:
        lines.append(f"  • 중복 사람 {len(dup_people)}건:")
        for d in dup_people[:5]:
            lines.append(f"      - '{d['name']}' ids={d['ids']}")
        lines.append("      LLM 한테 'merge_people 호출해줘' 시키면 합쳐줘")
    else:
        lines.append("  • 중복 사람: 없음")
    if dup_facts:
        lines.append(f"  • 중복 facts {len(dup_facts)}건:")
        for d in dup_facts[:5]:
            lines.append(f"      - '{d['key']}' ids={d['ids']}")
    else:
        lines.append("  • 중복 facts: 없음")
    lines.append(f"  • 고아 relations 삭제: {orphan}건")
    await update.message.reply_text("\n".join(lines))


DOC_ANALYZE_PROMPT = (
    "다음 문서를 *비서 입장*에서 사용자가 놓칠 만한 위험·중요 조항·서명 전 물어봐야 할 "
    "질문을 한국어로 분석해. 출력 마크다운 3 섹션:\n"
    "## 🔴 위험 / 주의\n"
    "## 🔑 핵심 조항\n"
    "## ❓ 사용자가 상대에게 물어볼 질문\n\n"
    "문서 종류 힌트: {hint}\n\n문서:\n{text}"
)


async def tool_analyze_document(chat_id: int, args: Dict) -> Dict:
    text = (args.get("document_text") or "").strip()
    if not text:
        return {"ok": False, "error": "document_text required"}
    hint = (args.get("doc_kind") or "unknown").strip()
    prompt = DOC_ANALYZE_PROMPT.format(text=text[:8000], hint=hint)
    try:
        data = await chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="ask", max_tokens=1500,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "analysis_md": content, "doc_kind": hint}


def tool_solar_term_check(chat_id: int, args: Dict) -> Dict:
    d = (args.get("date_local") or "").strip()
    if not d:
        return {"ok": False, "error": "date_local required"}
    return {
        "ok": True, "date": d,
        "today_term": lunar.solar_term_on(d),
        "next_term": lunar.next_solar_term(d),
    }


def tool_lunar_to_solar(chat_id: int, args: Dict) -> Dict:
    m = int(args.get("month") or 0)
    d = int(args.get("day") or 0)
    y = int(args.get("solar_year") or datetime.now(TZ).year)
    if not m or not d:
        return {"ok": False, "error": "month + day required"}
    sd = lunar.lunar_to_solar_recurring(m, d, y)
    if not sd:
        return {"ok": False, "error": f"lunar {m}/{d} can't be resolved for {y}"}
    return {"ok": True, "solar_date": sd.isoformat(), "lunar": {"month": m, "day": d}}


def tool_korean_holiday_check(chat_id: int, args: Dict) -> Dict:
    d = (args.get("date_local") or "").strip()
    if not d:
        return {"ok": False, "error": "date_local required"}
    is_h = korean_calendar.is_holiday(d)
    nxt = korean_calendar.next_holiday(d)
    return {"ok": True, "date": d, "is_holiday": is_h, "next_holiday": nxt}


async def tool_weather(chat_id: int, args: Dict) -> Dict:
    return await weather_mod.weather(args["location"], days=int(args.get("days", 1)))


async def tool_track_parcel(chat_id: int, args: Dict) -> Dict:
    return await weather_mod.track_parcel(
        tracking_no=args["tracking_no"],
        carrier=args.get("carrier"),
    )


async def tool_transit_text_query(chat_id: int, args: Dict) -> Dict:
    return await weather_mod.transit_text_query(
        origin=args["origin"],
        destination=args["destination"],
        arrive_by_iso=args.get("arrive_by_iso"),
    )


def tool_add_wedding_timeline(chat_id: int, args: Dict) -> Dict:
    """One-shot template: creates canonical engagement→marriage→honeymoon goals."""
    propose = args.get("propose_date_local")
    marry = args.get("marriage_date_local")
    honey = args.get("honeymoon_date_local")
    partner = (args.get("partner_name") or "").strip()
    created: List[int] = []

    if propose:
        gid = db.add_goal(
            chat_id,
            title=(f"{partner} 프로포즈 여행 준비" if partner else "프로포즈 여행 준비"),
            why="장기 추적 — 사용자가 가장 잊기 쉬운 큰 일정",
            target_date_local=propose,
            horizon="long",
            sub_tasks=[
                "장소 후보 3곳 추리기",
                "항공권/숙소 평균가 모니터링",
                "반지 사이즈 자연스럽게 파악",
                "당일 동선/타이밍 시뮬",
                "사진/영상 기록 담당 정하기",
            ],
            watch_query=("제주 호텔 12월" if propose.startswith(("2026-12", "2027-12")) else None),
        )
        created.append(gid)
        gid2 = db.add_goal(
            chat_id,
            title="약혼반지 구입 (상품권 활용)",
            why="비용 효율 — 상품권 할인 윈도우가 비정기적이라 미리 모니터링 필요",
            target_date_local=propose,
            horizon="long",
            sub_tasks=[
                "백화점 상품권 할인율 비교",
                "반지 디자인 후보 3개 좁히기",
                "사이즈 확인",
                "구입 D-30 결정",
            ],
            watch_query="신세계상품권 5% 할인",
        )
        created.append(gid2)

    if marry:
        gid = db.add_goal(
            chat_id,
            title="결혼식 준비",
            why="식장·스드메·하객 일정 등 다층 의존",
            target_date_local=marry,
            horizon="long",
            sub_tasks=[
                "양가 인사 일정",
                "예식장 답사 후보",
                "스드메 견적 비교",
                "청첩장 디자인/발송 일정",
                "본식 도우미 결정",
            ],
        )
        created.append(gid)

    if honey:
        gid = db.add_goal(
            chat_id,
            title="신혼여행 예약 윈도우",
            why="얼리버드 vs 라스트미닛 가격 차이 큼 → 모니터링 필요",
            target_date_local=honey,
            horizon="long",
            sub_tasks=[
                "지역 후보 2-3개 선정",
                "비행기 예약 골든 윈도우 확인",
                "호텔/풀빌라 가격 추이 모니터",
                "비자/여권 만료일 확인",
            ],
            watch_query="신혼여행 항공권 특가",
        )
        created.append(gid)

    if partner:
        # Try to ensure partner is registered as a person (idempotent)
        db.add_person(chat_id, partner, role="약혼녀")

    scheduler.ensure_proactive_for(chat_id)
    scheduler.ensure_daily_rhythm_for(chat_id)
    return {"ok": True, "goal_ids": created, "partner": partner or None}


# ---------------- people / relationships ----------------


def tool_add_person(chat_id: int, args: Dict) -> Dict:
    name = (args.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "name required"}
    pid = db.add_person(
        chat_id,
        name,
        aliases=args.get("aliases") or [],
        role=args.get("role"),
        notes=args.get("notes"),
        important_dates=args.get("important_dates") or [],
        preferred_language=args.get("preferred_language"),
    )
    return {"ok": True, "person_id": pid, "name": name}


def tool_update_person(chat_id: int, args: Dict) -> Dict:
    pid = args.get("person_id")
    if not pid:
        return {"ok": False, "error": "person_id required"}
    ok = db.update_person(
        int(pid), chat_id,
        name=args.get("name"),
        role=args.get("role"),
        notes=args.get("notes"),
        add_aliases=args.get("add_aliases") or [],
        important_dates=args.get("important_dates") or [],
    )
    return {"ok": ok, "person_id": int(pid)}


def tool_list_people(chat_id: int, args: Dict) -> Dict:
    rows = db.list_people(chat_id, role=args.get("role"))
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "name": r["name"], "role": r["role"],
            "aliases": json.loads(r["aliases_json"] or "[]"),
            "important_dates": _enrich_important_dates(
                json.loads(r["important_dates_json"] or "[]")
            ),
            "last_contact_utc": r["last_contact_utc"],
        })
    return {"ok": True, "people": out}


def _enrich_important_dates(raw: List[Dict]) -> List[Dict]:
    """Add days_until/next_occurrence to each important_date so the model never
    has to compute the calendar itself."""
    today = datetime.now(TZ).date()
    out = []
    for d in raw:
        item = dict(d)
        try:
            dt = datetime.fromisoformat(d["date_local"]).date()
            if d.get("recurring_yearly"):
                this_year = dt.replace(year=today.year)
                if this_year < today:
                    this_year = dt.replace(year=today.year + 1)
                item["next_occurrence"] = this_year.isoformat()
                item["days_until"] = (this_year - today).days
            else:
                item["days_until"] = (dt - today).days
        except Exception:
            pass
        out.append(item)
    return out


def tool_recall_person(chat_id: int, args: Dict) -> Dict:
    needle = (args.get("name_or_alias") or "").strip()
    if not needle:
        return {"ok": False, "error": "name_or_alias required"}
    hits = db.find_people_in_text(chat_id, needle)
    if not hits:
        return {"ok": False, "error": f"no person matches {needle!r}"}
    p = hits[0]
    # Also gather any chat_log mentions
    mentions = db.search_chat_log(chat_id, p["name"], limit=5)
    dates = _enrich_important_dates(json.loads(p["important_dates_json"] or "[]"))
    return {
        "ok": True,
        "person": {
            "id": p["id"], "name": p["name"], "role": p["role"],
            "aliases": json.loads(p["aliases_json"] or "[]"),
            "notes": p["notes"],
            "important_dates": dates,
            "last_contact_utc": p["last_contact_utc"],
        },
        "recent_mentions": mentions,
    }


def tool_log_contact_with(chat_id: int, args: Dict) -> Dict:
    needle = (args.get("name_or_alias") or "").strip()
    if not needle:
        return {"ok": False, "error": "name_or_alias required"}
    hits = db.find_people_in_text(chat_id, needle)
    if not hits:
        return {"ok": False, "error": f"no person matches {needle!r}"}
    p = hits[0]
    db.mark_contact(p["id"])
    # Also save a note so chat_log/notes have it
    channel = args.get("channel") or "message"
    note_text = f"[contact:{channel}] {p['name']}"
    if args.get("notes"):
        note_text += f" — {args['notes']}"
    db.add_note(chat_id, note_text, tags="contact")
    return {"ok": True, "person_id": p["id"], "name": p["name"]}


# ---------------- notes + episodic memory ----------------


def tool_save_note(chat_id: int, args: Dict) -> Dict:
    content = (args.get("content") or "").strip()
    if not content:
        return {"ok": False, "error": "content required"}
    nid = db.add_note(chat_id, content, args.get("tags"))
    return {"ok": True, "note_id": nid}


def tool_search_memory(chat_id: int, args: Dict) -> Dict:
    query = (args.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "query required"}
    kind = args.get("kind", "all")
    limit = int(args.get("limit", 5))
    out: Dict = {"ok": True, "query": query}
    if kind in ("notes", "all"):
        out["notes"] = db.search_notes(chat_id, query, limit)
    if kind in ("chat", "all"):
        out["chat_history"] = db.search_chat_log(chat_id, query, limit)
    return out


# ---------------- async external tool handlers ----------------


async def tool_web_search(chat_id: int, args: Dict) -> Dict:
    return await external.naver_search(
        query=args["query"],
        kind=args.get("kind", "blog"),
        display=int(args.get("display", 5)),
    )


async def tool_kakao_local(chat_id: int, args: Dict) -> Dict:
    return await external.kakao_local_keyword(
        query=args["query"],
        x=args.get("x"),
        y=args.get("y"),
        radius_m=args.get("radius_m"),
        size=int(args.get("size", 5)),
    )


async def tool_kakao_drive(chat_id: int, args: Dict) -> Dict:
    return await external.kakao_directions(
        origin_x=float(args["origin_x"]),
        origin_y=float(args["origin_y"]),
        dest_x=float(args["dest_x"]),
        dest_y=float(args["dest_y"]),
    )


async def tool_fetch_url(chat_id: int, args: Dict) -> Dict:
    return await external.fetch_url(
        url=args["url"], max_chars=int(args.get("max_chars", 8000))
    )


# ---------------- Google Calendar tool handlers ----------------


async def tool_gcal_list_events(chat_id: int, args: Dict) -> Dict:
    try:
        events = await gcal.list_events(
            chat_id,
            time_min_iso=args.get("time_min_iso"),
            time_max_iso=args.get("time_max_iso"),
            max_results=int(args.get("max_results", 25)),
        )
        return {"ok": True, "events": events}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def tool_gcal_create_event(chat_id: int, args: Dict) -> Dict:
    # If the LLM already called add_event for the same time and we created the
    # local row, we can link the GCal id back to it. Best-effort linkage by
    # matching most-recent local event without gcal_event_id in the same window.
    local_link_id = args.get("link_local_event_id")  # optional hint
    try:
        ev = await gcal.create_event(
            chat_id,
            summary=args["summary"],
            start_iso=args["start_iso"],
            end_iso=args.get("end_iso"),
            description=args.get("description"),
            location=args.get("location"),
        )
    except gcal.NotConnected as e:
        # If a local id was indicated, mark it pending so nightly retry attempts again.
        if local_link_id:
            try:
                db.set_event_gcal(int(local_link_id), None, "not_connected")
            except Exception:
                pass
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        if local_link_id:
            try:
                db.set_event_gcal(int(local_link_id), None, "pending")
            except Exception:
                pass
        return {"ok": False, "error": str(e), "gcal_sync_state": "pending"}
    gid = ev.get("id")
    if local_link_id and gid:
        try:
            db.set_event_gcal(int(local_link_id), gid, "synced")
        except Exception:
            logger.exception("set_event_gcal failed for local id %s", local_link_id)
    return {"ok": True, "event_id": gid, "html_link": ev.get("htmlLink")}


async def tool_gcal_update_event(chat_id: int, args: Dict) -> Dict:
    try:
        ev = await gcal.update_event(
            chat_id,
            event_id=args["event_id"],
            summary=args.get("summary"),
            start_iso=args.get("start_iso"),
            end_iso=args.get("end_iso"),
            description=args.get("description"),
            location=args.get("location"),
        )
        return {"ok": True, "event_id": ev.get("id")}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def tool_gcal_delete_event(chat_id: int, args: Dict) -> Dict:
    try:
        await gcal.delete_event(chat_id, event_id=args["event_id"])
        return {"ok": True, "event_id": args["event_id"]}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------- Gmail (shared Google OAuth) ----------------


async def tool_gmail_search(chat_id: int, args: Dict) -> Dict:
    try:
        ids = await gmail_mod.list_messages(
            chat_id, query=args["query"], max_results=int(args.get("max_results", 10))
        )
        # Fetch lightweight metadata for each
        out = []
        for mid in ids[:10]:
            try:
                m = await gmail_mod.get_message(chat_id, mid, body_max_chars=0)
                out.append({
                    "id": m["id"], "subject": m["subject"],
                    "from": m["from"], "date": m["date"], "snippet": m["snippet"],
                })
            except Exception as e:
                logger.warning("gmail_search get failed for %s: %s", mid, e)
        return {"ok": True, "messages": out}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def tool_gmail_get_message(chat_id: int, args: Dict) -> Dict:
    try:
        m = await gmail_mod.get_message(
            chat_id, args["message_id"],
            body_max_chars=int(args.get("body_max_chars", 4000)),
        )
        return {"ok": True, **m}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def tool_gmail_recent_summary(chat_id: int, args: Dict) -> Dict:
    try:
        msgs = await gmail_mod.recent_summary(
            chat_id,
            hours=min(max(int(args.get("hours", 24)), 1), 168),
            max_messages=min(max(int(args.get("max_messages", 10)), 1), 50),
        )
        return {"ok": True, "messages": msgs}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------- v4 tool handlers (outbound, rules, recall, streaks) ----------------


async def tool_gmail_send_email(chat_id: int, args: Dict) -> Dict:
    try:
        res = await gmail_mod.send_message(
            chat_id, args["to"], args["subject"], args["body_text"],
        )
        action_id = db.log_agent_action(
            chat_id, "gmail_send",
            summary=f"메일 보냄 → {args['to']} · {args['subject'][:40]}",
            payload={"to": args["to"], "subject": args["subject"],
                     "body_preview": args["body_text"][:200],
                     "gmail_message_id": res.get("id")},
            reversible=None,  # email send is not undoable
        )
        return {"ok": True, "gmail_message_id": res.get("id"), "agent_action_id": action_id}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        logger.exception("gmail_send_email failed")
        return {"ok": False, "error": str(e)}


async def tool_gmail_save_draft(chat_id: int, args: Dict) -> Dict:
    try:
        res = await gmail_mod.draft_message(
            chat_id, args["to"], args["subject"], args["body_text"])
        return {"ok": True, "draft_id": res.get("id")}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def tool_gmail_reply_to(chat_id: int, args: Dict) -> Dict:
    try:
        # Resolve recipient + subject from source if not supplied
        to_addr = args.get("to")
        subject = args.get("subject")
        if not to_addr or not subject:
            src = await gmail_mod.get_message(chat_id, args["in_reply_to_msg_id"], body_max_chars=100)
            if not to_addr:
                to_addr = (src.get("from") or "").strip()
            if not subject:
                src_sub = src.get("subject") or ""
                subject = src_sub if src_sub.lower().startswith("re:") else f"Re: {src_sub}"
        res = await gmail_mod.send_message(
            chat_id, to_addr, subject, args["body_text"],
            in_reply_to_msg_id=args["in_reply_to_msg_id"],
        )
        action_id = db.log_agent_action(
            chat_id, "gmail_reply",
            summary=f"메일 답장 → {to_addr} · {subject[:40]}",
            payload={"to": to_addr, "subject": subject,
                     "in_reply_to": args["in_reply_to_msg_id"],
                     "body_preview": args["body_text"][:200]},
        )
        return {"ok": True, "gmail_message_id": res.get("id"), "agent_action_id": action_id}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        logger.exception("gmail_reply_to failed")
        return {"ok": False, "error": str(e)}


async def tool_gcal_rsvp(chat_id: int, args: Dict) -> Dict:
    try:
        ev = await gcal.respond_to_invite(chat_id, args["event_id"], args["response"])
        summary = ev.get("summary") or "(제목 없음)"
        action_id = db.log_agent_action(
            chat_id, "gcal_rsvp",
            summary=f"RSVP {args['response']} → {summary[:40]}",
            payload={"event_id": args["event_id"], "response": args["response"]},
            reversible={"kind": "gcal_rsvp", "args": {"event_id": args["event_id"], "response": "tentative"}},
        )
        return {"ok": True, "event_id": args["event_id"],
                "response": args["response"], "agent_action_id": action_id}
    except gcal.NotConnected as e:
        return {"ok": False, "error": str(e), "needs_connect": True}
    except Exception as e:
        logger.exception("gcal_rsvp failed")
        return {"ok": False, "error": str(e)}


def tool_add_auto_rule(chat_id: int, args: Dict) -> Dict:
    rule_kind = args.get("rule_kind")
    cond = args.get("condition") or {}
    if rule_kind not in ("gmail_auto_add_event", "gcal_auto_rsvp",
                            "gcal_auto_decline", "meeting_auto_negotiate"):
        return {"ok": False, "error": f"unsupported rule_kind {rule_kind}"}
    rid = db.add_auto_rule(chat_id, rule_kind, cond)
    return {"ok": True, "rule_id": rid}


def tool_list_auto_rules(chat_id: int, args: Dict) -> Dict:
    rows = db.list_auto_rules(chat_id, only_enabled=False)
    return {"ok": True, "rules": [
        {"id": r["id"], "rule_kind": r["rule_kind"],
         "condition": json.loads(r["condition_json"]),
         "enabled": bool(r["enabled"])}
        for r in rows]}


def tool_disable_auto_rule(chat_id: int, args: Dict) -> Dict:
    rid = int(args.get("rule_id") or 0)
    if not rid:
        return {"ok": False, "error": "rule_id required"}
    return {"ok": db.disable_auto_rule(chat_id, rid)}


def tool_cross_recall(chat_id: int, args: Dict) -> Dict:
    """Aggregate everything mentioning `entity` across all silos."""
    entity = (args.get("entity") or "").strip()
    if not entity:
        return {"ok": False, "error": "entity required"}
    days = int(args.get("days") or 180)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    lowered = entity.lower()
    like = f"%{entity}%"
    out: Dict = {"entity": entity}
    # People
    out["people"] = [dict(p) for p in db.find_people_in_text(chat_id, entity)]
    # Events
    with db._conn() as c:
        out["events"] = [{"id": r["id"], "title": r["title"], "when_utc": r["when_utc"],
                          "location": r["location"], "notes": r["notes"]}
                          for r in c.execute(
            "SELECT * FROM events WHERE chat_id=? AND (title LIKE ? OR notes LIKE ? OR location LIKE ?) "
            "AND when_utc>=? ORDER BY when_utc DESC LIMIT 20",
            (chat_id, like, like, like, cutoff))]
        out["goals"] = [{"id": r["id"], "title": r["title"], "why": r["why"],
                          "status": r["status"], "target": r["target_date_local"]}
                         for r in c.execute(
            "SELECT * FROM goals WHERE chat_id=? AND "
            "(title LIKE ? OR why LIKE ? OR sub_tasks_json LIKE ?) ORDER BY id DESC LIMIT 10",
            (chat_id, like, like, like))]
        out["notes"] = [{"id": r["id"], "content": r["content"][:200], "created_at": r["created_at"]}
                         for r in c.execute(
            "SELECT * FROM notes WHERE chat_id=? AND (content LIKE ? OR tags LIKE ?) "
            "AND created_at>=? ORDER BY created_at DESC LIMIT 10",
            (chat_id, like, like, cutoff))]
        out["expenses"] = [{"id": r["id"], "amount_won": r["amount_won"],
                             "merchant": r["merchant"], "category": r["category"],
                             "when_local": r["when_local"]}
                            for r in c.execute(
            "SELECT * FROM expenses WHERE chat_id=? AND (merchant LIKE ? OR notes LIKE ?) "
            "AND when_local>=? ORDER BY when_local DESC LIMIT 15",
            (chat_id, like, like,
             (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")))]
        out["chat_log"] = [{"role": r["role"], "content": r["content"][:200],
                             "created_at": r["created_at"]}
                            for r in c.execute(
            "SELECT * FROM chat_log WHERE chat_id=? AND content LIKE ? AND created_at>=? "
            "ORDER BY created_at DESC LIMIT 8",
            (chat_id, like, cutoff))]
    return {"ok": True, **out}


def tool_detect_routines(chat_id: int, args: Dict) -> Dict:
    return {"ok": True, **routines.detect_all(chat_id)}


def tool_get_habit_streaks(chat_id: int, args: Dict) -> Dict:
    rows = db.list_habit_streaks(chat_id)
    return {"ok": True, "streaks": [
        {"habit_key": r["habit_key"], "current_streak": r["current_streak"],
         "best_streak": r["best_streak"], "last_log_date": r["last_log_date"]}
        for r in rows]}


SYNC_HANDLERS = {
    "add_event": tool_add_event,
    "list_events": tool_list_events,
    "update_event": tool_update_event,
    "delete_event": tool_delete_event,
    "remember_fact": tool_remember_fact,
    "forget_fact": tool_forget_fact,
    "add_recurring_task": tool_add_recurring_task,
    "list_recurring_tasks": tool_list_recurring_tasks,
    "delete_recurring_task": tool_delete_recurring_task,
    "enable_recurring_task": tool_enable_recurring_task,
    "disable_recurring_task": tool_disable_recurring_task,
    "save_note": tool_save_note,
    "search_memory": tool_search_memory,
    "add_goal": tool_add_goal,
    "list_goals": tool_list_goals,
    "update_goal": tool_update_goal,
    "complete_goal": tool_complete_goal,
    "add_goal_subtask": tool_add_goal_subtask,
    "complete_goal_subtask": tool_complete_goal_subtask,
    "add_person": tool_add_person,
    "update_person": tool_update_person,
    "list_people": tool_list_people,
    "recall_person": tool_recall_person,
    "log_contact_with": tool_log_contact_with,
    "log_expense": tool_log_expense,
    "summarize_expenses": tool_summarize_expenses,
    "log_habit": tool_log_habit,
    "summarize_habits": tool_summarize_habits,
    "korean_holiday_check": tool_korean_holiday_check,
    "solar_term_check": tool_solar_term_check,
    "lunar_to_solar": tool_lunar_to_solar,
    "add_wedding_timeline": tool_add_wedding_timeline,
    "add_auto_rule": tool_add_auto_rule,
    "list_auto_rules": tool_list_auto_rules,
    "disable_auto_rule": tool_disable_auto_rule,
    "cross_recall": tool_cross_recall,
    "detect_routines": tool_detect_routines,
    "get_habit_streaks": tool_get_habit_streaks,
    # v5 tool handlers (start_mission/cancel_mission/list_missions) are
    # registered later via SYNC_HANDLERS.update(...) once their function
    # definitions exist, to avoid forward-reference NameErrors at import.
}

ASYNC_HANDLERS = {
    "web_search": tool_web_search,
    "kakao_local_search": tool_kakao_local,
    "kakao_directions_drive": tool_kakao_drive,
    "fetch_url": tool_fetch_url,
    "gcal_list_events": tool_gcal_list_events,
    "gcal_create_event": tool_gcal_create_event,
    "gcal_update_event": tool_gcal_update_event,
    "gcal_delete_event": tool_gcal_delete_event,
    "gmail_search": tool_gmail_search,
    "gmail_get_message": tool_gmail_get_message,
    "gmail_recent_summary": tool_gmail_recent_summary,
    "gmail_send_email": tool_gmail_send_email,
    "gmail_save_draft": tool_gmail_save_draft,
    "gmail_reply_to": tool_gmail_reply_to,
    "gcal_rsvp": tool_gcal_rsvp,
    "weather": tool_weather,
    "track_parcel": tool_track_parcel,
    "transit_text_query": tool_transit_text_query,
}


# ---------------- the agent loop (reusable for live chat AND recurring tasks) ----------------


async def run_agent(chat_id: int, user_text: str, history: Optional[List[Dict]] = None,
                    max_hops: int = 6, kind: str = "chat") -> str:
    """Run the agent loop with tool-use until it returns a text answer.

    `kind` is forwarded to chat_completion so model_router can pick the right
    tier (chat→sonnet, agent→opus, ask→opus, etc.)."""
    history = history if history is not None else []
    history.append({"role": "user", "content": user_text})

    final_text = ""
    for hop in range(max_hops):
        messages = [_system_message(chat_id, recent_user_text=user_text), *history]
        data = await chat_completion(messages, tools=TOOLS, chat_id=chat_id, kind=kind)
        msg = data["choices"][0]["message"]
        history.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": msg.get("tool_calls") or None,
            }
        )
        tool_calls = parse_tool_calls(msg)
        if not tool_calls:
            final_text = (msg.get("content") or "").strip()
            break

        for tc in tool_calls:
            name = tc["name"]
            try:
                if name in SYNC_HANDLERS:
                    result = SYNC_HANDLERS[name](chat_id, tc["arguments"])
                elif name in ASYNC_HANDLERS:
                    result = await ASYNC_HANDLERS[name](chat_id, tc["arguments"])
                else:
                    result = {"ok": False, "error": f"unknown tool {name}"}
            except Exception as exc:
                logger.exception("tool %s failed", name)
                result = {"ok": False, "error": str(exc)}
            logger.info("tool %s args=%s result_keys=%s",
                        name, tc["arguments"],
                        list(result.keys()) if isinstance(result, dict) else type(result).__name__)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False)[:8000],
                }
            )

    return final_text or "처리 완료."


RECURRING_PROMPT_MAX = 2000
# Anything that looks like a raw tool invocation in the saved prompt is rejected.
# Saved prompts should describe intent in natural language, not call tools directly.
_RECURRING_PROMPT_BLOCKLIST = re.compile(
    r"^\s*(tool_|SYNC_HANDLERS|ASYNC_HANDLERS|run_agent\b|chat_completion\b)",
    re.IGNORECASE | re.MULTILINE,
)


def _sanitize_recurring_prompt(prompt: str) -> Optional[str]:
    """Return cleaned prompt or None if it looks like injection / too long."""
    if not prompt:
        return None
    if len(prompt) > RECURRING_PROMPT_MAX:
        return None
    if _RECURRING_PROMPT_BLOCKLIST.search(prompt):
        return None
    return prompt.strip()


async def run_recurring_task(task_id: int) -> None:
    """Wired into scheduler: load the saved prompt, run agent, send result to chat."""
    row = db.get_recurring_task(task_id)
    if row is None or not row["enabled"]:
        return
    chat_id = row["chat_id"]
    safe_prompt = _sanitize_recurring_prompt(row["prompt"])
    if safe_prompt is None:
        logger.warning("recurring task %s prompt rejected (sanitize)", task_id)
        bot = _app.bot if _app else None
        if bot:
            await bot.send_message(
                chat_id=chat_id,
                text=f"⚠️ 정기작업 #{task_id} 프롬프트가 안전 검사에서 차단됨. /tasks 에서 확인 후 재등록 부탁.",
            )
        return
    logger.info("running recurring task_id=%s chat=%s", task_id, chat_id)
    try:
        reply = await run_agent(chat_id, safe_prompt, history=[])
    except Exception as exc:
        logger.exception("recurring task %s agent failed", task_id)
        reply = f"⚠️ 정기 작업 실패: {exc}"
    bot = _app.bot if _app else None
    if bot:
        header = f"🔔 정기 알림 (#{task_id} · {row['cron_kst']} KST)\n\n"
        for i in range(0, len(reply), 4000):
            chunk = (header + reply) if i == 0 else reply[i : i + 4000]
            await bot.send_message(chat_id=chat_id, text=chunk[:4000])
    db.mark_recurring_run(task_id)


# ---------------- proactive runners (weekly review + daily imminent) ----------------


def _days_until(target_date_local: Optional[str]) -> Optional[int]:
    if not target_date_local:
        return None
    try:
        target = datetime.fromisoformat(target_date_local).date()
    except ValueError:
        return None
    today = datetime.now(TZ).date()
    return (target - today).days


def _format_goal_summary(goal_row) -> str:
    d = _days_until(goal_row["target_date_local"])
    head = f"#{goal_row['id']} · {goal_row['title']}"
    if d is not None:
        head += f"  (D-{d})" if d >= 0 else f"  (D+{-d})"
    if goal_row["why"]:
        head += f"\n  ↳ {goal_row['why']}"
    subs = _parse_sub_tasks(goal_row["sub_tasks_json"])
    open_subs = [s for s in subs if not s.get("done")]
    if open_subs:
        head += "\n  ↳ 남은 작업: " + ", ".join(s["text"] for s in open_subs[:3])
    if goal_row["watch_query"]:
        head += f"\n  ↳ watch: {goal_row['watch_query']}"
    return head


WEEKLY_REVIEW_PROMPT = (
    "이번 주 골 리뷰 시간이야. 사용자가 잊지 않도록 봇이 능동적으로 챙기는 자리.\n"
    "아래는 현재 open 상태인 모든 goals (sub_tasks 포함). 각 goal에 대해:\n"
    "  1) 일정대로 굴러가는지 (마감 D-n 보고)\n"
    "  2) 이번 주 안에 하면 좋은 액션 1~2개를 sub_task로 추가하거나 add_event\n"
    "  3) watch_query가 있고 'watch_due=true'로 표시된 골만 web_search/fetch_url로 최근 정보(가격/혜택/이벤트) 확인. 그 외 watch는 이번 주 건너뜀.\n"
    "  4) 너무 늦거나 흐려진 goal은 사용자에게 status 변경(paused/dropped) 제안\n"
    "필요한 도구는 자유롭게 사용. 마지막 답은 사용자에게 보낼 깔끔한 한국어 요약.\n\n"
    "현재 goals:\n{goals_block}"
)


async def run_weekly_goal_review(chat_id: int) -> None:
    """Proactive weekly review fired by scheduler (Sun 09:00 KST or forced trigger)."""
    rows = db.list_goals(chat_id, status="open")
    bot = _app.bot if _app else None
    if not rows:
        return
    formatted_blocks = []
    watch_due_ids: List[int] = []
    for r in rows:
        s = _format_goal_summary(r)
        if r["watch_query"] and db.goal_watch_due(r):
            watch_due_ids.append(r["id"])
            s += "  ↳ watch_due=true"
        elif r["watch_query"]:
            s += "  ↳ watch_due=false (이번 주 건너뜀)"
        formatted_blocks.append(s)
    goals_block = "\n\n".join(formatted_blocks)
    user_msg = WEEKLY_REVIEW_PROMPT.format(goals_block=goals_block)
    logger.info("weekly review for chat=%s with %d goals (watch_due=%d)",
                chat_id, len(rows), len(watch_due_ids))
    try:
        reply = await run_agent(chat_id, user_msg, history=[], max_hops=10)
    except Exception as exc:
        logger.exception("weekly review agent failed")
        reply = f"⚠️ 주간 골 리뷰 중 오류: {exc}"
    for r in rows:
        db.mark_goal_reviewed(r["id"])
    for gid in watch_due_ids:
        db.mark_goal_watch_run(gid)
    if bot:
        header = "📋 이번 주 골 리뷰\n\n"
        text = header + reply
        for i in range(0, len(text), 4000):
            await bot.send_message(chat_id=chat_id, text=text[i : i + 4000])


# ---------------- daily rhythm: morning briefing + evening reflection ----------------


async def run_morning_briefing(chat_id: int) -> None:
    """v10: deterministic, factual, score-free morning briefing. 흡수 대상 —
    evening_preview / midday_checkin / birthday_solo / inbox_triage. 하나의
    아침 메시지로 끝. 점수·% 0. 응원·기분 질문 0. 빈 섹션은 생략."""
    if _app is None or _app.bot is None:
        return
    today_local = datetime.now(TZ).date()
    today_iso = today_local.isoformat()
    state = db.get_daily_state(chat_id, today_iso)
    if state and state["briefing_sent"]:
        return
    if _is_quiet_now(chat_id):
        return
    # 48h 침묵한 사용자 (휴가·아픔·바쁨) → 봇도 침묵
    if not _user_was_active(chat_id, hours=48):
        logger.info("briefing: chat %s silent 48h+, skip", chat_id)
        return

    now_local = datetime.now(TZ)
    end_today = now_local.replace(hour=23, minute=59, second=59)
    today_items = await _merge_schedule(
        chat_id, now_local.astimezone(timezone.utc), end_today.astimezone(timezone.utc))

    # 임박 골 — v10 D-14 → D-3 (D-14는 noise)
    goals_imminent = [
        g for g in db.goals_due_within(chat_id, days=3)
        if g["target_date_local"]
    ]

    # 오늘 important_date 매칭 (음력 인식)
    birthdays_today: List[str] = []
    person_ids_today: List[int] = []
    for p in db.list_people(chat_id):
        for d in json.loads(p["important_dates_json"] or "[]"):
            try:
                if _date_matches_today(d, today_local):
                    tag = " (음력)" if d.get("is_lunar") else ""
                    role = f" {p['role']}" if p["role"] else ""
                    label = d.get("label") or "기념일"
                    birthdays_today.append(f"{p['name']}{role} {label}{tag}")
                    if p["id"] not in person_ids_today:
                        person_ids_today.append(p["id"])
            except Exception:
                pass
    db.mark_birthdays_today(chat_id, today_iso, person_ids_today)

    # 잊을 만한 거 (predictive)
    blindspots: List[str] = []
    if today_items and not _toggle_off_local(chat_id, "predictive_nudge_enabled"):
        try:
            blindspots = await _predict_blindspots(chat_id, today_items)
        except Exception:
            logger.exception("briefing predict failed (non-fatal)")

    # Inline inbox triage — Gmail 연결돼 있을 때만 (별도 cron 흡수)
    inbox_starred: List[Dict] = []
    inbox_archived = 0
    if db.get_oauth_token(chat_id, "google"):
        try:
            inbox_starred, inbox_archived = await _inline_inbox_triage(chat_id, cap=10)
        except Exception:
            logger.exception("briefing inbox triage failed (non-fatal)")

    # 결정형 텍스트 빌드 — 점수·응원·기분질문 0
    lines = [f"☀️ {today_local.strftime('%m월 %d일 (%a)')}", ""]
    if today_items:
        lines.append("오늘")
        for it in today_items[:6]:
            when = it["when_utc"].astimezone(TZ).strftime("%H:%M")
            loc = f" @{it['location']}" if it.get("location") else ""
            lines.append(f"  • {when} {it['title']}{loc}")
        lines.append("")
    if blindspots:
        lines.append("잊을 만한 거")
        for b in blindspots:
            lines.append(f"  • {b}")
        lines.append("")
    if inbox_starred or inbox_archived:
        lines.append("📬 메일")
        for m in inbox_starred[:5]:
            subj = (m.get("subject") or "")[:60]
            lines.append(f"  • ⭐ {subj}")
        if inbox_archived:
            lines.append(f"  • {inbox_archived}건 자동 정리")
        lines.append("")
    for b in birthdays_today[:3]:
        lines.append(f"🎂 {b}")
    for g in goals_imminent[:3]:
        d = _days_until(g["target_date_local"])
        lines.append(f"🎯 {g['title']} D-{d}")
    # v12 W5: 식어가는 관계 1줄 (opt-in via crm_pulse_enabled)
    if _toggle_on_local(chat_id, "crm_pulse_enabled"):
        try:
            crm_line = crm.briefing_line(chat_id)
            if crm_line:
                lines.append(crm_line)
        except Exception:
            logger.exception("crm briefing line failed (non-fatal)")
    # v12 W2: 구독 가격 변동 1-2줄 (opt-in via finance_scan_enabled)
    if _toggle_on_local(chat_id, "finance_scan_enabled"):
        try:
            for fin_line in finance.briefing_lines(chat_id):
                lines.append(fin_line)
        except Exception:
            logger.exception("finance briefing failed (non-fatal)")

    while lines and not lines[-1]:
        lines.pop()
    # 신호 0건이면 침묵
    if len(lines) <= 2:
        return
    await _app.bot.send_message(chat_id=chat_id, text="\n".join(lines))
    db.mark_briefing_sent(chat_id, today_iso)
    db.record_nudge(chat_id, "morning_briefing")


async def _inline_inbox_triage(chat_id: int, cap: int = 10) -> tuple:
    """run_inbox_triage 경량판 — (starred, archived_count) 반환. briefing이 호출."""
    try:
        ids = await gmail_mod.list_messages(
            chat_id, query="is:unread newer_than:1d", max_results=cap * 2)
    except Exception:
        return ([], 0)
    starred: List[Dict] = []
    archived = 0
    for mid in ids[:cap]:
        try:
            msg = await gmail_mod.get_message(chat_id, mid, body_max_chars=1200)
        except Exception:
            continue
        body = (msg.get("body") or msg.get("snippet") or "")[:1000]
        sender = msg.get("from") or ""
        subject = msg.get("subject") or ""
        prompt = INBOX_TRIAGE_PROMPT.format(
            sender=sender, subject=subject, body=body)
        try:
            data = await chat_completion(
                [{"role": "user", "content": prompt}],
                tools=None, chat_id=chat_id, kind="classify_content",
                max_tokens=140,
            )
            content = (data["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            continue
        import re as _re
        m = _re.search(r"\{[\s\S]*\}", content)
        if not m:
            continue
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            continue
        cat = parsed.get("category", "")
        if cat in ("important", "draft_reply"):
            starred.append({"id": mid, "subject": subject, "from": sender,
                              "summary": (parsed.get("summary") or "")[:80]})
        elif cat in ("archive", "noise"):
            try:
                await gmail_mod.archive(chat_id, mid)
                archived += 1
            except Exception:
                pass
    return (starred, archived)
    db.record_nudge(chat_id, "morning_briefing")


def _toggle_off_local(chat_id: int, key: str) -> bool:
    for row in db.list_facts(chat_id):
        if row["key"] == key and row["value"].strip().lower() in {"false", "off", "0", "no"}:
            return True
    return False


REFLECTION_OPENERS = [
    "오늘 어땠어? 한 줄로라도 좋아.",
    "🌙 하루 어떻게 흘렀는지 한마디만.",
    "오늘 가장 기억에 남는 순간은?",
    "🌙 오늘 컨디션·기분 어땠어?",
    "오늘 잘한 거 하나 + 아쉬운 거 하나만.",
]


async def run_evening_reflection(chat_id: int) -> None:
    if _app is None or _app.bot is None:
        return
    today_local = datetime.now(TZ).date()
    state = db.get_daily_state(chat_id, today_local.isoformat())
    if state and state["reflection_prompted"]:
        return  # already asked today
    import random
    opener = random.choice(REFLECTION_OPENERS)
    await _app.bot.send_message(chat_id=chat_id, text=opener)
    db.mark_reflection_prompted(chat_id, today_local.isoformat())
    db.record_nudge(chat_id, "evening_reflection")


async def run_daily_imminent_check(chat_id: int) -> None:
    """Daily 08:00 — three responsibilities in one cron:
      1. Push the legacy D-7 imminent goal digest (unchanged).
      2. Fire one-off D-30 / D-14 / D-3 / D-1 milestone alerts (idempotent via
         goal_milestones_sent table) so long-horizon goals get earlier signal.
      3. Skip both if user opted out via fact goal_review_enabled=false (caught
         upstream by scheduler._goal_review_disabled).
    """
    bot = _app.bot if _app else None
    if not bot:
        return
    today_local = datetime.now(TZ).date()
    open_goals = db.list_goals(chat_id, status="open")

    # ----- 1. legacy D-7 digest -----
    soon = [r for r in open_goals if r["target_date_local"]
            and 0 <= _days_until(r["target_date_local"]) <= 7]
    if soon:
        lines = ["📌 임박한 골 (D-7 이내)"]
        for r in soon:
            d = _days_until(r["target_date_local"])
            lines.append(f"• D-{d} · #{r['id']} {r['title']}")
        await bot.send_message(chat_id=chat_id, text="\n".join(lines))

    # ----- 2. milestone alerts -----
    if _toggle_off_local(chat_id, "goal_milestone_enabled"):
        return
    milestones = (30, 14, 3, 1)
    for r in open_goals:
        td = r["target_date_local"]
        if not td:
            continue
        try:
            target = datetime.fromisoformat(td).date()
        except ValueError:
            continue
        days_left = (target - today_local).days
        if days_left not in milestones:
            continue
        if db.milestone_already_sent(r["id"], days_left):
            continue
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ 완료", callback_data=f"act:goal_done:{r['id']}"),
            InlineKeyboardButton("⏸ 보류", callback_data=f"act:goal_pause:{r['id']}"),
        ]])
        text = f"🎯 D-{days_left} · #{r['id']} {r['title']}"
        if r["why"]:
            text += f"\n  ↳ {r['why'][:80]}"
        try:
            await bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)
            db.record_milestone_sent(chat_id, r["id"], days_left)
        except Exception:
            logger.exception("milestone send failed for goal %s D-%s", r["id"], days_left)


# ---------------- pre-emptive runners (gmail scan, evening preview, birthday, midday, leave-by) ----------------


# Short-lived in-memory candidates from gmail event scan — token → payload.
_pending_mail_cards: Dict[str, Dict] = {}


EMAIL_EVENT_EXTRACT_PROMPT = (
    "다음 이메일 본문에서 일정 한 건을 한 줄 JSON으로 뽑아.\n"
    'JSON: {"title": "짧은 한국어 제목 ≤30자", "when_local": "YYYY-MM-DDTHH:MM:SS (사용자 KST)", '
    '"location": "장소(없으면 빈 문자열)", "source_summary": "≤80자 요약"}\n'
    "정확한 날짜·시간이 본문에 없으면 빈 JSON {} 만 출력. JSON만, 그 외 금지."
)


async def _extract_event_from_email_text(body: str, *, hint_now_kst: str) -> Optional[Dict]:
    """Micro-LLM call: best-effort structured event extraction from one email body.
    Returns dict on success, None on any failure / empty result."""
    if not body or not body.strip():
        return None
    prompt = (
        EMAIL_EVENT_EXTRACT_PROMPT
        + f"\n현재 시각(KST): {hint_now_kst}\n\n본문:\n"
        + body[:2400]
    )
    try:
        data = await chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None,
            chat_id=0,
            kind="mail_extract",
            max_tokens=240,
        )
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception:
        logger.exception("mail extract LLM call failed")
        return None
    import re as _re
    blob = _re.sub(r"^\s*```(?:json)?\s*", "", content.strip(), flags=_re.IGNORECASE)
    blob = _re.sub(r"\s*```\s*$", "", blob)
    m = _re.search(r"\{[\s\S]*\}", blob)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return None
    if not parsed or not parsed.get("title") or not parsed.get("when_local"):
        return None
    return parsed


_NOISY_EMAIL_KEYWORDS = ("광고", "홍보", "스팸", "마케팅", "unsubscribe", "프로모션", "이벤트 안내")


def _gmail_auto_add_rule_match(chat_id: int, msg: Dict, cls: Dict) -> Optional[Dict]:
    """Check if any active gmail_auto_add_event rule matches this message + classifier result.
    Returns the matched rule dict (or None). All rules conjunctively check sender + min_conf."""
    sender = (msg.get("from") or "").lower()
    conf = float(cls.get("confidence", 0.0))
    for r in db.list_auto_rules(chat_id):
        if r["rule_kind"] != "gmail_auto_add_event":
            continue
        try:
            cond = json.loads(r["condition_json"])
        except Exception:
            continue
        sp = (cond.get("sender_pattern") or "").lower().strip()
        if sp and sp not in sender:
            continue
        if conf < float(cond.get("min_confidence", 0.85)):
            continue
        return cond
    return None


async def run_gmail_event_scan(chat_id: int) -> None:
    """Poll Gmail for recently received messages that look like events; surface
    if _is_quiet_now(chat_id):
        return
    a card with [✅ 추가] / [✏️ 수정] / [❌ 아님] buttons. Every msg_id is dedup'd
    in processed_gmail_msg_ids so we never re-offer the same message."""
    if _app is None or _app.bot is None:
        return
    if _toggle_off_local(chat_id, "gmail_event_scan_enabled"):
        return
    if not db.get_oauth_token(chat_id, "google"):
        return
    try:
        ids = await gmail_mod.list_messages(chat_id, query="newer_than:1d", max_results=20)
    except Exception:
        logger.exception("gmail_event_scan: list_messages failed")
        return
    for mid in ids:
        if db.is_gmail_processed(chat_id, mid):
            continue
        try:
            msg = await gmail_mod.get_message(chat_id, mid, body_max_chars=4000)
        except Exception:
            logger.exception("gmail_event_scan: get_message failed for %s", mid)
            continue
        body = msg.get("body") or msg.get("snippet") or ""
        subject = msg.get("subject") or ""
        combined = (subject + "\n" + body).strip()
        # Cheap noise filter.
        low = combined.lower()
        if any(kw in low for kw in _NOISY_EMAIL_KEYWORDS):
            db.mark_gmail_processed(chat_id, mid, "skipped")
            continue
        try:
            cls = await transcribe.classify_content(combined, hint="email")
        except Exception:
            logger.exception("gmail_event_scan: classify failed")
            db.mark_gmail_processed(chat_id, mid, "skipped")
            continue
        if cls.get("kind") != "event" or cls.get("confidence", 0.0) < 0.8:
            db.mark_gmail_processed(chat_id, mid, "low_conf")
            continue
        now_kst = datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S")
        candidate = await _extract_event_from_email_text(combined, hint_now_kst=now_kst)
        if not candidate:
            db.mark_gmail_processed(chat_id, mid, "low_conf")
            continue
        # v4: auto-rule check (gmail_auto_add_event). If any rule matches the
        # message + confidence + sender pattern, skip the card and execute now.
        rule_auto = _gmail_auto_add_rule_match(chat_id, msg, cls)
        if rule_auto and not _toggle_off_local(chat_id, "auto_rules_enabled"):
            try:
                when_utc = _parse_local_iso(candidate["when_local"])
                eid = db.add_event(
                    chat_id, candidate["title"][:80], when_utc,
                    notes=candidate.get("source_summary"),
                    remind_lead_minutes=30,
                    location=(candidate.get("location") or "").strip() or None,
                )
                db.mark_gmail_processed(chat_id, mid, "added")
                db.log_agent_action(
                    chat_id, "event_auto_add",
                    summary=f"메일 → 일정 자동 추가 · {candidate['title'][:40]}",
                    payload={"gmail_id": mid, "event_id": eid,
                             "when_local": candidate["when_local"], "from": msg.get("from")},
                    reversible={"kind": "event_delete", "args": {"event_id": eid}},
                )
                row = db.get_event(eid)
                if row:
                    scheduler.schedule_for(row)
                continue
            except Exception:
                logger.exception("auto add_event failed; falling back to card")
        # Build card.
        token = secrets.token_urlsafe(8)
        _pending_mail_cards[token] = {
            "chat_id": chat_id,
            "msg_id": mid,
            "title": candidate.get("title", "")[:80],
            "when_local": candidate.get("when_local"),
            "location": (candidate.get("location") or "").strip() or None,
            "source_summary": (candidate.get("source_summary") or "")[:160],
            "subject": subject,
        }
        body_lines = [
            "📬 메일에서 일정 후보를 찾았어",
            f"• 제목: {candidate.get('title', '')}",
            f"• 시각: {candidate.get('when_local')}",
        ]
        if candidate.get("location"):
            body_lines.append(f"• 장소: {candidate['location']}")
        if candidate.get("source_summary"):
            body_lines.append(f"• 메일: {candidate['source_summary']}")
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ 추가", callback_data=f"act:mail_add:{token}"),
            InlineKeyboardButton("❌ 아님", callback_data=f"act:mail_skip:{token}"),
        ]])
        try:
            await _app.bot.send_message(chat_id=chat_id, text="\n".join(body_lines), reply_markup=kb)
            db.mark_gmail_processed(chat_id, mid, "offered")
        except Exception:
            logger.exception("gmail_event_scan: send card failed")


# ----- Leave-by: compute ETA + schedule the one-shot APScheduler job -----


async def _resolve_to_coord(text: str) -> Optional[Dict]:
    """Try address lookup first, then keyword search. Returns {x, y} or None."""
    if not text or not text.strip():
        return None
    try:
        addr = await external.kakao_address_to_coord(text)
        if addr.get("ok") and addr.get("match"):
            m = addr["match"]
            return {"x": m["x"], "y": m["y"]}
    except Exception:
        pass
    try:
        kw = await external.kakao_local_keyword(query=text, size=1)
        if kw.get("ok") and kw.get("places"):
            p = kw["places"][0]
            return {"x": float(p["x"]), "y": float(p["y"])}
    except Exception:
        pass
    return None


# v9: travel mode auto-detection

_TRAVEL_KEYWORDS = ("출장", "여행", "휴가", "trip", "vacation", "business trip",
                     "신혼여행", "워크샵", "워크숍")
_TRAVEL_CITIES = ("도쿄", "오사카", "후쿠오카", "교토", "삿포로", "오키나와",
                   "타이베이", "방콕", "치앙마이", "발리", "보라카이",
                   "다낭", "하노이", "호치민", "싱가포르",
                   "뉴욕", "LA", "샌프란시스코", "보스턴", "시애틀",
                   "런던", "파리", "로마", "바르셀로나", "프라하",
                   "부산", "제주", "강릉", "여수")


async def _maybe_detect_travel(chat_id: int, event_row) -> None:
    """Best-effort: classify a new event as travel-related. If yes, register a
    travel_period covering the event's local date + nearby days."""
    title = (event_row["title"] or "").strip().lower()
    notes = (event_row["notes"] or "").lower()
    location = (event_row["location"] or "").strip()
    text = title + " " + notes + " " + location.lower()
    matched_kw = any(kw in text for kw in _TRAVEL_KEYWORDS)
    matched_city = next((c for c in _TRAVEL_CITIES if c in title or c in location), None)
    if not matched_kw and not matched_city:
        return
    # Skip if already in an active travel period
    try:
        when_local = datetime.fromisoformat(event_row["when_utc"]).astimezone(TZ).date()
    except Exception:
        return
    today_iso = when_local.isoformat()
    if db.active_travel_for(chat_id, today_iso):
        return
    destination = matched_city or location or (
        # try to extract place name from title — pick longest non-keyword token
        max((t for t in title.split() if t not in _TRAVEL_KEYWORDS), default="여행")
    )
    destination = destination.strip()[:40] or "여행"
    # Default window: event day ± 1 (will be refined by user)
    start = when_local
    end = when_local + timedelta(days=2)
    tid = db.add_travel_period(
        chat_id, destination, start.isoformat(), end.isoformat(),
        detected_from_event_id=event_row["id"],
        notes=f"auto-detected from event #{event_row['id']}",
    )
    if _app and _app.bot:
        try:
            await _app.bot.send_message(
                chat_id=chat_id,
                text=(f"🛫 여행 감지 — '{event_row['title']}' → {destination}\n"
                      f"  • 기간 {start.isoformat()} ~ {end.isoformat()}\n"
                      "  • 그 기간에는 날씨/leave-by가 destination 기준으로 전환\n"
                      f"  • 수정/삭제: /travel"),
            )
            db.log_agent_action(
                chat_id, "travel_detect",
                summary=f"{destination} 여행 자동 등록 #{tid}",
                payload={"travel_id": tid, "event_id": event_row["id"]},
            )
        except Exception:
            logger.exception("travel detect notify failed")


async def cmd_travel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/travel` 목록 + 활성 · `/travel add <dest> <YYYY-MM-DD> <YYYY-MM-DD>` ·
    `/travel delete <id>`."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        today_iso = datetime.now(TZ).date().isoformat()
        active = db.active_travel_for(chat_id, today_iso)
        rows = db.list_travel_periods(chat_id)
        lines = ["🛫 여행 기록"]
        if active:
            lines.append(f"  ▶️ 현재: {active['destination']} ({active['start_date_local']}~{active['end_date_local']})")
        else:
            lines.append("  ▶️ 현재: 여행 중 아님")
        lines.append("")
        for r in rows[:10]:
            lines.append(f"  #{r['id']} {r['destination']}: {r['start_date_local']}~{r['end_date_local']}")
        if not rows:
            lines.append("  (등록 없음)")
        await update.message.reply_text("\n".join(lines))
        return
    op = args[0].lower()
    if op == "add" and len(args) >= 4:
        dest = args[1]
        start = args[2]; end = args[3]
        try:
            datetime.fromisoformat(start); datetime.fromisoformat(end)
        except Exception:
            await update.message.reply_text("날짜는 YYYY-MM-DD")
            return
        tid = db.add_travel_period(chat_id, dest, start, end)
        await update.message.reply_text(f"🛫 #{tid} {dest} {start}~{end} 등록")
        return
    if op == "delete" and len(args) >= 2:
        try:
            tid = int(args[1])
        except ValueError:
            await update.message.reply_text("travel_id는 숫자")
            return
        ok = db.delete_travel_period(chat_id, tid)
        await update.message.reply_text(f"🗑 #{tid} 삭제" if ok else "없음")
        return
    await update.message.reply_text(
        "사용:\n  /travel\n  /travel add 도쿄 2026-07-10 2026-07-13\n  /travel delete <id>")


async def _try_schedule_leave_by(chat_id: int, event_row) -> None:
    """Best-effort: resolve origin (home_address fact) + dest (event.location),
    compute ETA via Kakao driving, schedule one-shot leave-by job. Silent on
    any failure so add_event never appears to fail because of nudge plumbing."""
    if _toggle_off_local(chat_id, "leave_by_enabled"):
        return
    location = event_row["location"] if event_row else None
    if not location:
        return
    # Resolve origin: home_address fact (mandatory — no fallback to last_known).
    home = None
    for r in db.list_facts(chat_id):
        if r["key"] == "home_address":
            home = r["value"]
            break
    if not home:
        logger.debug("leave-by: no home_address fact for chat %s — silent skip", chat_id)
        return
    o = await _resolve_to_coord(home)
    d = await _resolve_to_coord(location)
    if not o or not d:
        return
    try:
        route = await external.kakao_directions(
            origin_x=o["x"], origin_y=o["y"], dest_x=d["x"], dest_y=d["y"],
        )
    except Exception:
        logger.exception("leave-by: kakao_directions failed for event %s", event_row["id"])
        return
    if not route.get("ok"):
        return
    summary = route.get("summary") or {}
    duration_s = summary.get("duration_s")
    if not duration_s:
        return
    try:
        buf_min = int(_fact_value_local(chat_id, "leave_by_buffer_min") or "5")
    except ValueError:
        buf_min = 5
    eta_min = int(duration_s // 60) + max(0, buf_min)
    when_utc = datetime.fromisoformat(event_row["when_utc"])
    leave_at_utc = when_utc - timedelta(minutes=eta_min)
    if leave_at_utc <= datetime.now(timezone.utc):
        return
    job_id = scheduler.schedule_leave_by(event_row["id"], leave_at_utc)
    if job_id:
        db.set_event_leave_by_job(event_row["id"], job_id)
        logger.info("leave-by armed event=%s eta=%dmin fire=%s",
                    event_row["id"], eta_min, leave_at_utc.isoformat())


def _fact_value_local(chat_id: int, key: str) -> Optional[str]:
    for row in db.list_facts(chat_id):
        if row["key"] == key:
            return row["value"]
    return None


def _is_quiet_now(chat_id: int) -> bool:
    """True if current KST time falls in user's quiet window (default 23:00-07:00).
    Used by nudge runners to self-suppress. Critical alerts (mission done,
    D-0 reminders) ignore this."""
    if _toggle_off_local(chat_id, "quiet_hours_enabled"):
        return False
    start = _fact_value_local(chat_id, "quiet_hours_start") or "23:00"
    end = _fact_value_local(chat_id, "quiet_hours_end") or "07:00"
    try:
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
    except Exception:
        return False
    now = datetime.now(TZ)
    cur_min = now.hour * 60 + now.minute
    s_min = sh * 60 + sm
    e_min = eh * 60 + em
    if s_min <= e_min:
        return s_min <= cur_min < e_min
    # wrap (23 → 07)
    return cur_min >= s_min or cur_min < e_min


def _user_was_active(chat_id: int, hours: int = 48) -> bool:
    """v10: 사용자가 최근 N시간 안에 한 마디라도 했으면 True. nudge cron이
    이 가드를 먼저 보고 묵음 결정 — 휴가·아픈 날·바쁜 시즌엔 봇도 묵음."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with db._conn() as c:
        row = c.execute(
            "SELECT 1 FROM chat_log WHERE chat_id=? AND role='user' AND created_at>=? LIMIT 1",
            (chat_id, cutoff)).fetchone()
        return row is not None


# v12 W3: 짧은 메시지를 가까운 일정과 연결
_CONTEXT_ACTION_KEYWORDS = ("도착", "출발", "끝남", "끝났", "왔어", "갔어",
                              "감", "늦", "지각", "안 갔", "안갔", "출근",
                              "퇴근", "들어옴", "나감")


def _infer_short_message_context(chat_id: int, message: str) -> str:
    """짧은 메시지(≤12자) + 액션 키워드면 ±2시간 가장 가까운 일정을 한 줄
    [현재 가까운 일정 …]로 반환. user_text 앞에 prepend 돼 LLM 답변에 사용됨.
    매칭 안 되면 빈 문자열 — 일반 답변 흐름."""
    msg = (message or "").strip()
    if len(msg) > 12 or len(msg) < 1:
        return ""
    if not any(k in msg for k in _CONTEXT_ACTION_KEYWORDS):
        return ""
    now = datetime.now(timezone.utc)
    try:
        items = db.list_events(
            chat_id, now - timedelta(hours=2), now + timedelta(hours=2))
    except Exception:
        return ""
    if not items:
        return ""
    nearest = min(items,
                   key=lambda e: abs(datetime.fromisoformat(e["when_utc"]) - now))
    try:
        when_local = datetime.fromisoformat(nearest["when_utc"]).astimezone(TZ)
    except Exception:
        return ""
    title = (nearest["title"] or "")[:40]
    loc = nearest["location"] if "location" in nearest.keys() and nearest["location"] else ""
    loc_str = f" @{loc}" if loc else ""
    return f"[현재 가까운 일정 — id={nearest['id']}, {when_local.strftime('%H:%M')} {title}{loc_str}]"


# event_id → utc timestamp when the leave-by fired; consumed by run_late_check.
_leave_by_sent_at: Dict[int, datetime] = {}


async def _send_leave_by_for_event(event_id: int) -> None:
    if _app is None or _app.bot is None:
        return
    row = db.get_event(event_id)
    if not row:
        return
    when_local = datetime.fromisoformat(row["when_utc"]).astimezone(TZ)
    delta_min = max(0, int((when_local - datetime.now(TZ)).total_seconds() // 60))
    text = (
        f"🚗 지금 나가\n"
        f"• {row['title']}"
    )
    if row["location"]:
        text += f"\n• 장소: {row['location']}"
    text += f"\n• 도착 목표: {when_local.strftime('%H:%M')} ({delta_min}분 뒤)"
    try:
        await _app.bot.send_message(chat_id=row["chat_id"], text=text)
        # Arm the 8-min late check (a no-op if user moves / replies in time).
        sent_at = datetime.now(timezone.utc)
        _leave_by_sent_at[event_id] = sent_at
        scheduler.schedule_late_check(event_id, sent_at + timedelta(minutes=8))
    except Exception:
        logger.exception("leave-by send failed for event %s", event_id)


# Short-lived in-memory map for late-card buttons.
_pending_late_cards: Dict[str, Dict] = {}


async def run_late_check(event_id: int) -> None:
    """8 min after a leave-by fired: if the user hasn't sent any message since
    (suggesting they're not actually moving), surface a 'looks like you're
    running late — want me to tell anyone?' card."""
    if _app is None or _app.bot is None:
        return
    row = db.get_event(event_id)
    if not row:
        return
    sent_at = _leave_by_sent_at.pop(event_id, None)
    if not sent_at:
        return
    chat_id = row["chat_id"]
    # Activity check — any user message since the leave-by went out?
    with db._conn() as c:
        recent = c.execute(
            "SELECT 1 FROM chat_log WHERE chat_id=? AND role='user' "
            "AND created_at>? LIMIT 1",
            (chat_id, sent_at.isoformat()),
        ).fetchone()
    if recent:
        return  # user is engaging — assume they're on the move
    # Don't nudge for events long past.
    when_utc = datetime.fromisoformat(row["when_utc"])
    if (datetime.now(timezone.utc) - when_utc).total_seconds() > 30 * 60:
        return
    # Best-effort: look up GCal organizer email so we can offer "mail them" button.
    organizer = None
    if row["gcal_event_id"] and db.get_oauth_token(chat_id, "google"):
        try:
            ev = await gcal._api(
                chat_id, "GET", f"/calendars/primary/events/{row['gcal_event_id']}")
            organizer = (ev.get("organizer") or {}).get("email")
        except Exception:
            organizer = None

    token = secrets.token_urlsafe(8)
    _pending_late_cards[token] = {
        "chat_id": chat_id, "event_id": event_id,
        "event_title": row["title"], "organizer_email": organizer,
    }
    btn_row = [InlineKeyboardButton("📋 메시지 초안", callback_data=f"act:late_msg:{token}")]
    if organizer:
        btn_row.append(InlineKeyboardButton("📨 그분께 메일",
                                              callback_data=f"act:late_mail:{token}"))
    kb = InlineKeyboardMarkup([
        btn_row,
        [InlineKeyboardButton("👀 안 늦었어", callback_data=f"act:late_dismiss:{token}")],
    ])
    text = f"🚗 {row['title']} — 늦어지는 것 같아. 알릴까?"
    try:
        await _app.bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)
        db.record_nudge(chat_id, "late_check", str(event_id))
    except Exception:
        logger.exception("late check send failed for event %s", event_id)


async def run_leave_by_recompute(chat_id: int) -> None:
    """Daily 03:30 — re-arm leave-by jobs for upcoming 48h. ETA may have shifted
    (traffic, location change). Idempotent: schedule_leave_by uses replace_existing."""
    if _toggle_off_local(chat_id, "leave_by_enabled"):
        return
    for row in db.events_with_location_in_window(chat_id, hours=48):
        try:
            await _try_schedule_leave_by(chat_id, row)
        except Exception:
            logger.exception("leave-by recompute failed for event %s", row["id"])


# ----- Evening preview (tomorrow's first event + weather + holiday, terse) -----


PREDICTIVE_PROMPT = (
    "사용자의 내일 일정 + persona + 최근 행동 패턴을 보고 "
    "*잊을 만한 한 두 가지*를 짧게 예측해. 한국어, 한 줄당 ≤40자, 최대 3개. "
    "확실하지 않으면 빈 배열. JSON 배열만 출력: [\"등산화 챙겨\", \"축의금 5만\"].\n\n"
    "내일 일정:\n{events}\n\n"
    "persona snapshot:\n{persona}\n\n"
    "최근 30일 지출 카테고리:\n{spend}\n\n"
    "감지된 루틴:\n{routines}"
)


async def _predict_blindspots(chat_id: int, items: list) -> list:
    """오늘/내일 이벤트 리스트 기준 사용자가 잊을 만한 것 ≤3개. 실패 시 빈 리스트."""
    if not items:
        return []
    persona = db.get_latest_persona(chat_id)
    persona_md = (persona["content_md"][:800] if persona else "(없음)")
    spend = db.summarize_expenses(chat_id, days=30)
    spend_summary = ", ".join(f"{c[0]}: ₩{c[1]:,}" for c in spend.get("by_category", [])[:5]) or "(없음)"
    try:
        rts = routines.detect_all(chat_id)
        routines_md = json.dumps(rts, ensure_ascii=False)[:800]
    except Exception:
        routines_md = "(없음)"
    events_md = "\n".join(
        f"- {it['when_utc'].astimezone(TZ).strftime('%H:%M')} {it['title']}"
        + (f" @{it['location']}" if it.get("location") else "")
        for it in items[:6])
    prompt = PREDICTIVE_PROMPT.format(
        events=events_md, persona=persona_md,
        spend=spend_summary, routines=routines_md,
    )
    try:
        data = await chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="mood_classify",  # haiku tier (cheap)
            max_tokens=200,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("predictive nudge LLM failed")
        return []
    import re as _re
    m = _re.search(r"\[[\s\S]*\]", content)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(x).strip()[:50] for x in parsed if str(x).strip()][:3]


async def run_evening_preview(chat_id: int) -> None:
    """22:00 KST — single-line preview of tomorrow. Silent unless tomorrow has
    if _is_quiet_now(chat_id):
        return
    an event, is a Korean holiday, or weather flags rain/snow > 30%."""
    if _app is None or _app.bot is None:
        return
    if _toggle_off_local(chat_id, "weather_preview_enabled"):
        return
    now_local = datetime.now(TZ)
    tomorrow_local = (now_local + timedelta(days=1)).date()
    tomorrow_start = datetime.combine(tomorrow_local, datetime.min.time()).replace(tzinfo=TZ)
    tomorrow_end = tomorrow_start + timedelta(days=1)
    items = await _merge_schedule(
        chat_id, tomorrow_start.astimezone(timezone.utc), tomorrow_end.astimezone(timezone.utc))
    holiday = korean_calendar.is_holiday(tomorrow_local.isoformat())

    # Choose a weather location: first event with location, else default fact, else skip.
    loc = None
    for it in items:
        if it.get("location"):
            loc = it["location"]
            break
    if not loc:
        loc = _fact_value_local(chat_id, "default_weather_location") \
              or _fact_value_local(chat_id, "home_address")
    weather_info = None
    if loc:
        try:
            w = await weather_mod.weather(loc, days=2)
            if w.get("ok"):
                # Find tomorrow's max rain probability in samples.
                td = tomorrow_local.isoformat()
                rain_max = 0
                tmin, tmax = None, None
                for h in w.get("hourly", []):
                    if h.get("time", "").startswith(td):
                        if h.get("rain_pct") is not None and h["rain_pct"] > rain_max:
                            rain_max = h["rain_pct"]
                        if h.get("temp") is not None:
                            tmin = h["temp"] if tmin is None else min(tmin, h["temp"])
                            tmax = h["temp"] if tmax is None else max(tmax, h["temp"])
                weather_info = {
                    "location": w.get("location") or loc,
                    "rain_pct": rain_max,
                    "tmin": tmin, "tmax": tmax,
                }
        except Exception:
            logger.exception("evening_preview: weather lookup failed")

    # v6: also surface 24절기 (입추, 동지 etc.) if tomorrow is one.
    solar_term = lunar.solar_term_on(tomorrow_local.isoformat())
    # Heat / cold extremes from weather_info: 30+℃ → polite warning, ≤-5℃ → 한파
    heat = bool(weather_info and (weather_info.get("tmax") or 0) >= 30)
    cold = bool(weather_info and weather_info.get("tmin") is not None and weather_info["tmin"] <= -5)

    # Silence rule — skip if no signal.
    has_event = bool(items)
    is_holiday = holiday["is_holiday"]
    has_rain = bool(weather_info and (weather_info["rain_pct"] or 0) >= 30)
    if not (has_event or is_holiday or has_rain or solar_term or heat or cold):
        return

    parts = ["🌙 내일 미리보기"]
    if is_holiday:
        parts.append(f"  • 공휴일: {holiday['name']}")
    if solar_term:
        parts.append(f"  • 절기: {solar_term['name']}")
    if heat:
        parts.append("  • ⚠️ 폭염 — 물 자주 챙겨")
    if cold:
        parts.append("  • ⚠️ 한파 — 두꺼운 옷")
    if items:
        first = items[0]
        when = first["when_utc"].astimezone(TZ).strftime("%H:%M")
        loc_str = f" @{first['location']}" if first.get("location") else ""
        parts.append(f"  • 첫 일정 {when} {first['title']}{loc_str}")
        if len(items) > 1:
            parts.append(f"  • 그 외 {len(items) - 1}건")
    if weather_info:
        bits = [weather_info["location"]]
        if weather_info["tmin"] is not None:
            bits.append(f"{weather_info['tmin']:.0f}~{weather_info['tmax']:.0f}℃")
        if has_rain:
            bits.append(f"비/눈 {weather_info['rain_pct']}% — 우산 챙겨")
        parts.append("  • " + ", ".join(bits))
    # v7: predictive nudge — what might the user forget?
    if items and not _toggle_off_local(chat_id, "predictive_nudge_enabled"):
        try:
            blind = await _predict_blindspots(chat_id, items)
            if blind:
                parts.append("")
                parts.append("🔮 잊을 만한 거:")
                for b in blind:
                    parts.append(f"  • {b}")
        except Exception:
            logger.exception("predictive nudge attach failed (non-fatal)")
    try:
        await _app.bot.send_message(chat_id=chat_id, text="\n".join(parts))
    except Exception:
        logger.exception("evening_preview send failed")


# ----- Birthday solo (09:00, opt-in via fact birthday_alert_separate_enabled) -----


async def run_birthday_solo(chat_id: int) -> None:
    if _is_quiet_now(chat_id):
        return
    if _app is None or _app.bot is None:
        return
    if not _toggle_on_local(chat_id, "birthday_alert_separate_enabled"):
        return
    today_local = datetime.now(TZ).date()
    for p in db.list_people(chat_id):
        for d in json.loads(p["important_dates_json"] or "[]"):
            if not _date_matches_today(d, today_local):
                continue
            role_bit = f" ({p['role']})" if p["role"] else ""
            label = d.get("label") or "기념일"
            lunar_tag = " (음력)" if d.get("is_lunar") else ""
            text = f"🎂 오늘 {p['name']}{role_bit} {label}{lunar_tag} — 한마디 어때?"
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("📩 메시지 초안", callback_data=f"act:bday_draft:{p['id']}"),
            ]])
            try:
                await _app.bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)
            except Exception:
                logger.exception("birthday solo send failed for person %s", p["id"])


def _toggle_on_local(chat_id: int, key: str) -> bool:
    v = _fact_value_local(chat_id, key)
    return v is not None and v.strip().lower() in {"true", "on", "1", "yes"}


# ----- Midday check-in (13:00, terse if there are afternoon events) -----


async def run_midday_checkin(chat_id: int) -> None:
    if _is_quiet_now(chat_id):
        return
    if _app is None or _app.bot is None:
        return
    if _toggle_off_local(chat_id, "midday_checkin_enabled"):
        return
    today_local = datetime.now(TZ).date()
    state = db.get_daily_state(chat_id, today_local.isoformat())
    if state and state["midday_checkin_sent"]:
        return
    now_local = datetime.now(TZ)
    end_today = now_local.replace(hour=23, minute=59, second=59)
    items = await _merge_schedule(
        chat_id, now_local.astimezone(timezone.utc), end_today.astimezone(timezone.utc))
    if not items:
        # Silent — no afternoon plans to nudge about.
        return
    first = items[0]
    when = first["when_utc"].astimezone(TZ).strftime("%H:%M")
    title = first["title"][:30]
    loc = f" @{first['location']}" if first.get("location") else ""
    text = f"☀️ {when} {title}{loc} — 준비 됐어?"
    try:
        await _app.bot.send_message(chat_id=chat_id, text=text)
        db.mark_midday_checkin_sent(chat_id, today_local.isoformat())
        db.record_nudge(chat_id, "midday_checkin")
    except Exception:
        logger.exception("midday checkin send failed")


# ---------------- v4 runners: persona, invites, digest, streaks, scorecard, learning, budget ----------------


PERSONA_REBUILD_PROMPT = (
    "다음 사용자 데이터를 보고 비서가 항상 참고할 '인물 요약'을 작성해. "
    "한국어 마크다운. 6 섹션 — 각 섹션 2-3줄, 총 600자 이내. 추측 금지, "
    "데이터에 근거한 것만. 빈 섹션은 '(아직 데이터 부족)'.\n\n"
    "### 1) 루틴 (요일·시간 패턴)\n"
    "### 2) 관심사 (자주 등장 주제)\n"
    "### 3) 관계 (핵심 인물 + 마지막 상호작용)\n"
    "### 4) 감정·스트레스 (리플렉션 톤)\n"
    "### 5) 목표 진행 (활성 골 + 진척)\n"
    "### 6) 말투 (사용자 말투 메모 — 봇이 따라야 할 것)\n\n"
    "데이터:\n{payload}"
)


async def run_persona_rebuild(chat_id: int) -> None:
    """Sun 09:30 — rebuild persona_doc from chat_log + reflections + facts +
    people + spending + habits + routines. Saves new version, surfaces 1-line
    confirmation in chat."""
    if _toggle_off_local(chat_id, "persona_rebuild_enabled"):
        return
    # Gather inputs
    facts = [(f["key"], f["value"]) for f in db.list_facts(chat_id)]
    people = [(p["name"], p["role"] or "", p["last_contact_utc"] or "")
              for p in db.list_people(chat_id)]
    goals_open = [(g["title"], g["target_date_local"] or "", g["why"] or "")
                  for g in db.list_goals(chat_id, status="open")]
    reflections = [(r["date_local"], r["reflection_response"] or "", r["mood_sentiment"] or "")
                   for r in db.recent_reflections(chat_id, days=28)]
    spend = db.summarize_expenses(chat_id, days=30)
    habits_sum = db.summarize_habits(chat_id, days=28)
    routine_hints = routines.detect_all(chat_id)
    chat_log_excerpt = []
    if not _toggle_off_local(chat_id, "persona_chat_scan_enabled"):
        # last 28d of user messages, capped
        cutoff = (datetime.now(timezone.utc) - timedelta(days=28)).isoformat()
        with db._conn() as c:
            for r in c.execute(
                "SELECT content FROM chat_log WHERE chat_id=? AND role='user' AND created_at>=? "
                "ORDER BY created_at DESC LIMIT 80",
                (chat_id, cutoff)):
                chat_log_excerpt.append(r["content"][:200])

    payload = {
        "facts": facts,
        "people": people,
        "open_goals": goals_open,
        "reflections": reflections[:14],
        "spending_30d": {"total_won": spend.get("total_won", 0),
                          "by_category": spend.get("by_category", [])[:6]},
        "habits_28d": habits_sum.get("by_habit", []),
        "detected_routines": routine_hints,
        "recent_user_messages_sample": chat_log_excerpt[:30],
    }
    prompt = PERSONA_REBUILD_PROMPT.format(payload=json.dumps(payload, ensure_ascii=False))
    try:
        data = await chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="persona_rebuild", max_tokens=1200,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("persona rebuild LLM failed")
        return
    if not content:
        return
    sources = {
        "facts_n": len(facts), "people_n": len(people),
        "goals_n": len(goals_open), "reflections_n": len(reflections),
        "chat_log_n": len(chat_log_excerpt),
    }
    db.save_persona(chat_id, content, sources)
    if _app and _app.bot:
        try:
            latest = db.get_latest_persona(chat_id)
            await _app.bot.send_message(
                chat_id=chat_id,
                text=f"🧬 인물 요약 v{latest['version']} 갱신 완료. `/persona`로 열람.",
            )
        except Exception:
            pass


# ---------- Rule engine helpers ----------


def _rule_matches_gcal_invite(rule_cond: Dict, invite: Dict, chat_id: int) -> Optional[str]:
    """Return RSVP response string if rule matches, else None."""
    names = rule_cond.get("from_people_names") or []
    emails = rule_cond.get("from_emails") or []
    response = rule_cond.get("response", "accepted")
    org = (invite.get("organizer_email") or "").lower()
    if emails and any(e.lower() in org for e in emails):
        return response
    if names:
        # match by people table: any registered person whose name appears in invite
        people = db.list_people(chat_id)
        for p in people:
            if p["name"] in names:
                # check if any attendee email roughly matches their notes (best-effort)
                notes = (p["notes"] or "").lower()
                for att in invite.get("attendees", []):
                    if att and notes and att.lower() in notes:
                        return response
                # if organizer name is in subject/title hint
                if (invite.get("summary") or "") and p["name"] in (invite["summary"] or ""):
                    return response
    return None


def _invite_in_protected(invite: Dict, protected_spec: str) -> bool:
    """v10: invite의 시작 시각(KST)이 'HH:MM-HH:MM' 보호 구간 안인가."""
    start_iso = invite.get("start")
    if not start_iso:
        return False
    try:
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(TZ)
    except Exception:
        return False
    try:
        a, b = protected_spec.split("-")
        sh, sm = (int(x) for x in a.split(":"))
        eh, em = (int(x) for x in b.split(":"))
    except Exception:
        return False
    cur = start_dt.hour * 60 + start_dt.minute
    s = sh * 60 + sm
    e = eh * 60 + em
    if s <= e:
        return s <= cur < e
    return cur >= s or cur < e


async def _handle_protected_invite(chat_id: int, invite: Dict, protected: str) -> None:
    """v10: 보호 시간에 들어온 invite — 자동 tentative + 정중한 답장 초안 카드."""
    try:
        await gcal.respond_to_invite(chat_id, invite["id"], "tentative")
    except Exception:
        logger.exception("protected invite tentative failed")
    organizer = (invite.get("organizer_email") or "").strip()
    title = (invite.get("summary") or "(제목 없음)")[:60]
    draft = (
        f"안녕하세요, {title} 잘 받았습니다. "
        f"해당 시간({protected})은 깊은 작업 시간으로 비워두고 있어서, "
        "다른 시간 가능하실까요? 양해 부탁드립니다."
    )
    token = secrets.token_urlsafe(8)
    _pending_late_cards[token] = {
        "chat_id": chat_id, "kind": "protected_decline",
        "to": organizer, "subject": f"Re: {title}", "body": draft,
        "event_id": invite["id"],
    }
    btn_rows = []
    if organizer:
        btn_rows.append([InlineKeyboardButton(
            "📧 이대로 보내기", callback_data=f"act:proto_send:{token}")])
    btn_rows.append([
        InlineKeyboardButton("그래도 수락", callback_data=f"act:proto_accept:{token}"),
        InlineKeyboardButton("거절", callback_data=f"act:proto_decline:{token}"),
    ])
    kb = InlineKeyboardMarkup(btn_rows)
    text = (
        f"🛡 보호 시간({protected}) 침범 invite\n"
        f"• {title}\n"
        f"• 시작 {invite.get('start')}\n"
        f"• 자동 tentative 처리됨"
    )
    if organizer:
        text += f"\n\n초안:\n{draft}"
    try:
        await _app.bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)
    except Exception:
        logger.exception("protected invite card failed")


async def run_gcal_invite_watch(chat_id: int) -> None:
    """Every 30min: pull needsAction invites. Auto-RSVP on matched rules,
    otherwise surface a card with [✅ Yes] [❔ Maybe] [❌ No]."""
    if _toggle_off_local(chat_id, "auto_rules_enabled"):
        return
    if not db.get_oauth_token(chat_id, "google"):
        return
    try:
        pending = await gcal.list_pending_invites(chat_id, days_ahead=14)
    except Exception:
        logger.exception("list_pending_invites failed")
        return
    if not pending:
        return
    rules = [r for r in db.list_auto_rules(chat_id)
             if r["rule_kind"] in ("gcal_auto_rsvp", "gcal_auto_decline")]
    # v10: accept rules win over decline rules (사용자가 명시한 yes 우선)
    rules.sort(key=lambda r: 0 if r["rule_kind"] == "gcal_auto_rsvp" else 1)
    # v10: protected_hours_kst — '10:00-12:00' 같은 fact가 있으면 그 시간에
    # 시작하는 invite는 자동 tentative + 정중한 거절 메일 초안 카드.
    protected = _fact_value_local(chat_id, "protected_hours_kst")
    for inv in pending:
        matched = None
        for r in rules:
            try:
                cond = json.loads(r["condition_json"])
            except Exception:
                continue
            resp = _rule_matches_gcal_invite(cond, inv, chat_id)
            if resp:
                matched = resp
                break
        # protected_hours: 명시 룰이 없을 때만 적용 (사용자 명시 동의는 룰)
        if not matched and protected and _invite_in_protected(inv, protected):
            await _handle_protected_invite(chat_id, inv, protected)
            continue
        if matched:
            try:
                await gcal.respond_to_invite(chat_id, inv["id"], matched)
                db.log_agent_action(
                    chat_id, "gcal_rsvp",
                    summary=f"자동 RSVP {matched} → {(inv.get('summary') or '(제목 없음)')[:40]}",
                    payload={"event_id": inv["id"], "response": matched,
                             "rule_match": True},
                    reversible={"kind": "gcal_rsvp",
                                 "args": {"event_id": inv["id"], "response": "tentative"}},
                )
                logger.info("auto-RSVP'd invite %s as %s", inv["id"], matched)
            except Exception:
                logger.exception("auto-RSVP failed for %s", inv["id"])
            continue
        # No rule → manual card
        if _app and _app.bot:
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Yes", callback_data=f"act:rsvp_yes:{inv['id']}"),
                InlineKeyboardButton("❔ Maybe", callback_data=f"act:rsvp_maybe:{inv['id']}"),
                InlineKeyboardButton("❌ No", callback_data=f"act:rsvp_no:{inv['id']}"),
            ]])
            text = f"📨 GCal 초대 미응답\n• {inv.get('summary') or '(제목 없음)'}"
            if inv.get("start"):
                text += f"\n• {inv['start']}"
            if inv.get("location"):
                text += f"\n• {inv['location']}"
            try:
                await _app.bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)
            except Exception:
                logger.exception("invite card send failed")


async def run_agent_digest(chat_id: int) -> None:
    """21:45 — single message listing today's autonomous actions, each with [↩️]."""
    if _is_quiet_now(chat_id):
        return
    if _toggle_off_local(chat_id, "agent_digest_enabled"):
        return
    rows = db.list_undigested_actions(chat_id)
    if not rows:
        return
    if _app is None or _app.bot is None:
        return
    lines = ["📋 오늘 봇이 자율로 한 일"]
    ids: List[int] = []
    for r in rows[:15]:
        lines.append(f"  • #{r['id']} {r['summary']}")
        ids.append(r["id"])
    text = "\n".join(lines)
    # Inline keyboard with single-button per action (up to 5; the rest text-only)
    btn_rows = []
    for r in rows[:5]:
        if r["reversible_json"]:
            btn_rows.append([InlineKeyboardButton(
                f"↩️ #{r['id']} 취소", callback_data=f"act:agent_undo:{r['id']}")])
    kb = InlineKeyboardMarkup(btn_rows) if btn_rows else None
    try:
        await _app.bot.send_message(chat_id=chat_id, text=text, reply_markup=kb)
        db.mark_actions_digested(chat_id, ids)
    except Exception:
        logger.exception("agent_digest send failed")


async def run_streak_compute(chat_id: int) -> None:
    """00:30 — for each habit_key, decide today's streak based on yesterday's log."""
    if _toggle_off_local(chat_id, "streak_compute_enabled"):
        return
    today = datetime.now(TZ).date()
    yesterday = today - timedelta(days=1)
    for key in db.all_habit_keys(chat_id):
        dates = set(db.habit_logs_by_date(chat_id, key, days=60))
        prev = db.get_habit_streak(chat_id, key)
        if yesterday.isoformat() in dates:
            new_streak = (prev["current_streak"] + 1) if prev else 1
            db.upsert_habit_streak(chat_id, key, new_streak, yesterday.isoformat())
        else:
            # streak broken
            if prev and prev["current_streak"] > 0:
                db.upsert_habit_streak(chat_id, key, 0, prev["last_log_date"] or "")


async def run_weekly_scorecard(chat_id: int) -> None:
    """Sun 18:00 — 4-section scorecard (goals / habits / spending / mood)."""
    if _app is None or _app.bot is None:
        return
    if _toggle_off_local(chat_id, "weekly_scorecard_enabled"):
        return
    lines = ["📊 이번 주 스코어카드", ""]

    # 1) goals
    goals = db.list_goals(chat_id, status="open")
    lines.append("🎯 골 진척")
    if not goals:
        lines.append("  (활성 골 없음)")
    else:
        for g in goals[:5]:
            subs = json.loads(g["sub_tasks_json"] or "[]")
            done = sum(1 for s in subs if s.get("done"))
            total = len(subs)
            pct = int(100 * done / total) if total else 0
            d = ""
            if g["target_date_local"]:
                try:
                    days = (datetime.fromisoformat(g["target_date_local"]).date()
                            - datetime.now(TZ).date()).days
                    d = f" · D-{days}"
                except Exception:
                    pass
            lines.append(f"  • {g['title']}{d}: {done}/{total} ({pct}%)")
    lines.append("")

    # 2) habits
    lines.append("💪 습관 스트릭")
    streaks = db.list_habit_streaks(chat_id)
    if streaks:
        for s in streaks[:6]:
            lines.append(f"  {s['habit_key']}: {s['current_streak']}일")
    lines.append("")

    # 3) spending
    cur = db.summarize_expenses(chat_id, days=7)
    prev = db.summarize_expenses(chat_id, days=14)
    prev_total = max(prev["total_won"] - cur["total_won"], 0)
    lines.append("💰 지출")
    lines.append(f"  • 이번 주 ₩{cur['total_won']:,} ({cur['count']}건)")
    if prev_total:
        delta = cur["total_won"] - prev_total
        sign = "+" if delta >= 0 else ""
        pct = int(100 * delta / prev_total) if prev_total else 0
        lines.append(f"  • 지난주 ₩{prev_total:,} 대비 {sign}{delta:,} ({sign}{pct}%)")
    if cur["by_category"]:
        top = cur["by_category"][0]
        lines.append(f"  • 1위 카테고리: {top[0]} ₩{top[1]:,}")
    lines.append("")

    # 4) mood
    moods = db.recent_mood_stats(chat_id, days=7)
    if moods:
        total = sum(moods.values())
        bits = []
        for k in ("positive", "neutral", "negative"):
            if k in moods:
                bits.append(f"{k} {int(100 * moods[k] / total)}%")
        lines.append("🌙 무드: " + ", ".join(bits))
    else:
        lines.append("🌙 무드: (회고 응답 부족)")
    try:
        await _app.bot.send_message(chat_id=chat_id, text="\n".join(lines))
    except Exception:
        logger.exception("weekly scorecard send failed")


# ---- Active learning: hourly gap detection ----


_GAP_QUESTIONS: List[Dict] = [
    {"key": "home_address",
     "ask": "🏠 집 주소 한 번 알려줄래? leave-by 알림이 정확해져.",
     "needed": lambda chat_id: "home_address" not in {f["key"] for f in db.list_facts(chat_id)}},
    {"key": "work_address",
     "ask": "🏢 직장(혹은 자대) 주소 알려줄래? 출근/이동 시간 계산에 써.",
     "needed": lambda chat_id: "work_address" not in {f["key"] for f in db.list_facts(chat_id)}},
    {"key": "monthly_food_budget",
     "ask": "💰 한 달 식비 예산 얼마로 잡고 있어? (예: 30만)",
     "needed": lambda chat_id: "monthly_food_budget" not in {f["key"] for f in db.list_facts(chat_id)}},
    {"key": "people_role_missing",
     "ask": None,  # filled at runtime
     "needed": lambda chat_id: any(not p["role"] for p in db.list_people(chat_id))},
]


def _next_gap_question(chat_id: int) -> Optional[Dict]:
    for q in _GAP_QUESTIONS:
        try:
            if q["needed"](chat_id):
                if q["key"] == "people_role_missing":
                    person = next((p for p in db.list_people(chat_id) if not p["role"]), None)
                    if person:
                        return {"key": f"role:{person['id']}",
                                "ask": f"👤 {person['name']}은(는) 어떤 관계야? (친구 / 가족 / 약혼녀 / 동료 / 동기 ...)"}
                return q
        except Exception:
            continue
    return None


async def run_active_learning(chat_id: int) -> None:
    """Daily HH:MM — ask one memory-gap question. Skip if already asked today."""
    if _is_quiet_now(chat_id):
        return
    if _app is None or _app.bot is None:
        return
    if _toggle_off_local(chat_id, "active_learning_enabled"):
        return
    today_iso = datetime.now(TZ).date().isoformat()
    state = db.get_daily_state(chat_id, today_iso)
    if state and state["learning_question_asked"]:
        return
    q = _next_gap_question(chat_id)
    if not q:
        return
    try:
        await _app.bot.send_message(chat_id=chat_id, text=q["ask"])
        db.mark_learning_question_asked(chat_id, today_iso, q["key"])
    except Exception:
        logger.exception("active_learning send failed")


_MOOD_PROMPT = (
    "다음 한 줄 회고의 감정 톤을 분류해. "
    "JSON 한 줄: {\"mood\": \"positive\"|\"neutral\"|\"negative\"}\n\n회고: "
)


async def _classify_and_store_mood(chat_id: int, date_iso: str, text: str) -> None:
    """Micro-LLM (~$0.0002): classify reflection text → daily_state.mood_sentiment."""
    text = (text or "").strip()
    if not text or len(text) < 2:
        return
    try:
        data = await chat_completion(
            [{"role": "user", "content": _MOOD_PROMPT + text[:500]}],
            tools=None, chat_id=chat_id, kind="mood_classify", max_tokens=40,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("mood classify failed (non-fatal)")
        return
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if not m:
        return
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return
    mood = parsed.get("mood")
    if mood in ("positive", "neutral", "negative"):
        db.set_mood_sentiment(chat_id, date_iso, mood)


async def run_budget_check(chat_id: int) -> None:
    """Daily 08:30 — for each `monthly_<cat>_budget` fact, alert when ≥80% or ≥100%."""
    if _app is None or _app.bot is None:
        return
    if _toggle_off_local(chat_id, "budget_alert_enabled"):
        return
    facts_map = {f["key"]: f["value"] for f in db.list_facts(chat_id)}
    today = datetime.now(TZ).date()
    days_in_month = today.day
    cur = db.summarize_expenses(chat_id, days=days_in_month)
    by_cat = dict(cur["by_category"])
    alerts: List[str] = []
    for key, val in facts_map.items():
        if not key.startswith("monthly_") or not key.endswith("_budget"):
            continue
        cat = key[len("monthly_"):-len("_budget")]
        try:
            budget = int(val)
        except ValueError:
            continue
        # category match by Korean alias (식비 vs food etc.)
        spent = by_cat.get(cat, 0)
        # try Korean alias too
        for k2 in by_cat:
            if k2 and (cat in k2 or k2 in cat):
                spent = max(spent, by_cat[k2])
        if budget <= 0:
            continue
        pct = int(100 * spent / budget)
        if pct >= 100:
            alerts.append(f"⚠️ {cat} 예산 ₩{budget:,} 초과 — 이번 달 ₩{spent:,} ({pct}%)")
        elif pct >= 80:
            alerts.append(f"💸 {cat} 예산 ₩{budget:,} 의 {pct}% 도달 — 이번 달 ₩{spent:,}")
    if alerts:
        try:
            await _app.bot.send_message(chat_id=chat_id, text="\n".join(alerts))
        except Exception:
            logger.exception("budget alert send failed")


# ---------------- Telegram handlers ----------------


_app: Optional[Application] = None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "안녕하세요! Dailylife입니다. 일정·기억·검색·장기 골 챙김 다 도와드려요.\n\n"
        "예시:\n"
        "  • '내일 3시에 치과, 30분 전 알림'\n"
        "  • '내 집은 DMC 파크뷰 자이야. 기억해줘'\n"
        "  • '12월 25일 경서 프로포즈 여행 골 등록'\n"
        "  • '신촌 근처 지금 열린 한식집 추천'\n"
        "  • '매일 7시에 우도 운항 정보 알려줘'\n"
        "  • 음성 메시지 / 사진 (포스터·영수증) 그냥 보내도 OK\n\n"
        "명령어:\n"
        "  /setup — 가이드 온보딩 (이름·집·자대·큰 일정 한 번에 등록)\n"
        "  /today /week /agenda — 일정 조회\n"
        "  /goals — 장기 골 목록    /review — 지금 골 리뷰 돌리기\n"
        "  /notes — 메모 모음       /facts — 기억하는 정보\n"
        "  /tasks — 정기 작업       /cost — OpenRouter 사용량\n"
        "  /reset — 대화 메모리 초기화 (DB는 보존)"
    )


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    now = datetime.now(TZ)
    end = now.replace(hour=23, minute=59, second=59)
    items = await _merge_schedule(chat_id, now.astimezone(timezone.utc),
                                  end.astimezone(timezone.utc))
    if not items:
        await update.message.reply_text("오늘 남은 일정 없음 ✨")
        return
    legend = "(📍 local · 🟦 gcal · ✅ both)"
    body = "\n".join(_format_merged_event(it) for it in items)
    await update.message.reply_text(f"오늘 일정 {legend}:\n{body}")


async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    now = datetime.now(TZ)
    end = now + timedelta(days=7)
    items = await _merge_schedule(chat_id, now.astimezone(timezone.utc),
                                  end.astimezone(timezone.utc))
    if not items:
        await update.message.reply_text("앞으로 7일 일정 없음 ✨")
        return
    legend = "(📍 local · 🟦 gcal · ✅ both)"
    body = "\n".join(_format_merged_event(it) for it in items)
    await update.message.reply_text(f"이번 주 일정 {legend}:\n{body}")


async def cmd_agenda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    now = datetime.now(timezone.utc)
    items = await _merge_schedule(chat_id, now, now + timedelta(days=60))
    if not items:
        await update.message.reply_text("등록된 일정 없음 ✨")
        return
    legend = "(📍 local · 🟦 gcal · ✅ both)"
    body = "\n".join(_format_merged_event(it) for it in items)
    await update.message.reply_text(f"앞으로 60일 일정 {legend}:\n{body}")


async def cmd_people(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_people(chat_id)
    if not rows:
        await update.message.reply_text(
            "등록된 사람 없음. 자연어로 '경서는 약혼녀, 생일 7/12야'라고 하시면 자동으로 기억할게요."
        )
        return
    now = datetime.now(timezone.utc)
    today = datetime.now(TZ).date()
    lines = []
    for r in rows:
        bits = [f"#{r['id']} · {r['name']}"]
        if r["role"]:
            bits.append(r["role"])
        if r["last_contact_utc"]:
            try:
                lc = datetime.fromisoformat(r["last_contact_utc"])
                bits.append(f"last {(now - lc).days}d")
            except Exception:
                pass
        head = " · ".join(bits)
        dates = json.loads(r["important_dates_json"] or "[]")
        date_strs = []
        for d in dates:
            try:
                dt = datetime.fromisoformat(d["date_local"]).date()
                if d.get("recurring_yearly"):
                    this_year = dt.replace(year=today.year)
                    if this_year < today:
                        this_year = dt.replace(year=today.year + 1)
                    days_left = (this_year - today).days
                else:
                    days_left = (dt - today).days
                date_strs.append(f"{d['label']} D-{days_left}")
            except Exception:
                pass
        if date_strs:
            head += "\n     ↳ " + ", ".join(date_strs)
        lines.append(head)
    await update.message.reply_text("👥 등록된 사람:\n" + "\n\n".join(lines))


async def cmd_facts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_facts(chat_id)
    if not rows:
        await update.message.reply_text("기억하고 있는 정보가 아직 없어요. 알려주시면 저장할게요!")
        return
    body = "\n".join(f"• {r['key']}: {r['value']}" for r in rows)
    await update.message.reply_text("기억하고 있는 정보:\n" + body)


async def cmd_goals(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_goals(chat_id, status="open")
    if not rows:
        await update.message.reply_text(
            "등록된 골 없음. 자유롭게 말씀해주시면 큰 일정·장기 목표는 자동으로 골로 넣을게요."
        )
        return
    body = "\n\n".join(_format_goal_summary(r) for r in rows)
    await update.message.reply_text("📋 진행 중인 골:\n\n" + body)


async def cmd_review(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Force the weekly review for the current chat (otherwise it fires Sun 09:00 KST)."""
    chat_id = update.effective_chat.id
    rows = db.list_goals(chat_id, status="open")
    if not rows:
        await update.message.reply_text("리뷰할 골이 없어요. 먼저 큰 목표 몇 개 알려주세요.")
        return
    await update.message.reply_text("주간 리뷰 돌리는 중… 잠시만요. 도구 호출이 많을 수 있어요.")
    await run_weekly_goal_review(chat_id)


async def cmd_notes(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_notes(chat_id, limit=15)
    if not rows:
        await update.message.reply_text("저장된 메모가 없어요. 자유롭게 흘려도 알아서 저장할게요.")
        return
    lines = []
    for r in rows:
        when_local = datetime.fromisoformat(r["created_at"].rstrip("Z") + "+00:00").astimezone(TZ)
        tag = f" · #{r['tags']}" if r["tags"] else ""
        lines.append(f"#{r['id']} · {when_local.strftime('%m-%d %H:%M')}{tag}\n   {r['content']}")
    await update.message.reply_text("최근 메모:\n" + "\n\n".join(lines))


async def cmd_tasks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_recurring_tasks(chat_id)
    if not rows:
        await update.message.reply_text("등록된 정기 작업 없음.")
        return
    lines = []
    for r in rows:
        last = r["last_run_utc"] or "—"
        lines.append(f"#{r['id']} · {r['cron_kst']} KST · {'on' if r['enabled'] else 'off'}\n   ↳ {r['prompt']}\n   ↳ 마지막 실행: {last}")
    await update.message.reply_text("정기 작업:\n" + "\n\n".join(lines))


async def cmd_spending(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    days = 30
    if context.args:
        try:
            days = max(1, min(int(context.args[0]), 365))
        except Exception:
            pass
    s = db.summarize_expenses(chat_id, days=days)
    if s["count"] == 0:
        await update.message.reply_text(f"최근 {days}일 지출 기록 없음. 자연어로 '스벅 6500원' 같이 흘려도 자동 저장돼요.")
        return
    lines = [f"💸 최근 {days}일 지출 — 합계 ₩{s['total_won']:,} ({s['count']}건)"]
    for cat, won in s["by_category"][:8]:
        lines.append(f"  • {cat or '기타'}: ₩{won:,}")
    await update.message.reply_text("\n".join(lines))


async def cmd_habits(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    days = 7
    if context.args:
        try:
            days = max(1, min(int(context.args[0]), 90))
        except Exception:
            pass
    s = db.summarize_habits(chat_id, days=days)
    if not s["by_habit"]:
        await update.message.reply_text(f"최근 {days}일 습관 기록 없음.")
        return
    lines = [f"🌿 최근 {days}일 습관:"]
    for h in s["by_habit"]:
        total_min = h.get("mins") or 0
        bit = f"  • {h['habit_key']}: {h['n']}회"
        if total_min:
            bit += f", 총 {total_min}분"
        lines.append(bit)
    await update.message.reply_text("\n".join(lines))


async def cmd_briefing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/briefing` → force now; `/briefing off`/`on` → toggle; `/briefing 07:30` → time."""
    chat_id = update.effective_chat.id
    arg = (context.args[0] if context.args else "").strip().lower()
    if arg in {"off", "0", "false"}:
        db.remember_fact(chat_id, "briefing_enabled", "false")
        scheduler.disable_daily_rhythm_for(chat_id)
        scheduler.ensure_daily_rhythm_for(chat_id)  # reflection only
        await update.message.reply_text("☀️ 아침 브리핑 OFF.")
        return
    if arg in {"on", "1", "true"}:
        db.forget_fact(chat_id, "briefing_enabled")
        scheduler.ensure_daily_rhythm_for(chat_id)
        await update.message.reply_text("☀️ 아침 브리핑 ON (기본 07:30 KST).")
        return
    if re.fullmatch(r"\d{1,2}:\d{2}", arg):
        db.remember_fact(chat_id, "briefing_time", arg)
        scheduler.ensure_daily_rhythm_for(chat_id)
        await update.message.reply_text(f"☀️ 아침 브리핑 시간을 {arg} KST 로 변경.")
        return
    await update.message.reply_text("☀️ 지금 한 번 브리핑 보내드릴게요…")
    scheduler.trigger_morning_briefing_now(chat_id)


async def cmd_reflect(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/reflect` → force now; `/reflect on`/`off` → toggle; `/reflect 21:30` → time."""
    chat_id = update.effective_chat.id
    arg = (context.args[0] if context.args else "").strip().lower()
    if arg in {"off", "0", "false"}:
        db.remember_fact(chat_id, "reflection_enabled", "false")
        scheduler.disable_daily_rhythm_for(chat_id)
        scheduler.ensure_daily_rhythm_for(chat_id)
        await update.message.reply_text("🌙 저녁 회고 OFF.")
        return
    if arg in {"on", "1", "true"}:
        db.forget_fact(chat_id, "reflection_enabled")
        scheduler.ensure_daily_rhythm_for(chat_id)
        await update.message.reply_text("🌙 저녁 회고 ON (기본 21:30 KST).")
        return
    if re.fullmatch(r"\d{1,2}:\d{2}", arg):
        db.remember_fact(chat_id, "reflection_time", arg)
        scheduler.ensure_daily_rhythm_for(chat_id)
        await update.message.reply_text(f"🌙 저녁 회고 시간을 {arg} KST 로 변경.")
        return
    scheduler.trigger_evening_reflection_now(chat_id)


_NUDGE_KEYS = {
    # alias → (fact_key, default_on, label)
    "메일":       ("gmail_event_scan_enabled",         True,  "메일→일정 감지"),
    "gmail":      ("gmail_event_scan_enabled",         True,  "메일→일정 감지"),
    "leaveby":    ("leave_by_enabled",                  True,  "Leave-by"),
    "출발":       ("leave_by_enabled",                  True,  "Leave-by"),
    "preview":    ("weather_preview_enabled",           False, "전날 밤 프리뷰"),
    "프리뷰":     ("weather_preview_enabled",           False, "전날 밤 프리뷰"),
    "milestone":  ("goal_milestone_enabled",            True,  "골 마일스톤"),
    "마일스톤":   ("goal_milestone_enabled",            True,  "골 마일스톤"),
    "birthday":   ("birthday_alert_separate_enabled",   False, "생일 단독 알림"),
    "생일":       ("birthday_alert_separate_enabled",   False, "생일 단독 알림"),
    "checkin":    ("midday_checkin_enabled",            False, "점심 체크인"),
    "점심":       ("midday_checkin_enabled",            False, "점심 체크인"),
    "briefing":   ("briefing_enabled",                  True,  "아침 브리핑"),
    "브리핑":     ("briefing_enabled",                  True,  "아침 브리핑"),
    "reflect":    ("reflection_enabled",                True,  "저녁 회고"),
    "회고":       ("reflection_enabled",                True,  "저녁 회고"),
    # v4
    "persona":    ("persona_rebuild_enabled",           True,  "인물 요약 재생성"),
    "rules":      ("auto_rules_enabled",                True,  "자동 규칙 마스터"),
    "digest":     ("agent_digest_enabled",              True,  "어제 봇이 한 일 디지스트"),
    "decompose":  ("auto_decompose_enabled",            True,  "골 자동 분해"),
    "scorecard":  ("weekly_scorecard_enabled",          False, "주간 스코어카드"),
    "streak":     ("streak_compute_enabled",            True,  "스트릭 계산"),
    "budget":     ("budget_alert_enabled",              True,  "예산 알림"),
    "learning":   ("active_learning_enabled",           True,  "하루 1 질문"),
    "voice":      ("voice_replies_enabled",             False, "음성 답장 (cost)"),
}


def _nudge_status(chat_id: int, key: str, default_on: bool) -> bool:
    v = _fact_value_local(chat_id, key)
    if v is None:
        return default_on
    s = v.strip().lower()
    if default_on:  # default-on toggles use the "off" set
        return s not in {"false", "off", "0", "no"}
    return s in {"true", "on", "1", "yes"}


async def cmd_nudges(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/nudges` → 현재 상태; `/nudges on|off <alias> [HH:MM]` → 토글/시간 변경."""
    chat_id = update.effective_chat.id
    args = [a.lower() for a in (context.args or [])]
    # Status print
    if not args:
        lines = ["📣 활성 알림"]
        rows = [
            ("gmail_event_scan_enabled",       True,  "메일→일정 감지", None,
             f"매 {_fact_value_local(chat_id, 'gmail_event_scan_minutes') or '60'}분"),
            ("leave_by_enabled",               True,  "Leave-by",          None, None),
            ("weather_preview_enabled",        False, "전날 밤 프리뷰",
             _fact_value_local(chat_id, "weather_preview_time") or "22:00", None),
            ("goal_milestone_enabled",         True,  "골 마일스톤",        None, "D-30/14/3/1"),
            ("birthday_alert_separate_enabled", False, "생일 단독 알림",     "09:00", None),
            ("midday_checkin_enabled",         False, "점심 체크인",
             _fact_value_local(chat_id, "midday_checkin_time") or "13:00", None),
            ("briefing_enabled",               True,  "아침 브리핑",
             _fact_value_local(chat_id, "briefing_time") or "07:30", None),
            ("reflection_enabled",             True,  "저녁 회고",
             _fact_value_local(chat_id, "reflection_time") or "21:30", None),
        ]
        for key, default_on, label, time_str, extra in rows:
            on = _nudge_status(chat_id, key, default_on)
            bits = ["ON" if on else "OFF"]
            if time_str:
                bits.append(f"@{time_str}")
            if extra:
                bits.append(extra)
            lines.append(f"  • {label}: {' '.join(bits)}")
        lines.append("")
        lines.append("끄기/켜기: /nudges off 메일 (또는 leaveby/preview/milestone/birthday/checkin/briefing/reflect)")
        lines.append("시간 변경: /nudges preview 22:30 / /nudges checkin 12:45")
        await update.message.reply_text("\n".join(lines))
        return

    # Mutation forms.
    op = args[0]
    if op in {"on", "off"} and len(args) >= 2:
        alias = args[1]
        if alias not in _NUDGE_KEYS:
            await update.message.reply_text(f"모르는 알림: {alias}")
            return
        fact_key, default_on, label = _NUDGE_KEYS[alias]
        if op == "off":
            if default_on:
                db.remember_fact(chat_id, fact_key, "false")
            else:
                db.forget_fact(chat_id, fact_key)
            # v5: mark recent nudges of this kind as 'toggled_off' for self-improve.
            nudge_kind_guess = {
                "gmail_event_scan_enabled": "gmail_event_scan",
                "leave_by_enabled": "late_check",
                "weather_preview_enabled": "evening_preview",
                "goal_milestone_enabled": "goal_milestone",
                "birthday_alert_separate_enabled": "birthday_solo",
                "midday_checkin_enabled": "midday_checkin",
                "briefing_enabled": "morning_briefing",
                "reflection_enabled": "evening_reflection",
            }.get(fact_key)
            if nudge_kind_guess:
                db.react_to_recent_nudge(chat_id, nudge_kind_guess,
                                          "toggled_off", within_hours=72)
            await update.message.reply_text(f"🔕 {label} OFF.")
        else:
            if default_on:
                db.forget_fact(chat_id, fact_key)
            else:
                db.remember_fact(chat_id, fact_key, "true")
            await update.message.reply_text(f"🔔 {label} ON.")
        scheduler.ensure_daily_rhythm_for(chat_id)
        return
    # Time-change form: /nudges <alias> HH:MM
    if op in _NUDGE_KEYS and len(args) >= 2 and re.fullmatch(r"\d{1,2}:\d{2}", args[1]):
        fact_key, _, label = _NUDGE_KEYS[op]
        time_fact_map = {
            "weather_preview_enabled": "weather_preview_time",
            "midday_checkin_enabled": "midday_checkin_time",
            "briefing_enabled": "briefing_time",
            "reflection_enabled": "reflection_time",
        }
        if fact_key not in time_fact_map:
            await update.message.reply_text(f"{label} 은 시간 변경 미지원.")
            return
        db.remember_fact(chat_id, time_fact_map[fact_key], args[1])
        scheduler.ensure_daily_rhythm_for(chat_id)
        await update.message.reply_text(f"⏰ {label} 시간 → {args[1]} KST.")
        return
    await update.message.reply_text(
        "사용법: /nudges (상태) | /nudges on|off <alias> | /nudges <alias> HH:MM"
    )


async def cmd_rules(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/rules` 목록 · `/rules add <자연어>` LLM이 JSON 변환 · `/rules off <id>` 비활성화."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        rows = db.list_auto_rules(chat_id, only_enabled=False)
        if not rows:
            await update.message.reply_text(
                "🤖 자동 규칙 없음.\n"
                "사용: /rules add 신세계상품권 메일은 자동으로 일정 추가\n"
                "      /rules add 경서 보낸 GCal 초대는 자동 yes\n"
                "      /rules off <id>"
            )
            return
        lines = ["🤖 자동 규칙"]
        for r in rows:
            status = "ON " if r["enabled"] else "OFF"
            cond = r["condition_json"]
            lines.append(f"  • #{r['id']} [{status}] {r['rule_kind']} {cond}")
        await update.message.reply_text("\n".join(lines))
        return
    op = args[0].lower()
    if op == "off" and len(args) >= 2:
        try:
            rid = int(args[1])
        except ValueError:
            await update.message.reply_text("rule_id는 숫자여야 해.")
            return
        ok = db.disable_auto_rule(chat_id, rid)
        await update.message.reply_text(
            f"🔕 규칙 #{rid} 비활성화" if ok else f"#{rid} 못 찾음.")
        return
    if op == "add" and len(args) >= 2:
        raw = " ".join(args[1:]).strip()
        # LLM converts NL → JSON
        prompt = (
            "사용자가 비서 자동 규칙을 자연어로 묘사했어. JSON으로 변환해서 한 줄만 출력:\n"
            '{"rule_kind": "gmail_auto_add_event"|"gcal_auto_rsvp", "condition": {...}}\n'
            "examples:\n"
            "  - 신세계 메일 자동 일정 → {\"rule_kind\":\"gmail_auto_add_event\","
            "\"condition\":{\"sender_pattern\":\"@shinsegae\",\"min_confidence\":0.85}}\n"
            "  - 경서 초대 자동 yes → {\"rule_kind\":\"gcal_auto_rsvp\","
            "\"condition\":{\"from_people_names\":[\"경서\"],\"response\":\"accepted\"}}\n"
            f"\n사용자 문장: {raw}\n\nJSON만 출력."
        )
        try:
            data = await chat_completion(
                [{"role": "user", "content": prompt}],
                tools=None, chat_id=chat_id, kind="rule_parse", max_tokens=200,
            )
            content = (data["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            await update.message.reply_text("⚠ LLM 변환 실패.")
            return
        import re as _re
        m = _re.search(r"\{[\s\S]*\}", content)
        if not m:
            await update.message.reply_text(f"⚠ JSON 파싱 실패:\n{content[:200]}")
            return
        try:
            parsed = json.loads(m.group(0))
            rid = db.add_auto_rule(chat_id, parsed["rule_kind"], parsed["condition"])
            await update.message.reply_text(
                f"🤖 규칙 #{rid} 등록\n  • kind: {parsed['rule_kind']}\n  • cond: {parsed['condition']}"
            )
        except Exception as e:
            await update.message.reply_text(f"⚠ 등록 실패: {e}")
        return
    await update.message.reply_text(
        "사용: /rules | /rules add <자연어> | /rules off <id>")


async def cmd_persona(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/persona` 최신 인물 요약 · `/persona rebuild` 지금 재생성."""
    chat_id = update.effective_chat.id
    arg = (context.args[0] if context.args else "").strip().lower()
    if arg == "rebuild":
        await update.message.reply_text("🧬 인물 요약 재생성 중… (조금 걸려)")
        scheduler.trigger_persona_rebuild_now(chat_id)
        return
    p = db.get_latest_persona(chat_id)
    if not p:
        await update.message.reply_text(
            "🧬 아직 인물 요약 없음. /persona rebuild 로 생성 가능."
        )
        return
    text = (
        f"🧬 인물 요약 v{p['version']} (갱신 {p['generated_at'][:10]})\n\n"
        + p["content_md"]
    )
    for i in range(0, len(text), 4000):
        await update.message.reply_text(text[i:i + 4000])


async def cmd_recall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/recall <엔티티>` cross-table 회수."""
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("사용: /recall 경서")
        return
    entity = " ".join(context.args).strip()
    res = tool_cross_recall(chat_id, {"entity": entity})
    lines = [f"🔎 '{entity}' 회수"]
    if res.get("people"):
        lines.append(f"  👤 사람 {len(res['people'])}: " +
                     ", ".join(p["name"] for p in res["people"][:5]))
    if res.get("events"):
        lines.append(f"  📅 이벤트 {len(res['events'])}: " +
                     ", ".join(e["title"][:20] for e in res["events"][:5]))
    if res.get("goals"):
        lines.append(f"  🎯 골 {len(res['goals'])}: " +
                     ", ".join(g["title"][:20] for g in res["goals"][:5]))
    if res.get("notes"):
        lines.append(f"  📝 노트 {len(res['notes'])}")
    if res.get("expenses"):
        total = sum(e["amount_won"] for e in res["expenses"])
        lines.append(f"  💰 지출 {len(res['expenses'])}건 ₩{total:,}")
    if res.get("chat_log"):
        lines.append(f"  💬 채팅 {len(res['chat_log'])} 히트")
    if len(lines) == 1:
        lines.append("  (관련 데이터 없음)")
    await update.message.reply_text("\n".join(lines))


async def cmd_scorecard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/scorecard` 지금 스코어카드 발사."""
    chat_id = update.effective_chat.id
    await update.message.reply_text("📊 스코어카드 만들고 있어…")
    scheduler.trigger_weekly_scorecard_now(chat_id)


async def cmd_agent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/agent <목표>` 다단계 자율 에이전트 모드 — max_hops 20, 외부 액션은 auto_rule만."""
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text(
            "사용: /agent 다음주 부산 KTX 예약 시각 확인 + 호텔 후보\n"
            "    (web_search·fetch_url·gmail·gcal 등 다단계 자율)"
        )
        return
    goal = " ".join(context.args).strip()
    db.log_chat(chat_id, "user", f"[/agent] {goal}")
    await update.message.reply_text(f"🤖 에이전트 시작 — '{goal[:60]}'")
    try:
        reply = await run_agent(
            chat_id,
            f"[AGENT MODE] {goal}\n\n자율로 도구를 조합해 단계적으로 해결해. "
            "외부 액션(메일 전송·RSVP)은 auto_rule 통과한 것만 자동. 나머지는 초안 제안.",
            history=[], max_hops=20, kind="agent",
        )
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i:i + 4000])
    except Exception as e:
        logger.exception("/agent failed")
        await update.message.reply_text(f"⚠ {e}")


# ---------------- v5: bank SMS bridge ----------------


BANK_SMS_PROMPT = (
    "한국 카드/은행 SMS 한 줄에서 거래 정보를 JSON으로 뽑아. "
    'JSON: {{"amount_won": <int>, "merchant": "<상호명>", '
    '"category": "<식비/교통/쇼핑/카페/기타>"}}. '
    "거래가 아니거나 (잔액 알림, 광고) 파싱 불가면 빈 객체 {{}} 반환. JSON만.\n\n"
    "SMS: {sms}"
)


async def _parse_bank_sms(sms: str) -> Optional[Dict]:
    try:
        data = await chat_completion(
            [{"role": "user", "content": BANK_SMS_PROMPT.format(sms=sms[:400])}],
            tools=None, chat_id=0, kind="bank_sms_parse", max_tokens=160,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("bank_sms parse failed")
        return None
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return None
    if not parsed or not parsed.get("amount_won"):
        return None
    return parsed


async def cmd_bank_sms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/bank_sms <텍스트>` 카드 SMS 자동 가계부."""
    chat_id = update.effective_chat.id
    sms = " ".join(context.args or []).strip()
    if not sms:
        await update.message.reply_text(
            "사용: /bank_sms KB체크 9,500원 스타벅스 09:32\n"
            "(Apple Shortcut / Tasker로 SMS 받으면 자동 forward 가능)"
        )
        return
    parsed = await _parse_bank_sms(sms)
    if not parsed:
        await update.message.reply_text("⚠ 거래 정보 추출 실패. (광고/잔액 알림일 수도)")
        return
    amount = int(parsed["amount_won"])
    merchant = (parsed.get("merchant") or "").strip()
    category = (parsed.get("category") or "기타").strip()
    auto_add = _toggle_on_local(chat_id, "bank_sms_auto_add")
    if auto_add:
        eid = db.log_expense(chat_id, amount_won=amount, merchant=merchant,
                              category=category, notes=f"[bank_sms] {sms[:120]}")
        tok = secrets.token_urlsafe(8)
        db.push_deleted_audit(chat_id, "expense",
                               {"id": eid, "amount_won": amount,
                                "merchant": merchant, "category": category},
                               tok)
        _pending_undos[tok] = ("expense", chat_id)
        db.log_agent_action(chat_id, "bank_sms_auto_add",
                             summary=f"SMS 자동 ₩{amount:,} {merchant}",
                             payload={"sms": sms[:200], "expense_id": eid})
        kb = _undo_keyboard(tok, "취소")
        await update.message.reply_text(
            f"💳 자동 기록: ₩{amount:,} · {merchant} · {category}",
            reply_markup=kb,
        )
    else:
        # Confirmation card
        token = secrets.token_urlsafe(8)
        _pending_mail_cards[token] = {  # reuse the in-memory dict
            "chat_id": chat_id, "kind": "bank_sms_confirm",
            "amount": amount, "merchant": merchant, "category": category,
            "sms": sms,
        }
        kb = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"✅ ₩{amount:,} {merchant}",
                                  callback_data=f"act:bank_add:{token}"),
            InlineKeyboardButton("❌ 아님",
                                  callback_data=f"act:bank_skip:{token}"),
        ]])
        await update.message.reply_text(
            f"💳 SMS 분석:\n  • ₩{amount:,}\n  • {merchant}\n  • {category}",
            reply_markup=kb,
        )


# ---------------- v5: image generation ----------------


async def tool_generate_image(chat_id: int, args: Dict) -> Dict:
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt required"}
    # daily cap
    try:
        cap = int(_fact_value_local(chat_id, "image_daily_cap") or "3")
    except ValueError:
        cap = 3
    with db._conn() as c:
        today_iso = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        used = c.execute(
            "SELECT COUNT(*) AS n FROM agent_actions WHERE chat_id=? "
            "AND action_kind='image_gen' AND executed_at>=?",
            (chat_id, today_iso)).fetchone()["n"]
    if used >= cap:
        return {"ok": False, "error": f"daily image cap reached ({used}/{cap})"}
    try:
        png = await transcribe.generate_image(prompt)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if _app and _app.bot:
        try:
            await _app.bot.send_photo(chat_id=chat_id, photo=png, caption=prompt[:200])
        except Exception:
            logger.exception("image send failed")
    db.log_agent_action(chat_id, "image_gen",
                         summary=f"이미지 생성: {prompt[:50]}",
                         payload={"prompt": prompt, "bytes": len(png)})
    return {"ok": True, "bytes": len(png), "remaining_today": max(0, cap - used - 1)}


ASYNC_HANDLERS["generate_image"] = tool_generate_image
ASYNC_HANDLERS["analyze_document"] = tool_analyze_document
ASYNC_HANDLERS["lifelog_search"] = tool_lifelog_search_async
ASYNC_HANDLERS["web_scrape"] = tool_web_scrape
ASYNC_HANDLERS["web_screenshot"] = tool_web_screenshot
ASYNC_HANDLERS["place_phone_call"] = tool_place_phone_call


async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/image <prompt>` 즉시 이미지 생성 + 발송."""
    chat_id = update.effective_chat.id
    prompt = " ".join(context.args or []).strip()
    if not prompt:
        await update.message.reply_text("사용: /image 분홍 노을 발리 결혼식")
        return
    await update.message.reply_text("🎨 그리는 중…")
    res = await tool_generate_image(chat_id, {"prompt": prompt})
    if not res["ok"]:
        await update.message.reply_text(f"⚠ {res.get('error')}")
    # success case already sent the photo via send_photo


# ---------------- v5: voice reply auto chain ----------------


def _is_voice_reply_active(chat_id: int) -> bool:
    """ON if fact voice_reply_auto=true OR a /voice session is active."""
    if _toggle_on_local(chat_id, "voice_reply_auto"):
        return True
    until_iso = _fact_value_local(chat_id, "voice_reply_session_until_utc")
    if not until_iso:
        return False
    try:
        until = datetime.fromisoformat(until_iso)
        return datetime.now(timezone.utc) < until
    except Exception:
        return False


async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/voice [N분]` 다음 N분간 모든 응답을 음성으로 (기본 30분)."""
    chat_id = update.effective_chat.id
    arg = (context.args[0] if context.args else "30").strip().lower()
    if arg in {"off", "0"}:
        db.forget_fact(chat_id, "voice_reply_session_until_utc")
        await update.message.reply_text("🔇 음성 답장 모드 OFF.")
        return
    try:
        minutes = max(1, min(int(arg), 240))
    except ValueError:
        minutes = 30
    until = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    db.remember_fact(chat_id, "voice_reply_session_until_utc", until)
    await update.message.reply_text(f"🔊 음성 답장 {minutes}분간 ON.")


async def cmd_say(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/say <텍스트>` TTS로 음성 답장."""
    chat_id = update.effective_chat.id
    text = " ".join(context.args or []).strip()
    if not text:
        await update.message.reply_text("사용: /say 텍스트")
        return
    try:
        audio = await transcribe.synthesize_voice(text)
    except Exception as e:
        await update.message.reply_text(f"⚠ TTS 실패: {e}")
        return
    try:
        await context.bot.send_voice(chat_id=chat_id, voice=audio, caption=text[:120])
    except Exception as e:
        await update.message.reply_text(f"⚠ 음성 전송 실패: {e}")


# ---------------- v5: MISSION MODE ----------------


MISSION_SYSTEM_PROMPT = (
    "[MISSION MODE] 당신은 자율 multi-hop 에이전트. 한 mission 안에서 여러 "
    "tick에 걸쳐 작업을 이어감. 매 tick마다 1-3개 tool call로 단계를 전진시키고, "
    "최종 답이 준비됐을 때만 tool call 없이 짧은 마크다운 결과를 반환. "
    "외부 액션 (메일/RSVP)은 사용자가 미리 등록한 auto_rules 통과한 것만 자동 실행. "
    "그 외는 초안 제안 후 종료. cost가 한정돼 있으니 가능한 적은 hop으로 끝내.\n\n"
    "🔗 최종 결과 마크다운에는 모든 사실 주장 (가격·시간·운영시간·평점 등) 옆에 "
    "URL 출처를 '[출처](url)' 형식으로 inline 표시. fetch_url/web_search 결과의 "
    "원본 URL을 그대로 인용. 추측은 '?' 표시 후 검증 권유."
)

MISSION_FINAL_MARKER_TURNS = (5, 30, 90)  # minutes — checkpoint cadence buckets


def _mission_max_hops_per_tick(chat_id: int) -> int:
    try:
        v = int(_fact_value_local(chat_id, "mission_max_hops_per_tick") or "3")
    except ValueError:
        v = 3
    return max(1, min(v, 5))


def _agent_daily_cost_limit(chat_id: int) -> float:
    try:
        return float(_fact_value_local(chat_id, "agent_daily_cost_limit_usd") or "1.0")
    except ValueError:
        return 1.0


async def run_mission_tick(mission_id: int) -> None:
    """One pump cycle for a single mission. Advances 1-3 hops, persists state.

    Termination conditions (in priority):
      1. status changed to cancelled/done externally → exit silently
      2. cost guard exceeded → status=paused + alert
      3. LLM returned text answer with no tool calls → status=done + final msg
      4. current_hop >= max_hops → status=done with whatever we have"""
    m = db.get_mission(mission_id)
    if not m or m["status"] != "running":
        return
    chat_id = m["chat_id"]
    # Cost guard — Wave 1 risk mitigation
    if m["cost_usd_running"] >= _agent_daily_cost_limit(chat_id):
        db.update_mission(mission_id, status="paused",
                           result_md=(m["result_md"] or "")
                           + f"\n\n⏸ 비용 한도 (${m['cost_usd_running']:.4f}) 도달 — 사용자 확인 필요")
        if _app and _app.bot:
            await _app.bot.send_message(
                chat_id=chat_id,
                text=(f"⏸ Mission #{mission_id} 일시정지 — "
                      f"비용 한도 ${m['cost_usd_running']:.4f} 도달.\n"
                      f"재개하려면 facts에서 agent_daily_cost_limit_usd 올려줘."),
            )
        return

    history = json.loads(m["history_json"] or "[]")
    if not history:
        # First tick — seed with the system + user goal.
        history.append({"role": "user", "content": MISSION_SYSTEM_PROMPT + "\n\n목표:\n" + m["goal_text"]})

    per_tick = _mission_max_hops_per_tick(chat_id)
    started_at = datetime.now(timezone.utc)
    cost_before = m["cost_usd_running"]
    final_text: Optional[str] = None
    hops_this_tick = 0
    for _ in range(per_tick):
        if m["current_hop"] + hops_this_tick >= m["max_hops"]:
            final_text = "(max_hops 도달 — mission 종료)"
            break
        messages = [_system_message(chat_id), *history]
        try:
            data = await chat_completion(
                messages, tools=TOOLS, chat_id=chat_id, kind="mission",
            )
        except Exception as e:
            logger.exception("mission %s LLM call failed", mission_id)
            db.update_mission(mission_id, status="failed",
                               result_md=(m["result_md"] or "") + f"\n\n❌ LLM 오류: {e}")
            if _app and _app.bot:
                await _app.bot.send_message(
                    chat_id=chat_id,
                    text=f"❌ Mission #{mission_id} 실패 — LLM 오류: {e}")
            return
        # Track cost delta best-effort from usage table
        try:
            new_cost = _last_usage_cost(chat_id) + cost_before
        except Exception:
            new_cost = cost_before
        msg = data["choices"][0]["message"]
        history.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or None,
        })
        tool_calls = parse_tool_calls(msg)
        hops_this_tick += 1
        if not tool_calls:
            final_text = (msg.get("content") or "").strip()
            break
        for tc in tool_calls:
            name = tc["name"]
            try:
                if name in SYNC_HANDLERS:
                    result = SYNC_HANDLERS[name](chat_id, tc["arguments"])
                elif name in ASYNC_HANDLERS:
                    result = await ASYNC_HANDLERS[name](chat_id, tc["arguments"])
                else:
                    result = {"ok": False, "error": f"unknown tool {name}"}
            except Exception as exc:
                logger.exception("mission %s tool %s failed", mission_id, name)
                result = {"ok": False, "error": str(exc)}
            history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": json.dumps(result, ensure_ascii=False)[:8000],
            })

    # Persist state after the tick
    db.update_mission(
        mission_id,
        history_json=json.dumps(history, ensure_ascii=False),
        current_hop=m["current_hop"] + hops_this_tick,
        cost_usd_running=cost_before + max(0, _last_usage_cost(chat_id, since=started_at)),
        last_checkpoint_at=datetime.now(timezone.utc).isoformat(),
    )

    if final_text is not None:
        result_md = (m["result_md"] or "") + "\n\n" + final_text
        db.update_mission(mission_id, status="done", result_md=result_md.strip())
        await _send_mission_result(mission_id, result_md.strip())
        return

    # v8: Live progress UI — edit a single Telegram message in place each tick
    # so the user can watch the mission move forward.
    if _app and _app.bot:
        recent = history[-1] if history else {}
        snippet = (recent.get("content") or "")[:200] if isinstance(recent, dict) else ""
        if not snippet and len(history) >= 2 and isinstance(history[-2], dict):
            snippet = (history[-2].get("content") or "")[:200]
        cost_now = cost_before + max(0, _last_usage_cost(chat_id, since=started_at))
        progress_text = (
            f"🛠 Mission #{mission_id} 진행 중 — {m['title'][:40]}\n"
            f"  • {m['current_hop'] + hops_this_tick}/{m['max_hops']} hop · ${cost_now:.4f}\n"
            f"  • 최근: {snippet[:200]}"
        )
        try:
            if m["progress_message_id"]:
                await _app.bot.edit_message_text(
                    chat_id=chat_id, message_id=m["progress_message_id"],
                    text=progress_text)
            else:
                sent = await _app.bot.send_message(chat_id=chat_id, text=progress_text)
                db.update_mission(mission_id, progress_message_id=sent.message_id)
        except Exception:
            # Telegram errors on 'message not modified' or expired msg — non-fatal
            pass


def _last_usage_cost(chat_id: int, since: Optional[datetime] = None) -> float:
    """Sum of cost_usd from usage table for this chat since `since` (or last 5min)."""
    since = since or (datetime.now(timezone.utc) - timedelta(minutes=5))
    with db._conn() as c:
        r = c.execute(
            "SELECT COALESCE(SUM(cost_usd),0) AS s FROM usage "
            "WHERE chat_id=? AND created_at>=?",
            (chat_id, since.isoformat())).fetchone()
        return float(r["s"] or 0.0)


async def _send_mission_result(mission_id: int, result_md: str) -> None:
    if _app is None or _app.bot is None:
        return
    m = db.get_mission(mission_id)
    if not m:
        return
    text = f"✅ Mission #{mission_id} 완성 — {m['title']}\n\n{result_md[:3500]}"
    try:
        await _app.bot.send_message(chat_id=m["chat_id"], text=text)
    except Exception:
        logger.exception("mission result send failed for %s", mission_id)


# Tool handlers


def tool_start_mission(chat_id: int, args: Dict) -> Dict:
    title = (args.get("title") or "").strip()
    goal = (args.get("goal_text") or args.get("goal") or "").strip()
    if not title or not goal:
        return {"ok": False, "error": "title + goal_text required"}
    max_hops = int(args.get("max_hops") or 80)
    mid = db.add_mission(chat_id, title, goal, max_hops=max(5, min(max_hops, 200)))
    # Kick off first tick right away.
    scheduler.trigger_mission_tick_now(mid)
    return {"ok": True, "mission_id": mid}


def tool_cancel_mission(chat_id: int, args: Dict) -> Dict:
    mid = int(args.get("mission_id") or 0)
    if not mid:
        return {"ok": False, "error": "mission_id required"}
    return {"ok": db.cancel_mission(chat_id, mid)}


def tool_list_missions(chat_id: int, args: Dict) -> Dict:
    status = args.get("status")
    rows = db.list_missions(chat_id, status=status)
    return {"ok": True, "missions": [
        {"id": r["id"], "title": r["title"], "status": r["status"],
         "current_hop": r["current_hop"], "max_hops": r["max_hops"],
         "cost_usd": round(r["cost_usd_running"], 4)}
        for r in rows[:20]]}


# Register v5 mission tools now that their handlers exist.
SYNC_HANDLERS.update({
    "start_mission": tool_start_mission,
    "cancel_mission": tool_cancel_mission,
    "list_missions": tool_list_missions,
})


# ---------------- v5: knowledge graph tools ----------------


def tool_graph_query(chat_id: int, args: Dict) -> Dict:
    kind = args.get("start_kind")
    sid = args.get("start_id")
    if not kind or sid is None:
        return {"ok": False, "error": "start_kind + start_id required"}
    depth = max(1, min(int(args.get("max_depth") or 2), 4))
    rel_filter = args.get("relation_filter") or None
    g = db.graph_bfs(chat_id, kind, int(sid), max_depth=depth, relation_filter=rel_filter)
    # Hydrate nodes with their titles where possible (small N, ok)
    hydrated = []
    for n in g["nodes"]:
        label = None
        try:
            if n["kind"] == "person":
                row = db.get_person(int(n["id"]))
                label = row["name"] if row else None
            elif n["kind"] == "event":
                row = db.get_event(int(n["id"]))
                label = row["title"] if row else None
            elif n["kind"] == "goal":
                row = db.get_goal(int(n["id"])) if hasattr(db, "get_goal") else None
                label = row["title"] if row else None
        except Exception:
            label = None
        hydrated.append({**n, "label": label})
    return {"ok": True, "nodes": hydrated, "edges": g["edges"]}


def tool_add_relation(chat_id: int, args: Dict) -> Dict:
    rid = db.add_relation(
        chat_id,
        args["from_kind"], int(args["from_id"]),
        args["to_kind"], int(args["to_id"]),
        args["relation_kind"], source="manual",
    )
    return {"ok": rid is not None, "relation_id": rid}


SYNC_HANDLERS.update({
    "graph_query": tool_graph_query,
    "add_relation": tool_add_relation,
    "find_duplicates": tool_find_duplicates,
    "merge_people": tool_merge_people,
    "add_watch": tool_add_watch,
    "list_watches": tool_list_watches,
})


# /ask command — multi-hop reasoning using graph + cross_recall

ASK_SYSTEM = (
    "사용자가 사람·이벤트·골·노트·지출에 걸친 multi-hop 질문을 함. "
    "graph_query로 핵심 엔티티에서 BFS 탐색 + cross_recall로 silo별 합본. "
    "마지막에 한국어로 1-2문단 자연 요약. tool call은 최대 3개."
)


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text(
            "사용: /ask 경서랑 1년 안에 갔다온 여행 다 알려줘"
        )
        return
    q = " ".join(context.args).strip()
    db.log_chat(chat_id, "user", f"[/ask] {q}")
    await update.message.reply_text("🔎 그래프 traversal 중…")
    try:
        reply = await run_agent(
            chat_id,
            f"{ASK_SYSTEM}\n\n질문: {q}",
            history=[], max_hops=6, kind="ask",
        )
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i:i + 4000])
    except Exception as e:
        logger.exception("/ask failed")
        await update.message.reply_text(f"⚠ {e}")


# Slash commands


async def cmd_mission(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/mission <목표>` 시작 · `/mission cancel <id>` 취소 · `/mission <id>` 결과."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        await update.message.reply_text(
            "사용:\n"
            "  /mission <자연어 목표>  — 자율 프로젝트 시작\n"
            "  /mission cancel <id>    — 취소\n"
            "  /mission <id>           — 결과 보기"
        )
        return
    if args[0].lower() == "cancel" and len(args) >= 2:
        try:
            mid = int(args[1])
        except ValueError:
            await update.message.reply_text("mission_id는 숫자.")
            return
        ok = db.cancel_mission(chat_id, mid)
        await update.message.reply_text(f"⏹ Mission #{mid} 취소" if ok else "취소 실패 (없거나 이미 종료)")
        return
    # /mission <id> = show result
    if len(args) == 1 and args[0].isdigit():
        m = db.get_mission(int(args[0]))
        if not m or m["chat_id"] != chat_id:
            await update.message.reply_text("없는 mission")
            return
        text = (f"Mission #{m['id']} [{m['status']}] — {m['title']}\n"
                f"  • hops {m['current_hop']}/{m['max_hops']} · cost ${m['cost_usd_running']:.4f}\n\n"
                f"{(m['result_md'] or '(아직 결과 없음)')[:3500]}")
        await update.message.reply_text(text)
        return
    # Start new mission
    goal = " ".join(args).strip()
    title = goal[:40] + ("…" if len(goal) > 40 else "")
    mid = db.add_mission(chat_id, title, goal, max_hops=80)
    scheduler.trigger_mission_tick_now(mid)
    await update.message.reply_text(
        f"🚀 Mission #{mid} 시작 — '{title}'\n"
        "매 5분마다 1-3 hop 전진. 완성되면 알려줄게. /missions 로 진행 보기."
    )


async def cmd_missions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/missions` 모든 mission 목록 (최근 20)."""
    chat_id = update.effective_chat.id
    rows = db.list_missions(chat_id)
    if not rows:
        await update.message.reply_text("아직 mission 없음. /mission <목표> 로 시작.")
        return
    lines = ["🚀 Missions"]
    for r in rows[:20]:
        emoji = {"running": "▶️", "done": "✅", "failed": "❌",
                 "cancelled": "⏹", "paused": "⏸"}.get(r["status"], "·")
        lines.append(f"  {emoji} #{r['id']} [{r['status']}] {r['title'][:50]} "
                     f"— {r['current_hop']}/{r['max_hops']} hop · ${r['cost_usd_running']:.4f}")
    await update.message.reply_text("\n".join(lines))


# ---------------- v5: relation auto-extract ----------------


RELATION_EXTRACT_PROMPT = (
    "다음 이벤트/노트 기록에서 관계 (edges)를 추출해. "
    'JSON 배열만: [{{"from_kind": "event"|"note", "from_id": <id>, '
    '"to_kind": "person"|"event"|"goal", "to_id": <id>, "relation_kind": '
    '"attended"|"mentions"|"sponsored"|"location_of"|"paid_for"|"about"}}]. '
    "확실하지 않으면 비워. 한 호출당 최대 8개. JSON만, 다른 텍스트 금지.\n\n"
    "기존 사람 후보:\n{people}\n\n"
    "어제 추가된 항목:\n{items}"
)


async def run_relation_extract(chat_id: int) -> None:
    """Daily 04:00 — find yesterday's new events/notes, ask LLM to identify
    relations to existing people / other events, INSERT into relations table."""
    if _toggle_off_local(chat_id, "graph_extract_enabled"):
        return
    yesterday_iso = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    # Yesterday's events
    with db._conn() as c:
        events = list(c.execute(
            "SELECT id, title, notes, location FROM events "
            "WHERE chat_id=? AND created_at>=? LIMIT 20",
            (chat_id, yesterday_iso)))
        notes = list(c.execute(
            "SELECT id, content FROM notes "
            "WHERE chat_id=? AND created_at>=? LIMIT 20",
            (chat_id, yesterday_iso)))
    if not events and not notes:
        return
    people = [(p["id"], p["name"]) for p in db.list_people(chat_id)]
    items_md_parts = []
    for e in events:
        items_md_parts.append(
            f"event #{e['id']}: title={e['title']!r}, notes={(e['notes'] or '')[:80]!r}, "
            f"location={e['location']!r}")
    for n in notes:
        items_md_parts.append(f"note #{n['id']}: {(n['content'] or '')[:120]!r}")
    items_md = "\n".join(items_md_parts)
    people_md = "\n".join(f"person #{pid}: {name}" for pid, name in people) or "(없음)"
    prompt = RELATION_EXTRACT_PROMPT.format(people=people_md, items=items_md)
    try:
        data = await chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="relation_extract", max_tokens=600,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("relation extract LLM failed")
        return
    import re as _re
    m = _re.search(r"\[[\s\S]*\]", content)
    if not m:
        return
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return
    if not isinstance(parsed, list):
        return
    inserted = 0
    for rel in parsed[:12]:
        try:
            rid = db.add_relation(
                chat_id,
                rel["from_kind"], int(rel["from_id"]),
                rel["to_kind"], int(rel["to_id"]),
                rel["relation_kind"],
                source="auto_extract",
            )
            if rid:
                inserted += 1
        except Exception:
            continue
    if inserted:
        logger.info("relation_extract chat=%s inserted=%d", chat_id, inserted)


# ---------------- v5: self-improving reinforcement ----------------


SELF_IMPROVE_PROMPT = (
    "다음은 봇 nudge 반응 통계 + 사용자 persona snapshot이야. "
    "사용자가 *반복적으로* 무시 (≥70% no_response/dismissed) 또는 *명시적으로 끔* "
    "(toggled_off) 하는 nudge가 있으면 자동으로 OFF로 전환할 추천 + 봇 자체 톤/말투 "
    "개선 한 줄을 JSON으로 답해. 추가 변경 사항이 없다면 빈 객체 {{}} 반환.\n\n"
    'JSON: {{"disable_nudges": ["<fact_key>", ...], "tone_override": "<≤300자 한국어 메모>"}}\n\n'
    "nudge stats (지난 4주):\n{stats}\n\n"
    "persona snapshot:\n{persona}\n\n"
    "현재 봇 override:\n{current_override}"
)


# nudge_kind → fact key used by /nudges to disable it
_NUDGE_FACT_MAP = {
    "midday_checkin": "midday_checkin_enabled",
    "evening_preview": "weather_preview_enabled",
    "morning_briefing": "briefing_enabled",
    "evening_reflection": "reflection_enabled",
    "gmail_event_scan": "gmail_event_scan_enabled",
    "goal_milestone": "goal_milestone_enabled",
    "birthday_solo": "birthday_alert_separate_enabled",
    "late_check": "leave_by_enabled",
}


async def run_self_improve(chat_id: int) -> None:
    """Sun 10:00 — meta-LLM looks at nudge engagement + persona, optionally
    auto-toggles noisy nudges OFF and writes a tone override."""
    if _toggle_off_local(chat_id, "self_improve_enabled"):
        return
    stats = db.nudge_stats_by_kind(chat_id, days=28)
    if not stats:
        return  # nothing to learn from
    persona = db.get_latest_persona(chat_id)
    cur_override = db.get_prompt_override(chat_id) or "(없음)"
    prompt = SELF_IMPROVE_PROMPT.format(
        stats=json.dumps(stats, ensure_ascii=False),
        persona=(persona["content_md"][:1500] if persona else "(없음)"),
        current_override=cur_override,
    )
    try:
        data = await chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="self_improve", max_tokens=500,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("self_improve LLM failed")
        return
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", content)
    if not m:
        return
    try:
        parsed = json.loads(m.group(0))
    except Exception:
        return
    changes: List[str] = []
    disabled_names: List[str] = []
    for fact_key in (parsed.get("disable_nudges") or []):
        if fact_key in {v for v in _NUDGE_FACT_MAP.values()}:
            db.remember_fact(chat_id, fact_key, "false")
            scheduler.disable_daily_rhythm_for(chat_id)
            scheduler.ensure_daily_rhythm_for(chat_id)
            disabled_names.append(fact_key.replace("_enabled", ""))
            changes.append(fact_key)
    tone = (parsed.get("tone_override") or "").strip()
    if tone:
        db.set_prompt_override(chat_id, tone[:600])
        changes.append("tone_override")
    # v10: 한 줄. 사용자가 자세히 보고 싶으면 /improvements 직접.
    if disabled_names and _app and _app.bot:
        try:
            await _app.bot.send_message(
                chat_id=chat_id,
                text=f"🔕 {', '.join(disabled_names)} 자동 OFF.",
            )
            db.log_agent_action(
                chat_id, "self_improve",
                summary=f"자기 조정 {len(changes)}건",
                payload={"changes": changes, "raw": content[:300]},
                reversible={"kind": "prompt_override_revert",
                             "args": {"previous": cur_override}},
            )
        except Exception:
            logger.exception("self_improve notify failed")


async def cmd_improvements(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/improvements` 최근 자기 조정 기록 · `/improvements undo` 모든 override 제거."""
    chat_id = update.effective_chat.id
    arg = (context.args[0] if context.args else "").strip().lower()
    if arg == "undo":
        if db.clear_prompt_override(chat_id):
            await update.message.reply_text("🔄 self-improve override 제거됨.")
        else:
            await update.message.reply_text("override 없음.")
        return
    # Show recent self_improve agent_actions
    with db._conn() as c:
        rows = list(c.execute(
            "SELECT * FROM agent_actions WHERE chat_id=? AND action_kind='self_improve' "
            "ORDER BY executed_at DESC LIMIT 10", (chat_id,)))
    current = db.get_prompt_override(chat_id)
    lines = ["🤖 자기 조정 이력"]
    if not rows:
        lines.append("  (아직 없음 — Sun 10:00에 자동 실행)")
    for r in rows:
        lines.append(f"  • {r['executed_at'][:10]} — {r['summary']}")
    if current:
        lines.append("")
        lines.append("📝 현재 톤 override:")
        lines.append("  " + current[:300])
    lines.append("")
    lines.append("취소: /improvements undo")
    await update.message.reply_text("\n".join(lines))


async def cmd_metrics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/metrics` — observability pulse: 24h errors, cron health, cost trend."""
    chat_id = update.effective_chat.id
    errs = db.recent_errors(hours=24, limit=200)
    by_src = db.error_counts_by_source(hours=24)
    cost = db.usage_summary(chat_id)
    by_model = cost.get("by_model_month", [])
    # Cron snapshot via scheduler introspection
    jobs = scheduler._scheduler.get_jobs() if scheduler._scheduler else []
    pending_gcal = len([r for r in db.list_pending_gcal_sync() if r["chat_id"] == chat_id])
    nudge_stats = db.nudge_stats_by_kind(chat_id, days=7)
    miss_running = len([m for m in db.list_missions(chat_id, status="running")])

    lines = ["📈 시스템 헬스 (24h)"]
    lines.append(f"  • 에러 {len(errs)}건 — 상위:")
    for s in by_src[:5]:
        lines.append(f"      - {s['source']}: {s['n']}")
    if not by_src:
        lines.append("      (없음)")
    lines.append(f"  • 활성 cron job {len(jobs)}개")
    lines.append(f"  • GCal pending sync {pending_gcal}건")
    lines.append(f"  • 진행 중 mission {miss_running}건")
    lines.append("")
    lines.append("💰 OpenRouter 비용")
    lines.append(f"  • 오늘 ${cost['today']['cost']:.4f} / 이번달 ${cost['month']['cost']:.4f}")
    if by_model:
        lines.append("  • 모델별 (30일):")
        for r in by_model[:5]:
            lines.append(f"      - {r['model']}: {r['n']}콜 ${r['cost']:.4f}")
    lines.append("")
    lines.append("📣 nudge 반응 (7일)")
    if nudge_stats:
        for s in nudge_stats[:8]:
            engaged = s.get("engaged") or 0
            total = s.get("sent") or 0
            rate = (engaged * 100 // total) if total else 0
            lines.append(f"  • {s['nudge_kind']}: {total} sent, {engaged} engaged ({rate}%)")
    else:
        lines.append("  (아직 기록 없음)")
    await update.message.reply_text("\n".join(lines))


async def cmd_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/models` — current routing table + override syntax."""
    chat_id = update.effective_chat.id
    import model_router
    table = model_router.routing_table()
    overrides = [(f["key"], f["value"]) for f in db.list_facts(chat_id)
                 if f["key"].startswith("model_override_")]
    lines = ["🧠 모델 라우팅"]
    # group by tier
    tiers: Dict[str, List[str]] = {}
    for kind, model in table.items():
        tiers.setdefault(model, []).append(kind)
    for model, kinds in sorted(tiers.items()):
        lines.append(f"  • {model}")
        lines.append(f"      {', '.join(sorted(kinds))}")
    if overrides:
        lines.append("")
        lines.append("👤 내 override:")
        for k, v in overrides:
            lines.append(f"  • {k[len('model_override_'):]} → {v}")
    lines.append("")
    lines.append("override: `remember_fact model_override_chat anthropic/claude-opus-4.8`")
    await update.message.reply_text("\n".join(lines))


async def cmd_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """v10: 오늘 + 임박 골 D-3 + 활성 mission. 스트릭/% 0. 빈 섹션 생략."""
    chat_id = update.effective_chat.id
    today_local = datetime.now(TZ).date()
    now_local = datetime.now(TZ)
    end_today = now_local.replace(hour=23, minute=59, second=59)
    items = await _merge_schedule(
        chat_id, now_local.astimezone(timezone.utc),
        end_today.astimezone(timezone.utc))
    goals_imminent = [
        g for g in db.goals_due_within(chat_id, days=3) if g["target_date_local"]
    ]
    missions = db.list_missions(chat_id, status="running")

    bdays = []
    for p in db.list_people(chat_id):
        for d in json.loads(p["important_dates_json"] or "[]"):
            if _date_matches_today(d, today_local):
                bdays.append(f"{p['name']}: {d['label']}")

    lines = [f"📋 {today_local.strftime('%m월 %d일 (%a)')}", ""]
    if items:
        for it in items[:6]:
            when = it["when_utc"].astimezone(TZ).strftime("%H:%M")
            loc = f" @{it['location']}" if it.get("location") else ""
            lines.append(f"  • {when} {it['title']}{loc}")
        lines.append("")
    for g in goals_imminent[:3]:
        d = _days_until(g["target_date_local"])
        lines.append(f"🎯 {g['title']} D-{d}")
    for m in missions[:3]:
        lines.append(f"🚀 #{m['id']} {m['title']} ({m['current_hop']}/{m['max_hops']})")
    for b in bdays[:3]:
        lines.append(f"🎂 {b}")
    while lines and not lines[-1]:
        lines.pop()
    await update.message.reply_text("\n".join(lines))


async def cmd_now(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """v10: 지금 알아야 할 4줄. 다음 일정 / 활성 mission / 미답 메일 / D-3 골."""
    chat_id = update.effective_chat.id
    now_utc = datetime.now(timezone.utc)
    end_today = datetime.now(TZ).replace(hour=23, minute=59, second=59).astimezone(timezone.utc)
    items = await _merge_schedule(chat_id, now_utc, end_today)
    missions = db.list_missions(chat_id, status="running")
    goals_imminent = [
        g for g in db.goals_due_within(chat_id, days=3) if g["target_date_local"]
    ]
    lines = []
    if items:
        nxt = items[0]
        when = nxt["when_utc"].astimezone(TZ).strftime("%H:%M")
        loc = f" @{nxt['location']}" if nxt.get("location") else ""
        lines.append(f"📅 다음 {when} {nxt['title']}{loc}")
    for m in missions[:1]:
        lines.append(f"🚀 #{m['id']} {m['title']} ({m['current_hop']}/{m['max_hops']})")
    for g in goals_imminent[:1]:
        d = _days_until(g["target_date_local"])
        lines.append(f"🎯 {g['title']} D-{d}")
    if not lines:
        lines.append("(조용한 시간)")
    await update.message.reply_text("\n".join(lines))


async def cmd_translate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/translate <텍스트>` 또는 `/translate en <텍스트>` — 한국어↔외국어 통역.
    target 미지정 시 입력 언어 감지 → 반대 방향 (외국어→한국어 또는 한국어→영어)."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if not args:
        await update.message.reply_text(
            "사용: /translate <텍스트>\n"
            "     /translate en|ja|zh|ko <텍스트>")
        return
    target = None
    if args[0].lower() in ("ko", "en", "ja", "zh"):
        target = args[0].lower()
        text = " ".join(args[1:]).strip()
    else:
        text = " ".join(args).strip()
    if not text:
        await update.message.reply_text("텍스트 없음.")
        return
    if not target:
        src = await translator.detect_language(text)
        target = "ko" if src != "ko" else "en"
    result = await translator.translate(text, target)
    if not result:
        await update.message.reply_text("⚠ 통역 실패")
        return
    await update.message.reply_text(result)


# ---------------- v12 W7: /expert ----------------


async def cmd_expert(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/expert <도메인> [주제]` 시작 · `/expert end` 종료 · `/expert` 상태."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    active = db.active_expert_session(chat_id)
    if not args:
        if active:
            await update.message.reply_text(
                f"🎓 활성: [{active['domain']}] {active['topic'] or ''}\n"
                "종료: /expert end")
        else:
            domains = ", ".join(experts.PERSONAS.keys())
            await update.message.reply_text(
                f"도메인: {domains}\n"
                "사용: /expert <도메인> [주제]\n예: /expert legal 임대 계약서 점검")
        return
    if args[0].lower() in ("end", "exit", "끝", "종료"):
        if active:
            db.end_expert_session(chat_id, active["id"])
            await update.message.reply_text("✓ 일반 모드로 복귀")
        else:
            await update.message.reply_text("활성 세션 없음")
        return
    domain = experts.resolve_domain(args[0])
    if not domain:
        await update.message.reply_text(
            f"모르는 도메인. 가능: {', '.join(experts.PERSONAS.keys())}")
        return
    persona = experts.get_persona(domain)
    topic = " ".join(args[1:]).strip() or None
    if active:
        db.end_expert_session(chat_id, active["id"])
    sid = db.create_expert_session(chat_id, domain, persona["system"], topic)
    msg = f"{persona['disclaimer']}\n— session #{sid}"
    if topic:
        msg += f" · {topic}"
    await update.message.reply_text(msg)


# ---------------- v12 W5: /relationships ----------------


async def cmd_relationships(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/relationships` 이번 주 top + 식어가는 사람. 점수 비공개."""
    chat_id = update.effective_chat.id
    now = datetime.now(timezone.utc)
    week_iso = crm._week_iso(now - timedelta(days=3))
    top = db.top_relationships(chat_id, week_iso, limit=10)
    fading = db.fading_relationships(chat_id, threshold_pct=0.5)
    lines = []
    if top:
        lines.append("이번 주 자주 본 사람")
        for r in top[:6]:
            if r["meeting_count"] == 0 and r["message_count"] == 0:
                continue
            sig = []
            if r["meeting_count"]:
                sig.append(f"만남 {r['meeting_count']}")
            if r["message_count"]:
                sig.append(f"언급 {r['message_count']}")
            lines.append(f"  • {r['name']} — {' · '.join(sig) or '활동 없음'}")
    if fading:
        lines.append("")
        lines.append("식어가는 듯")
        for f in fading[:2]:
            days = db.last_contact_days_ago(chat_id, f["person_id"])
            day_str = f"{days}일째" if days is not None else "?"
            lines.append(f"  • {f['name']} — {day_str}")
    if not lines:
        lines.append("(이번 주 데이터 부족)")
    await update.message.reply_text("\n".join(lines))


# ---------------- v12 W2: /charges ----------------


async def cmd_charges(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/charges` 등록 구독 · `/charges cancel <id>` 해지 표시 ·
    `/charges scan` 즉시 자동 후보 식별."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if args and args[0].lower() == "cancel" and len(args) >= 2:
        try:
            cid = int(args[1])
        except ValueError:
            await update.message.reply_text("id는 숫자.")
            return
        ok = db.cancel_recurring_charge(chat_id, cid)
        await update.message.reply_text(
            f"🗑 #{cid} 해지" if ok else "없거나 이미 해지")
        return
    if args and args[0].lower() == "scan":
        await update.message.reply_text("스캔 중…")
        try:
            cands = await finance.detect_new_subscriptions(chat_id)
        except Exception as e:
            await update.message.reply_text(f"⚠ {e}")
            return
        if not cands:
            await update.message.reply_text("새 구독 후보 없음.")
            return
        lines = ["새 구독 후보"]
        for c in cands:
            lines.append(f"  • {c['merchant']}: ₩{c['amount_won']:,} ({c['occurrences']}회)")
        lines.append("")
        lines.append("등록: 봇한테 '<merchant> 구독 등록해줘'")
        await update.message.reply_text("\n".join(lines))
        return
    rows = db.list_recurring_charges(chat_id)
    if not rows:
        await update.message.reply_text(
            "등록 구독 없음.\n"
            "자동 식별: /charges scan\n"
            "수동 등록: 봇한테 '넷플릭스 17000 매월 구독 등록해줘'")
        return
    lines = ["💳 활성 구독"]
    total = 0
    for r in rows:
        total += r["amount_won"]
        period_mark = "/월" if r["period"] == "monthly" else "/년"
        lines.append(f"  • #{r['id']} {r['merchant']}: ₩{r['amount_won']:,}{period_mark}")
    lines.append("")
    lines.append(f"월 합계 ≈ ₩{total:,}")
    lines.append("해지: /charges cancel <id>")
    await update.message.reply_text("\n".join(lines))


# ---------------- v12 W4: /write ----------------


async def cmd_write(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/write <목적>` 시작 · `/write_continue` 다음 · `/write_done` 종료."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    cur = db.current_writing(chat_id)
    if not args:
        if cur:
            await update.message.reply_text(
                f"✍️ 활성: #{cur['id']} {cur['purpose']}\n"
                "다음: /write_continue · 종료: /write_done")
        else:
            await update.message.reply_text(
                "사용: /write <목적> [— 청중 — 글자수]\n"
                "예: /write 분기보고서 — 임원진 — 1500자")
        return
    purpose = " ".join(args).strip()
    # 가벼운 파싱: '—' 또는 '-' 구분자
    parts = [p.strip() for p in purpose.replace("—", "-").split("-")]
    audience = parts[1] if len(parts) > 1 else None
    length_str = parts[2] if len(parts) > 2 else None
    length_target = None
    if length_str:
        import re as _re
        m = _re.search(r"(\d+)", length_str)
        if m:
            length_target = int(m.group(1))
    wid = db.create_writing(chat_id, parts[0], audience=audience,
                              length_target=length_target)
    await update.message.reply_text(
        f"✍️ writing #{wid} — 톤 학습 + 개요 작성 중…")
    try:
        outline = await writing.build_outline(chat_id, wid)
    except Exception as e:
        await update.message.reply_text(f"⚠ {e}")
        return
    if not outline:
        await update.message.reply_text("⚠ 개요 생성 실패")
        return
    await update.message.reply_text(
        f"개요:\n\n{outline}\n\n"
        "이대로 OK면 `/write_continue` — 수정은 redline 답장")


async def cmd_write_continue(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    cur = db.current_writing(chat_id)
    if not cur:
        await update.message.reply_text("활성 writing 없음.")
        return
    next_section = (cur["current_section"] or 0) + 1
    await update.message.reply_text(f"섹션 {next_section} 작성 중…")
    try:
        text = await writing.build_section(chat_id, cur["id"], next_section)
    except Exception as e:
        await update.message.reply_text(f"⚠ {e}")
        return
    if not text:
        await update.message.reply_text("⚠ 섹션 생성 실패")
        return
    for i in range(0, len(text), 4000):
        await update.message.reply_text(text[i:i + 4000])


async def cmd_write_done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    cur = db.current_writing(chat_id)
    if not cur:
        await update.message.reply_text("활성 writing 없음.")
        return
    db.update_writing(cur["id"], status="done")
    await update.message.reply_text(f"✓ writing #{cur['id']} 완료")


# ---------------- v12 W6: /plan_trip /trips ----------------


async def cmd_plan_trip(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/plan_trip <목적지> <YYYY-MM-DD> <YYYY-MM-DD> [인원] [예산만]`"""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if len(args) < 3:
        await update.message.reply_text(
            "사용: /plan_trip <목적지> <시작> <끝> [인원] [예산만]\n"
            "예: /plan_trip 발리 2026-12-20 2026-12-23 2 800")
        return
    destination = args[0]
    start_date = args[1]
    end_date = args[2]
    try:
        datetime.fromisoformat(start_date)
        datetime.fromisoformat(end_date)
    except ValueError:
        await update.message.reply_text("날짜는 YYYY-MM-DD")
        return
    party_size = int(args[3]) if len(args) > 3 else 1
    budget_total = None
    if len(args) > 4:
        try:
            budget_total = int(args[4]) * 10000
        except ValueError:
            pass
    tid = trip_planner.start_trip(
        chat_id, destination, start_date, end_date,
        party_size=party_size, budget_total=budget_total)
    trip = db.get_trip(tid)
    mid = trip["mission_id"]
    scheduler.trigger_mission_tick_now(mid)
    await update.message.reply_text(
        f"🛫 trip #{tid} mission #{mid} 시작\n"
        f"  • {destination} {start_date}~{end_date} {party_size}인\n"
        f"  • 5분마다 진행. 완료 시 itinerary 알려줄게.")


async def cmd_trips(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_trips(chat_id)
    if not rows:
        await update.message.reply_text("trip 없음. /plan_trip 시작.")
        return
    lines = ["🛫 trips"]
    for r in rows[:10]:
        status_emoji = {"planning": "🟡", "done": "✓", "cancelled": "✗"}.get(
            r["status"], "·")
        lines.append(
            f"  {status_emoji} #{r['id']} {r['destination']} "
            f"{r['start_date']}~{r['end_date']}")
    await update.message.reply_text("\n".join(lines))


# ---------------- v12 W1: /negotiate /negotiations ----------------


async def cmd_negotiate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """`/negotiate <email> <주제>` 수동 협상 시작 — 봇이 빈 시간 3개 메일."""
    chat_id = update.effective_chat.id
    args = list(context.args or [])
    if len(args) < 2:
        await update.message.reply_text(
            "사용: /negotiate <email> <주제>\n예: /negotiate kim@x.com Q3 리뷰")
        return
    email = args[0]
    topic = " ".join(args[1:])
    slots = negotiator.find_free_slots(chat_id, duration_min=60, max_slots=3)
    if not slots:
        await update.message.reply_text("⚠ 향후 2주에 빈 시간 없음")
        return
    subject, body = negotiator.format_offer_email("", topic, slots)
    nid = db.create_negotiation(chat_id, email, topic)
    db.update_negotiation(nid, state="awaiting",
                            history_json=json.dumps([{
                                "kind": "bot_offer",
                                "slots": slots,
                                "at": datetime.now(timezone.utc).isoformat(),
                            }]))
    # Gmail 전송
    if not db.get_oauth_token(chat_id, "google"):
        await update.message.reply_text(
            f"⚠ Gmail 미연결. 메일 초안 below — 수동 전송:\n\n"
            f"To: {email}\nSubject: {subject}\n\n{body}")
        return
    try:
        await gmail_mod.send_message(chat_id, email, subject, body)
        await update.message.reply_text(
            f"📨 negotiation #{nid} → {email}\n"
            f"  • 제안 시간: {', '.join(slots)}")
    except Exception as e:
        await update.message.reply_text(f"⚠ 메일 전송 실패: {e}")


async def cmd_negotiations(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_negotiations(chat_id)
    if not rows:
        await update.message.reply_text("협상 없음. /negotiate 시작.")
        return
    lines = ["📨 negotiations"]
    for r in rows[:10]:
        emoji = {"proposed": "🟡", "awaiting": "⏳", "agreed": "✓",
                 "abandoned": "✗"}.get(r["state"], "·")
        lines.append(
            f"  {emoji} #{r['id']} {r['counterparty_email']} — {r['topic'][:40]}")
    await update.message.reply_text("\n".join(lines))


async def cmd_diag(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    pending_gcal = [r for r in db.list_pending_gcal_sync() if r["chat_id"] == chat_id]
    missed = db.missed_reminders(chat_id, hours=72)
    cost = db.usage_summary(chat_id)
    facts = db.list_facts(chat_id)
    people = db.list_people(chat_id)
    goals = db.list_goals(chat_id, status="open")
    expenses = db.summarize_expenses(chat_id, days=30)
    lines = [
        "🛠 진단",
        f"  • 누적 비용 이번달: ${cost['month']['cost']:.4f} ({cost['month']['n']}콜)",
        f"  • 등록 facts: {len(facts)} / 사람: {len(people)} / open 골: {len(goals)}",
        f"  • 최근 30일 지출 합계: ₩{expenses['total_won']:,} ({expenses['count']}건)",
        f"  • GCal pending sync: {len(pending_gcal)}",
        f"  • 최근 72h missed 리마인더: {len(missed)}",
    ]
    await update.message.reply_text("\n".join(lines))


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a snapshot of the user's local data as a markdown summary."""
    chat_id = update.effective_chat.id
    lines = ["# Dailylife 데이터 스냅샷", ""]
    lines.append("## Facts")
    for f in db.list_facts(chat_id):
        lines.append(f"- **{f['key']}**: {f['value']}")
    lines.append("")
    lines.append("## People")
    for p in db.list_people(chat_id):
        bits = [p["name"]]
        if p["role"]:
            bits.append(p["role"])
        lines.append(f"- {' · '.join(bits)}")
    lines.append("")
    lines.append("## Open goals")
    for g in db.list_goals(chat_id, status="open"):
        lines.append(f"- #{g['id']} {g['title']} (target {g['target_date_local']})")
    lines.append("")
    lines.append("## Recent notes (10)")
    for n in db.list_notes(chat_id, limit=10):
        lines.append(f"- {n['created_at'][:10]}: {n['content'][:120]}")
    body = "\n".join(lines)
    # Send as a regular text message (chunked) — user can copy/save
    for i in range(0, len(body), 4000):
        await update.message.reply_text(body[i:i + 4000])


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_history.pop(update.effective_chat.id, None)
    await update.message.reply_text("대화 메모리 초기화 완료. 일정/기억은 그대로 보존.")


SETUP_PROMPT = (
    "사용자가 /setup을 실행해서 온보딩 모드야. 친근한 한국어로 다음을 멀티턴 흐름처럼 한번에 안내하고, "
    "사용자가 '시작' 또는 '응' 등으로 응답하면 한 가지씩 받아서 remember_fact로 즉시 저장해. "
    "묻는 순서:\n"
    "  1) 어떻게 불러드리면 좋을지 (이름·별명) → fact 'preferred_name'\n"
    "  2) 집 주소나 동네 → fact 'home_address' (가능하면 kakao_local_search로 좌표도 조회)\n"
    "  3) 자대·직장 위치 → fact 'unit_location' or 'workplace'\n"
    "  4) 외출/외박 정기 패턴 (매주 X요일 등) → 필요하면 add_recurring_task로 안내 메시지 등록\n"
    "  5) 장기적으로 신경 쓰고 있는 큰 일정 (3개월+ 남은 것) — add_goal 여러 개\n"
    "  6) 매주 일요일 9시 KST에 골 리뷰 보내드릴 거라 안내. 끄려면 'goal_review_enabled=false' 기억해달라고 말하면 됨.\n"
    "한 번에 하나씩만 묻고, 사용자가 답하면 즉시 도구로 저장해. 친근하게."
)


async def cmd_setup(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    # Reset short-term history so the setup turn starts clean
    chat_history.pop(chat_id, None)
    history = _history(chat_id)
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        reply = await run_agent(chat_id, SETUP_PROMPT, history=history)
    except Exception as exc:
        logger.exception("/setup failed")
        await update.message.reply_text(f"⚠️ 온보딩 시작 실패: {exc}")
        return
    db.log_chat(chat_id, "assistant", reply)
    for i in range(0, len(reply), 4000):
        await update.message.reply_text(reply[i : i + 4000])


async def cmd_connect_gcal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not gcal.is_configured():
        await update.message.reply_text(
            "Google Calendar 연동이 아직 설정되지 않았어요. "
            "관리자가 GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET을 추가해야 합니다."
        )
        return
    auth_url = gcal.build_auth_url(chat_id)
    await update.message.reply_text(
        "📅 Google 캘린더 연동\n\n"
        "아래 링크에서 본인 Google 계정으로 로그인 + 캘린더 권한을 허용해주세요.\n"
        "허용이 끝나면 자동으로 이 채팅에 연동 완료 메시지가 옵니다.\n\n"
        f"{auth_url}\n\n"
        "(링크는 한 번만 사용 가능. 만료되면 /connect_gcal 다시 입력)"
    )


async def cmd_gcal_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    row = db.get_oauth_token(chat_id, "google")
    if not row:
        await update.message.reply_text("Google 캘린더 미연결. /connect_gcal 로 연결하세요.")
        return
    txt = (
        "✅ Google 캘린더 연결됨\n"
        f"• scope: {row['scopes']}\n"
        f"• 갱신: {row['updated_at']}\n"
        f"• 만료: {row['expires_at_utc']}"
    )
    await update.message.reply_text(txt)


async def cmd_disconnect_gcal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    ok = db.delete_oauth_token(chat_id, "google")
    await update.message.reply_text(
        "🔌 Google 캘린더 연동 해제됨." if ok else "이미 연동되어 있지 않아요."
    )


async def cmd_cost(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    s = db.usage_summary(chat_id)
    today = s["today"]
    month = s["month"]
    by_model = s["by_model_month"]
    lines = [
        "💰 OpenRouter 사용량",
        "",
        f"오늘:  {today['n']}콜  ·  ${today['cost']:.4f}  ·  in {today['pt']}/out {today['ct']} tok",
        f"이번달: {month['n']}콜  ·  ${month['cost']:.4f}  ·  in {month['pt']}/out {month['ct']} tok",
    ]
    if by_model:
        lines.append("")
        lines.append("모델별 (이번달):")
        for r in by_model[:5]:
            lines.append(f"  • {r['model']}: {r['n']}콜  ${r['cost']:.4f}")
    await update.message.reply_text("\n".join(lines))


# ---------------- inline-keyboard undo ----------------


def _undo_keyboard(token: str, label: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text=f"↩️ {label}", callback_data=f"undo:{token}")]]
    )


def _collect_undo_offers(history_after_run: List[Dict]) -> List[tuple]:
    """Walk the most recent assistant->tool turns and collect any (token, label)
    from delete-style tool results so we can attach undo buttons."""
    offers = []
    for msg in history_after_run[-12:]:
        if msg.get("role") != "tool":
            continue
        try:
            content = json.loads(msg.get("content") or "{}")
        except Exception:
            continue
        token = content.get("undo_token")
        label = content.get("undo_label")
        if token and label:
            offers.append((token, label))
    return offers


async def on_callback_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline action keyboard for events/goals — done / pause / remind."""
    cq = update.callback_query
    if not cq or not (cq.data or "").startswith("act:"):
        return
    parts = cq.data.split(":")
    if len(parts) < 3:
        await cq.answer("잘못된 액션")
        return
    _, kind, rest = parts[0], parts[1], ":".join(parts[2:])
    chat_id = cq.message.chat_id if cq.message else None
    if chat_id is None:
        await cq.answer("권한 없음")
        return
    try:
        if kind == "evt_done":
            eid = int(rest)
            row = db.get_event(eid)
            if not row or row["chat_id"] != chat_id:
                await cq.answer("이미 처리됨", show_alert=False)
                return
            payload = dict(row)
            db.delete_event(eid, chat_id)
            tok = secrets.token_urlsafe(8)
            db.push_deleted_audit(chat_id, "event", payload, tok)
            _pending_undos[tok] = ("event", chat_id)
            await cq.answer("완료 처리 — 취소 가능", show_alert=False)
            try:
                await cq.edit_message_reply_markup(
                    reply_markup=_undo_keyboard(tok, f"이벤트 #{eid} 완료 취소"))
            except Exception:
                pass
        elif kind == "evt_remind":
            eid = int(rest)
            row = db.get_event(eid)
            if row and row["chat_id"] == chat_id:
                await scheduler._send_reminder(eid)
                await cq.answer("리마인더 발사", show_alert=False)
            else:
                await cq.answer("없는 이벤트")
        elif kind == "goal_done":
            gid = int(rest)
            db.update_goal(gid, chat_id, status="done")
            await cq.answer("골 완료 ✅", show_alert=False)
            try:
                await cq.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
        elif kind == "goal_pause":
            gid = int(rest)
            db.update_goal(gid, chat_id, status="paused")
            await cq.answer("골 보류 ⏸", show_alert=False)
        elif kind == "mail_add":
            cand = _pending_mail_cards.pop(rest, None)
            if not cand or cand["chat_id"] != chat_id:
                await cq.answer("만료된 카드", show_alert=False)
                return
            try:
                when_utc = _parse_local_iso(cand["when_local"])
            except ValueError:
                await cq.answer("시각 파싱 실패", show_alert=True)
                return
            eid = db.add_event(
                chat_id, cand["title"], when_utc,
                notes=cand.get("source_summary"),
                remind_lead_minutes=30, location=cand.get("location"),
            )
            row = db.get_event(eid)
            if row:
                scheduler.schedule_for(row)
                if cand.get("location") and not _toggle_off_local(chat_id, "leave_by_enabled"):
                    asyncio.create_task(_try_schedule_leave_by(chat_id, row))
            db.mark_gmail_processed(chat_id, cand["msg_id"], "added")
            await cq.answer(f"✅ 일정 #{eid} 추가", show_alert=False)
            try:
                await cq.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"📌 추가 완료 — #{eid} {cand['title']} ({cand['when_local']})",
            )
        elif kind == "mail_skip":
            cand = _pending_mail_cards.pop(rest, None)
            if cand:
                db.mark_gmail_processed(cand["chat_id"], cand["msg_id"], "skipped")
            await cq.answer("건너뜀", show_alert=False)
            try:
                await cq.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
        elif kind in ("rsvp_yes", "rsvp_maybe", "rsvp_no"):
            response = {"rsvp_yes": "accepted", "rsvp_maybe": "tentative",
                        "rsvp_no": "declined"}[kind]
            try:
                ev = await gcal.respond_to_invite(chat_id, rest, response)
                db.log_agent_action(
                    chat_id, "gcal_rsvp",
                    summary=f"RSVP {response} → {(ev.get('summary') or '(제목 없음)')[:40]}",
                    payload={"event_id": rest, "response": response},
                    reversible={"kind": "gcal_rsvp",
                                 "args": {"event_id": rest, "response": "tentative"}},
                )
                await cq.answer(f"✅ {response}", show_alert=False)
                try:
                    await cq.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
            except Exception as e:
                await cq.answer(f"⚠ {e}", show_alert=True)
        elif kind == "agent_undo":
            aid = int(rest)
            row = db.get_agent_action(chat_id, aid)
            if not row or not row["reversible_json"]:
                await cq.answer("취소 불가", show_alert=False)
                return
            try:
                rev = json.loads(row["reversible_json"])
                if rev.get("kind") == "gcal_rsvp":
                    a = rev["args"]
                    await gcal.respond_to_invite(chat_id, a["event_id"], a["response"])
                    db.mark_action_reversed(chat_id, aid)
                    await cq.answer("취소 처리됨 (tentative로 되돌림)", show_alert=False)
                else:
                    await cq.answer("이 액션은 취소 미지원", show_alert=False)
            except Exception as e:
                await cq.answer(f"⚠ {e}", show_alert=True)
        elif kind in ("bank_add", "bank_skip"):
            cand = _pending_mail_cards.pop(rest, None)
            if not cand or cand.get("chat_id") != chat_id or cand.get("kind") != "bank_sms_confirm":
                await cq.answer("만료된 카드")
                return
            try:
                await cq.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
            if kind == "bank_skip":
                await cq.answer("건너뜀")
                return
            # bank_add
            eid = db.log_expense(chat_id, amount_won=cand["amount"],
                                  merchant=cand["merchant"], category=cand["category"],
                                  notes=f"[bank_sms] {cand['sms'][:120]}")
            db.log_agent_action(chat_id, "bank_sms_add",
                                 summary=f"SMS 기록 ₩{cand['amount']:,} {cand['merchant']}",
                                 payload={"sms": cand["sms"][:200], "expense_id": eid})
            await cq.answer("💳 기록됨")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"💳 expense #{eid} 저장 — ₩{cand['amount']:,} {cand['merchant']}",
            )
        elif kind in ("proto_send", "proto_accept", "proto_decline"):
            cand = _pending_late_cards.pop(rest, None)
            if not cand or cand.get("chat_id") != chat_id or cand.get("kind") != "protected_decline":
                await cq.answer("만료된 카드")
                return
            try:
                await cq.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
            if kind == "proto_accept":
                try:
                    await gcal.respond_to_invite(chat_id, cand["event_id"], "accepted")
                    db.log_agent_action(
                        chat_id, "gcal_rsvp",
                        summary="보호 시간 침범 수락",
                        payload={"event_id": cand["event_id"]},
                    )
                    await cq.answer("수락")
                except Exception as e:
                    await cq.answer(f"⚠ {e}", show_alert=True)
            elif kind == "proto_decline":
                try:
                    await gcal.respond_to_invite(chat_id, cand["event_id"], "declined")
                    db.log_agent_action(
                        chat_id, "gcal_rsvp",
                        summary="보호 시간 거절",
                        payload={"event_id": cand["event_id"]},
                    )
                    await cq.answer("거절")
                except Exception as e:
                    await cq.answer(f"⚠ {e}", show_alert=True)
            elif kind == "proto_send":
                to_addr = cand.get("to")
                if not to_addr:
                    await cq.answer("organizer 이메일 없음")
                    return
                try:
                    await gmail_mod.send_message(
                        chat_id, to_addr, cand["subject"], cand["body"])
                    await gcal.respond_to_invite(chat_id, cand["event_id"], "declined")
                    db.log_agent_action(
                        chat_id, "gmail_send",
                        summary=f"보호 시간 거절 메일 → {to_addr}",
                        payload={"to": to_addr, "subject": cand["subject"]},
                    )
                    await cq.answer(f"✅ {to_addr}에게 메일")
                except Exception as e:
                    await cq.answer(f"⚠ {e}", show_alert=True)
        elif kind in ("late_msg", "late_mail", "late_dismiss"):
            cand = _pending_late_cards.pop(rest, None)
            if not cand or cand["chat_id"] != chat_id:
                await cq.answer("만료된 카드", show_alert=False)
                return
            try:
                await cq.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass
            if kind == "late_dismiss":
                await cq.answer("OK", show_alert=False)
            elif kind == "late_msg":
                draft = (f"지금 가는 중인데 {cand['event_title']} 5분 정도 "
                         "늦을 것 같아요. 죄송합니다 🙏")
                await context.bot.send_message(chat_id=chat_id,
                                                text=f"📋 복사용 초안:\n\n{draft}")
                await cq.answer("초안 보냄", show_alert=False)
            elif kind == "late_mail":
                org = cand.get("organizer_email")
                if not org:
                    await cq.answer("organizer 이메일 없음", show_alert=False)
                    return
                subject = f"Re: {cand['event_title']} — 5분 정도 늦겠습니다"
                body = (
                    f"안녕하세요,\n\n{cand['event_title']}에 5분 정도 늦을 것 같습니다. "
                    "양해 부탁드립니다.\n감사합니다."
                )
                try:
                    res = await gmail_mod.send_message(chat_id, org, subject, body)
                    db.log_agent_action(
                        chat_id, "gmail_send",
                        summary=f"늦음 안내 메일 → {org} · {cand['event_title'][:40]}",
                        payload={"to": org, "subject": subject,
                                  "event_id": cand["event_id"],
                                  "gmail_message_id": res.get("id")},
                    )
                    await cq.answer(f"✅ {org}에게 메일", show_alert=False)
                except Exception as e:
                    logger.exception("late mail send failed")
                    await cq.answer(f"⚠ {e}", show_alert=True)
        elif kind == "bday_draft":
            pid = int(rest)
            people = [p for p in db.list_people(chat_id) if p["id"] == pid]
            if not people:
                await cq.answer("없는 사람", show_alert=False)
                return
            p = people[0]
            await cq.answer("초안 생성 중…")
            prompt = (
                f"한국어로 {p['name']}({p['role'] or '지인'})에게 보낼 짧고 따뜻한 "
                f"생일/기념일 축하 메시지 초안 1-2줄만 써. 이모지 1개 정도 OK. "
                f"인사말이나 부연 설명 없이 메시지 본문만."
            )
            try:
                data = await chat_completion(
                    [{"role": "user", "content": prompt}],
                    tools=None, chat_id=chat_id, kind="bday_draft", max_tokens=200,
                )
                draft = (data["choices"][0]["message"].get("content") or "").strip()
            except Exception:
                draft = f"{p['name']}야, 오늘 좋은 하루 보내! 🎂"
            await context.bot.send_message(
                chat_id=chat_id, text=f"📩 {p['name']}에게:\n\n{draft}",
            )
        else:
            await cq.answer(f"미구현 액션: {kind}")
    except Exception as e:
        logger.exception("callback action failed")
        await cq.answer(f"⚠ {e}", show_alert=True)


async def on_callback_undo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    cq = update.callback_query
    if not cq or not (cq.data or "").startswith("undo:"):
        return
    token = cq.data.split(":", 1)[1]
    audit = db.get_deleted_audit(token)
    if not audit:
        await cq.answer("이미 처리됐거나 만료된 취소 요청이에요.", show_alert=True)
        return
    chat_id = audit["chat_id"]
    if cq.from_user and cq.message and cq.message.chat_id != chat_id:
        await cq.answer("권한 없음.", show_alert=True)
        return
    payload = json.loads(audit["payload_json"])
    kind = audit["kind"]
    msg = ""
    if kind == "event":
        eid = db.add_event(
            chat_id,
            payload["title"],
            datetime.fromisoformat(payload["when_utc"]),
            payload.get("notes"),
            payload.get("remind_lead_minutes"),
        )
        new_row = db.get_event(eid)
        if new_row:
            scheduler.schedule_for(new_row)
        msg = f"이벤트 복구 완료 (#{eid})."
    elif kind == "fact":
        db.remember_fact(chat_id, payload["key"], payload["value"])
        msg = f"기억 '{payload['key']}' 복구 완료."
    elif kind == "recurring":
        tid = db.add_recurring_task(chat_id, payload["cron_kst"], payload["prompt"])
        new_row = db.get_recurring_task(tid)
        if new_row:
            scheduler.schedule_recurring(new_row)
        msg = f"정기 작업 복구 완료 (#{tid})."
    else:
        msg = "복구 미지원 종류."
    db.mark_audit_restored(audit["id"])
    _pending_undos.pop(token, None)
    await cq.answer("복구 완료.", show_alert=False)
    try:
        await cq.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass
    await context.bot.send_message(chat_id=chat_id, text=f"↩️ {msg}")


async def _process_user_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_text: str,
    *,
    log_prefix: str = "",
) -> None:
    """Shared agent dispatch used by text, voice, and photo handlers."""
    chat_id = update.effective_chat.id
    db.log_chat(chat_id, "user", f"{log_prefix}{user_text}")
    # v12 W3: 짧은 메시지 + 액션 키워드면 가까운 일정을 컨텍스트로 prepend
    if not _toggle_off_local(chat_id, "context_inference_enabled"):
        try:
            ctx_note = _infer_short_message_context(chat_id, user_text)
            if ctx_note:
                user_text = f"{ctx_note}\n{user_text}"
        except Exception:
            logger.exception("context inference failed (non-fatal)")
    # If we asked for a reflection today and haven't captured it yet, this
    # message is the response (best-effort heuristic — works for short replies).
    try:
        today_iso = datetime.now(TZ).date().isoformat()
        state = db.get_daily_state(chat_id, today_iso)
        if state and state["reflection_prompted"] and not state["reflection_response"]:
            db.save_reflection_response(chat_id, today_iso, user_text)
            # Fire-and-forget mood classifier (~$0.0002) — fills /scorecard mood bar.
            asyncio.create_task(_classify_and_store_mood(chat_id, today_iso, user_text))
    except Exception:
        logger.exception("reflection capture failed (non-fatal)")
    # Opportunistic cleanup of idle chats (no extra cost — only sweeps every msg).
    _sweep_idle_chats()
    history = _history(chat_id)
    # Compact long history before the next call (saves cost + improves recall).
    await _compact_history_if_needed(chat_id)
    history = _history(chat_id)  # refresh in case compaction replaced contents
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        reply = await run_agent(chat_id, user_text, history=history)
    except Exception as exc:
        logger.exception("agent loop failed")
        await update.message.reply_text(f"⚠️ 처리 실패: {exc}")
        if history and history[-1].get("role") == "user":
            history.pop()
        return

    db.log_chat(chat_id, "assistant", reply)
    _trim_history(chat_id)
    undo_offers = _collect_undo_offers(history)
    for i in range(0, len(reply), 4000):
        chunk = reply[i : i + 4000]
        # Attach undo button (only first one) to the FINAL message chunk.
        is_last = i + 4000 >= len(reply)
        if is_last and undo_offers:
            token, label = undo_offers[0]
            await update.message.reply_text(chunk, reply_markup=_undo_keyboard(token, label))
        else:
            await update.message.reply_text(chunk)
    # v5: voice-reply chain — if voice mode is active AND reply is short, also
    # send TTS audio so the conversation feels truly bidir.
    if _is_voice_reply_active(chat_id) and reply.strip() and len(reply) <= 500:
        try:
            audio = await transcribe.synthesize_voice(reply)
            await context.bot.send_voice(chat_id=chat_id, voice=audio)
        except Exception:
            logger.exception("voice reply TTS failed (non-fatal)")


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not (update.message.voice or update.message.audio):
        return
    chat_id = update.effective_chat.id
    voice = update.message.voice or update.message.audio
    mime = getattr(voice, "mime_type", "audio/ogg") or "audio/ogg"
    logger.info("voice from %s: duration=%s mime=%s", chat_id, voice.duration, mime)

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        tg_file = await voice.get_file()
        bio = await tg_file.download_as_bytearray()
        text = await transcribe.transcribe_voice(bytes(bio), mime)
    except transcribe.TranscribeUnavailable:
        await update.message.reply_text(
            "음성 인식이 아직 켜져 있지 않아요. OPENAI_API_KEY를 Railway에 추가하면 켜집니다."
        )
        return
    except Exception as exc:
        logger.exception("transcribe failed")
        await update.message.reply_text(f"⚠️ 음성 처리 실패: {exc}")
        return

    if not text:
        await update.message.reply_text("음성에서 텍스트를 찾지 못했어요.")
        return

    logger.info("transcribed (%d chars): %r", len(text), text[:160])
    await update.message.reply_text(f"🎙️ 들었어요: {text[:300]}\n\n처리 중…")
    await _process_user_text(update, context, text, log_prefix="[voice] ")


DOC_IMAGE_MIMES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}
DOC_MAX_BYTES = 19 * 1024 * 1024  # Telegram bot file size cap is 20MB


async def on_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle PDF or image-as-document attachments. Extract content and feed
    through the agent so it picks the right tool (save_note / add_event / ...)."""
    if not update.message or not update.message.document:
        return
    chat_id = update.effective_chat.id
    doc = update.message.document
    mime = (doc.mime_type or "").lower()
    fname = doc.file_name or "file"
    caption = (update.message.caption or "").strip()
    size = doc.file_size or 0
    logger.info("doc from %s: name=%r mime=%s size=%s caption=%r",
                chat_id, fname, mime, size, caption[:80])

    if size and size > DOC_MAX_BYTES:
        await update.message.reply_text(
            f"⚠ 파일이 너무 커요 ({size // (1024*1024)}MB). "
            "Telegram 봇 한도가 20MB라서 못 받아요."
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        tg_file = await doc.get_file()
        bio = await tg_file.download_as_bytearray()
    except Exception as exc:
        logger.exception("doc download failed")
        await update.message.reply_text(f"⚠ 파일 받기 실패: {exc}")
        return

    extracted = ""
    extraction_kind = ""

    if mime == "application/pdf" or fname.lower().endswith(".pdf"):
        extraction_kind = "pdf"
        extracted = transcribe.extract_pdf_text(bytes(bio))
        if not extracted:
            await update.message.reply_text(
                "📄 PDF에서 텍스트를 못 뽑았어요 (스캔본이거나 암호화된 듯). "
                "필요하면 사진으로 다시 찍어 보내주세요 — 비전으로 읽어볼게요."
            )
            return
        preview = extracted[:600] + ("…" if len(extracted) > 600 else "")
        await update.message.reply_text(f"📄 PDF 텍스트 추출 ({len(extracted)}자):\n\n{preview}\n\n처리 중…")

    elif mime in DOC_IMAGE_MIMES or fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        extraction_kind = "image"
        try:
            extracted = await transcribe.describe_image(bytes(bio), mime=mime or "image/jpeg",
                                                         caption=caption or None)
        except Exception as exc:
            logger.exception("vision (doc) failed")
            await update.message.reply_text(f"⚠ 이미지 처리 실패: {exc}")
            return
        preview = extracted[:1200]
        await update.message.reply_text(f"🖼️ 이미지에서 추출:\n\n{preview}\n\n처리 중…")

    else:
        await update.message.reply_text(
            f"📎 {fname} ({mime or '알 수 없는 형식'}) — 아직 PDF랑 이미지만 읽을 수 있어요."
        )
        return

    # Cheap classifier hint to help the main agent pick the right tool fast.
    try:
        cls = await transcribe.classify_content(extracted, hint=caption or fname)
    except Exception:
        cls = {"kind": "none", "confidence": 0.0, "summary": ""}
    user_text = (
        f"[{extraction_kind} 첨부 · file={fname} · "
        f"classified={cls['kind']}({cls['confidence']:.1f})]\n"
        f"요약: {cls['summary']}\n"
        f"caption: {caption!r}\n\n"
        f"추출된 내용:\n{extracted}"
    )
    await _process_user_text(update, context, user_text, log_prefix=f"[{extraction_kind}] ")


async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return
    chat_id = update.effective_chat.id
    caption = (update.message.caption or "").strip()
    photo = update.message.photo[-1]   # highest resolution
    logger.info("photo from %s: %sx%s caption=%r", chat_id, photo.width, photo.height, caption[:80])

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    try:
        tg_file = await photo.get_file()
        bio = await tg_file.download_as_bytearray()
        description = await transcribe.describe_image(bytes(bio), mime="image/jpeg", caption=caption or None)
    except Exception as exc:
        logger.exception("vision failed")
        await update.message.reply_text(f"⚠️ 사진 처리 실패: {exc}")
        return

    logger.info("vision (%d chars): %r", len(description), description[:160])
    await update.message.reply_text(f"📷 사진에서 추출:\n\n{description[:1500]}\n\n처리 중…")

    try:
        cls = await transcribe.classify_content(description, hint=caption or None)
    except Exception:
        cls = {"kind": "none", "confidence": 0.0, "summary": ""}
    user_text = (
        f"[사진 첨부 · classified={cls['kind']}({cls['confidence']:.1f})]\n"
        f"요약: {cls['summary']}\n"
        f"caption: {caption!r}\n"
        f"추출된 정보:\n{description}"
    )
    await _process_user_text(update, context, user_text, log_prefix="[photo] ")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    user_text = update.message.text
    # v6: Forward detection — if the user forwarded a message from elsewhere
    # (a friend's KakaoTalk-style note, a channel announcement, an email
    # snippet), prepend a header so the agent treats it as third-party content
    # to summarize/extract/file rather than as a direct conversation turn.
    msg = update.message
    prefix = ""
    if getattr(msg, "forward_origin", None) or getattr(msg, "forward_from", None) \
            or getattr(msg, "forward_from_chat", None) or getattr(msg, "forward_sender_name", None):
        sender = (getattr(msg, "forward_sender_name", None)
                  or (getattr(msg, "forward_from", None) and msg.forward_from.full_name)
                  or (getattr(msg, "forward_from_chat", None) and msg.forward_from_chat.title)
                  or "(이름 없음)")
        prefix = f"[forwarded from {sender}]\n"
    logger.info("msg from %s%s: %r", update.effective_chat.id,
                 " (forward)" if prefix else "", user_text[:200])
    await _process_user_text(update, context, prefix + user_text)


BOT_COMMANDS: List[BotCommand] = [
    # v10: 메뉴 17개만. 나머지 명령은 handler 살아있어서 직접 타이핑하면 작동.
    BotCommand("now", "지금 알아야 할 것"),
    BotCommand("today", "오늘 일정"),
    BotCommand("week", "이번 주"),
    BotCommand("agenda", "앞으로 60일"),
    BotCommand("notes", "최근 메모"),
    BotCommand("facts", "기억"),
    BotCommand("people", "사람"),
    BotCommand("goals", "장기 골"),
    BotCommand("spending", "이번 달 지출"),
    BotCommand("habits", "습관"),
    BotCommand("mission", "자율 프로젝트"),
    BotCommand("dashboard", "오늘 상태"),
    BotCommand("nudges", "알림 토글"),
    BotCommand("rules", "자동 액션 규칙"),
    BotCommand("connect_gcal", "Google 연동"),
    BotCommand("setup", "온보딩"),
    BotCommand("help", "사용법"),
]


async def post_init(app: Application) -> None:
    global _app
    _app = app
    db.init_db()
    scheduler.init(
        app.bot,
        run_recurring_task,
        run_weekly_goal_review,
        run_daily_imminent_check,
        morning_briefing_runner=run_morning_briefing,
        evening_reflection_runner=run_evening_reflection,
        gmail_event_scan_runner=run_gmail_event_scan,
        evening_preview_runner=run_evening_preview,
        birthday_solo_runner=run_birthday_solo,
        midday_checkin_runner=run_midday_checkin,
        leave_by_recompute_runner=run_leave_by_recompute,
        leave_by_runner=_send_leave_by_for_event,
        persona_rebuild_runner=run_persona_rebuild,
        gcal_invite_watch_runner=run_gcal_invite_watch,
        agent_digest_runner=run_agent_digest,
        streak_compute_runner=run_streak_compute,
        weekly_scorecard_runner=run_weekly_scorecard,
        active_learning_runner=run_active_learning,
        budget_check_runner=run_budget_check,
        late_check_runner=run_late_check,
        mission_tick_runner=run_mission_tick,
        relation_extract_runner=run_relation_extract,
        self_improve_runner=run_self_improve,
        subscription_runner=run_subscription,
        inbox_triage_runner=run_inbox_triage,
        experiment_followup_runner=run_experiment_followup,
        lifelog_index_runner=run_lifelog_index,
        watch_pump_runner=run_watch_pump,
        relationship_pulse_runner=run_relationship_pulse,
        finance_scan_runner=run_finance_scan,
    )
    # Register the slash-command menu so Telegram clients show autocomplete.
    # Failure is non-fatal (the bot still works without the menu).
    try:
        await app.bot.set_my_commands(BOT_COMMANDS)
        logger.info("set_my_commands: registered %d commands", len(BOT_COMMANDS))
    except Exception:
        logger.exception("set_my_commands failed (non-fatal)")
    # Start the aiohttp OAuth/health server alongside polling. Failure here is
    # non-fatal — bot keeps polling, only the Google Calendar OAuth flow breaks.
    try:
        await oauth_server.start_oauth_server(app.bot)
        logger.info("post_init: db + scheduler + oauth server ready")
    except Exception:
        logger.exception("oauth server failed to start (bot continues without it)")


def main() -> None:
    logger.info("Starting Dailylife bot")
    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("now", cmd_now))
    # v12: 메뉴 등록 X — 직접 타이핑만
    app.add_handler(CommandHandler("translate", cmd_translate))
    app.add_handler(CommandHandler("expert", cmd_expert))
    app.add_handler(CommandHandler("relationships", cmd_relationships))
    app.add_handler(CommandHandler("charges", cmd_charges))
    app.add_handler(CommandHandler("write", cmd_write))
    app.add_handler(CommandHandler("write_continue", cmd_write_continue))
    app.add_handler(CommandHandler("write_done", cmd_write_done))
    app.add_handler(CommandHandler("plan_trip", cmd_plan_trip))
    app.add_handler(CommandHandler("trips", cmd_trips))
    app.add_handler(CommandHandler("negotiate", cmd_negotiate))
    app.add_handler(CommandHandler("negotiations", cmd_negotiations))
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("agenda", cmd_agenda))
    app.add_handler(CommandHandler("facts", cmd_facts))
    app.add_handler(CommandHandler("people", cmd_people))
    app.add_handler(CommandHandler("goals", cmd_goals))
    app.add_handler(CommandHandler("review", cmd_review))
    app.add_handler(CommandHandler("notes", cmd_notes))
    app.add_handler(CommandHandler("tasks", cmd_tasks))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CommandHandler("cost", cmd_cost))
    app.add_handler(CommandHandler("setup", cmd_setup))
    app.add_handler(CommandHandler("briefing", cmd_briefing))
    app.add_handler(CommandHandler("reflect", cmd_reflect))
    app.add_handler(CommandHandler("spending", cmd_spending))
    app.add_handler(CommandHandler("habits", cmd_habits))
    app.add_handler(CommandHandler("nudges", cmd_nudges))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("metrics", cmd_metrics))
    app.add_handler(CommandHandler("dashboard", cmd_dashboard))
    app.add_handler(CommandHandler("cleanup", cmd_cleanup))
    app.add_handler(CommandHandler("macro", cmd_macro))
    app.add_handler(CommandHandler("subscribe", cmd_subscribe))
    app.add_handler(CommandHandler("unsubscribe", cmd_unsubscribe))
    app.add_handler(CommandHandler("quiet", cmd_quiet))
    app.add_handler(CommandHandler("travel", cmd_travel))
    app.add_handler(CommandHandler("experiment", cmd_experiment))
    app.add_handler(CommandHandler("decide", cmd_decide))
    app.add_handler(CommandHandler("watch", cmd_watch))
    app.add_handler(CommandHandler("watches", cmd_watch))  # alias
    app.add_handler(CommandHandler("watch_cancel", cmd_watch_cancel))
    app.add_handler(CommandHandler("vault", cmd_vault))
    app.add_handler(CommandHandler("call", cmd_call))
    app.add_handler(CommandHandler("calls", cmd_call))  # alias
    app.add_handler(CommandHandler("cancel_call", cmd_cancel_call))
    app.add_handler(CommandHandler("improvements", cmd_improvements))
    app.add_handler(CommandHandler("rules", cmd_rules))
    app.add_handler(CommandHandler("persona", cmd_persona))
    app.add_handler(CommandHandler("recall", cmd_recall))
    app.add_handler(CommandHandler("scorecard", cmd_scorecard))
    app.add_handler(CommandHandler("agent", cmd_agent))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("mission", cmd_mission))
    app.add_handler(CommandHandler("missions", cmd_missions))
    app.add_handler(CommandHandler("bank_sms", cmd_bank_sms))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("voice", cmd_voice))
    app.add_handler(CommandHandler("say", cmd_say))
    app.add_handler(CommandHandler("diag", cmd_diag))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("connect_gcal", cmd_connect_gcal))
    app.add_handler(CommandHandler("gcal_status", cmd_gcal_status))
    app.add_handler(CommandHandler("disconnect_gcal", cmd_disconnect_gcal))
    app.add_handler(CallbackQueryHandler(on_callback_undo, pattern=r"^undo:"))
    app.add_handler(CallbackQueryHandler(on_callback_action, pattern=r"^act:"))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
