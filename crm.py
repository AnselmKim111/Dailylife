"""Personal CRM — 관계 강도 자동 트래킹.

매주 일요일 09:00 cron이 사람별 점수 계산. *식어가는* 1-2명만 briefing surface.
점수 직접 노출 X — '경서랑 한 달째 연락 없네' 같은 자연어만.
v12 W5.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import db

logger = logging.getLogger(__name__)


def _week_iso(d: datetime) -> str:
    """ISO 주 형식 '2026-W23'."""
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def _person_aliases(person_row) -> List[str]:
    import json as _json
    try:
        return [person_row["name"]] + _json.loads(person_row["aliases_json"] or "[]")
    except Exception:
        return [person_row["name"]]


def _count_meetings(chat_id: int, person_row, start: datetime, end: datetime) -> int:
    """person이 attendee로 잡힌 events — title/notes에 이름·alias 포함."""
    names = _person_aliases(person_row)
    if not names:
        return 0
    rows = db.list_events(chat_id, start, end)
    count = 0
    for r in rows:
        title = (r["title"] or "")
        notes = (r["notes"] or "")
        combined = title + " " + notes
        if any(n and n in combined for n in names):
            count += 1
    return count


def _count_messages(chat_id: int, person_row, start: datetime, end: datetime) -> int:
    """chat_log에서 person 이름·alias 언급 카운트."""
    names = _person_aliases(person_row)
    if not names:
        return 0
    cutoff_start = start.isoformat()
    cutoff_end = end.isoformat()
    count = 0
    with db._conn() as c:
        rows = c.execute(
            "SELECT content FROM chat_log WHERE chat_id=? AND created_at BETWEEN ? AND ?",
            (chat_id, cutoff_start, cutoff_end))
        for r in rows:
            content = r["content"] or ""
            if any(n and n in content for n in names):
                count += 1
    return count


def _sentiment_for(chat_id: int, person_row, start: datetime, end: datetime) -> Optional[float]:
    """daily_state.reflection_response 중 person 언급된 날들의 mood_sentiment 평균.
    positive=+1, neutral=0, negative=-1."""
    names = _person_aliases(person_row)
    if not names:
        return None
    cutoff_start_date = start.date().isoformat()
    cutoff_end_date = end.date().isoformat()
    scores = []
    with db._conn() as c:
        rows = c.execute(
            "SELECT reflection_response, mood_sentiment FROM daily_state "
            "WHERE chat_id=? AND date_local BETWEEN ? AND ? "
            "AND reflection_response IS NOT NULL",
            (chat_id, cutoff_start_date, cutoff_end_date))
        for r in rows:
            response = r["reflection_response"] or ""
            mood = r["mood_sentiment"]
            if not any(n and n in response for n in names):
                continue
            if mood == "positive":
                scores.append(1.0)
            elif mood == "negative":
                scores.append(-1.0)
            elif mood == "neutral":
                scores.append(0.0)
    if not scores:
        return None
    return sum(scores) / len(scores)


def _compute_score(meeting: int, message: int, sentiment: Optional[float]) -> float:
    """log scale로 노이즈 줄임. sentiment ±2 가중."""
    m = math.log(meeting + 1) * 5.0
    msg = math.log(message + 1) * 2.0
    s = (sentiment or 0.0) * 2.0
    return m + msg + s


def compute_pulse_for_week(chat_id: int, week_end: Optional[datetime] = None) -> List[Dict]:
    """주말 cron — 지난 7일 모든 사람 score 계산 + relationship_pulse UPSERT.
    반환: 계산된 사람 리스트 (디버그용)."""
    end = week_end or datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    week_iso = _week_iso(end - timedelta(days=3))  # 주 중간 기준
    people = db.list_people(chat_id)
    out = []
    for p in people:
        meet = _count_meetings(chat_id, p, start, end)
        msg = _count_messages(chat_id, p, start, end)
        senti = _sentiment_for(chat_id, p, start, end)
        score = _compute_score(meet, msg, senti)
        db.upsert_relationship_pulse(
            chat_id, p["id"], week_iso, meet, msg, senti, score)
        out.append({
            "person_id": p["id"], "name": p["name"],
            "meeting": meet, "message": msg,
            "sentiment": senti, "score": score,
        })
    return out


def briefing_line(chat_id: int) -> Optional[str]:
    """morning_briefing이 호출 — 식어가는 사람 *최대 1명*만 자연어 1줄.

    상위 룰:
    1. 친밀 점수 baseline 대비 50% 이하 + 마지막 contact 21일+
    2. role이 가족·약혼녀·친구·동기 일 때만 (직장 동료 등은 surface X)
    3. 위축감 안 주게 1명 max
    """
    fading = db.fading_relationships(chat_id, threshold_pct=0.5)
    if not fading:
        return None
    intimate_roles = ("가족", "약혼녀", "약혼남", "친구", "동기", "베프", "절친")
    for f in fading:
        if f["role"] not in intimate_roles:
            continue
        days = db.last_contact_days_ago(chat_id, f["person_id"])
        if days is None or days < 21:
            continue
        return f"🤔 {f['name']} {days}일째 — 한 마디 어때?"
    return None
