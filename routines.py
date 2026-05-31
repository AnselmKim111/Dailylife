"""Rule-based routine + pattern detection over events, habits, and expenses.

No numpy / sklearn — pure Python, run periodically by the weekly review and
scorecard runners. Output is a list of suggestion dicts the bot can surface
to the user as 'should I make this a recurring task?' style nudges.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import db

_TZ = ZoneInfo(db.USER_TZ_NAME)


def _dt_local(when_utc_iso: str) -> datetime:
    return datetime.fromisoformat(when_utc_iso).astimezone(_TZ)


# ---------------- weekday × time-of-day clusters on events ----------------


def detect_weekly_event_routines(chat_id: int, weeks: int = 6) -> List[Dict]:
    """Look back `weeks` weeks; group events by normalized title keyword.
    For each group with ≥3 occurrences clustered on the same weekday & hour
    (±1h), surface a 'looks like a weekly routine' suggestion."""
    end = datetime.now(_TZ)
    start = end - timedelta(weeks=weeks)
    rows = db.list_events(chat_id, start.astimezone(_TZ).replace(tzinfo=_TZ), end)
    buckets: Dict[str, List[datetime]] = defaultdict(list)
    for r in rows:
        title = (r["title"] or "").strip()
        if not title or len(title) < 2:
            continue
        key = _title_key(title)
        try:
            buckets[key].append(_dt_local(r["when_utc"]))
        except Exception:
            continue
    suggestions: List[Dict] = []
    for key, times in buckets.items():
        if len(times) < 3:
            continue
        weekday_hour_counts: Counter = Counter((t.weekday(), t.hour) for t in times)
        (wd, hr), n = weekday_hour_counts.most_common(1)[0]
        if n < 3:
            continue
        weekday_kr = "월화수목금토일"[wd]
        suggestions.append({
            "kind": "weekly_event_routine",
            "key": key,
            "weekday": weekday_kr,
            "hour": hr,
            "n": n,
            "summary": f"매주 {weekday_kr}요일 {hr:02d}시에 {key} ({n}회 관찰)",
        })
    return suggestions


def _title_key(title: str) -> str:
    """Normalize an event title to a routine key (drop dates / numbers / spaces)."""
    import re
    t = re.sub(r"\d+", "", title)
    t = re.sub(r"[^\w가-힣]+", "", t)
    return t[:20].lower()


# ---------------- recurring spending ----------------


def detect_recurring_spending(chat_id: int, days: int = 30) -> List[Dict]:
    """Merchants the user pays ≥4 times in last `days` days — likely a routine.
    Return amount totals so the bot can ask 'spend ₩X/mo on Y — keep going?'."""
    rows = db.list_expenses(chat_id, days=days)
    by_merchant: Dict[str, List[int]] = defaultdict(list)
    for r in rows:
        m = (r["merchant"] or "").strip()
        if not m:
            continue
        by_merchant[m].append(int(r["amount_won"]))
    suggestions: List[Dict] = []
    for merchant, amounts in by_merchant.items():
        if len(amounts) < 4:
            continue
        total = sum(amounts)
        suggestions.append({
            "kind": "recurring_spend",
            "merchant": merchant,
            "n": len(amounts),
            "total_won": total,
            "summary": f"{merchant}에 {days}일간 {len(amounts)}회 · 합계 ₩{total:,}",
        })
    suggestions.sort(key=lambda s: -s["total_won"])
    return suggestions[:6]


# ---------------- habit gaps ----------------


def detect_habit_gaps(chat_id: int, days: int = 60) -> List[Dict]:
    """For each known habit_key, find the mean interval between logs.
    If the current gap (today - last log) > 2x mean, surface a 'haven't done
    X in N days' nudge so the user can re-engage. Skips brand-new habits."""
    suggestions: List[Dict] = []
    today = datetime.now(_TZ).date()
    for key in db.all_habit_keys(chat_id):
        dates = db.habit_logs_by_date(chat_id, key, days=days)
        if len(dates) < 4:
            continue
        date_objs = sorted({datetime.fromisoformat(d).date() for d in dates})
        if len(date_objs) < 4:
            continue
        intervals = [(date_objs[i] - date_objs[i - 1]).days for i in range(1, len(date_objs))]
        mean_interval = sum(intervals) / len(intervals)
        gap = (today - date_objs[-1]).days
        if gap > max(2 * mean_interval, mean_interval + 2):
            suggestions.append({
                "kind": "habit_gap",
                "habit_key": key,
                "days_since": gap,
                "usual_interval": round(mean_interval, 1),
                "summary": f"{key} {gap}일째 안 함 (평소 {mean_interval:.1f}일 간격)",
            })
    return suggestions


# ---------------- composite entry point ----------------


def detect_all(chat_id: int) -> Dict[str, List[Dict]]:
    return {
        "weekly_events": detect_weekly_event_routines(chat_id),
        "recurring_spend": detect_recurring_spending(chat_id),
        "habit_gaps": detect_habit_gaps(chat_id),
    }
