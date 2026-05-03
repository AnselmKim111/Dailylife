"""Reminder scheduler — fires Telegram messages at (event_time - lead) minutes."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot

import db
from llm import USER_TZ

logger = logging.getLogger(__name__)
TZ = ZoneInfo(USER_TZ)
_scheduler: AsyncIOScheduler | None = None
_bot: Bot | None = None


def init(bot: Bot) -> None:
    global _scheduler, _bot
    _bot = bot
    _scheduler = AsyncIOScheduler(timezone=TZ)
    _scheduler.start()
    # Re-arm anything still pending after a restart.
    rearmed = 0
    for row in db.pending_reminders():
        if schedule_for(row):
            rearmed += 1
    logger.info("scheduler started; re-armed %d pending reminders", rearmed)


def _fire_at(event_id: int) -> datetime | None:
    row = db.get_event(event_id)
    if row is None or row["reminded"] or row["remind_lead_minutes"] is None:
        return None
    when_utc = datetime.fromisoformat(row["when_utc"])
    fire_utc = when_utc - timedelta(minutes=row["remind_lead_minutes"])
    return fire_utc


def schedule_for(row) -> bool:
    """Schedule (or skip) a reminder for the given event row. Returns True if armed."""
    assert _scheduler is not None
    if row["reminded"] or row["remind_lead_minutes"] is None:
        return False
    fire_utc = _fire_at(row["id"])
    if fire_utc is None:
        return False
    now = datetime.now(timezone.utc)
    if fire_utc <= now:
        # Already past; fire immediately if event is still in the future.
        when_utc = datetime.fromisoformat(row["when_utc"])
        if when_utc <= now:
            return False
        fire_utc = now + timedelta(seconds=2)
    _scheduler.add_job(
        _send_reminder,
        "date",
        run_date=fire_utc,
        args=[row["id"]],
        id=f"event-{row['id']}",
        replace_existing=True,
        misfire_grace_time=300,
    )
    logger.info("armed reminder event_id=%s fire=%s", row["id"], fire_utc.isoformat())
    return True


async def _send_reminder(event_id: int) -> None:
    assert _bot is not None
    row = db.get_event(event_id)
    if row is None or row["reminded"]:
        return
    when_local = datetime.fromisoformat(row["when_utc"]).astimezone(TZ)
    delta = when_local - datetime.now(TZ)
    minutes = max(0, int(delta.total_seconds() // 60))
    text = (
        f"⏰ 리마인더\n"
        f"• {row['title']}\n"
        f"• {when_local.strftime('%Y-%m-%d %H:%M')} ({USER_TZ})  •  {minutes}분 뒤"
    )
    if row["notes"]:
        text += f"\n• 메모: {row['notes']}"
    try:
        await _bot.send_message(chat_id=row["chat_id"], text=text)
        db.mark_reminded(event_id)
    except Exception:
        logger.exception("failed to send reminder for event %s", event_id)
