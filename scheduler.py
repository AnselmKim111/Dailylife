"""APScheduler-driven reminders + recurring agent tasks."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from telegram import Bot

import db
from llm import USER_TZ

logger = logging.getLogger(__name__)
TZ = ZoneInfo(USER_TZ)
_scheduler: Optional[AsyncIOScheduler] = None
_bot: Optional[Bot] = None
_recurring_runner: Optional[Callable[[int], Awaitable[None]]] = None


def init(bot: Bot, recurring_runner: Callable[[int], Awaitable[None]]) -> None:
    """Boot the scheduler. recurring_runner is an async fn(task_id) -> None."""
    global _scheduler, _bot, _recurring_runner
    _bot = bot
    _recurring_runner = recurring_runner
    _scheduler = AsyncIOScheduler(timezone=TZ)
    _scheduler.start()

    rearmed_reminders = 0
    for row in db.pending_reminders():
        if schedule_for(row):
            rearmed_reminders += 1

    rearmed_tasks = 0
    for row in db.list_recurring_tasks():
        if schedule_recurring(row):
            rearmed_tasks += 1

    logger.info(
        "scheduler started; re-armed %d reminders, %d recurring tasks",
        rearmed_reminders,
        rearmed_tasks,
    )


# ---------------- one-shot event reminders ----------------


def _fire_at(event_id: int) -> Optional[datetime]:
    row = db.get_event(event_id)
    if row is None or row["reminded"] or row["remind_lead_minutes"] is None:
        return None
    when_utc = datetime.fromisoformat(row["when_utc"])
    return when_utc - timedelta(minutes=row["remind_lead_minutes"])


def schedule_for(row) -> bool:
    assert _scheduler is not None
    if row["reminded"] or row["remind_lead_minutes"] is None:
        return False
    fire_utc = _fire_at(row["id"])
    if fire_utc is None:
        return False
    now = datetime.now(timezone.utc)
    if fire_utc <= now:
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


# ---------------- recurring agent tasks ----------------


def schedule_recurring(row) -> bool:
    """Arm an APScheduler cron job for a recurring task row."""
    assert _scheduler is not None and _recurring_runner is not None
    if not row["enabled"]:
        return False
    cron = (row["cron_kst"] or "").strip()
    try:
        hh, mm = cron.split(":")
        hour, minute = int(hh), int(mm)
    except ValueError:
        logger.error("bad cron_kst %r for task %s", cron, row["id"])
        return False
    _scheduler.add_job(
        _run_recurring,
        CronTrigger(hour=hour, minute=minute, timezone=TZ),
        args=[row["id"]],
        id=f"recurring-{row['id']}",
        replace_existing=True,
        misfire_grace_time=600,
    )
    logger.info("armed recurring task_id=%s daily at %02d:%02d KST", row["id"], hour, minute)
    return True


def cancel_recurring(task_id: int) -> None:
    if _scheduler is None:
        return
    job_id = f"recurring-{task_id}"
    if _scheduler.get_job(job_id):
        _scheduler.remove_job(job_id)
        logger.info("cancelled recurring task_id=%s", task_id)


async def _run_recurring(task_id: int) -> None:
    assert _recurring_runner is not None
    try:
        await _recurring_runner(task_id)
    except Exception:
        logger.exception("recurring task %s failed", task_id)
