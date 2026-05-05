"""APScheduler-driven reminders + recurring agent tasks + proactive goal cron."""

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
_weekly_review_runner: Optional[Callable[[int], Awaitable[None]]] = None
_daily_imminent_runner: Optional[Callable[[int], Awaitable[None]]] = None


def init(
    bot: Bot,
    recurring_runner: Callable[[int], Awaitable[None]],
    weekly_review_runner: Callable[[int], Awaitable[None]],
    daily_imminent_runner: Callable[[int], Awaitable[None]],
) -> None:
    global _scheduler, _bot, _recurring_runner, _weekly_review_runner, _daily_imminent_runner
    _bot = bot
    _recurring_runner = recurring_runner
    _weekly_review_runner = weekly_review_runner
    _daily_imminent_runner = daily_imminent_runner
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

    proactive_users = 0
    for chat_id in db.all_chat_ids_with_goals():
        ensure_proactive_for(chat_id)
        proactive_users += 1

    logger.info(
        "scheduler started; re-armed %d reminders, %d recurring tasks, %d goal-tracking chats",
        rearmed_reminders,
        rearmed_tasks,
        proactive_users,
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


# ---------------- proactive goal cron (per-chat, idempotent) ----------------


def _goal_review_disabled(chat_id: int) -> bool:
    """User can opt out by setting fact goal_review_enabled=false."""
    for row in db.list_facts(chat_id):
        if row["key"] == "goal_review_enabled" and row["value"].strip().lower() in {"false", "off", "0", "no"}:
            return True
    return False


def ensure_proactive_for(chat_id: int) -> None:
    """Idempotently arm the weekly goal review (Sun 09:00 KST) and daily imminent check
    (08:00 KST) for this chat. Safe to call repeatedly — replace_existing=True."""
    if _scheduler is None:
        return
    if _goal_review_disabled(chat_id):
        return
    _scheduler.add_job(
        _run_weekly_review,
        CronTrigger(day_of_week="sun", hour=9, minute=0, timezone=TZ),
        args=[chat_id],
        id=f"weekly-review-{chat_id}",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _run_daily_imminent,
        CronTrigger(hour=8, minute=0, timezone=TZ),
        args=[chat_id],
        id=f"daily-imminent-{chat_id}",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    logger.info("proactive cron armed for chat %s (Sun 09:00 + daily 08:00 KST)", chat_id)


def disable_proactive_for(chat_id: int) -> None:
    if _scheduler is None:
        return
    for jid in (f"weekly-review-{chat_id}", f"daily-imminent-{chat_id}"):
        if _scheduler.get_job(jid):
            _scheduler.remove_job(jid)
    logger.info("proactive cron disabled for chat %s", chat_id)


def trigger_weekly_review_now(chat_id: int) -> None:
    """Force-fire weekly review immediately (for testing or explicit user request)."""
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_weekly_review,
        "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"weekly-review-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


def trigger_daily_imminent_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_daily_imminent,
        "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"daily-imminent-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


async def _run_weekly_review(chat_id: int) -> None:
    assert _weekly_review_runner is not None
    if _goal_review_disabled(chat_id):
        return
    try:
        await _weekly_review_runner(chat_id)
    except Exception:
        logger.exception("weekly goal review failed for chat %s", chat_id)


async def _run_daily_imminent(chat_id: int) -> None:
    assert _daily_imminent_runner is not None
    if _goal_review_disabled(chat_id):
        return
    try:
        await _daily_imminent_runner(chat_id)
    except Exception:
        logger.exception("daily imminent check failed for chat %s", chat_id)
