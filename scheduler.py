"""APScheduler-driven reminders + recurring agent tasks + proactive goal cron."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
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
_morning_briefing_runner: Optional[Callable[[int], Awaitable[None]]] = None
_evening_reflection_runner: Optional[Callable[[int], Awaitable[None]]] = None
_gmail_event_scan_runner: Optional[Callable[[int], Awaitable[None]]] = None
_evening_preview_runner: Optional[Callable[[int], Awaitable[None]]] = None
_birthday_solo_runner: Optional[Callable[[int], Awaitable[None]]] = None
_midday_checkin_runner: Optional[Callable[[int], Awaitable[None]]] = None
_leave_by_recompute_runner: Optional[Callable[[int], Awaitable[None]]] = None
_leave_by_runner: Optional[Callable[[int], Awaitable[None]]] = None  # arg = event_id
# v4 runners
_persona_rebuild_runner: Optional[Callable[[int], Awaitable[None]]] = None
_gcal_invite_watch_runner: Optional[Callable[[int], Awaitable[None]]] = None
_agent_digest_runner: Optional[Callable[[int], Awaitable[None]]] = None
_streak_compute_runner: Optional[Callable[[int], Awaitable[None]]] = None
_weekly_scorecard_runner: Optional[Callable[[int], Awaitable[None]]] = None
_active_learning_runner: Optional[Callable[[int], Awaitable[None]]] = None
_budget_check_runner: Optional[Callable[[int], Awaitable[None]]] = None


def init(
    bot: Bot,
    recurring_runner: Callable[[int], Awaitable[None]],
    weekly_review_runner: Callable[[int], Awaitable[None]],
    daily_imminent_runner: Callable[[int], Awaitable[None]],
    morning_briefing_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    evening_reflection_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    gmail_event_scan_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    evening_preview_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    birthday_solo_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    midday_checkin_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    leave_by_recompute_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    leave_by_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    persona_rebuild_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    gcal_invite_watch_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    agent_digest_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    streak_compute_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    weekly_scorecard_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    active_learning_runner: Optional[Callable[[int], Awaitable[None]]] = None,
    budget_check_runner: Optional[Callable[[int], Awaitable[None]]] = None,
) -> None:
    global _scheduler, _bot, _recurring_runner, _weekly_review_runner, _daily_imminent_runner
    global _morning_briefing_runner, _evening_reflection_runner
    global _gmail_event_scan_runner, _evening_preview_runner, _birthday_solo_runner
    global _midday_checkin_runner, _leave_by_recompute_runner, _leave_by_runner
    global _persona_rebuild_runner, _gcal_invite_watch_runner, _agent_digest_runner
    global _streak_compute_runner, _weekly_scorecard_runner, _active_learning_runner
    global _budget_check_runner
    _bot = bot
    _recurring_runner = recurring_runner
    _weekly_review_runner = weekly_review_runner
    _daily_imminent_runner = daily_imminent_runner
    _morning_briefing_runner = morning_briefing_runner
    _evening_reflection_runner = evening_reflection_runner
    _gmail_event_scan_runner = gmail_event_scan_runner
    _evening_preview_runner = evening_preview_runner
    _birthday_solo_runner = birthday_solo_runner
    _midday_checkin_runner = midday_checkin_runner
    _leave_by_recompute_runner = leave_by_recompute_runner
    _leave_by_runner = leave_by_runner
    _persona_rebuild_runner = persona_rebuild_runner
    _gcal_invite_watch_runner = gcal_invite_watch_runner
    _agent_digest_runner = agent_digest_runner
    _streak_compute_runner = streak_compute_runner
    _weekly_scorecard_runner = weekly_scorecard_runner
    _active_learning_runner = active_learning_runner
    _budget_check_runner = budget_check_runner
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
        ensure_daily_rhythm_for(chat_id)
        proactive_users += 1

    # House-keeping crons (idempotent — replace_existing).
    _scheduler.add_job(
        _run_audit_cleanup,
        CronTrigger(day_of_week="sun", hour=3, minute=30, timezone=TZ),
        id="audit-cleanup-weekly",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _run_gcal_pending_retry,
        CronTrigger(hour=3, minute=15, timezone=TZ),
        id="gcal-pending-retry",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _run_gcal_mirror,
        CronTrigger(hour=3, minute=0, timezone=TZ),
        id="gcal-mirror",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.add_job(
        _run_v4_storage_cleanup,
        CronTrigger(day_of_week="sun", hour=4, minute=0, timezone=TZ),
        id="v4-storage-cleanup-weekly",
        replace_existing=True,
        misfire_grace_time=3600,
    )

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


# ---------------- daily rhythm (morning briefing + evening reflection) ----------------


def _fact_value(chat_id: int, key: str) -> Optional[str]:
    for row in db.list_facts(chat_id):
        if row["key"] == key:
            return row["value"]
    return None


def _toggle_off(chat_id: int, key: str) -> bool:
    v = _fact_value(chat_id, key)
    return v is not None and v.strip().lower() in {"false", "off", "0", "no"}


def _parse_hhmm(s: str, default_h: int, default_m: int) -> tuple:
    try:
        hh, mm = s.split(":")
        return (int(hh), int(mm))
    except Exception:
        return (default_h, default_m)


def ensure_daily_rhythm_for(chat_id: int) -> None:
    """Idempotently arm all per-chat rhythm crons: morning briefing, evening
    reflection, evening preview, birthday-solo, midday check-in, leave-by
    recompute, gmail event scan. Each one respects its own fact toggle. Safe
    to call repeatedly — `replace_existing=True` on every job id."""
    if _scheduler is None:
        return
    if _morning_briefing_runner and not _toggle_off(chat_id, "briefing_enabled"):
        t = _fact_value(chat_id, "briefing_time") or "07:30"
        hour, minute = _parse_hhmm(t, 7, 30)
        _scheduler.add_job(
            _run_morning_briefing,
            CronTrigger(hour=hour, minute=minute, timezone=TZ),
            args=[chat_id],
            id=f"morning-briefing-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _evening_reflection_runner and not _toggle_off(chat_id, "reflection_enabled"):
        t = _fact_value(chat_id, "reflection_time") or "21:30"
        hour, minute = _parse_hhmm(t, 21, 30)
        _scheduler.add_job(
            _run_evening_reflection,
            CronTrigger(hour=hour, minute=minute, timezone=TZ),
            args=[chat_id],
            id=f"evening-reflection-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    # Pre-emptive nudges — each gated by its own fact toggle.
    if _evening_preview_runner and not _toggle_off(chat_id, "weather_preview_enabled"):
        t = _fact_value(chat_id, "weather_preview_time") or "22:00"
        hour, minute = _parse_hhmm(t, 22, 0)
        _scheduler.add_job(
            _run_evening_preview,
            CronTrigger(hour=hour, minute=minute, timezone=TZ),
            args=[chat_id],
            id=f"evening-preview-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _birthday_solo_runner and _toggle_on(chat_id, "birthday_alert_separate_enabled"):
        _scheduler.add_job(
            _run_birthday_solo,
            CronTrigger(hour=9, minute=0, timezone=TZ),
            args=[chat_id],
            id=f"birthday-solo-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _midday_checkin_runner and not _toggle_off(chat_id, "midday_checkin_enabled"):
        t = _fact_value(chat_id, "midday_checkin_time") or "13:00"
        hour, minute = _parse_hhmm(t, 13, 0)
        _scheduler.add_job(
            _run_midday_checkin,
            CronTrigger(hour=hour, minute=minute, timezone=TZ),
            args=[chat_id],
            id=f"midday-checkin-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _leave_by_recompute_runner and not _toggle_off(chat_id, "leave_by_enabled"):
        _scheduler.add_job(
            _run_leave_by_recompute,
            CronTrigger(hour=3, minute=30, timezone=TZ),
            args=[chat_id],
            id=f"leave-by-recompute-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _gmail_event_scan_runner and not _toggle_off(chat_id, "gmail_event_scan_enabled"):
        try:
            minutes = int(_fact_value(chat_id, "gmail_event_scan_minutes") or "60")
        except ValueError:
            minutes = 60
        minutes = max(15, min(minutes, 360))
        if minutes < 60:
            trig = CronTrigger(minute=f"*/{minutes}", timezone=TZ)
        else:
            trig = CronTrigger(minute="0", hour=f"*/{max(1, minutes // 60)}", timezone=TZ)
        _scheduler.add_job(
            _run_gmail_event_scan, trig,
            args=[chat_id],
            id=f"gmail-event-scan-{chat_id}",
            replace_existing=True,
            misfire_grace_time=600,
        )
    # ===== v4 nudges =====
    if _persona_rebuild_runner and not _toggle_off(chat_id, "persona_rebuild_enabled"):
        _scheduler.add_job(
            _run_persona_rebuild,
            CronTrigger(day_of_week="sun", hour=9, minute=30, timezone=TZ),
            args=[chat_id],
            id=f"persona-rebuild-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _gcal_invite_watch_runner and not _toggle_off(chat_id, "auto_rules_enabled"):
        _scheduler.add_job(
            _run_gcal_invite_watch,
            CronTrigger(minute="*/30", timezone=TZ),
            args=[chat_id],
            id=f"gcal-invite-watch-{chat_id}",
            replace_existing=True,
            misfire_grace_time=600,
        )
    if _agent_digest_runner and not _toggle_off(chat_id, "agent_digest_enabled"):
        _scheduler.add_job(
            _run_agent_digest,
            CronTrigger(hour=21, minute=45, timezone=TZ),
            args=[chat_id],
            id=f"agent-digest-{chat_id}",
            replace_existing=True,
            misfire_grace_time=1800,
        )
    if _streak_compute_runner and not _toggle_off(chat_id, "streak_compute_enabled"):
        _scheduler.add_job(
            _run_streak_compute,
            CronTrigger(hour=0, minute=30, timezone=TZ),
            args=[chat_id],
            id=f"streak-compute-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _weekly_scorecard_runner and not _toggle_off(chat_id, "weekly_scorecard_enabled"):
        _scheduler.add_job(
            _run_weekly_scorecard,
            CronTrigger(day_of_week="sun", hour=18, minute=0, timezone=TZ),
            args=[chat_id],
            id=f"weekly-scorecard-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _active_learning_runner and not _toggle_off(chat_id, "active_learning_enabled"):
        t = _fact_value(chat_id, "learning_question_time") or "14:00"
        hour, minute = _parse_hhmm(t, 14, 0)
        _scheduler.add_job(
            _run_active_learning,
            CronTrigger(hour=hour, minute=minute, timezone=TZ),
            args=[chat_id],
            id=f"active-learning-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    if _budget_check_runner and not _toggle_off(chat_id, "budget_alert_enabled"):
        _scheduler.add_job(
            _run_budget_check,
            CronTrigger(hour=8, minute=30, timezone=TZ),
            args=[chat_id],
            id=f"budget-check-{chat_id}",
            replace_existing=True,
            misfire_grace_time=3600,
        )
    logger.info("daily rhythm armed for chat %s", chat_id)


def _toggle_on(chat_id: int, key: str) -> bool:
    """Default OFF — opt-in. True only when fact is explicitly true/on/1/yes."""
    v = _fact_value(chat_id, key)
    return v is not None and v.strip().lower() in {"true", "on", "1", "yes"}


def disable_daily_rhythm_for(chat_id: int) -> None:
    if _scheduler is None:
        return
    for jid in (
        f"morning-briefing-{chat_id}",
        f"evening-reflection-{chat_id}",
        f"evening-preview-{chat_id}",
        f"birthday-solo-{chat_id}",
        f"midday-checkin-{chat_id}",
        f"leave-by-recompute-{chat_id}",
        f"gmail-event-scan-{chat_id}",
        f"persona-rebuild-{chat_id}",
        f"gcal-invite-watch-{chat_id}",
        f"agent-digest-{chat_id}",
        f"streak-compute-{chat_id}",
        f"weekly-scorecard-{chat_id}",
        f"active-learning-{chat_id}",
        f"budget-check-{chat_id}",
    ):
        if _scheduler.get_job(jid):
            _scheduler.remove_job(jid)


async def _run_morning_briefing(chat_id: int) -> None:
    if _morning_briefing_runner is None or _toggle_off(chat_id, "briefing_enabled"):
        return
    try:
        await _morning_briefing_runner(chat_id)
    except Exception:
        logger.exception("morning briefing failed for chat %s", chat_id)


async def _run_evening_reflection(chat_id: int) -> None:
    if _evening_reflection_runner is None or _toggle_off(chat_id, "reflection_enabled"):
        return
    try:
        await _evening_reflection_runner(chat_id)
    except Exception:
        logger.exception("evening reflection failed for chat %s", chat_id)


# ---------------- pre-emptive runners (gmail scan, previews, birthday, midday, leave-by) ----------------


async def _run_gmail_event_scan(chat_id: int) -> None:
    if _gmail_event_scan_runner is None or _toggle_off(chat_id, "gmail_event_scan_enabled"):
        return
    try:
        await _gmail_event_scan_runner(chat_id)
    except Exception:
        logger.exception("gmail event scan failed for chat %s", chat_id)


async def _run_evening_preview(chat_id: int) -> None:
    if _evening_preview_runner is None or _toggle_off(chat_id, "weather_preview_enabled"):
        return
    try:
        await _evening_preview_runner(chat_id)
    except Exception:
        logger.exception("evening preview failed for chat %s", chat_id)


async def _run_birthday_solo(chat_id: int) -> None:
    if _birthday_solo_runner is None or not _toggle_on(chat_id, "birthday_alert_separate_enabled"):
        return
    try:
        await _birthday_solo_runner(chat_id)
    except Exception:
        logger.exception("birthday solo failed for chat %s", chat_id)


async def _run_midday_checkin(chat_id: int) -> None:
    if _midday_checkin_runner is None or _toggle_off(chat_id, "midday_checkin_enabled"):
        return
    try:
        await _midday_checkin_runner(chat_id)
    except Exception:
        logger.exception("midday checkin failed for chat %s", chat_id)


async def _run_leave_by_recompute(chat_id: int) -> None:
    if _leave_by_recompute_runner is None or _toggle_off(chat_id, "leave_by_enabled"):
        return
    try:
        await _leave_by_recompute_runner(chat_id)
    except Exception:
        logger.exception("leave-by recompute failed for chat %s", chat_id)


async def _run_leave_by_fire(event_id: int) -> None:
    """One-shot job entry-point: fires the actual leave-by message via the runner."""
    if _leave_by_runner is None:
        return
    try:
        await _leave_by_runner(event_id)
    except Exception:
        logger.exception("leave-by fire failed for event %s", event_id)


def schedule_leave_by(event_id: int, leave_at_utc: datetime) -> Optional[str]:
    """Arm a one-shot APScheduler job to fire the leave-by message at `leave_at_utc`.
    Returns the job id (or None if scheduler not available / fire-time already past)."""
    if _scheduler is None or _leave_by_runner is None:
        return None
    now = datetime.now(timezone.utc)
    if leave_at_utc <= now:
        return None
    job_id = f"leave-by-{event_id}"
    _scheduler.add_job(
        _run_leave_by_fire,
        DateTrigger(run_date=leave_at_utc),
        args=[event_id],
        id=job_id,
        replace_existing=True,
        misfire_grace_time=300,
    )
    return job_id


def cancel_leave_by(event_id: int) -> None:
    if _scheduler is None:
        return
    jid = f"leave-by-{event_id}"
    if _scheduler.get_job(jid):
        _scheduler.remove_job(jid)


def trigger_evening_preview_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_evening_preview, "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"evening-preview-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


def trigger_midday_checkin_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_midday_checkin, "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"midday-checkin-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


def trigger_gmail_scan_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_gmail_event_scan, "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"gmail-scan-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


# ---------------- v4 wrappers ----------------


async def _run_persona_rebuild(chat_id: int) -> None:
    if _persona_rebuild_runner is None or _toggle_off(chat_id, "persona_rebuild_enabled"):
        return
    try:
        await _persona_rebuild_runner(chat_id)
    except Exception:
        logger.exception("persona rebuild failed for chat %s", chat_id)


async def _run_gcal_invite_watch(chat_id: int) -> None:
    if _gcal_invite_watch_runner is None or _toggle_off(chat_id, "auto_rules_enabled"):
        return
    try:
        await _gcal_invite_watch_runner(chat_id)
    except Exception:
        logger.exception("gcal invite watch failed for chat %s", chat_id)


async def _run_agent_digest(chat_id: int) -> None:
    if _agent_digest_runner is None or _toggle_off(chat_id, "agent_digest_enabled"):
        return
    try:
        await _agent_digest_runner(chat_id)
    except Exception:
        logger.exception("agent digest failed for chat %s", chat_id)


async def _run_streak_compute(chat_id: int) -> None:
    if _streak_compute_runner is None or _toggle_off(chat_id, "streak_compute_enabled"):
        return
    try:
        await _streak_compute_runner(chat_id)
    except Exception:
        logger.exception("streak compute failed for chat %s", chat_id)


async def _run_weekly_scorecard(chat_id: int) -> None:
    if _weekly_scorecard_runner is None or _toggle_off(chat_id, "weekly_scorecard_enabled"):
        return
    try:
        await _weekly_scorecard_runner(chat_id)
    except Exception:
        logger.exception("weekly scorecard failed for chat %s", chat_id)


async def _run_active_learning(chat_id: int) -> None:
    if _active_learning_runner is None or _toggle_off(chat_id, "active_learning_enabled"):
        return
    try:
        await _active_learning_runner(chat_id)
    except Exception:
        logger.exception("active learning failed for chat %s", chat_id)


async def _run_budget_check(chat_id: int) -> None:
    if _budget_check_runner is None or _toggle_off(chat_id, "budget_alert_enabled"):
        return
    try:
        await _budget_check_runner(chat_id)
    except Exception:
        logger.exception("budget check failed for chat %s", chat_id)


def trigger_persona_rebuild_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_persona_rebuild, "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"persona-rebuild-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


def trigger_weekly_scorecard_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_weekly_scorecard, "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"weekly-scorecard-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


def trigger_agent_digest_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_agent_digest, "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"agent-digest-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


# v4 weekly housekeeping (called once at init below)


async def _run_v4_storage_cleanup() -> None:
    """Sun 04:00 KST: trim persona_doc versions + ancient digested agent_actions."""
    try:
        n = db.cleanup_old_personas(keep_recent=8)
        if n:
            logger.info("persona cleanup: dropped %d old versions", n)
    except Exception:
        logger.exception("persona cleanup failed")
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        with db._conn() as c:
            cur = c.execute(
                "DELETE FROM agent_actions WHERE status IN ('reversed','digested') "
                "AND executed_at < ?", (cutoff,))
            if cur.rowcount:
                logger.info("agent_actions cleanup: dropped %d rows", cur.rowcount)
    except Exception:
        logger.exception("agent_actions cleanup failed")


def trigger_morning_briefing_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_morning_briefing, "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"morning-briefing-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


def trigger_evening_reflection_now(chat_id: int) -> None:
    if _scheduler is None:
        return
    _scheduler.add_job(
        _run_evening_reflection, "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=2),
        args=[chat_id],
        id=f"evening-reflection-{chat_id}-once-{int(datetime.now(timezone.utc).timestamp())}",
        misfire_grace_time=120,
    )


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


# ---------------- house-keeping crons ----------------


async def _run_audit_cleanup() -> None:
    """Weekly: delete restored or stale deleted_audit rows + log how many."""
    try:
        n = db.cleanup_deleted_audit(max_age_days=30)
        logger.info("deleted_audit cleanup: %d rows removed", n)
    except Exception:
        logger.exception("audit cleanup failed")


async def _run_gcal_pending_retry() -> None:
    """Nightly: retry GCal create for events whose dual-write previously failed."""
    try:
        pending = db.list_pending_gcal_sync()
    except Exception:
        logger.exception("pending gcal list failed")
        return
    if not pending:
        return
    try:
        import gcal  # local to avoid circular at module load
    except Exception:
        logger.exception("gcal import failed in pending retry")
        return
    for row in pending:
        chat_id = row["chat_id"]
        try:
            when_local = datetime.fromisoformat(row["when_utc"]).astimezone(TZ).isoformat()
            ev = await gcal.create_event(
                chat_id,
                summary=row["title"],
                start_iso=when_local,
                description=row["notes"] or None,
            )
            db.set_event_gcal(row["id"], ev.get("id"), "synced")
            logger.info("gcal retry success for event %s", row["id"])
        except Exception:
            logger.exception("gcal retry failed for event %s; leaving pending", row["id"])


async def _run_gcal_mirror() -> None:
    """Nightly: pull GCal events into local DB so /today /week show them too.

    Direction is GCal → local only (one-way). Linked via events.gcal_event_id;
    no remind_lead_minutes set so they don't double-fire reminders. Bot-created
    events are dual-written elsewhere, this only catches what the user added
    on the Google side."""
    try:
        import gcal
    except Exception:
        logger.exception("gcal import failed in mirror cron")
        return
    # We don't keep a list of chats in scheduler; iterate oauth_tokens for 'google' rows.
    try:
        chat_ids = db.all_chat_ids_with_google_oauth()
    except Exception:
        logger.exception("listing google-connected chats failed")
        return
    if not chat_ids:
        return
    from_utc = datetime.now(timezone.utc) - timedelta(hours=24)
    to_utc = datetime.now(timezone.utc) + timedelta(days=14)
    mirrored_total = 0
    for chat_id in chat_ids:
        try:
            events = await gcal.list_events(
                chat_id,
                time_min_iso=from_utc.astimezone(TZ).isoformat(),
                time_max_iso=to_utc.astimezone(TZ).isoformat(),
                max_results=100,
            )
        except Exception:
            logger.exception("gcal list failed for chat %s", chat_id)
            continue
        for e in events:
            gid = e.get("id")
            if not gid:
                continue
            if db.find_event_by_gcal_id(chat_id, gid):
                continue
            start = e.get("start")
            if not start:
                continue
            try:
                start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            except ValueError:
                continue
            title = (e.get("summary") or "(제목 없음)").strip()
            db.add_event(
                chat_id=chat_id,
                title=title,
                when_utc=start_dt,
                notes=(e.get("description") or None),
                remind_lead_minutes=None,  # gcal mirror events don't fire bot reminders
                gcal_event_id=gid,
                gcal_sync_state="mirrored",
            )
            mirrored_total += 1
    if mirrored_total:
        logger.info("gcal mirror: pulled %d new events across %d chats",
                    mirrored_total, len(chat_ids))
