"""Dailylife — Telegram personal assistant powered by an OpenRouter LLM with tool-use."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import db
import external
import scheduler
from llm import TOOLS, USER_TZ, chat_completion, parse_tool_calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dailylife")

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "16"))
TZ = ZoneInfo(USER_TZ)

SYSTEM_PROMPT_TEMPLATE = (
    "You are Dailylife, the user's personal Telegram assistant for schedule, errands, and "
    "general questions. Default to Korean unless the user writes another language. Be warm and concise.\n\n"
    "All datetimes the user mentions are in {tz} timezone. Convert relative times "
    "('내일 3시', 'in 2 hours', '다음 주 월요일') against current_time below.\n"
    "current_time: {now} ({tz})\n\n"
    "When the user mentions a personal/stable fact about themselves (집 주소, 자대 위치, 가족 이름, "
    "선호도 등), call remember_fact to persist it — don't just acknowledge in chat.\n"
    "When asked something you can answer better with live data (영업시간, 길찾기, 운항정보, 시간표), "
    "use the available tools (web_search, kakao_local_search, fetch_url, kakao_directions_drive) "
    "instead of guessing.\n"
    "When the user wants a recurring briefing ('매일 X시에 ~ 알려줘'), use add_recurring_task.\n\n"
    "Known facts about this user:\n{facts_block}"
)

# Per-chat in-memory short-term history (raw tool turns retained).
chat_history: Dict[int, List[Dict]] = {}


# ---------------- helpers ----------------


def _history(chat_id: int) -> List[Dict]:
    return chat_history.setdefault(chat_id, [])


def _trim_history(chat_id: int) -> None:
    h = chat_history.get(chat_id, [])
    if len(h) > HISTORY_LIMIT * 2:
        chat_history[chat_id] = h[-HISTORY_LIMIT * 2 :]


def _facts_block(chat_id: int) -> str:
    rows = db.list_facts(chat_id)
    if not rows:
        return "  (none yet)"
    return "\n".join(f"  - {r['key']}: {r['value']}" for r in rows)


def _system_message(chat_id: int) -> Dict:
    now_local = datetime.now(TZ).strftime("%Y-%m-%d %H:%M (%a)")
    return {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(
            tz=USER_TZ, now=now_local, facts_block=_facts_block(chat_id)
        ),
    }


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
    lead = args.get("remind_lead_minutes")
    lead = 30 if lead is None else int(lead)
    lead = lead if lead > 0 else None
    eid = db.add_event(chat_id, title, when_utc, notes, lead)
    row = db.get_event(eid)
    armed = scheduler.schedule_for(row) if row else False
    return {
        "ok": True,
        "event_id": eid,
        "when_local": when_utc.astimezone(TZ).isoformat(),
        "remind_lead_minutes": lead,
        "reminder_armed": armed,
    }


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
    ok = db.delete_event(int(eid), chat_id)
    return {"ok": ok, "event_id": int(eid)}


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
    return {"ok": db.forget_fact(chat_id, key), "key": key}


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
    ok = db.delete_recurring_task(int(tid), chat_id)
    if ok:
        scheduler.cancel_recurring(int(tid))
    return {"ok": ok, "task_id": int(tid)}


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
}

ASYNC_HANDLERS = {
    "web_search": tool_web_search,
    "kakao_local_search": tool_kakao_local,
    "kakao_directions_drive": tool_kakao_drive,
    "fetch_url": tool_fetch_url,
}


# ---------------- the agent loop (reusable for live chat AND recurring tasks) ----------------


async def run_agent(chat_id: int, user_text: str, history: Optional[List[Dict]] = None,
                    max_hops: int = 6) -> str:
    """Run the agent loop with tool-use until it returns a text answer."""
    history = history if history is not None else []
    history.append({"role": "user", "content": user_text})

    final_text = ""
    for hop in range(max_hops):
        messages = [_system_message(chat_id), *history]
        data = await chat_completion(messages, tools=TOOLS)
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


async def run_recurring_task(task_id: int) -> None:
    """Wired into scheduler: load the saved prompt, run agent, send result to chat."""
    row = db.get_recurring_task(task_id)
    if row is None or not row["enabled"]:
        return
    chat_id = row["chat_id"]
    logger.info("running recurring task_id=%s chat=%s", task_id, chat_id)
    try:
        reply = await run_agent(chat_id, row["prompt"], history=[])
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


# ---------------- Telegram handlers ----------------


_app: Optional[Application] = None


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "안녕하세요! Dailylife입니다. 일정 + 잡일 + 검색 도와드려요.\n\n"
        "예) '내일 3시에 치과, 30분 전 알림'\n"
        "    '내 집은 DMC 파크뷰 자이야. 기억해줘'\n"
        "    '신촌 근처 지금 열린 한식집 찾아줘'\n"
        "    '매일 7시에 우도 가는 배 운항 정보 알려줘'\n\n"
        "/today /week /agenda — 일정 조회\n"
        "/facts — 기억하고 있는 정보\n"
        "/tasks — 정기 작업 목록\n"
        "/reset — 대화 메모리 초기화"
    )


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    now = datetime.now(TZ)
    end = now.replace(hour=23, minute=59, second=59)
    rows = db.list_events(chat_id, now.astimezone(timezone.utc), end.astimezone(timezone.utc))
    if not rows:
        await update.message.reply_text("오늘 남은 일정 없음 ✨")
        return
    await update.message.reply_text("오늘 일정:\n" + "\n".join(_format_event_row(r) for r in rows))


async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    now = datetime.now(TZ)
    end = now + timedelta(days=7)
    rows = db.list_events(chat_id, now.astimezone(timezone.utc), end.astimezone(timezone.utc))
    if not rows:
        await update.message.reply_text("앞으로 7일 일정 없음 ✨")
        return
    await update.message.reply_text("이번 주 일정:\n" + "\n".join(_format_event_row(r) for r in rows))


async def cmd_agenda(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_events(chat_id, datetime.now(timezone.utc))
    if not rows:
        await update.message.reply_text("등록된 일정 없음 ✨")
        return
    await update.message.reply_text("전체 일정:\n" + "\n".join(_format_event_row(r) for r in rows))


async def cmd_facts(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    rows = db.list_facts(chat_id)
    if not rows:
        await update.message.reply_text("기억하고 있는 정보가 아직 없어요. 알려주시면 저장할게요!")
        return
    body = "\n".join(f"• {r['key']}: {r['value']}" for r in rows)
    await update.message.reply_text("기억하고 있는 정보:\n" + body)


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


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_history.pop(update.effective_chat.id, None)
    await update.message.reply_text("대화 메모리 초기화 완료. 일정/기억은 그대로 보존.")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text
    logger.info("msg from %s: %r", chat_id, user_text[:200])

    history = _history(chat_id)
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        reply = await run_agent(chat_id, user_text, history=history)
    except Exception as exc:
        logger.exception("agent loop failed")
        await update.message.reply_text(f"⚠️ 처리 실패: {exc}")
        # Drop the user turn we just appended in run_agent.
        if history and history[-1].get("role") == "user":
            history.pop()
        return

    _trim_history(chat_id)
    for i in range(0, len(reply), 4000):
        await update.message.reply_text(reply[i : i + 4000])


async def post_init(app: Application) -> None:
    global _app
    _app = app
    db.init_db()
    scheduler.init(app.bot, run_recurring_task)
    logger.info("post_init: db ready, scheduler running")


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
    app.add_handler(CommandHandler("today", cmd_today))
    app.add_handler(CommandHandler("week", cmd_week))
    app.add_handler(CommandHandler("agenda", cmd_agenda))
    app.add_handler(CommandHandler("facts", cmd_facts))
    app.add_handler(CommandHandler("tasks", cmd_tasks))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
