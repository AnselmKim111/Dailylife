"""Dailylife — Telegram personal assistant powered by an OpenRouter LLM with tool-use."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
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
    "You are Dailylife, the user's personal Telegram assistant for schedule and chores. "
    "Default to Korean unless the user writes another language. Be warm and concise.\n\n"
    "All datetimes the user mentions are in {tz} timezone. Convert relative times "
    "('내일 3시', 'in 2 hours', '다음 주 월요일 아침') against current_time below.\n"
    "current_time: {now} ({tz})\n\n"
    "Use tools whenever the user mentions a schedule item, asks what's on their agenda, "
    "or wants to change/remove an entry. Do NOT just chat about scheduling — call the tool. "
    "After tool execution, summarize the result for the user in plain language. "
    "When listing events, format like '#<id> · <date> <time> · <title>'."
)

# In-memory short-term chat history per chat (raw tool turns kept so model has context).
chat_history: Dict[int, List[Dict]] = {}


def _history(chat_id: int) -> List[Dict]:
    return chat_history.setdefault(chat_id, [])


def _trim_history(chat_id: int) -> None:
    h = chat_history.get(chat_id, [])
    if len(h) > HISTORY_LIMIT * 2:
        chat_history[chat_id] = h[-HISTORY_LIMIT * 2 :]


def _system_message() -> Dict:
    now_local = datetime.now(TZ).strftime("%Y-%m-%d %H:%M (%a)")
    return {
        "role": "system",
        "content": SYSTEM_PROMPT_TEMPLATE.format(tz=USER_TZ, now=now_local),
    }


def _parse_local_iso(s: str) -> datetime:
    """Parse an ISO 8601 string assumed to be in USER_TZ (or with explicit offset)."""
    s = s.strip()
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Best-effort: trim trailing Z.
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


# ---------------- Tool implementations ----------------


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


TOOL_HANDLERS = {
    "add_event": tool_add_event,
    "list_events": tool_list_events,
    "update_event": tool_update_event,
    "delete_event": tool_delete_event,
}


# ---------------- Telegram handlers ----------------


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "안녕하세요! Dailylife입니다. 일정 관리와 잡일 도와드려요.\n\n"
        "예) '내일 3시에 치과 예약, 1시간 전 알려줘'\n"
        "    '오늘 뭐 있어?'\n"
        "    '#3 일정 취소'\n\n"
        "/today  /week  /agenda — 빠른 조회\n"
        "/reset  대화 메모리 초기화"
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


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_history.pop(update.effective_chat.id, None)
    await update.message.reply_text("대화 메모리 초기화 완료. 일정 데이터는 그대로 보존됨.")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    user_text = update.message.text
    logger.info("msg from %s: %r", chat_id, user_text[:200])

    history = _history(chat_id)
    history.append({"role": "user", "content": user_text})

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Tool-use loop: call model, execute any tool calls, feed results back, repeat.
    final_text = None
    for hop in range(4):
        messages = [_system_message(), *history]
        try:
            data = await chat_completion(messages, tools=TOOLS)
        except Exception as exc:
            logger.exception("OpenRouter call failed")
            await update.message.reply_text(f"⚠️ 모델 호출 실패: {exc}")
            history.pop()
            return

        msg = data["choices"][0]["message"]
        # Persist the assistant turn (including any tool_calls) into history.
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
            handler = TOOL_HANDLERS.get(tc["name"])
            if handler is None:
                result = {"ok": False, "error": f"unknown tool {tc['name']}"}
            else:
                try:
                    result = handler(chat_id, tc["arguments"])
                except Exception as exc:
                    logger.exception("tool %s failed", tc["name"])
                    result = {"ok": False, "error": str(exc)}
            logger.info("tool %s args=%s result=%s", tc["name"], tc["arguments"], result)
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    if not final_text:
        final_text = "처리 완료."

    _trim_history(chat_id)

    for i in range(0, len(final_text), 4000):
        await update.message.reply_text(final_text[i : i + 4000])


async def post_init(app: Application) -> None:
    db.init_db()
    scheduler.init(app.bot)
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
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
