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
import transcribe
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
    "You are Dailylife, the user's personal Telegram assistant for schedule, errands, memory, and "
    "general questions. Default to Korean unless the user writes another language. Be warm and concise.\n\n"
    "All datetimes the user mentions are in {tz} timezone. Convert relative times "
    "('내일 3시', 'in 2 hours', '다음 주 월요일') against current_time below.\n"
    "current_time: {now} ({tz})\n\n"
    "Tool routing:\n"
    "- Stable identity facts (집 주소, 자대 위치, 가족 이름, 선호도) → remember_fact.\n"
    "- Free-form short notes the user wants saved (점심 약속 메모, 책 추천, 선물 후보) → save_note.\n"
    "- 'What did I say about X' / '지난주에 X 얘기 어땠지' style recall → search_memory.\n"
    "- Time-bound appointment with a clock time → add_event.\n"
    "- Daily recurring briefing ('매일 X시에 …') → add_recurring_task.\n"
    "- Live data (영업시간, 길찾기, 운항정보, 시간표) → use web_search / kakao_local_search / "
    "  fetch_url / kakao_directions_drive — don't guess.\n\n"
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


# ---------------- goals (long-horizon, proactive) ----------------


def tool_add_goal(chat_id: int, args: Dict) -> Dict:
    title = (args.get("title") or "").strip()
    if not title:
        return {"ok": False, "error": "title required"}
    gid = db.add_goal(
        chat_id,
        title,
        why=args.get("why"),
        target_date_local=args.get("target_date_local"),
        horizon=args.get("horizon", "long"),
        sub_tasks=args.get("sub_tasks") or [],
        watch_query=args.get("watch_query"),
    )
    # First time a goal is added on this chat, ensure the proactive crons are armed.
    scheduler.ensure_proactive_for(chat_id)
    return {"ok": True, "goal_id": gid}


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
    "save_note": tool_save_note,
    "search_memory": tool_search_memory,
    "add_goal": tool_add_goal,
    "list_goals": tool_list_goals,
    "update_goal": tool_update_goal,
    "complete_goal": tool_complete_goal,
    "add_goal_subtask": tool_add_goal_subtask,
    "complete_goal_subtask": tool_complete_goal_subtask,
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
    "  3) watch_query가 있으면 web_search 또는 fetch_url로 최근 정보(가격/혜택/이벤트)를 한 번 확인해 알림\n"
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
    goals_block = "\n\n".join(_format_goal_summary(r) for r in rows)
    user_msg = WEEKLY_REVIEW_PROMPT.format(goals_block=goals_block)
    logger.info("weekly review for chat=%s with %d goals", chat_id, len(rows))
    try:
        reply = await run_agent(chat_id, user_msg, history=[], max_hops=10)
    except Exception as exc:
        logger.exception("weekly review agent failed")
        reply = f"⚠️ 주간 골 리뷰 중 오류: {exc}"
    for r in rows:
        db.mark_goal_reviewed(r["id"])
    if bot:
        header = "📋 이번 주 골 리뷰\n\n"
        text = header + reply
        for i in range(0, len(text), 4000):
            await bot.send_message(chat_id=chat_id, text=text[i : i + 4000])


async def run_daily_imminent_check(chat_id: int) -> None:
    """Daily 08:00 — push only if any goal target_date is within D-7."""
    rows = db.goals_due_within(chat_id, days=7)
    bot = _app.bot if _app else None
    if not rows or not bot:
        return
    lines = ["📌 임박한 골 (D-7 이내)"]
    for r in rows:
        d = _days_until(r["target_date_local"])
        lines.append(f"• D-{d} · #{r['id']} {r['title']}")
    await bot.send_message(chat_id=chat_id, text="\n".join(lines))


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


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_history.pop(update.effective_chat.id, None)
    await update.message.reply_text("대화 메모리 초기화 완료. 일정/기억은 그대로 보존.")


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
    history = _history(chat_id)
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
    for i in range(0, len(reply), 4000):
        await update.message.reply_text(reply[i : i + 4000])


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

    user_text = f"[사진 첨부] caption={caption!r}\n추출된 정보:\n{description}"
    await _process_user_text(update, context, user_text, log_prefix="[photo] ")


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    user_text = update.message.text
    logger.info("msg from %s: %r", update.effective_chat.id, user_text[:200])
    await _process_user_text(update, context, user_text)


async def post_init(app: Application) -> None:
    global _app
    _app = app
    db.init_db()
    scheduler.init(
        app.bot,
        run_recurring_task,
        run_weekly_goal_review,
        run_daily_imminent_check,
    )
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
    app.add_handler(CommandHandler("goals", cmd_goals))
    app.add_handler(CommandHandler("review", cmd_review))
    app.add_handler(CommandHandler("notes", cmd_notes))
    app.add_handler(CommandHandler("tasks", cmd_tasks))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.PHOTO, on_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()
