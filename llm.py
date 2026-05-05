"""OpenRouter client + tool-use schema for the schedule agent."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HTTP_REFERER = os.environ.get("OPENROUTER_REFERER", "https://t.me/AnselmsSlave7bot")
APP_TITLE = os.environ.get("OPENROUTER_TITLE", "Dailylife Telegram Bot")
USER_TZ = os.environ.get("USER_TZ", "Asia/Seoul")

TOOLS: List[Dict] = [
    # ---- schedule ----
    {
        "type": "function",
        "function": {
            "name": "add_event",
            "description": (
                "Save a new schedule entry for the user. Use whenever the user mentions a "
                "future appointment, plan, deadline, or task with a specific time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short, glanceable title (Korean OK)."},
                    "when_iso": {
                        "type": "string",
                        "description": (
                            f"ISO 8601 datetime in the user's local timezone ({USER_TZ}). "
                            "Example: 2026-05-04T15:00:00. If the user gave a relative time "
                            "('내일 3시'), resolve it using the current_time provided in the system prompt."
                        ),
                    },
                    "notes": {"type": "string", "description": "Optional details."},
                    "remind_lead_minutes": {
                        "type": "integer",
                        "description": (
                            "Minutes before the event to send a reminder. Default 30 unless the "
                            "user specifies otherwise. Set to 0 for no reminder."
                        ),
                    },
                },
                "required": ["title", "when_iso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": (
                "Look up the user's saved schedule entries. Use for any 'what's on my "
                "schedule' / '오늘 뭐 있어?' / 'agenda' style query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_iso": {"type": "string", "description": f"Inclusive start datetime in {USER_TZ}. Omit for 'now'."},
                    "to_iso": {"type": "string", "description": f"Inclusive end datetime in {USER_TZ}. Omit for no upper bound."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_event",
            "description": "Modify an existing event the user previously saved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "when_iso": {"type": "string"},
                    "notes": {"type": "string"},
                    "remind_lead_minutes": {"type": "integer"},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": "Remove an event from the user's schedule.",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "integer"}},
                "required": ["event_id"],
            },
        },
    },
    # ---- long-term facts about the user ----
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": (
                "Persist a piece of personal information the user wants you to remember "
                "long-term (home address, workplace, military unit, family names, "
                "preferences, etc.). Call this whenever the user shares a stable fact "
                "about themselves, OR when they explicitly say '기억해줘'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short snake_case identifier, e.g. 'home_address', 'unit_location'."},
                    "value": {"type": "string", "description": "The actual value to store, in natural language."},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": "Delete a previously remembered fact by its key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    # ---- recurring tasks (e.g. daily 7am udo ferry briefing) ----
    {
        "type": "function",
        "function": {
            "name": "add_recurring_task",
            "description": (
                "Schedule a daily task that runs at a fixed local time and runs the given "
                "prompt through this same agent (with full tool access), then sends the "
                "answer to the user. Use whenever the user wants a recurring briefing — "
                "'매일 7시에 우도 배 운항 알려줘' / 'every morning summarize ...'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cron_kst": {
                        "type": "string",
                        "description": "Daily fire time in HH:MM (24h, KST). Example: '07:00'.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The instruction to run, written as if the user is sending it. "
                            "Example: '우도 운항 정보 사이트 확인해서 오늘 운항여부와 시간표 알려줘.'"
                        ),
                    },
                },
                "required": ["cron_kst", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_recurring_tasks",
            "description": "List all recurring tasks the user has set up.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_recurring_task",
            "description": "Cancel a recurring task by its id.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
    },
    # ---- external lookups ----
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search Korean web (Naver). Use for general questions, schedules, news, "
                "operating-hours lookups. Returns title/snippet/link list."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["blog", "news", "webkr", "encyc"],
                        "description": "blog (default) | news | webkr | encyc",
                    },
                    "display": {"type": "integer", "description": "Number of results (1-10). Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kakao_local_search",
            "description": (
                "Search Korean places (restaurants, addresses, landmarks) via Kakao Local. "
                "Returns place name, address, phone, lat/long. Optional radius search around (x,y)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "x": {"type": "number", "description": "Optional center longitude."},
                    "y": {"type": "number", "description": "Optional center latitude."},
                    "radius_m": {"type": "integer", "description": "Optional search radius in meters (max 20000)."},
                    "size": {"type": "integer", "description": "Number of results (1-15). Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kakao_directions_drive",
            "description": (
                "Get DRIVING (car) directions between two coords via Kakao Mobility. "
                "Returns distance and duration. Note: no public-transit option on this tier."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin_x": {"type": "number"},
                    "origin_y": {"type": "number"},
                    "dest_x": {"type": "number"},
                    "dest_y": {"type": "number"},
                },
                "required": ["origin_x", "origin_y", "dest_x", "dest_y"],
            },
        },
    },
    # ---- long-horizon goals (proactive engine) ----
    {
        "type": "function",
        "function": {
            "name": "add_goal",
            "description": (
                "Save a LONG-HORIZON goal (weeks/months away, big enough that the user is likely "
                "to forget without a system). Examples: '12월 말 프로포즈 여행', '6월 동기들과 제주', "
                "'약혼반지 살 상품권 미리 알아보기', '신혼여행 2월'. Different from add_event "
                "(specific clock time soon) and save_note (no horizon). "
                "If the goal involves periodic price/availability watching (gift cards, hotel rates, "
                "concert tickets), set watch_query so the weekly review can fetch fresh info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "why": {"type": "string", "description": "Why this matters / context."},
                    "target_date_local": {
                        "type": "string",
                        "description": f"YYYY-MM-DD in {USER_TZ}. Omit if vague.",
                    },
                    "horizon": {
                        "type": "string",
                        "enum": ["short", "medium", "long"],
                        "description": "short (~1mo), medium (~3mo), long (3mo+).",
                    },
                    "sub_tasks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional initial breakdown.",
                    },
                    "watch_query": {
                        "type": "string",
                        "description": (
                            "Optional Naver-search-friendly query for periodic monitoring. "
                            "Example: '신세계상품권 5% 할인'. Bot will run web_search on this weekly."
                        ),
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_goals",
            "description": "List the user's goals (default: open).",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "done", "paused", "dropped", "all"],
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_goal",
            "description": "Modify fields of an existing goal. Use to push out target_date, refine why, or change watch_query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "integer"},
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                    "target_date_local": {"type": "string"},
                    "horizon": {"type": "string", "enum": ["short", "medium", "long"]},
                    "status": {"type": "string", "enum": ["open", "done", "paused", "dropped"]},
                    "watch_query": {"type": "string"},
                },
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_goal",
            "description": "Mark a goal as done.",
            "parameters": {
                "type": "object",
                "properties": {"goal_id": {"type": "integer"}},
                "required": ["goal_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_goal_subtask",
            "description": "Add a sub-step to an existing goal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["goal_id", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_goal_subtask",
            "description": "Mark a sub-step done by its 0-based index in the sub_tasks list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "integer"},
                    "sub_index": {"type": "integer"},
                },
                "required": ["goal_id", "sub_index"],
            },
        },
    },
    # ---- notes + episodic memory ----
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": (
                "Save a free-form short note for later recall. Use for quick captures the "
                "user wants to remember but isn't a stable identity fact (use remember_fact "
                "for that), an event with a time (use add_event), or a long-horizon goal "
                "(use add_goal). Examples: '오늘 점심에 김철수 만남 — 인하대 후배', "
                "'스벅에서 본 책 추천: ...', '아빠 생신 선물 후보들'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "tags": {"type": "string", "description": "Optional comma-separated tags."},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Full-text search the user's saved notes AND past chat history (episodic). "
                "Use for any 'what did I say about X', '내가 언제 X 얘기했지', "
                "'지난주에 X 어땠어' style recall. Each query token ≥3 chars works best."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["notes", "chat", "all"],
                        "description": "notes (saved notes only) | chat (past chat only) | all (default)",
                    },
                    "limit": {"type": "integer", "description": "Max hits per source (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "GET a URL and return its text (HTML stripped). Use for direct page lookups "
                "like ferry operation status, train timetables, business pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "description": "Truncation limit (default 8000)."},
                },
                "required": ["url"],
            },
        },
    },
    # ---- Google Calendar (per-user OAuth) ----
    {
        "type": "function",
        "function": {
            "name": "gcal_list_events",
            "description": (
                "List events on the user's Google Calendar in a time window. "
                "Use any time the user asks about Google Calendar specifically OR when "
                "they ask 'what's on my schedule' and you need authoritative data. "
                "Returns google event ids that can be passed to gcal_update_event / gcal_delete_event."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min_iso": {"type": "string", "description": f"Inclusive start in {USER_TZ}. Omit = now."},
                    "time_max_iso": {"type": "string", "description": f"Inclusive end in {USER_TZ}. Omit = no upper bound."},
                    "max_results": {"type": "integer", "description": "Default 25, max 100."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gcal_create_event",
            "description": (
                "Create an event on the user's Google Calendar. Use whenever the user "
                "wants something on their actual Google Calendar (not just bot's local store). "
                "If you also use add_event for a local reminder, that's fine — both can coexist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title."},
                    "start_iso": {"type": "string", "description": f"Start datetime in {USER_TZ} ISO 8601."},
                    "end_iso": {"type": "string", "description": "Optional. Defaults to start + 1h."},
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["summary", "start_iso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gcal_update_event",
            "description": "Modify a Google Calendar event. event_id comes from gcal_list_events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "start_iso": {"type": "string"},
                    "end_iso": {"type": "string"},
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["event_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gcal_delete_event",
            "description": "Delete a Google Calendar event by id.",
            "parameters": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
            },
        },
    },
]


async def chat_completion(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[str] = None,
    chat_id: Optional[int] = None,
    kind: str = "chat",
) -> Dict:
    payload: Dict = {"model": OPENROUTER_MODEL, "messages": messages}
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": HTTP_REFERER,
        "X-Title": APP_TITLE,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("OpenRouter %s: %s", resp.status_code, resp.text)
            resp.raise_for_status()
        data = resp.json()
    # Best-effort usage logging — never let logging failure poison the call.
    try:
        u = data.get("usage") or {}
        # OpenRouter sometimes returns string cost; coerce.
        cost = u.get("cost") or 0
        if isinstance(cost, str):
            try:
                cost = float(cost)
            except ValueError:
                cost = 0.0
        import db as _db   # local import to avoid circular
        _db.log_usage(
            chat_id=chat_id,
            model=data.get("model") or OPENROUTER_MODEL,
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            cost_usd=float(cost or 0),
            kind=kind,
        )
    except Exception:
        logger.exception("usage logging failed (ignored)")
    return data


def parse_tool_calls(message: Dict) -> List[Dict]:
    """Return list of {name, arguments(dict), id} from a model message."""
    out = []
    for tc in message.get("tool_calls") or []:
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        out.append({"id": tc["id"], "name": tc["function"]["name"], "arguments": args})
    return out
