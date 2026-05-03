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
]


async def chat_completion(
    messages: List[Dict],
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[str] = None,
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
        return resp.json()


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
