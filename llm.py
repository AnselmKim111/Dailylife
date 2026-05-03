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
                    "from_iso": {
                        "type": "string",
                        "description": f"Inclusive start datetime in {USER_TZ}. Omit for 'now'.",
                    },
                    "to_iso": {
                        "type": "string",
                        "description": f"Inclusive end datetime in {USER_TZ}. Omit for 'no upper bound'.",
                    },
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
