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
                    "location": {
                        "type": "string",
                        "description": (
                            "Optional address or place name. When provided, the bot will "
                            "auto-schedule a leave-by alert using kakao directions ETA."
                        ),
                    },
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
    {
        "type": "function",
        "function": {
            "name": "enable_recurring_task",
            "description": "Re-enable a previously disabled recurring task without recreating it.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_recurring_task",
            "description": "Pause a recurring task (keeps the row, stops firing) without deleting it.",
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
                "Full-text search the user's saved notes AND past chat history AND uploaded "
                "attachments (PDF/사진/음성). Use for any 'what did I say about X', "
                "'내가 언제 X 얘기했지', '지난주에 X 어땠어' style recall. "
                "v19: attachments도 'all'에 포함 — 별도 호출 필요 없음."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["notes", "chat", "attachments", "all"],
                        "description": "notes | chat | attachments | all (default)",
                    },
                    "limit": {"type": "integer", "description": "Max hits per source (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    # ---- v19 attachments: 사용자가 보낸 파일 직접 검색 + 원본 재전송 ----
    {
        "type": "function",
        "function": {
            "name": "search_attachments",
            "description": (
                "사용자가 보낸 PDF/사진/음성 파일 검색. PNR·예약번호·항공편·confirmation·"
                "티켓·영수증·문서 등 *raw 추출 텍스트* 매칭. 사용자가 '내 항공권', "
                "'그 PDF', '예약번호', '5월에 보낸 사진' 류 언급 시 *항상 먼저* 호출. "
                "Each query token ≥3 chars works best."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "file_kind": {
                        "type": "string",
                        "enum": ["pdf", "image", "voice", "doc_other"],
                        "description": "Optional: restrict to one kind.",
                    },
                    "limit": {"type": "integer", "description": "Max hits (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resend_attachment",
            "description": (
                "사용자가 명시적으로 *원본 파일 재전송* 요청 시. '그 PDF 다시 보내줘', "
                "'#3 파일 줘', '5월에 보낸 사진 다시' 같은 표현. "
                "먼저 search_attachments로 attachment_id 찾고, 그 다음 이 도구 호출."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {"type": "integer"},
                },
                "required": ["attachment_id"],
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
                "If you also called add_event for the same event, pass that local event's id "
                "as link_local_event_id so the two records stay linked and undo works correctly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "Event title."},
                    "start_iso": {"type": "string", "description": f"Start datetime in {USER_TZ} ISO 8601."},
                    "end_iso": {"type": "string", "description": "Optional. Defaults to start + 1h."},
                    "description": {"type": "string"},
                    "location": {"type": "string"},
                    "link_local_event_id": {"type": "integer", "description": "Optional. The local event_id from a prior add_event call so the two records are linked."},
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
    # ---- spending + habits ----
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": (
                "Korean weather forecast for an address or place name (서대문구, "
                "DMC파크뷰자이아파트, 해운대 등). Uses Open-Meteo so no API key needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer", "description": "1-7, default 1."},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "korean_holiday_check",
            "description": "Check if a given KST date is a Korean public holiday + nearest upcoming one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_local": {"type": "string", "description": "YYYY-MM-DD (KST)."},
                },
                "required": ["date_local"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "solar_term_check",
            "description": (
                "Check if a date is a Korean 24절기 (입춘/하지/추분/입동 etc.) "
                "and return the next upcoming term."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_local": {"type": "string", "description": "YYYY-MM-DD (KST)."},
                },
                "required": ["date_local"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_duplicates",
            "description": (
                "Return potential duplicate people (same name) and duplicate facts (same key) "
                "in the user's memory. Use before suggesting consolidation."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "merge_people",
            "description": (
                "Merge multiple person rows into one. Moves important_dates + aliases + "
                "last_contact into keep_id, rewires all relations, then deletes drop_ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keep_id": {"type": "integer"},
                    "drop_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["keep_id", "drop_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_phone_call",
            "description": (
                "Place an outbound voice call on the user's behalf. Twilio "
                "speaks the `purpose` in Korean → records the recipient's "
                "response → hangs up. Transcript + summary auto-sent to user "
                "afterward. Use for: 식당 예약, 진료 예약 변경, 영업시간 확인, "
                "고객센터 문의. Always ask the user to confirm to_number first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to_number": {"type": "string", "description": "+82 prefix recommended (e.g., '+82212345678')."},
                    "purpose": {"type": "string", "description": "한국어 한 문장, 봇이 그대로 말함."},
                },
                "required": ["to_number", "purpose"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_scrape",
            "description": (
                "JS-rendered headless browser fetch. Use INSTEAD of fetch_url when "
                "the target page needs Javascript to render (SPA, infinite scroll, "
                "lazy-loaded content) or the user wants a specific selector waited "
                "for. Returns text content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "wait_for_selector": {"type": "string", "description": "Optional CSS selector to wait for."},
                    "max_chars": {"type": "integer", "description": "Default 6000."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_screenshot",
            "description": (
                "Take a screenshot of a webpage and send it to the user as a "
                "Telegram photo. Use when visual confirmation matters (booking "
                "confirmation, comparison table, status page)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "purpose": {"type": "string", "description": "Short caption."},
                    "full_page": {"type": "boolean"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_watch",
            "description": (
                "Register a long-running watch task. Bot will periodically check "
                "the condition (every check_interval_min, min 30) and fire ONCE "
                "when matched. Use when user says '~되면 알려줘', '~뜨면', "
                "'~떨어지면', '~매진 풀리면'. check_kind: 'web_search' for general "
                "queries, 'fetch_url' for a specific page, 'gmail_query' for "
                "Gmail search syntax."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": "Short Korean label ≤30자."},
                    "condition_md": {"type": "string"},
                    "check_kind": {"type": "string", "enum": ["fetch_url", "web_search", "gmail_query"]},
                    "check_args": {"type": "object"},
                    "check_interval_min": {"type": "integer", "description": "min 30, default 60."},
                },
                "required": ["label", "condition_md", "check_kind", "check_args"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_watches",
            "description": "List the user's watch tasks (active/triggered/cancelled).",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["active", "triggered", "cancelled"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lifelog_search",
            "description": (
                "Semantic search across the user's chat history, notes, events, "
                "and reflections via embeddings. USE FIRST when the user asks "
                "vaguely about something past ('그 영화 뭐였더라', '작년에 산 책', "
                "'그때 그 카페'). Returns top-k matches with similarity scores. "
                "Better than cross_recall for fuzzy recall — cross_recall is "
                "literal substring matching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "Top-k results (1-20, default 8)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_document",
            "description": (
                "Deep-read a document (contract / 계약서 / 약관 / 이력서 / 보고서) that "
                "the user already uploaded — bot pulls the extracted text from the most "
                "recent chat turn. Returns risk flags, key clauses, suggested questions "
                "to ask the counterparty. Use only when explicitly asked to 'read carefully' "
                "or 'check for risks' — don't auto-fire on every upload."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_text": {"type": "string", "description": "Full extracted text (paste from chat history)."},
                    "doc_kind": {"type": "string", "description": "Optional hint: contract|insurance|résumé|policy|other."},
                },
                "required": ["document_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lunar_to_solar",
            "description": (
                "Convert a lunar (음력) date to this year's solar date. Useful for "
                "computing when an elder's 음력 생신 falls this year."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {"type": "integer"},
                    "day": {"type": "integer"},
                    "solar_year": {"type": "integer", "description": "Year to resolve to (defaults to current)."},
                },
                "required": ["month", "day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_parcel",
            "description": "Best-effort 택배 추적. Returns raw page text the model can interpret.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_no": {"type": "string"},
                    "carrier": {"type": "string", "description": "Optional: cj, 한진, 우체국, etc."},
                },
                "required": ["tracking_no"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transit_text_query",
            "description": (
                "Stopgap public-transit query (until ODsay key arrives). Pulls a "
                "Naver mobile search page for the route and returns raw text; "
                "you then parse the times. Not precision-grade."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "arrive_by_iso": {"type": "string", "description": "Optional KST ISO 8601 target arrival."},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_wedding_timeline",
            "description": (
                "Bulk-create the canonical 약혼·결혼·신혼여행 long-horizon goals + "
                "their sub-tasks in one shot. Use when the user shares the dates "
                "for propose/marriage/honeymoon and wants the bot to start tracking."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "propose_date_local": {"type": "string", "description": "YYYY-MM-DD KST."},
                    "marriage_date_local": {"type": "string", "description": "YYYY-MM-DD KST."},
                    "honeymoon_date_local": {"type": "string", "description": "YYYY-MM-DD KST."},
                    "partner_name": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_expense",
            "description": (
                "Log a money spend. Use whenever the user mentions a price they paid "
                "('스타벅스 6500원', '점심 12000원 식비'). Korean Won (정수). category is "
                "free-form: 식비/카페/교통/선물/생활/문화/etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount_won": {"type": "integer"},
                    "category": {"type": "string"},
                    "merchant": {"type": "string"},
                    "when_local": {"type": "string", "description": "Optional KST ISO 8601 datetime; defaults to now."},
                    "notes": {"type": "string"},
                },
                "required": ["amount_won"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_expenses",
            "description": "Roll up expenses for /spending — last N days by category. Default 30 days.",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_habit",
            "description": (
                "Log a habit event. Use for '오늘 운동 1시간', '책 30분 읽음', '물 2L 마심'. "
                "habit_key is a short slug (exercise, reading, water, study, …)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "habit_key": {"type": "string"},
                    "duration_min": {"type": "integer"},
                    "notes": {"type": "string"},
                },
                "required": ["habit_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_habits",
            "description": "Roll up habits — last N days by habit_key. Default 7 days.",
            "parameters": {
                "type": "object",
                "properties": {"days": {"type": "integer"}},
            },
        },
    },
    # ---- people / relationships ----
    {
        "type": "function",
        "function": {
            "name": "add_person",
            "description": (
                "Persist a person the user mentioned (약혼녀, 가족, 동기, 친구, 동료, 후배 등). "
                "Use whenever the user introduces a named person or shares a fact like '내 약혼녀는 경서야'. "
                "If important_dates includes a birthday or anniversary, the assistant can later remind D-N."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Canonical short name, e.g. '경서'."},
                    "aliases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Other ways the user calls this person: ['경서야','우리 경서'].",
                    },
                    "role": {"type": "string", "description": "약혼녀 | 가족 | 동기 | 친구 | 직장 | 후배 | etc."},
                    "notes": {"type": "string", "description": "Free-form facts/context."},
                    "important_dates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "e.g. '생일', '기념일', '입대일', '음력 생신'."},
                                "date_local": {"type": "string", "description": "YYYY-MM-DD. If is_lunar=true, this is the LUNAR date (year ignored, only month+day used)."},
                                "recurring_yearly": {"type": "boolean"},
                                "is_lunar": {"type": "boolean", "description": "True for 음력 dates (양가 어른 생신 etc.); bot auto-converts to this year's solar date."},
                            },
                            "required": ["label", "date_local"],
                        },
                    },
                    "preferred_language": {
                        "type": "string",
                        "enum": ["ko", "en", "ja", "zh"],
                        "description": "Person이 모국어로 쓰는 언어. 외국인이면 'en'/'ja'/'zh' — 봇이 메일 도착 시 자동 한국어 요약하고 답장 시 자동 통역.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_person",
            "description": "Modify or extend an existing person. Aliases and important_dates are merged, not replaced.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "notes": {"type": "string"},
                    "add_aliases": {"type": "array", "items": {"type": "string"}},
                    "important_dates": {
                        "type": "array",
                        "items": {"type": "object"},
                    },
                },
                "required": ["person_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_people",
            "description": "List all known people for the user (optionally filtered by role).",
            "parameters": {
                "type": "object",
                "properties": {"role": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_person",
            "description": "Look up everything we know about one person — facts/notes/dates/last_contact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_alias": {"type": "string"},
                },
                "required": ["name_or_alias"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_contact_with",
            "description": "Record explicit contact with a person (call, message, meeting). Auto-updates last_contact_utc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_or_alias": {"type": "string"},
                    "channel": {"type": "string", "description": "예: 'call','telegram','meet','message','letter'."},
                    "notes": {"type": "string"},
                },
                "required": ["name_or_alias"],
            },
        },
    },
    # ---- Gmail read-only (shares Google OAuth with gcal) ----
    {
        "type": "function",
        "function": {
            "name": "gmail_search",
            "description": (
                "Search the user's Gmail with the same q syntax the Gmail UI accepts "
                "(e.g. 'from:airline subject:항공권 newer_than:7d'). Returns small "
                "message summaries (subject/from/date/snippet). Use get_message+id for full body."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "1-50, default 10."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_get_message",
            "description": "Fetch one Gmail message by id with full body (HTML stripped, up to 4000 chars).",
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string"},
                    "body_max_chars": {"type": "integer", "description": "Default 4000."},
                },
                "required": ["message_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_recent_summary",
            "description": (
                "Cheap overview of recent inbox — last N hours of messages "
                "(subject/from/snippet only, no body). Use for 'today's mail' style "
                "briefings or when the user asks 'anything important come in?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Default 24, max 168 (7d)."},
                    "max_messages": {"type": "integer", "description": "Default 10, max 50."},
                },
            },
        },
    },
    # ---- v4: outbound Gmail + GCal RSVP ----
    {
        "type": "function",
        "function": {
            "name": "gmail_send_email",
            "description": (
                "Send a new email on the user's behalf. Use only when the user "
                "explicitly asks to send a mail, or the request can clearly only be "
                "fulfilled by sending one (e.g. '내가 좀 늦는다고 메일 보내줘'). "
                "Always include a short body — no empty messages. Default subject if user didn't give one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email."},
                    "subject": {"type": "string"},
                    "body_text": {"type": "string"},
                },
                "required": ["to", "subject", "body_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_save_draft",
            "description": (
                "Save an email as a Gmail draft (does NOT send). Use when the user "
                "wants to review before sending, or when an auto-rule wants to "
                "stage a reply for human approval."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body_text": {"type": "string"},
                },
                "required": ["to", "subject", "body_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gmail_reply_to",
            "description": (
                "Reply to a specific Gmail message (threaded). `in_reply_to_msg_id` "
                "is the Gmail message id you saw in a recent search/get result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "in_reply_to_msg_id": {"type": "string"},
                    "body_text": {"type": "string"},
                    "to": {"type": "string", "description": "Optional override. Defaults to source's From: header."},
                    "subject": {"type": "string", "description": "Optional. Defaults to 'Re: <source subject>'."},
                },
                "required": ["in_reply_to_msg_id", "body_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gcal_rsvp",
            "description": (
                "RSVP to a Google Calendar invite — accepted/declined/tentative. "
                "Use when the user asks to accept/decline a meeting they were invited to, "
                "or when an auto-rule fires."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "response": {"type": "string", "enum": ["accepted", "declined", "tentative"]},
                },
                "required": ["event_id", "response"],
            },
        },
    },
    # ---- v4: rule engine ----
    {
        "type": "function",
        "function": {
            "name": "add_auto_rule",
            "description": (
                "Register a sent-consent rule for an autonomous action. "
                "rule_kind ∈ {'gmail_auto_add_event', 'gcal_auto_rsvp'}. "
                "Condition examples: "
                "{'sender_pattern': '@hyundaicard.com', 'min_confidence': 0.9} for gmail, "
                "{'from_people_names': ['경서','관현'], 'response': 'accepted'} for rsvp."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_kind": {"type": "string"},
                    "condition": {"type": "object"},
                },
                "required": ["rule_kind", "condition"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_auto_rules",
            "description": "List active rules so the user can review them.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "disable_auto_rule",
            "description": "Disable a single rule by id.",
            "parameters": {
                "type": "object",
                "properties": {"rule_id": {"type": "integer"}},
                "required": ["rule_id"],
            },
        },
    },
    # ---- v4: cross-table recall ----
    {
        "type": "function",
        "function": {
            "name": "cross_recall",
            "description": (
                "Pull every record across people + events + goals + notes + expenses "
                "+ chat_log that mentions a given entity (name or keyword). Use "
                "FIRST when the user asks 'X 관련된 거 다 보여줘' or 'tell me everything "
                "about X' — better than firing individual list/search tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "days": {"type": "integer", "description": "Lookback days for events/expenses/log. Default 180."},
                },
                "required": ["entity"],
            },
        },
    },
    # ---- v4: routines + streaks ----
    {
        "type": "function",
        "function": {
            "name": "detect_routines",
            "description": (
                "Run pattern mining on the user's events/expenses/habits and return "
                "weekly routines, recurring spending, and habit gaps. The user can then "
                "confirm to promote any of them to a recurring task or expense alert."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_habit_streaks",
            "description": "Return current and best streak for every habit the user logs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # ---- v5: mission mode (autonomous overnight projects) ----
    {
        "type": "function",
        "function": {
            "name": "start_mission",
            "description": (
                "Spawn a background autonomous mission that runs in 5-min ticks "
                "across hours/days. Use ONLY for tasks the user wants worked on "
                "asynchronously ('다음달 부산 코스 만들어 — 천천히 봐도 돼'). "
                "Don't use for instant questions. The agent loop persists state "
                "and resumes per tick until done or max_hops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short label (≤40 chars)."},
                    "goal_text": {"type": "string", "description": "Full goal description in Korean."},
                    "max_hops": {"type": "integer", "description": "Total hop budget (default 80, max 200)."},
                },
                "required": ["title", "goal_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_mission",
            "description": "Stop a running mission.",
            "parameters": {
                "type": "object",
                "properties": {"mission_id": {"type": "integer"}},
                "required": ["mission_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_missions",
            "description": "List recent missions for the user (optional status filter).",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["running", "done", "failed", "cancelled", "paused"]},
                },
            },
        },
    },
    # ---- v5: knowledge graph + reasoning ----
    {
        "type": "function",
        "function": {
            "name": "graph_query",
            "description": (
                "Multi-hop graph traversal across people/events/goals/notes/expenses. "
                "Returns the subgraph (nodes + edges) within max_depth hops of the start "
                "entity. Use BEFORE cross_recall for entity-relationship questions "
                "('경서랑 결혼 후 1년 안에 갔다온 여행', '내가 작년에 만난 사람들')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_kind": {"type": "string", "enum": ["person", "event", "goal", "note", "expense", "mission"]},
                    "start_id": {"type": "integer"},
                    "max_depth": {"type": "integer", "description": "Default 2."},
                    "relation_filter": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start_kind", "start_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_relation",
            "description": "Manually add a graph edge between two entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_kind": {"type": "string"},
                    "from_id": {"type": "integer"},
                    "to_kind": {"type": "string"},
                    "to_id": {"type": "integer"},
                    "relation_kind": {"type": "string"},
                },
                "required": ["from_kind", "from_id", "to_kind", "to_id", "relation_kind"],
            },
        },
    },
    # ---- v5: visual output ----
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate one PNG via OpenAI gpt-image-1 and send it as a Telegram "
                "photo to the user. Daily cap enforced by fact image_daily_cap "
                "(default 3). Use only when an image genuinely adds value "
                "(anniversary card, mood collage, illustration on request) — "
                "don't decorate routine answers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Detailed image prompt in Korean or English."},
                },
                "required": ["prompt"],
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
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> Dict:
    # v5: route by kind unless explicit `model=` was passed.
    if model is None:
        import model_router
        fact_override = None
        if chat_id is not None:
            try:
                import db as _db
                for r in _db.list_facts(chat_id):
                    if r["key"] == f"model_override_{kind}":
                        fact_override = r["value"]
                        break
            except Exception:
                fact_override = None
        model = model_router.pick_model(kind, fact_override=fact_override)
    payload: Dict = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
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
            model=data.get("model") or model or OPENROUTER_MODEL,
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
