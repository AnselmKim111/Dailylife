"""Gmail v1 read-only API wrappers, sharing the gcal OAuth token.

Same google identity for the same chat_id — we reuse gcal.get_valid_access_token
since the token was issued for the combined `calendar + gmail.readonly` scope.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

import db
import gcal

logger = logging.getLogger(__name__)

API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"


async def _gmail_api(
    chat_id: int,
    method: str,
    path: str,
    *,
    params: Optional[Dict] = None,
) -> Dict[str, Any]:
    token = await gcal.get_valid_access_token(chat_id)
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{API_BASE}{path}"
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.request(method, url, headers=headers, params=params)
        if r.status_code == 401:
            # One forced refresh
            row = db.get_oauth_token(chat_id, "google")
            if row and row["refresh_token"]:
                refreshed = await gcal.refresh_access(row["refresh_token"])
                gcal.store_token_response(chat_id, refreshed)
                headers["Authorization"] = f"Bearer {refreshed['access_token']}"
                r = await c.request(method, url, headers=headers, params=params)
        if r.status_code == 403:
            # Likely missing gmail.readonly scope — token issued for calendar only.
            raise gcal.NotConnected(
                "Gmail scope not granted. /connect_gcal 다시 눌러 권한 재동의 필요"
            )
        if r.status_code >= 400:
            logger.error("gmail %s %s: %s", method, path, r.text[:400])
            r.raise_for_status()
        return r.json()


def _b64url_decode(s: str) -> bytes:
    s = s.replace("-", "+").replace("_", "/")
    padding = (-len(s)) % 4
    return base64.b64decode(s + "=" * padding)


def _extract_text(payload: Dict) -> str:
    """Walk the MIME tree and pull text/plain (fallback to text/html stripped)."""
    if not payload:
        return ""
    mime = payload.get("mimeType", "")
    body = payload.get("body") or {}
    data = body.get("data")
    if data and mime.startswith("text/plain"):
        try:
            return _b64url_decode(data).decode("utf-8", errors="replace")
        except Exception:
            return ""
    if data and mime.startswith("text/html"):
        try:
            html = _b64url_decode(data).decode("utf-8", errors="replace")
            # cheap HTML strip
            return re.sub(r"<[^>]+>", " ", html)
        except Exception:
            return ""
    parts = payload.get("parts") or []
    plain = ""
    html_fallback = ""
    for p in parts:
        m = p.get("mimeType", "")
        if m.startswith("text/plain") and (p.get("body") or {}).get("data"):
            try:
                plain += _b64url_decode(p["body"]["data"]).decode("utf-8", errors="replace") + "\n"
            except Exception:
                pass
        elif m.startswith("text/html") and (p.get("body") or {}).get("data"):
            try:
                html_fallback += _b64url_decode(p["body"]["data"]).decode("utf-8", errors="replace")
            except Exception:
                pass
        elif m.startswith("multipart/"):
            sub = _extract_text(p)
            if sub:
                plain += sub + "\n"
    if plain.strip():
        return plain
    if html_fallback:
        return re.sub(r"<[^>]+>", " ", html_fallback)
    return ""


def _headers_dict(payload: Dict) -> Dict[str, str]:
    return {h["name"].lower(): h["value"] for h in (payload.get("headers") or [])}


async def list_messages(
    chat_id: int,
    query: str = "",
    max_results: int = 20,
) -> List[str]:
    """Return Gmail message IDs matching the search query (q syntax)."""
    params = {"maxResults": min(max(max_results, 1), 50)}
    if query:
        params["q"] = query
    data = await _gmail_api(chat_id, "GET", "/messages", params=params)
    return [m["id"] for m in data.get("messages", [])]


async def get_message(
    chat_id: int,
    message_id: str,
    body_max_chars: int = 4000,
) -> Dict[str, Any]:
    """Fetch one message; return id/subject/from/date/snippet/body."""
    data = await _gmail_api(
        chat_id, "GET", f"/messages/{message_id}", params={"format": "full"}
    )
    payload = data.get("payload") or {}
    headers = _headers_dict(payload)
    body = _extract_text(payload).strip()
    return {
        "id": data.get("id"),
        "thread_id": data.get("threadId"),
        "subject": headers.get("subject"),
        "from": headers.get("from"),
        "to": headers.get("to"),
        "date": headers.get("date"),
        "snippet": data.get("snippet"),
        "body": body[:body_max_chars],
        "truncated": len(body) > body_max_chars,
    }


async def recent_summary(
    chat_id: int,
    hours: int = 24,
    max_messages: int = 10,
) -> List[Dict[str, Any]]:
    """Top N messages newer than `hours` ago, returned as small dicts. Caller can
    feed these into an LLM to produce a one-paragraph briefing."""
    q = f"newer_than:{max(hours // 24, 1)}d"
    ids = await list_messages(chat_id, query=q, max_results=max_messages)
    out: List[Dict[str, Any]] = []
    for mid in ids[:max_messages]:
        try:
            m = await get_message(chat_id, mid, body_max_chars=600)
        except Exception:
            logger.exception("recent_summary get_message failed for %s", mid)
            continue
        out.append({
            "id": m["id"],
            "subject": m["subject"],
            "from": m["from"],
            "date": m["date"],
            "snippet": m["snippet"],
        })
    return out
