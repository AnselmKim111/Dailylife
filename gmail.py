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
    json_body: Optional[Dict] = None,
) -> Dict[str, Any]:
    token = await gcal.get_valid_access_token(chat_id)
    headers = {"Authorization": f"Bearer {token}"}
    if json_body is not None:
        headers["Content-Type"] = "application/json"
    url = f"{API_BASE}{path}"
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.request(method, url, headers=headers, params=params, json=json_body)
        if r.status_code == 401:
            # One forced refresh
            row = db.get_oauth_token(chat_id, "google")
            if row and row["refresh_token"]:
                refreshed = await gcal.refresh_access(row["refresh_token"])
                gcal.store_token_response(chat_id, refreshed)
                headers["Authorization"] = f"Bearer {refreshed['access_token']}"
                r = await c.request(method, url, headers=headers, params=params, json=json_body)
        if r.status_code == 403:
            # Likely missing scope — token issued for fewer scopes than needed.
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


def _b64url_encode(s: bytes) -> str:
    return base64.urlsafe_b64encode(s).decode("ascii").rstrip("=")


def _build_rfc822(
    to: str, subject: str, body_text: str,
    *, in_reply_to: Optional[str] = None, references: Optional[str] = None,
) -> str:
    """Compose a minimal RFC 822 email + return the base64url-encoded raw form
    Gmail's users.messages.send expects."""
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
        msg["References"] = references or in_reply_to
    msg.set_content(body_text)
    return _b64url_encode(msg.as_bytes())


async def send_message(
    chat_id: int,
    to: str,
    subject: str,
    body_text: str,
    *,
    in_reply_to_msg_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Send an email via Gmail. If `in_reply_to_msg_id` is set, we fetch that
    message's Message-ID header so the reply threads in Gmail."""
    in_reply_to = None
    thread_id = None
    if in_reply_to_msg_id:
        try:
            src = await get_message(chat_id, in_reply_to_msg_id, body_max_chars=200)
            thread_id = src.get("thread_id")
            # Fetch full headers for Message-ID
            data = await _gmail_api(
                chat_id, "GET", f"/messages/{in_reply_to_msg_id}",
                params={"format": "metadata", "metadataHeaders": "Message-ID"},
            )
            for h in (data.get("payload") or {}).get("headers", []):
                if h.get("name", "").lower() == "message-id":
                    in_reply_to = h.get("value")
                    break
        except Exception:
            logger.exception("failed to load source for in_reply_to")
    raw = _build_rfc822(to, subject, body_text, in_reply_to=in_reply_to)
    body = {"raw": raw}
    if thread_id:
        body["threadId"] = thread_id
    return await _gmail_api(chat_id, "POST", "/messages/send", json_body=body)


async def draft_message(
    chat_id: int,
    to: str,
    subject: str,
    body_text: str,
) -> Dict[str, Any]:
    """Save an email as a Gmail draft (does NOT send)."""
    raw = _build_rfc822(to, subject, body_text)
    body = {"message": {"raw": raw}}
    return await _gmail_api(chat_id, "POST", "/drafts", json_body=body)


async def modify_labels(
    chat_id: int, message_id: str,
    add: Optional[list] = None, remove: Optional[list] = None,
) -> Dict[str, Any]:
    """Apply Gmail labels. Common use: remove=['UNREAD'] to archive-as-read."""
    body = {}
    if add:
        body["addLabelIds"] = list(add)
    if remove:
        body["removeLabelIds"] = list(remove)
    return await _gmail_api(
        chat_id, "POST", f"/messages/{message_id}/modify", json_body=body,
    )


async def archive(chat_id: int, message_id: str) -> Dict[str, Any]:
    """Archive a message — removes INBOX label (Gmail's archive semantics)."""
    return await modify_labels(chat_id, message_id, remove=["INBOX"])


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
