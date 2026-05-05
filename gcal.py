"""Google Calendar OAuth 2.0 flow + Calendar v3 API wrappers.

Per-user tokens are stored in `oauth_tokens` keyed by (chat_id, 'google').
Refresh-token rotation: Google often omits refresh_token on refresh responses,
so db.save_oauth_token preserves the existing one when the new payload omits it.
"""

from __future__ import annotations

import logging
import os
import secrets as _secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx

import db

logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
PUBLIC_BASE_URL = os.environ.get(
    "PUBLIC_BASE_URL", "https://dailylife-production.up.railway.app"
).rstrip("/")
USER_TZ_NAME = os.environ.get("USER_TZ", "Asia/Seoul")
TZ = ZoneInfo(USER_TZ_NAME)

REDIRECT_PATH = "/oauth/google/callback"
SCOPE = "https://www.googleapis.com/auth/calendar"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CAL_API_BASE = "https://www.googleapis.com/calendar/v3"

# state token -> chat_id (in-memory, OAuth round-trip is short).
_pending_state: Dict[str, int] = {}


class NotConnected(Exception):
    pass


def redirect_uri() -> str:
    return f"{PUBLIC_BASE_URL}{REDIRECT_PATH}"


def is_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def build_auth_url(chat_id: int) -> str:
    """Generate the user-facing Google sign-in URL for this chat."""
    state = _secrets.token_urlsafe(16)
    _pending_state[state] = chat_id
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",
        "prompt": "consent",  # force refresh_token issuance
        "state": state,
    }
    return f"{AUTH_URL}?{urlencode(params)}"


async def exchange_code(code: str) -> Dict[str, Any]:
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri(),
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(TOKEN_URL, data=data)
        if r.status_code >= 400:
            logger.error("token exchange %s: %s", r.status_code, r.text[:300])
            r.raise_for_status()
        return r.json()


async def refresh_access(refresh_token: str) -> Dict[str, Any]:
    data = {
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(TOKEN_URL, data=data)
        if r.status_code >= 400:
            logger.error("token refresh %s: %s", r.status_code, r.text[:300])
            r.raise_for_status()
        return r.json()


def consume_state(state: str) -> Optional[int]:
    """Pop the chat_id associated with this state token (single-use)."""
    return _pending_state.pop(state, None)


def store_token_response(chat_id: int, payload: Dict[str, Any]) -> None:
    expires_in = int(payload.get("expires_in") or 0)
    expires_at = (
        (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
        if expires_in
        else None
    )
    db.save_oauth_token(
        chat_id=chat_id,
        provider="google",
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token"),
        expires_at_utc=expires_at,
        scopes=payload.get("scope") or SCOPE,
    )


async def get_valid_access_token(chat_id: int) -> str:
    """Return a non-expired access_token for the user, refreshing if needed."""
    row = db.get_oauth_token(chat_id, "google")
    if not row:
        raise NotConnected("Google Calendar not connected for this chat")
    expires_at = row["expires_at_utc"]
    needs_refresh = False
    if expires_at:
        try:
            ea = datetime.fromisoformat(expires_at)
            if ea <= datetime.now(timezone.utc) + timedelta(seconds=60):
                needs_refresh = True
        except ValueError:
            needs_refresh = True
    if needs_refresh:
        if not row["refresh_token"]:
            raise NotConnected("access token expired and no refresh_token; reconnect needed")
        refreshed = await refresh_access(row["refresh_token"])
        store_token_response(chat_id, refreshed)
        row = db.get_oauth_token(chat_id, "google")
        if row is None:
            raise NotConnected("token store error after refresh")
    return row["access_token"]


async def _api(
    chat_id: int,
    method: str,
    path: str,
    *,
    params: Optional[Dict] = None,
    json_body: Optional[Dict] = None,
) -> Dict[str, Any]:
    token = await get_valid_access_token(chat_id)
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{CAL_API_BASE}{path}"
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.request(method, url, headers=headers, params=params, json=json_body)
        if r.status_code == 401:
            # Try one forced refresh on stale tokens.
            row = db.get_oauth_token(chat_id, "google")
            if row and row["refresh_token"]:
                refreshed = await refresh_access(row["refresh_token"])
                store_token_response(chat_id, refreshed)
                headers["Authorization"] = f"Bearer {refreshed['access_token']}"
                r = await c.request(method, url, headers=headers, params=params, json=json_body)
        if r.status_code >= 400:
            logger.error("gcal %s %s: %s", method, path, r.text[:400])
            r.raise_for_status()
        if r.status_code == 204 or not r.content:
            return {}
        return r.json()


def _to_rfc3339(dt_iso_local_or_utc: str) -> str:
    """Normalize an ISO 8601 datetime string to RFC3339 with offset."""
    s = dt_iso_local_or_utc.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return dt.isoformat()


async def list_events(
    chat_id: int,
    time_min_iso: Optional[str] = None,
    time_max_iso: Optional[str] = None,
    max_results: int = 25,
    calendar_id: str = "primary",
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": min(max(max_results, 1), 100),
    }
    if time_min_iso:
        params["timeMin"] = _to_rfc3339(time_min_iso)
    else:
        params["timeMin"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if time_max_iso:
        params["timeMax"] = _to_rfc3339(time_max_iso)
    data = await _api(chat_id, "GET", f"/calendars/{calendar_id}/events", params=params)
    out = []
    for e in data.get("items", []):
        start = e.get("start", {})
        end = e.get("end", {})
        out.append({
            "id": e.get("id"),
            "summary": e.get("summary"),
            "description": e.get("description"),
            "location": e.get("location"),
            "start": start.get("dateTime") or start.get("date"),
            "end": end.get("dateTime") or end.get("date"),
            "html_link": e.get("htmlLink"),
            "status": e.get("status"),
        })
    return out


async def create_event(
    chat_id: int,
    summary: str,
    start_iso: str,
    end_iso: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    calendar_id: str = "primary",
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "summary": summary,
        "start": {"dateTime": _to_rfc3339(start_iso), "timeZone": USER_TZ_NAME},
        "end": {
            "dateTime": _to_rfc3339(end_iso) if end_iso else _to_rfc3339(start_iso),
            "timeZone": USER_TZ_NAME,
        },
    }
    if not end_iso:
        # Default to 1-hour duration
        from_dt = datetime.fromisoformat(_to_rfc3339(start_iso))
        body["end"]["dateTime"] = (from_dt + timedelta(hours=1)).isoformat()
    if description:
        body["description"] = description
    if location:
        body["location"] = location
    return await _api(chat_id, "POST", f"/calendars/{calendar_id}/events", json_body=body)


async def update_event(
    chat_id: int,
    event_id: str,
    summary: Optional[str] = None,
    start_iso: Optional[str] = None,
    end_iso: Optional[str] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    calendar_id: str = "primary",
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if summary is not None:
        body["summary"] = summary
    if start_iso:
        body["start"] = {"dateTime": _to_rfc3339(start_iso), "timeZone": USER_TZ_NAME}
    if end_iso:
        body["end"] = {"dateTime": _to_rfc3339(end_iso), "timeZone": USER_TZ_NAME}
    if description is not None:
        body["description"] = description
    if location is not None:
        body["location"] = location
    return await _api(chat_id, "PATCH", f"/calendars/{calendar_id}/events/{event_id}", json_body=body)


async def delete_event(chat_id: int, event_id: str, calendar_id: str = "primary") -> bool:
    await _api(chat_id, "DELETE", f"/calendars/{calendar_id}/events/{event_id}")
    return True
