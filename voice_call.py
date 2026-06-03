"""Twilio outbound voice calls.

Pragmatic v11 scope: bot speaks an instruction via TwiML <Say>, records
the recipient's response with <Record>, hangs up, then a webhook receives
the recording URL → Whisper transcript → LLM summary → user gets a
report. Full bidirectional Realtime streaming is deferred.

Required env (Railway):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_NUMBER (한국 발신 번호)
  PUBLIC_BASE_URL (Railway 도메인 — TwiML 콜백 URL 만드는 데 사용)

If creds missing, place_call returns a degraded error — bot stays alive.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

TWILIO_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_NUMBER = os.environ.get("TWILIO_NUMBER", "")
PUBLIC_BASE = os.environ.get(
    "PUBLIC_BASE_URL",
    os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
).rstrip("/")
if PUBLIC_BASE and not PUBLIC_BASE.startswith("http"):
    PUBLIC_BASE = "https://" + PUBLIC_BASE

try:
    from twilio.rest import Client as TwilioClient  # type: ignore
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False
    logger.info("twilio sdk not installed — voice call degraded")


def _client() -> Optional["TwilioClient"]:
    if not (TWILIO_AVAILABLE and TWILIO_SID and TWILIO_TOKEN):
        return None
    try:
        return TwilioClient(TWILIO_SID, TWILIO_TOKEN)
    except Exception:
        logger.exception("twilio client init failed")
        return None


def is_configured() -> bool:
    return bool(_client() and TWILIO_NUMBER and PUBLIC_BASE)


def build_twiml(instruction_md: str, call_id: int, max_record_sec: int = 60) -> str:
    """봇이 instruction을 한국어 TTS로 말하고 → 응답 녹음 → 끝."""
    # Use Twilio's built-in TTS (Polly) with Korean voice.
    safe_text = (instruction_md or "")[:600].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    record_action = f"{PUBLIC_BASE}/twilio/recording?call_id={call_id}"
    status_cb = f"{PUBLIC_BASE}/twilio/status?call_id={call_id}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response>'
        f'<Say language="ko-KR" voice="Polly.Seoyeon">{safe_text}</Say>'
        '<Pause length="1"/>'
        f'<Record maxLength="{max_record_sec}" timeout="3" '
        f'recordingStatusCallback="{record_action}" playBeep="true"/>'
        f'<Say language="ko-KR" voice="Polly.Seoyeon">감사합니다. 끊겠습니다.</Say>'
        '</Response>'
    )


async def place_call(
    to_number: str, instruction_md: str, call_id: int,
) -> Dict[str, Any]:
    """Place an outbound call. Returns dict with twilio_call_sid or error."""
    client = _client()
    if client is None:
        return {"ok": False, "error": "twilio not configured (SID/TOKEN/NUMBER + PUBLIC_BASE_URL needed)"}
    if not to_number:
        return {"ok": False, "error": "to_number required"}
    twiml = build_twiml(instruction_md, call_id)
    voice_cb = f"{PUBLIC_BASE}/twilio/twiml?call_id={call_id}"
    status_cb = f"{PUBLIC_BASE}/twilio/status?call_id={call_id}"
    try:
        # Pass TwiML directly via 'twiml' param (no separate hosted URL needed)
        call = client.calls.create(
            to=to_number,
            from_=TWILIO_NUMBER,
            twiml=twiml,
            status_callback=status_cb,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
        )
        return {"ok": True, "twilio_call_sid": call.sid}
    except Exception as e:
        logger.exception("twilio call failed")
        return {"ok": False, "error": str(e)}


async def cancel_call(twilio_call_sid: str) -> bool:
    client = _client()
    if client is None:
        return False
    try:
        client.calls(twilio_call_sid).update(status="completed")
        return True
    except Exception:
        logger.exception("twilio cancel failed")
        return False


async def fetch_recording_audio(recording_url: str) -> Optional[bytes]:
    """Twilio recordings need basic auth to download."""
    import httpx
    if not (TWILIO_SID and TWILIO_TOKEN):
        return None
    # Twilio recording URLs end without extension — append .mp3
    url = recording_url if recording_url.endswith(".mp3") else recording_url + ".mp3"
    try:
        async with httpx.AsyncClient(timeout=60.0,
                                       auth=(TWILIO_SID, TWILIO_TOKEN)) as c:
            r = await c.get(url)
            r.raise_for_status()
            return r.content
    except Exception:
        logger.exception("recording fetch failed")
        return None
