"""Tiny aiohttp server that runs alongside the Telegram polling bot.

Single responsibility: handle the Google OAuth redirect at /oauth/google/callback,
exchange the code for tokens, persist them, and notify the user via Telegram.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from aiohttp import web

import db
import gcal
import voice_call

logger = logging.getLogger(__name__)

PORT = int(os.environ.get("PORT", "8080"))
_bot = None  # set by start_oauth_server()


async def health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def google_callback(request: web.Request) -> web.Response:
    err = request.query.get("error")
    if err:
        logger.warning("oauth error from google: %s", err)
        return _html("❌ Google 인증이 거부되었습니다. 텔레그램으로 돌아가서 다시 /connect_gcal 해주세요.", 400)

    code = request.query.get("code")
    state = request.query.get("state")
    if not code or not state:
        return _html("⚠ code/state 누락", 400)

    chat_id = gcal.consume_state(state)
    if chat_id is None:
        return _html("⚠ 인증 세션이 만료됐어요. 텔레그램에서 /connect_gcal 다시 시도해주세요.", 400)

    try:
        token_payload = await gcal.exchange_code(code)
    except Exception:
        logger.exception("token exchange failed")
        return _html("⚠ 토큰 교환 실패. 다시 시도해주세요.", 500)

    gcal.store_token_response(chat_id, token_payload)
    logger.info("Google Calendar connected for chat %s", chat_id)

    if _bot is not None:
        try:
            await _bot.send_message(
                chat_id=chat_id,
                text="📅 Google 캘린더 연동 완료. 이제 일정 자연어로 말하면 자동 동기화돼요.",
            )
        except Exception:
            logger.exception("failed to notify chat after oauth")

    return _html(
        "✅ 연동 완료. 이 창은 닫고 텔레그램으로 돌아가세요.",
        200,
    )


def _html(msg: str, status: int) -> web.Response:
    body = (
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        "<title>Dailylife</title>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<style>body{font-family:-apple-system,system-ui,sans-serif;"
        "padding:32px;text-align:center;background:#fafafa;color:#222}"
        "div{max-width:480px;margin:80px auto;padding:32px;border-radius:16px;"
        "background:#fff;box-shadow:0 4px 24px rgba(0,0,0,0.06)}</style>"
        "</head><body><div><h2>Dailylife · 컨텍스트봇</h2>"
        f"<p>{msg}</p></div></body></html>"
    )
    return web.Response(text=body, content_type="text/html", status=status)


async def twilio_twiml(request: web.Request) -> web.Response:
    """TwiML callback — Twilio fetches this when the call connects."""
    try:
        call_id = int(request.query.get("call_id") or "0")
    except ValueError:
        return web.Response(status=400, text="bad call_id")
    if not call_id:
        return web.Response(status=400, text="bad call_id")
    row = db.get_voice_call(call_id)
    if not row:
        return web.Response(status=404, text="unknown call")
    db.update_voice_call(
        call_id, status="active",
        started_at=__import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat())
    twiml = voice_call.build_twiml(row["purpose"], call_id)
    return web.Response(text=twiml, content_type="text/xml")


async def twilio_recording(request: web.Request) -> web.Response:
    """Twilio fires this when <Record> finishes — we get RecordingUrl."""
    try:
        call_id = int(request.query.get("call_id") or "0")
    except ValueError:
        return web.Response(status=400)
    if not call_id:
        return web.Response(status=400)
    form = await request.post()
    rec_url = form.get("RecordingUrl")
    duration = form.get("RecordingDuration")
    logger.info("twilio recording for call_id=%s url=%s dur=%s",
                call_id, rec_url, duration)
    if rec_url:
        # fire-and-forget transcription
        import asyncio
        asyncio.create_task(_finalize_call(call_id, str(rec_url), int(duration or 0)))
    return web.Response(text="ok")


async def twilio_status(request: web.Request) -> web.Response:
    """Status callback — capture end + cost (best-effort)."""
    try:
        call_id = int(request.query.get("call_id") or "0")
    except ValueError:
        return web.Response(status=400)
    form = await request.post()
    status = form.get("CallStatus")
    if status == "completed":
        db.update_voice_call(
            call_id, status="done",
            ended_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat())
    elif status in ("failed", "no-answer", "busy"):
        db.update_voice_call(call_id, status="failed")
    return web.Response(text="ok")


async def _finalize_call(call_id: int, rec_url: str, duration_sec: int) -> None:
    """Download recording → Whisper → LLM summary → Telegram report."""
    import transcribe
    audio = await voice_call.fetch_recording_audio(rec_url)
    if not audio:
        db.update_voice_call(call_id, status="failed", duration_sec=duration_sec)
        return
    try:
        transcript = await transcribe.transcribe_voice(audio, mime="audio/mpeg")
    except Exception:
        logger.exception("call transcribe failed")
        transcript = "(녹음 전사 실패)"
    # Rough cost — Twilio voice ≈ $0.013/min outbound to KR + Whisper $0.006/min
    cost = max(0.0, duration_sec / 60.0 * 0.019)
    row = db.get_voice_call(call_id)
    summary_md = ""
    if _bot is not None and row:
        try:
            await _bot.send_message(
                chat_id=row["chat_id"],
                text=(f"📞 통화 #{call_id} 종료 — {duration_sec}초, ~${cost:.4f}\n\n"
                      f"녹음:\n{transcript[:1500]}"),
            )
            db.log_agent_action(
                row["chat_id"], "phone_call",
                summary=f"통화 {duration_sec}s → {row['to_number']}",
                payload={"call_id": call_id, "to": row["to_number"],
                         "purpose": row["purpose"][:200]},
            )
        except Exception:
            logger.exception("call summary send failed")
    db.update_voice_call(
        call_id, status="done", duration_sec=duration_sec,
        transcript_md=transcript[:4000],
        summary_md=summary_md[:1000],
        cost_usd=cost,
    )


async def start_oauth_server(bot) -> web.AppRunner:
    """Start aiohttp on $PORT (Railway sets this) — non-blocking."""
    global _bot
    _bot = bot
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_get(gcal.REDIRECT_PATH, google_callback)
    # v11: Twilio webhooks
    app.router.add_get("/twilio/twiml", twilio_twiml)
    app.router.add_post("/twilio/twiml", twilio_twiml)
    app.router.add_post("/twilio/recording", twilio_recording)
    app.router.add_post("/twilio/status", twilio_status)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    logger.info("oauth/health server listening on 0.0.0.0:%s", PORT)
    return runner
