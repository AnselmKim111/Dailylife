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


async def start_oauth_server(bot) -> web.AppRunner:
    """Start aiohttp on $PORT (Railway sets this) — non-blocking."""
    global _bot
    _bot = bot
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    app.router.add_get(gcal.REDIRECT_PATH, google_callback)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    logger.info("oauth/health server listening on 0.0.0.0:%s", PORT)
    return runner
