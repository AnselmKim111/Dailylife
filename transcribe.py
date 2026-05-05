"""Audio + image processing wrappers.

- transcribe_voice: OpenAI Whisper API (needs OPENAI_API_KEY)
- describe_image: OpenRouter vision (Haiku 4.5) — extracts schedule/note/place info
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_VISION_MODEL = os.environ.get("OPENROUTER_VISION_MODEL", "anthropic/claude-haiku-4.5")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

VISION_PROMPT = (
    "이 이미지에서 사용자에게 유용한 정보를 추출해서 한국어로 정리해줘.\n"
    "특히 다음을 명시적으로 뽑아내:\n"
    "- 일정/이벤트: 날짜·시간·장소·제목\n"
    "- 메모성 정보: 책 제목, 가게 이름, 인물, 가격, 연락처\n"
    "- 영수증/지출이라면: 가게, 금액, 날짜, 품목\n"
    "- 포스터/공지라면: 행사명, 일시, 장소, 비용, 참여 방법\n"
    "추출이 어려우면 '명확한 일정/메모 정보 없음'이라고 한 줄로만 답해.\n"
    "이미지에 안 보이는 정보는 만들지 마."
)


class TranscribeUnavailable(Exception):
    pass


async def transcribe_voice(file_bytes: bytes, mime: str = "audio/ogg") -> str:
    """Whisper STT. Telegram voice notes are typically OGG Opus."""
    if not OPENAI_API_KEY:
        raise TranscribeUnavailable("OPENAI_API_KEY not configured")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    # Telegram voice files come as .oga (Opus in OGG container). Whisper accepts ogg.
    ext = "ogg" if "ogg" in mime else mime.split("/")[-1].split(";")[0] or "ogg"
    files = {
        "file": (f"voice.{ext}", file_bytes, mime or "audio/ogg"),
        "model": (None, "whisper-1"),
        "language": (None, "ko"),
    }
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(OPENAI_TRANSCRIBE_URL, headers=headers, files=files)
        if r.status_code >= 400:
            logger.error("Whisper %s: %s", r.status_code, r.text[:300])
            r.raise_for_status()
        data = r.json()
    return (data.get("text") or "").strip()


async def describe_image(file_bytes: bytes, mime: str, caption: Optional[str] = None) -> str:
    """Send image to OpenRouter vision model and get a structured Korean summary."""
    if not OPENROUTER_API_KEY:
        return "[비전 분석 불가: OPENROUTER_API_KEY 없음]"
    b64 = base64.b64encode(file_bytes).decode("ascii")
    media_type = mime if mime else "image/jpeg"
    user_text = VISION_PROMPT
    if caption:
        user_text += f"\n\n사용자 첨부 메모: {caption}"
    payload = {
        "model": OPENROUTER_VISION_MODEL,
        "max_tokens": 800,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://t.me/AnselmsSlave7bot"),
        "X-Title": "Dailylife Vision",
    }
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(OPENROUTER_URL, json=payload, headers=headers)
        if r.status_code >= 400:
            logger.error("Vision %s: %s", r.status_code, r.text[:300])
            r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"].strip()
