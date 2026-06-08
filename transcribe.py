"""Audio + image processing wrappers.

- transcribe_voice: OpenAI Whisper API (needs OPENAI_API_KEY)
- describe_image: OpenRouter vision (Haiku 4.5) — extracts schedule/note/place info
- extract_pdf_text: local pypdf, no network call
"""

from __future__ import annotations

import base64
import io
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


CLASSIFY_PROMPT = (
    "다음 텍스트가 어떤 종류의 자료인지 한 줄 JSON으로 분류해. "
    "허용 카테고리: receipt(영수증·가격표), event(일정·행사·예약), "
    "business_card(명함·연락처), poster(포스터·공지·홍보), document_text(일반 글), none(분류 불가).\n"
    'JSON 형식: {"kind": <카테고리>, "confidence": 0.0~1.0, "summary": "한국어로 ≤140자 요약"}\n'
    "오직 JSON 한 줄만 출력. 그 외 텍스트 금지."
)


async def classify_content(text: str, hint: Optional[str] = None) -> dict:
    """Tiny LLM call (~$0.0002) that routes uploaded content to the right tool.

    Falls back to {kind: 'none'} on any failure so the caller can still pass
    raw text to the main agent. Tolerant JSON parser handles models that
    wrap output in code fences or add commentary."""
    if not text or not text.strip():
        return {"kind": "none", "confidence": 1.0, "summary": ""}
    if not OPENROUTER_API_KEY:
        return {"kind": "none", "confidence": 0.0, "summary": ""}
    user_msg = (f"hint: {hint}\n\n" if hint else "") + f"text:\n{text[:1500]}"
    payload = {
        "model": os.environ.get("OPENROUTER_CLASSIFY_MODEL",
                                  os.environ.get("OPENROUTER_MODEL",
                                                 "anthropic/claude-haiku-4.5")),
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": CLASSIFY_PROMPT + "\n\n" + user_msg},
        ],
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://t.me/AnselmsSlave7bot"),
        "X-Title": "Dailylife Classify",
    }
    async with httpx.AsyncClient(timeout=20.0) as c:
        try:
            r = await c.post(OPENROUTER_URL, json=payload, headers=headers)
            if r.status_code >= 400:
                logger.warning("classify_content %s: %s", r.status_code, r.text[:200])
                return {"kind": "none", "confidence": 0.0, "summary": ""}
            data = r.json()
        except Exception:
            logger.exception("classify_content network failure")
            return {"kind": "none", "confidence": 0.0, "summary": ""}
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    # Strip fences and find the JSON object.
    import json as _json
    import re as _re
    blob = content
    # remove ```json … ``` fences
    blob = _re.sub(r"^\s*```(?:json)?\s*", "", blob.strip(), flags=_re.IGNORECASE)
    blob = _re.sub(r"\s*```\s*$", "", blob)
    # If model said extra text, take first {…} block
    m = _re.search(r"\{[\s\S]*\}", blob)
    if m:
        blob = m.group(0)
    try:
        parsed = _json.loads(blob)
    except Exception:
        logger.warning("classify_content non-JSON: %r", content[:200])
        return {"kind": "none", "confidence": 0.0, "summary": ""}
    kind = parsed.get("kind", "none")
    if kind not in {"receipt", "event", "business_card", "poster",
                    "document_text", "none"}:
        kind = "none"
    try:
        conf = float(parsed.get("confidence") or 0.0)
    except Exception:
        conf = 0.0
    return {
        "kind": kind,
        "confidence": conf,
        "summary": (parsed.get("summary") or "")[:140],
    }


OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


async def embed_text(text: str, model: str = "text-embedding-3-small") -> bytes:
    """OpenAI Embeddings — text → float32 array bytes (1536 dims, ~6KB).

    Cost: \\$0.02 / 1M tokens. lifelog cron이 매일 약 10K tokens 사용 = \\$0.0002/일."""
    if not OPENAI_API_KEY:
        raise TranscribeUnavailable("OPENAI_API_KEY not configured")
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text for embedding")
    if len(text) > 8000:
        text = text[:8000]
    payload = {"model": model, "input": text}
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(OPENAI_EMBEDDINGS_URL, json=payload, headers=headers)
        if r.status_code >= 400:
            logger.error("embed %s: %s", r.status_code, r.text[:300])
            r.raise_for_status()
        data = r.json()
    vec = data["data"][0]["embedding"]
    import struct
    return struct.pack(f"{len(vec)}f", *vec)


def cosine_similarity(a_bytes: bytes, b_bytes: bytes) -> float:
    import struct
    n = len(a_bytes) // 4
    a = struct.unpack(f"{n}f", a_bytes)
    b = struct.unpack(f"{n}f", b_bytes)
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def generate_image(
    prompt: str,
    size: str = "1024x1024",
    model: str = "gpt-image-1",
) -> bytes:
    """Generate one PNG via OpenAI Images API. Returns raw PNG bytes.

    gpt-image-1 returns base64-encoded b64_json by default.
    Cost: ~$0.04 per 1024x1024 (varies). Daily cap enforced by caller."""
    if not OPENAI_API_KEY:
        raise TranscribeUnavailable("OPENAI_API_KEY not configured")
    payload = {
        "model": model,
        "prompt": prompt[:1000],
        "size": size,
        "n": 1,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(OPENAI_IMAGE_URL, json=payload, headers=headers)
        if r.status_code >= 400:
            logger.error("image gen %s: %s", r.status_code, r.text[:300])
            r.raise_for_status()
        data = r.json()
    item = (data.get("data") or [{}])[0]
    b64 = item.get("b64_json")
    if b64:
        return base64.b64decode(b64)
    url = item.get("url")
    if url:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r2 = await c.get(url)
            r2.raise_for_status()
            return r2.content
    raise RuntimeError("no image data returned")


async def synthesize_voice(
    text: str,
    voice: str = "nova",
    model: str = "gpt-4o-mini-tts",
    response_format: str = "opus",
) -> bytes:
    """Render Korean text → Telegram-friendly voice note (OGG Opus).

    Caller usually feeds the returned bytes into bot.send_voice. ~$0.015/1000
    chars on OpenAI gpt-4o-mini-tts; cap input to keep cost bounded."""
    if not OPENAI_API_KEY:
        raise TranscribeUnavailable("OPENAI_API_KEY not configured")
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text for TTS")
    if len(text) > 4000:
        text = text[:4000]
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(OPENAI_TTS_URL, json=payload, headers=headers)
        if r.status_code >= 400:
            logger.error("TTS %s: %s", r.status_code, r.text[:300])
            r.raise_for_status()
    return r.content


def extract_pdf_text(file_bytes: bytes, max_pages: int = 30, max_chars: int = 12000) -> str:
    """Extract text from a PDF using pypdf. Pure Python, no system deps.

    Returns plain text; '' if extraction fails (encrypted, scanned-only, etc.).
    Caps at max_pages and max_chars to keep prompts bounded."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.error("pypdf not installed")
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        logger.warning("pypdf failed to open: %s", e)
        return ""
    if reader.is_encrypted:
        try:
            reader.decrypt("")  # try empty password
        except Exception:
            return ""
    out: list[str] = []
    total = 0
    for i, page in enumerate(reader.pages[:max_pages]):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            out.append(f"--- page {i + 1} ---\n{t.strip()}")
            total += len(t)
            if total >= max_chars:
                break
    return "\n\n".join(out)[:max_chars]


PDF_VISION_PROMPT = (
    "이 PDF는 사용자가 보낸 파일이야 (보통 e-ticket, 영수증, 명세서). "
    "모든 페이지에서 *문자 그대로* 정보를 한국어로 추출해줘. 구조 그대로:\n"
    "- 날짜·시간·장소·이름·번호·코드는 한 글자도 빼지 말고.\n"
    "- 표는 항목별로 줄바꿈해서.\n"
    "- 예약번호/PNR/confirmation code 등은 *명확히 라벨 붙여서*.\n"
    "- 페이지 구분은 '--- page N ---'.\n"
    "- 추가 설명·요약 X. 추출만."
)


async def extract_pdf_with_vision_fallback(
    file_bytes: bytes,
    *,
    max_pages: int = 30,
    max_chars: int = 12000,
    max_vision_size: int = 5_000_000,
) -> tuple[str, str, Optional[str]]:
    """Try pypdf first; if extracted text < 40 chars (image-only PDF),
    fall back to Haiku 4.5 vision via OpenRouter (Anthropic supports inline PDF).

    Returns (text, method, failure_reason).
      method ∈ {'pypdf', 'vision_pdf', 'failed'}.
      failure_reason is None on success."""
    pypdf_text = extract_pdf_text(file_bytes, max_pages=max_pages, max_chars=max_chars)
    if len(pypdf_text.strip()) >= 40:
        return (pypdf_text, "pypdf", None)
    # vision fallback
    if not OPENROUTER_API_KEY:
        return ("", "failed", "no_openrouter_key")
    if len(file_bytes) > max_vision_size:
        return (pypdf_text, "failed", f"size_over_{max_vision_size}")
    try:
        b64 = base64.b64encode(file_bytes).decode("ascii")
        payload = {
            "model": OPENROUTER_VISION_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PDF_VISION_PROMPT},
                    {"type": "file",
                     "file": {"filename": "input.pdf",
                              "file_data": f"data:application/pdf;base64,{b64}"}},
                ],
            }],
            "max_tokens": 4000,
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://dailylife.bot",
            "X-Title": "Dailylife PDF vision fallback",
        }
        async with httpx.AsyncClient(timeout=90.0) as c:
            r = await c.post(OPENROUTER_URL, json=payload, headers=headers)
            if r.status_code >= 400:
                logger.warning("vision PDF fallback HTTP %s: %s", r.status_code, r.text[:300])
                return (pypdf_text, "failed", f"http_{r.status_code}")
            data = r.json()
        text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if len(text) < 20:
            return (pypdf_text, "failed", "vision_empty_response")
        return (text[:max_chars], "vision_pdf", None)
    except Exception as e:
        logger.exception("vision PDF fallback failed")
        return (pypdf_text, "failed", f"exception:{e}")
