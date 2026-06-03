"""Translator — 양방향 외국어 통역.

ko ↔ en/ja/zh. chat_completion(kind='translate') → Haiku tier (저렴/빠름).
v12 W8.
"""

from __future__ import annotations

import logging
from typing import Optional

import llm

logger = logging.getLogger(__name__)

LANG_NAMES = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
    "zh": "中文",
}


async def translate(text: str, target: str, source: Optional[str] = None) -> str:
    """text → target 언어로 통역. source 미지정 시 자동 감지."""
    if not text or not text.strip():
        return ""
    tgt = LANG_NAMES.get(target, target)
    src = f"from {LANG_NAMES.get(source, source)} " if source else ""
    prompt = (
        f"Translate the following text {src}to {tgt}. "
        "Output ONLY the translation, no commentary, no quotes, no explanation. "
        "Preserve tone, formality, and any names verbatim.\n\n"
        f"Text:\n{text[:4000]}"
    )
    try:
        data = await llm.chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=0, kind="translate", max_tokens=2000,
        )
        return (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("translate failed")
        return ""


async def detect_language(text: str) -> str:
    """짧은 한 단어 응답으로 언어 코드 — ko/en/ja/zh/other."""
    if not text or not text.strip():
        return "ko"
    snippet = text[:500]
    # 한글 비중으로 빠른 휴리스틱
    korean = sum(1 for c in snippet if 0xAC00 <= ord(c) <= 0xD7A3)
    if korean > len(snippet) * 0.2:
        return "ko"
    japanese = sum(1 for c in snippet if 0x3040 <= ord(c) <= 0x30FF)
    if japanese > 0:
        return "ja"
    chinese = sum(1 for c in snippet if 0x4E00 <= ord(c) <= 0x9FFF)
    if chinese > 0:
        return "zh"
    latin = sum(1 for c in snippet if c.isascii() and c.isalpha())
    if latin > len(snippet) * 0.3:
        return "en"
    return "other"
