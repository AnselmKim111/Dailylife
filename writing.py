"""Long-form writing — multi-pass 협업.

활성 writing 있으면 _system_message에 'WRITING MODE' 주입 (bot.py 처리).
lifelog로 사용자 톤 학습 + Opus tier multi-pass.
v12 W4.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import db
import lifelog
import llm

logger = logging.getLogger(__name__)

OUTLINE_PROMPT = (
    "사용자가 다음 글을 쓰려고 함:\n"
    "- 목적: {purpose}\n"
    "- 청중: {audience}\n"
    "- 길이: {length_target}\n\n"
    "사용자 *과거 글 톤* (lifelog 회상):\n{tone_samples}\n\n"
    "한국어로 3-5 섹션 개요만 제안 (각 섹션 한 줄 제목 + 1줄 요지). "
    "마크다운 ##  형식. 본문 X. 친근하지만 군더더기 없이."
)

SECTION_PROMPT = (
    "다음 개요 중 section #{section} 본문만 작성:\n\n{outline}\n\n"
    "사용자 톤 참고:\n{tone_samples}\n\n"
    "글 목적: {purpose} / 청중: {audience} / 전체 길이 약 {length_target}자\n"
    "section은 약 {section_length}자 한국어 마크다운. 인용 시 [출처](url) inline."
)


async def _gather_tone_samples(chat_id: int, k: int = 5) -> str:
    """lifelog에서 사용자 과거 글 회수 — note + reflection 위주."""
    try:
        results = await lifelog.search(chat_id, "긴 글 보고서 회고 글쓰기", k=k * 2)
    except Exception:
        return "(톤 샘플 부족 — 일반 톤)"
    if not results:
        return "(톤 샘플 부족 — 일반 톤)"
    samples = []
    for r in results[:k]:
        if r["kind"] in ("note", "reflection"):
            samples.append(f"- {r['content'][:200]}")
    if not samples:
        return "(톤 샘플 부족 — 일반 톤)"
    return "\n".join(samples)


async def build_outline(chat_id: int, writing_id: int) -> str:
    w = db.get_writing(writing_id)
    if not w:
        return ""
    tone = await _gather_tone_samples(chat_id)
    prompt = OUTLINE_PROMPT.format(
        purpose=w["purpose"],
        audience=w["audience"] or "일반",
        length_target=w["length_target"] or "유연",
        tone_samples=tone,
    )
    try:
        data = await llm.chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="writing", max_tokens=1000,
        )
        outline = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("outline build failed")
        return ""
    db.update_writing(writing_id, outline_md=outline)
    return outline


async def build_section(chat_id: int, writing_id: int, section: int) -> str:
    w = db.get_writing(writing_id)
    if not w or not w["outline_md"]:
        return ""
    tone = await _gather_tone_samples(chat_id)
    sections_count = w["outline_md"].count("##") or 4
    section_length = (w["length_target"] or 1500) // sections_count
    prompt = SECTION_PROMPT.format(
        section=section,
        outline=w["outline_md"],
        tone_samples=tone,
        purpose=w["purpose"],
        audience=w["audience"] or "일반",
        length_target=w["length_target"] or 1500,
        section_length=section_length,
    )
    try:
        data = await llm.chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=chat_id, kind="writing", max_tokens=2500,
        )
        text = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("section build failed")
        return ""
    # append to draft
    existing = w["draft_md"] or ""
    new_draft = existing + ("\n\n" if existing else "") + text
    db.update_writing(writing_id, draft_md=new_draft, current_section=section)
    return text
