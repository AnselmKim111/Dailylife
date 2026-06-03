"""Meeting negotiation — 외부인과 시간 조율 자동화.

상대방 메일 → 봇이 캘린더 빈 시간 3개 제안 답장 → 응답 받음 → 합의 시
양쪽 calendar 등록 → briefing 1줄 보고.

상태 머신: proposed → awaiting → agreed | abandoned
v12 W1.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import db
import llm

logger = logging.getLogger(__name__)


MEETING_PARSE_PROMPT = (
    "다음 이메일이 *회의/약속 요청*인지 판단. 맞으면 JSON 한 줄:\n"
    '{{"is_meeting": true, "duration_min": <int|null>, "topic": "<주제 ≤40자>", '
    '"proposed_times": ["YYYY-MM-DDTHH:MM", ...] }}\n'
    "아니면 {{\"is_meeting\": false}}.\n"
    "proposed_times는 본문에서 발신자가 *명시한* 시간만 (없으면 빈 배열).\n"
    "JSON만, 다른 텍스트 금지.\n\n"
    "보낸 사람: {sender}\n제목: {subject}\n본문:\n{body}"
)


REPLY_ACCEPT_PARSE_PROMPT = (
    "다음 이메일 답장에서 *시간 합의*를 추출. 봇이 제안한 후보 3개와 비교:\n"
    "제안: {offered_times}\n\n"
    "답장 본문:\n{body}\n\n"
    'JSON 한 줄: {{"agreed": true, "agreed_time": "YYYY-MM-DDTHH:MM"}} '
    '또는 {{"agreed": false, "counterproposal": "YYYY-MM-DDTHH:MM"|null, '
    '"reason": "<짧은 이유>"}}\n'
    "JSON만, 다른 텍스트 금지."
)


async def parse_meeting_request(sender: str, subject: str, body: str) -> Dict:
    """메일이 회의 요청인지 분류 + topic·시간 추출. Haiku tier (cheap)."""
    prompt = MEETING_PARSE_PROMPT.format(
        sender=sender, subject=subject, body=body[:2000])
    try:
        data = await llm.chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=0, kind="rule_parse", max_tokens=300,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("parse_meeting_request failed")
        return {"is_meeting": False}
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return {"is_meeting": False}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"is_meeting": False}


def find_free_slots(
    chat_id: int, duration_min: int = 60,
    days_ahead: int = 10, max_slots: int = 3,
) -> List[str]:
    """캘린더에서 평일 9-18시 중 빈 시간 후보 ≤3개 반환 (ISO 형식)."""
    now = datetime.now(timezone.utc).astimezone()
    start = now + timedelta(hours=1)  # 1시간 이후부터
    end = now + timedelta(days=days_ahead)
    try:
        events = db.list_events(chat_id, start, end)
    except Exception:
        events = []
    busy = []
    for e in events:
        try:
            t = datetime.fromisoformat(e["when_utc"])
            busy.append((t, t + timedelta(minutes=duration_min)))
        except Exception:
            continue
    slots = []
    # candidate hours: 10, 14, 16 KST
    for day_offset in range(1, days_ahead + 1):
        if len(slots) >= max_slots:
            break
        date = (now + timedelta(days=day_offset)).date()
        weekday = date.weekday()
        if weekday >= 5:  # 주말 skip
            continue
        for hour in (10, 14, 16):
            if len(slots) >= max_slots:
                break
            from zoneinfo import ZoneInfo
            kst = ZoneInfo("Asia/Seoul")
            candidate = datetime(date.year, date.month, date.day, hour, 0, tzinfo=kst)
            candidate_utc = candidate.astimezone(timezone.utc)
            cand_end = candidate_utc + timedelta(minutes=duration_min)
            # 충돌 체크
            conflict = any(
                bs < cand_end and be > candidate_utc for bs, be in busy)
            if conflict:
                continue
            slots.append(candidate.strftime("%Y-%m-%dT%H:%M"))
    return slots


def format_offer_email(
    sender_name: str, topic: str, slots: List[str],
) -> Tuple[str, str]:
    """답장 메일 subject + body. 한국어. 비서 톤."""
    subject = f"Re: {topic}" if topic else "회의 시간 조율"
    if not slots:
        body = (
            f"안녕하세요{', ' + sender_name if sender_name else ''}.\n\n"
            "회의 일정 확인했습니다. 다음 2주 일정이 빠듯해 아래 옵션 외에 "
            "구체적으로 가능한 시간 알려주시면 맞춰보겠습니다.\n\n감사합니다."
        )
    else:
        slot_lines = "\n".join(
            f"  · {s.replace('T', ' ')} (KST)" for s in slots)
        body = (
            f"안녕하세요{', ' + sender_name if sender_name else ''}.\n\n"
            f"회의 일정 확인했습니다. 아래 시간대 중 가능한 곳 있으실까요?\n\n"
            f"{slot_lines}\n\n"
            "선택 알려주시면 캘린더 등록 후 안내드리겠습니다.\n\n감사합니다."
        )
    return subject, body


async def parse_reply(body: str, offered_times: List[str]) -> Dict:
    """상대 답장에서 동의 시간 추출."""
    prompt = REPLY_ACCEPT_PARSE_PROMPT.format(
        offered_times=offered_times, body=body[:2000])
    try:
        data = await llm.chat_completion(
            [{"role": "user", "content": prompt}],
            tools=None, chat_id=0, kind="rule_parse", max_tokens=200,
        )
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        logger.exception("parse_reply failed")
        return {"agreed": False}
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return {"agreed": False}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"agreed": False}


def _match_auto_rule(chat_id: int, sender_email: str) -> bool:
    """auto_rules에서 meeting_auto_negotiate 룰 매칭. counterparty_pattern 일치."""
    rules = db.list_auto_rules(chat_id)
    for r in rules:
        if r["rule_kind"] != "meeting_auto_negotiate":
            continue
        try:
            cond = json.loads(r["condition_json"])
        except Exception:
            continue
        pattern = (cond.get("sender_pattern") or "").lower()
        from_people = cond.get("from_people_names") or []
        if pattern and pattern in sender_email.lower():
            return True
        if from_people:
            people = db.list_people(chat_id)
            for p in people:
                if p["name"] in from_people:
                    notes = (p["notes"] or "").lower()
                    if sender_email.lower() in notes:
                        return True
    return False
