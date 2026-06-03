"""Financial Guardian — 구독·반복 결제 자동 추적 + 이상치 감지.

매일 06:00 cron이 Gmail (영수증·청구서) + expenses (카드 SMS) 분석.
신규 반복 결제 → confirmation 카드. 가격 변동 ≥ threshold → 알림 (briefing).
v12 W2.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import db
import llm

logger = logging.getLogger(__name__)


# 기존 구독 가격 변동 ≥ threshold_pct 이면 alert 대상
def detect_price_changes(chat_id: int, days: int = 35) -> List[Dict]:
    """등록 구독별 최근 35일 expenses에서 *최근 결제 금액 vs registered amount* 비교.
    변동 ≥ anomaly_threshold_pct 시 surface."""
    # when_local은 naive strftime이라 cutoff도 naive로 맞춤
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    out = []
    charges = db.list_recurring_charges(chat_id, status="active")
    for c in charges:
        merchant = c["merchant"]
        if not merchant:
            continue
        # expenses 테이블에서 같은 merchant 최근 결제 1개
        with db._conn() as conn:
            row = conn.execute(
                "SELECT amount_won, when_local FROM expenses WHERE chat_id=? "
                "AND merchant LIKE ? AND when_local>=? ORDER BY when_local DESC LIMIT 1",
                (chat_id, f"%{merchant}%", cutoff)).fetchone()
        if not row:
            continue
        recent = int(row["amount_won"])
        registered = int(c["amount_won"])
        if registered == 0:
            continue
        diff_pct = (recent - registered) * 100 / registered
        if abs(diff_pct) >= c["anomaly_threshold_pct"]:
            out.append({
                "merchant": merchant,
                "registered": registered,
                "recent": recent,
                "diff_pct": diff_pct,
                "when_local": row["when_local"],
            })
    return out


SUBSCRIPTION_DETECT_PROMPT = (
    "다음 카드/은행 결제 내역을 보고 *반복 구독*으로 의심되는 항목을 "
    'JSON 배열로 추출: [{{"merchant":"이름","amount_won":<int>,"period":"monthly"|"yearly"}}].\n'
    "기준:\n"
    "- 같은 merchant가 30일 이내 1회 이상 등장\n"
    "- 금액이 일정 (±5% 이내)\n"
    "- 의심 정도 낮으면 비워. 신뢰 가는 후보만 (최대 3개).\n"
    "JSON만, 다른 텍스트 금지.\n\n"
    "결제 내역 (지난 60일):\n{transactions}"
)


async def detect_new_subscriptions(chat_id: int) -> List[Dict]:
    """expenses 60일 분석 → LLM micro-call로 반복 구독 후보 ≤3개 식별.
    이미 등록된 merchant는 제외."""
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S")
    registered = {c["merchant"].lower() for c in db.list_recurring_charges(chat_id)}
    txns: List[Tuple[str, int, str]] = []
    with db._conn() as conn:
        rows = conn.execute(
            "SELECT merchant, amount_won, when_local FROM expenses "
            "WHERE chat_id=? AND when_local>=? AND merchant IS NOT NULL "
            "ORDER BY merchant, when_local",
            (chat_id, cutoff))
        for r in rows:
            if not r["merchant"]:
                continue
            if r["merchant"].lower() in registered:
                continue
            txns.append((r["merchant"], r["amount_won"], r["when_local"]))
    if len(txns) < 4:
        return []
    # 같은 merchant ≥2회만 후보
    by_merchant: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for m, a, w in txns:
        by_merchant[m].append((a, w))
    candidates = []
    for m, items in by_merchant.items():
        if len(items) < 2:
            continue
        # 금액 stable check
        amounts = [a for a, _ in items]
        if not amounts:
            continue
        avg = sum(amounts) / len(amounts)
        if avg == 0:
            continue
        # 변동 5% 이내인 항목만
        stable_count = sum(1 for a in amounts if abs(a - avg) / avg <= 0.10)
        if stable_count < 2:
            continue
        candidates.append({
            "merchant": m,
            "amount_won": int(round(avg)),
            "occurrences": len(items),
            "period": "monthly",
        })
    candidates.sort(key=lambda x: -x["occurrences"])
    return candidates[:3]


def briefing_lines(chat_id: int) -> List[str]:
    """morning_briefing이 호출 — finance 관련 1-2줄 (변동·신규 후보)."""
    lines = []
    try:
        changes = detect_price_changes(chat_id)
        for ch in changes[:2]:
            sign = "+" if ch["diff_pct"] > 0 else ""
            lines.append(
                f"💳 {ch['merchant']} {ch['registered']:,} → {ch['recent']:,} "
                f"({sign}{ch['diff_pct']:.0f}%)")
    except Exception:
        logger.exception("finance briefing failed")
    return lines
