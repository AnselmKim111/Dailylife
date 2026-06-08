"""Lifelog — embed user history + semantic search across all silos.

매일 04:30 cron이 yesterdays_new_entities() 결과를 embed해 vector_store에 적재.
사용자 질문이 모호한 회상('그 영화 뭐였더라')일 때 lifelog_search가 cosine
similarity로 chat_log/notes/events/reflections 통합 top-k 회수.

비용: text-embedding-3-small @ $0.02/1M tokens. 매일 ~10K tokens = $0.0002/일.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import db
import transcribe

logger = logging.getLogger(__name__)


async def index_entity(chat_id: int, kind: str, entity_id: int, text: str) -> bool:
    """단일 entity embed 후 vector_store에 upsert. 이미 있으면 skip."""
    if db.already_embedded(chat_id, kind, entity_id):
        return False
    text = (text or "").strip()
    if len(text) < 12:
        return False
    try:
        emb = await transcribe.embed_text(text)
    except Exception:
        logger.exception("embed_text failed for %s:%s", kind, entity_id)
        return False
    db.upsert_embedding(chat_id, kind, entity_id, text, emb)
    return True


async def index_yesterday(chat_id: int) -> Dict[str, int]:
    """매일 cron: 어제 새로 추가된 entity들 embed. silos별 카운트 반환."""
    buckets = db.yesterdays_new_entities(chat_id)
    counts = {k: 0 for k in buckets}
    for kind, items in buckets.items():
        for item in items:
            ok = await index_entity(chat_id, kind, item["id"], item["content"])
            if ok:
                counts[kind] += 1
    return counts


async def backfill_all(chat_id: int, budget_usd: float = 0.05) -> Dict[str, int]:
    """v19 W4: one-time — walk chat_log + notes + attachments embedding gaps.
    Cost cap: text-embedding-3-small ≈ $0.02 / 1M tokens. budget_usd=0.05
    allows ~2.5M tokens which covers years of chat. Stops when budget hit."""
    counts: Dict[str, int] = {"chat_log": 0, "note": 0, "attachment": 0}
    # text-embedding-3-small is ~6.5 chars/token roughly for Korean+English mix
    char_budget = int((budget_usd / 0.02) * 1_000_000 * 6.5)
    chars_used = 0

    candidates: List[Dict] = []
    with db._conn() as c:  # noqa: SLF001 — internal backfill query
        for r in c.execute(
            "SELECT id, content FROM chat_log WHERE chat_id=? AND role='user' "
            "AND length(content) >= 20 ORDER BY id DESC LIMIT 2000", (chat_id,)
        ):
            candidates.append({"kind": "chat_log", "id": r["id"], "text": r["content"]})
        for r in c.execute(
            "SELECT id, content FROM notes WHERE chat_id=? "
            "ORDER BY id DESC LIMIT 1000", (chat_id,)
        ):
            candidates.append({"kind": "note", "id": r["id"], "text": r["content"]})
        for r in c.execute(
            "SELECT id, extracted_text FROM attachments WHERE chat_id=? "
            "AND extracted_text != '' ORDER BY id DESC LIMIT 500", (chat_id,)
        ):
            candidates.append({"kind": "attachment", "id": r["id"], "text": r["extracted_text"]})

    for cand in candidates:
        text = (cand["text"] or "").strip()
        if not text or len(text) < 20:
            continue
        if chars_used + len(text) > char_budget:
            logger.info("backfill chat %s: budget hit at %d chars", chat_id, chars_used)
            break
        ok = await index_entity(chat_id, cand["kind"], cand["id"], text)
        if ok:
            counts[cand["kind"]] = counts.get(cand["kind"], 0) + 1
            chars_used += len(text)

    logger.info("backfill chat %s done: %s (chars=%d/budget=%d)",
                 chat_id, counts, chars_used, char_budget)
    return counts


async def search(chat_id: int, query: str, k: int = 8) -> List[Dict]:
    """Query를 embed 후 chat의 모든 vector와 cosine 유사도 top-k."""
    try:
        q_emb = await transcribe.embed_text(query)
    except Exception:
        logger.exception("query embed failed")
        return []
    rows = db.list_all_embeddings(chat_id)
    if not rows:
        return []
    scored = []
    for r in rows:
        sim = transcribe.cosine_similarity(q_emb, r["embedding_blob"])
        scored.append({
            "score": sim,
            "kind": r["entity_kind"],
            "entity_id": r["entity_id"],
            "content": r["content_text"],
        })
    scored.sort(key=lambda x: -x["score"])
    return scored[:k]
