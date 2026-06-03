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
