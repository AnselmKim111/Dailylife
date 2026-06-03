"""Per-task model routing for OpenRouter chat completions.

Maps each call's `kind` to an appropriate Claude tier:
- Haiku 4.5 for cheap one-shot extraction / classification
- Sonnet 4.6 for everyday chat / briefings / reflection
- Opus 4.8 for multi-hop reasoning, weekly meta-reviews, autonomous missions

Resolution order on every call:
1. Explicit `model=` argument passed by caller
2. Per-chat fact `model_override_<kind>` if set
3. Default tier from KIND_TO_TIER below
4. Global env override `OPENROUTER_MODEL` (last resort, mostly for testing)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

HAIKU = os.environ.get("OPENROUTER_MODEL_HAIKU", "anthropic/claude-haiku-4.5")
SONNET = os.environ.get("OPENROUTER_MODEL_SONNET", "anthropic/claude-sonnet-4.6")
OPUS = os.environ.get("OPENROUTER_MODEL_OPUS", "anthropic/claude-opus-4.8")

# kind → model id. Add new kinds as we introduce them.
KIND_TO_TIER: Dict[str, str] = {
    # cheap one-shots
    "classify_content":  HAIKU,
    "mood_classify":     HAIKU,
    "rule_parse":        HAIKU,
    "mail_extract":      HAIKU,
    "auto_decompose":    HAIKU,
    "bday_draft":        HAIKU,
    "bank_sms_parse":    HAIKU,
    "relation_extract":  HAIKU,
    "translate":         HAIKU,
    "context_infer":     HAIKU,
    "finance_detect":    HAIKU,
    "intent_classify":   HAIKU,
    "ambiguity_detect":  HAIKU,
    "persona_verify":    HAIKU,
    "compaction":        HAIKU,
    # everyday assistant
    "chat":              SONNET,
    "briefing":          SONNET,
    "reflection":        SONNET,
    "vision":            SONNET,
    # heavy reasoning
    "weekly_review":     OPUS,
    "persona_rebuild":   OPUS,
    "self_improve":      OPUS,
    "mission":           OPUS,
    "agent":             OPUS,
    "writing":           OPUS,
    "ask":               OPUS,
}


def pick_model(kind: str, fact_override: Optional[str] = None) -> str:
    """Return the model id to use for this kind.

    `fact_override` is the value of `model_override_<kind>` fact if any (caller
    must look it up — keeps this module dependency-free)."""
    if fact_override:
        return fact_override
    return KIND_TO_TIER.get(kind, os.environ.get("OPENROUTER_MODEL", SONNET))


def routing_table() -> Dict[str, str]:
    """Snapshot for `/models` command."""
    return dict(KIND_TO_TIER)
