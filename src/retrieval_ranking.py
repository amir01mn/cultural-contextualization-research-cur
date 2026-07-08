"""Shared retrieval reranking: culture match + generated-triple penalty."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

from . import config
from .utils import normalize_text

T = TypeVar("T")


def culture_match_score(item_culture: str, question_culture: Optional[str]) -> float:
    """Return 1.0 for strong culture match, else mismatch penalty."""
    pc = normalize_text(item_culture).lower()
    qc = normalize_text(question_culture or "").lower()
    if not qc:
        return 1.0
    if not pc:
        return config.RETRIEVAL_CULTURE_UNKNOWN_SCORE
    if pc == qc or pc in qc or qc in pc:
        return 1.0
    # Common aliases
    aliases = {
        "usa": "american",
        "us": "american",
        "united states": "american",
        "uk": "british",
        "south korea": "korean",
        "north korea": "korean",
    }
    pc_norm = aliases.get(pc, pc)
    qc_norm = aliases.get(qc, qc)
    if pc_norm == qc_norm or pc_norm in qc_norm or qc_norm in pc_norm:
        return 1.0
    return config.RETRIEVAL_CULTURE_MISMATCH_PENALTY


def rerank_hits(
    hits: list[tuple[T, float]],
    *,
    top_k: int,
    question_culture: Optional[str] = None,
    get_culture: Callable[[T], str],
    get_generated: Callable[[T], bool] | None = None,
    allow_cross_culture_shared: Callable[[T], bool] | None = None,
    culture_first: bool = True,
) -> list[tuple[T, float]]:
    """Rerank retrieval hits with culture preference and generated penalty."""
    if not hits:
        return []

    get_generated = get_generated or (lambda _x: False)
    allow_cross_culture_shared = allow_cross_culture_shared or (lambda _x: False)

    scored: list[tuple[T, float, float]] = []
    for item, base in hits:
        cm = culture_match_score(get_culture(item), question_culture)
        gen = 1.0 if not get_generated(item) else config.RETRIEVAL_GENERATED_PENALTY
        if cm < 0.9 and allow_cross_culture_shared(item):
            cm = max(cm, config.RETRIEVAL_SHARED_PIVOT_CULTURE_SCORE)
        final = float(base) * cm * gen
        scored.append((item, final, cm))

    scored.sort(key=lambda x: x[1], reverse=True)

    if culture_first and question_culture:
        matched = [x for x in scored if x[2] >= 0.9]
        cross = [x for x in scored if x[2] < 0.9 and allow_cross_culture_shared(x[0])]
        rest = [x for x in scored if x[2] < 0.9 and not allow_cross_culture_shared(x[0])]
        pool = matched + cross + rest
        scored = pool

    return [(item, score) for item, score, _ in scored[:top_k]]


def triple_culture(triple) -> str:
    raw = triple.raw_fields or {}
    return normalize_text(
        raw.get("cultural_group") or raw.get("cultural group") or ""
    )


def passage_culture(doc: dict) -> str:
    return normalize_text(doc.get("cultural_group") or "")
