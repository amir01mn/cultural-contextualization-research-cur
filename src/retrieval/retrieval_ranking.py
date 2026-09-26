"""Shared retrieval reranking: culture match + generated-triple penalty."""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

from ..core import config
from ..core.culture_normalizer import canonical_culture
from ..core.utils import normalize_text

T = TypeVar("T")


def culture_match_score(item_culture: str, question_culture: Optional[str]) -> float:
    """Return 1.0 for strong culture match, else mismatch penalty.

    Matching is done on canonical culture keys, so country/demonym pairs
    ("Japan" vs "Japanese") and plurals ("American" vs "Americans") match. The
    previous six-entry alias table plus substring matching missed nearly every
    benchmark culture and also produced false positives via substring overlap.
    """
    qc = canonical_culture(question_culture)
    if not qc:
        return 1.0
    pc = canonical_culture(item_culture)
    if not pc:
        return config.RETRIEVAL_CULTURE_UNKNOWN_SCORE
    if pc == qc:
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
    min_score: Optional[float] = None,
    max_per_culture: Optional[int] = None,
) -> list[tuple[T, float]]:
    """Rerank retrieval hits with culture preference and generated penalty.

    Two precision controls beyond ranking:

    ``min_score`` drops hits below an absolute relevance floor. Callers can then
    return nothing at all, so the answering system falls back to closed-book
    rather than being handed irrelevant evidence. Measured on the broken run,
    injecting evidence unconditionally corrupted ~10-15% of the answers the base
    model would otherwise have got right.

    ``max_per_culture`` limits how many results come from one culture, which
    stops a single generic entry dominating (one pathway was returned for
    239 of 500 questions).
    """
    if not hits:
        return []

    if min_score is None:
        min_score = config.RETRIEVAL_MIN_SCORE
    if max_per_culture is None:
        max_per_culture = config.RETRIEVAL_MAX_PER_CULTURE

    get_generated = get_generated or (lambda _x: False)
    allow_cross_culture_shared = allow_cross_culture_shared or (lambda _x: False)

    scored: list[tuple[T, float, float]] = []
    for item, base in hits:
        if float(base) < min_score:
            continue
        cm = culture_match_score(get_culture(item), question_culture)
        gen = 1.0 if not get_generated(item) else config.RETRIEVAL_GENERATED_PENALTY
        if cm < 0.9 and allow_cross_culture_shared(item):
            cm = max(cm, config.RETRIEVAL_SHARED_PIVOT_CULTURE_SCORE)
        final = float(base) * cm * gen
        scored.append((item, final, cm))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)

    if culture_first and question_culture:
        matched = [x for x in scored if x[2] >= 0.9]
        cross = [x for x in scored if x[2] < 0.9 and allow_cross_culture_shared(x[0])]
        rest = [x for x in scored if x[2] < 0.9 and not allow_cross_culture_shared(x[0])]
        pool = matched + cross + rest
        scored = pool

    selected: list[tuple[T, float]] = []
    per_culture: dict[str, int] = {}
    overflow: list[tuple[T, float]] = []
    for item, score, _cm in scored:
        key = canonical_culture(get_culture(item))
        if max_per_culture > 0 and per_culture.get(key, 0) >= max_per_culture:
            overflow.append((item, score))
            continue
        per_culture[key] = per_culture.get(key, 0) + 1
        selected.append((item, score))
        if len(selected) >= top_k:
            return selected

    # Backfill from the capped-out results rather than returning fewer than
    # requested when the corpus genuinely has little cultural diversity.
    for item, score in overflow:
        if len(selected) >= top_k:
            break
        selected.append((item, score))
    return selected[:top_k]


def triple_culture(triple) -> str:
    raw = triple.raw_fields or {}
    return normalize_text(
        raw.get("cultural_group") or raw.get("cultural group") or ""
    )


def passage_culture(doc: dict) -> str:
    return normalize_text(doc.get("cultural_group") or "")
