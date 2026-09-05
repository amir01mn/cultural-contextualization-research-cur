"""Shared retrieval reranking: culture match + generated-triple penalty."""

from __future__ import annotations

import re
from typing import Callable, Optional, TypeVar

from . import config
from .utils import normalize_text

T = TypeVar("T")


_CULTURE_ALIASES = {
    "american": {
        "american", "americans", "usa", "us", "u s", "united states",
        "united states of america", "u.s.", "u.s.a.",
    },
    "british": {"british", "uk", "u k", "united kingdom", "english"},
    "south korea": {
        "south korea", "south korean", "south koreans", "korea", "korean", "koreans",
    },
    "north korea": {"north korea", "north korean", "north koreans"},
    "china": {"china", "chinese", "chinese people"},
    "japan": {"japan", "japanese", "japanese people", "japanese sushi chefs"},
    "iran": {"iran", "iranian", "iranians", "persian", "persians"},
    "bangladesh": {"bangladesh", "bangladeshi", "bangladeshis", "bengali", "bengalis"},
    "spain": {"spain", "spanish", "spaniard", "spaniards"},
    "netherlands": {"netherlands", "dutch", "dutch people", "holland"},
    "new zealand": {"new zealand", "new zealander", "new zealanders", "maori", "māori", "kiwi"},
    "turkey": {
        "turkey", "turkish", "turkish people", "turks", "germans of turkish heritage",
    },
    "russia": {"russia", "russian", "russians"},
    "indonesia": {"indonesia", "indonesian", "indonesians"},
    "philippines": {"philippines", "philippine", "filipino", "filipinos"},
    "pakistan": {"pakistan", "pakistani", "pakistanis"},
    "south africa": {"south africa", "south african", "south africans"},
    "zimbabwe": {"zimbabwe", "zimbabwean", "zimbabweans", "shona", "mashona"},
    "brazil": {"brazil", "brazilian", "brazilians"},
}

_ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in _CULTURE_ALIASES.items()
    for alias in aliases
}


def _clean_culture(text: str) -> str:
    text = normalize_text(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _canonical_culture(text: str) -> str:
    """Map country/demonym variants to one comparison label when possible."""
    cleaned = _clean_culture(text)
    if not cleaned:
        return ""
    if cleaned in _ALIAS_TO_CANONICAL:
        return _ALIAS_TO_CANONICAL[cleaned]

    # Some extracted culture strings contain modifiers, e.g. "Japanese sushi chefs"
    # or "Germans of Turkish heritage". Match known aliases as whole-word spans.
    for alias, canonical in sorted(_ALIAS_TO_CANONICAL.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", cleaned):
            return canonical
    return cleaned


def culture_match_score(item_culture: str, question_culture: Optional[str]) -> float:
    """Return 1.0 for strong culture match, else mismatch penalty."""
    pc = _clean_culture(item_culture)
    qc = _clean_culture(question_culture or "")
    if not qc:
        return 1.0
    if not pc:
        return config.RETRIEVAL_CULTURE_UNKNOWN_SCORE

    if pc == qc or pc in qc or qc in pc:
        return 1.0

    pc_norm = _canonical_culture(pc)
    qc_norm = _canonical_culture(qc)
    if pc_norm and qc_norm and (
        pc_norm == qc_norm or pc_norm in qc_norm or qc_norm in pc_norm
    ):
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
