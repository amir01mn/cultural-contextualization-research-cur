"""Cross-cultural abstraction alignment with embedding candidates + LLM verify."""

from __future__ import annotations

from typing import Optional

from . import config, local_llm, prompts
from .embedding_utils import cosine_similarity_matrix, embed_texts
from .schemas import CanonicalEntity, CrossCulturalAlignment
from .utils import get_logger, normalize_text, stable_id, truncate

LOGGER = get_logger("cross_cultural_aligner")

_ABSTRACT_LAYERS = {"LAYER_2_SOCIAL", "LAYER_3_DEEP_VALUE"}


def _candidate_pairs(entities: list[CanonicalEntity],
                     threshold: float) -> list[tuple[int, int, float]]:
    eligible = [
        (i, e) for i, e in enumerate(entities)
        if e.layer.value in _ABSTRACT_LAYERS and len(e.cultures) == 1
    ]
    if len(eligible) < 2:
        return []

    labels = [e.canonical_label for _, e in eligible]
    embeddings = embed_texts(labels, fit=True)
    sim = cosine_similarity_matrix(embeddings)

    pairs: list[tuple[int, int, float]] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            ei, ent_i = eligible[i]
            ej, ent_j = eligible[j]
            if ent_i.cultures[0].lower() == ent_j.cultures[0].lower():
                continue
            score = float(sim[i, j])
            if score >= threshold:
                pairs.append((ei, ej, score))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _format_verify_prompt(entity_a: CanonicalEntity,
                          entity_b: CanonicalEntity) -> str:
    return prompts.ALIGNMENT_VERIFY_USER.format(
        culture_a=entity_a.cultures[0] if entity_a.cultures else "unknown",
        label_a=entity_a.canonical_label,
        layer_a=entity_a.layer.value,
        culture_b=entity_b.cultures[0] if entity_b.cultures else "unknown",
        label_b=entity_b.canonical_label,
        layer_b=entity_b.layer.value,
    )


def _parse_verify_response(text: str) -> tuple[bool, str]:
    parsed = local_llm.safe_parse_json(text)
    if not parsed or not isinstance(parsed, dict):
        low = text.strip().lower()
        verified = low.startswith("yes") or '"verified": true' in low
        return verified, normalize_text(text)[:200]
    verified = bool(parsed.get("verified", False))
    rationale = normalize_text(parsed.get("rationale") or parsed.get("reason"))
    return verified, rationale


def align_cross_cultural(
    entities: list[CanonicalEntity],
    backend=None,
    *,
    similarity_threshold: Optional[float] = None,
    verify: bool = True,
    max_pairs: Optional[int] = None,
) -> tuple[list[CrossCulturalAlignment], list[CanonicalEntity]]:
    """Find cross-cultural abstraction pivots; optionally LLM-verify."""
    threshold = similarity_threshold or config.ALIGNMENT_SIMILARITY_THRESHOLD
    max_pairs = max_pairs or config.ALIGNMENT_MAX_PAIRS
    pairs = _candidate_pairs(entities, threshold)[:max_pairs]
    if not pairs:
        LOGGER.info("Cross-cultural alignment: no candidate pairs above threshold.")
        return [], list(entities)

    entities_out = [e.model_copy(deep=True) for e in entities]
    by_id = {e.entity_id: e for e in entities_out}
    alignments: list[CrossCulturalAlignment] = []

    verify_prompts = []
    verify_meta = []
    for ei, ej, score in pairs:
        ent_a = entities[ei]
        ent_b = entities[ej]
        verify_prompts.append(_format_verify_prompt(ent_a, ent_b))
        verify_meta.append((ent_a, ent_b, score))

    verify_outputs: list[str] = []
    if verify and backend is not None:
        verify_outputs = backend.generate(
            verify_prompts, system_prompt=prompts.ALIGNMENT_VERIFY_SYSTEM)
    else:
        verify_outputs = ['{"verified": true, "rationale": "embedding-only"}'] * len(pairs)

    for (ent_a, ent_b, score), text in zip(verify_meta, verify_outputs):
        verified, rationale = _parse_verify_response(text)
        if verify and backend is not None and not verified:
            continue

        pivot_label = ent_a.canonical_label
        if len(ent_b.canonical_label) > len(pivot_label):
            pivot_label = ent_b.canonical_label
        pivot_id = stable_id(
            "pivot", pivot_label, ent_a.cultures, ent_b.cultures, prefix="p_")

        cultures = sorted(set(ent_a.cultures) | set(ent_b.cultures))
        aliases = sorted(set(ent_a.aliases) | set(ent_b.aliases) | {ent_a.canonical_label, ent_b.canonical_label})
        instance_ids = sorted(set(ent_a.source_instance_ids) | set(ent_b.source_instance_ids))

        if pivot_id not in by_id:
            by_id[pivot_id] = CanonicalEntity(
                entity_id=pivot_id,
                canonical_label=pivot_label,
                layer=ent_a.layer,
                cultures=cultures,
                aliases=aliases,
                source_instance_ids=instance_ids,
                is_shared_abstraction=True,
            )
            entities_out.append(by_id[pivot_id])
        else:
            pivot = by_id[pivot_id]
            pivot.is_shared_abstraction = True
            pivot.cultures = sorted(set(pivot.cultures) | set(cultures))
            pivot.aliases = sorted(set(pivot.aliases) | set(aliases))

        ent_a.is_shared_abstraction = True
        ent_b.is_shared_abstraction = True

        alignments.append(CrossCulturalAlignment(
            alignment_id=stable_id(ent_a.entity_id, ent_b.entity_id, prefix="a_"),
            abstraction_node_id=pivot_id,
            culture_a=ent_a.cultures[0] if ent_a.cultures else "",
            culture_b=ent_b.cultures[0] if ent_b.cultures else "",
            entity_ids=[ent_a.entity_id, ent_b.entity_id],
            entity_label_a=ent_a.canonical_label,
            entity_label_b=ent_b.canonical_label,
            similarity=score,
            verified=verified or not verify,
            verification_rationale=rationale,
        ))

    LOGGER.info("Cross-cultural alignment: %d verified alignments from %d candidates.",
                len(alignments), len(pairs))
    return alignments, entities_out
