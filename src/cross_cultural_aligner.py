"""Cross-cultural abstraction alignment with embedding candidates + LLM verify."""

from __future__ import annotations

from typing import Optional

from . import config, local_llm, prompts
from .embedding_utils import cosine_similarity_matrix, embed_texts
from .schemas import CanonicalEntity, CrossCulturalAlignment
from .utils import get_logger, normalize_node_label, normalize_text, stable_id, truncate

LOGGER = get_logger("cross_cultural_aligner")

_ABSTRACT_LAYERS = {"LAYER_2_SOCIAL", "LAYER_3_DEEP_VALUE"}

# Actor/group labels can legitimately occur at L2, but they are not themselves
# the shared norm/value abstractions that A4 is intended to test.
_ACTOR_GROUP_MARKERS = (
    "people", "person", "persons", "customer", "customers", "tourist",
    "tourists", "foreigner", "foreigners", "family member", "family members",
    "guest", "guests", "host", "hosts", "author", "visitor", "visitors",
    "employee", "employees", "boss", "bosses", "service provider",
    "service providers", "individual", "individuals", "themselves", "others",
)
_NORM_VALUE_CUES = (
    "norm", "etiquette", "expectation", "obligation", "hierarchy", "respect",
    "politeness", "hospitality", "reciprocity", "gratitude", "appreciation",
    "custom", "rule", "relationship", "relation", "deference", "duty",
    "courtesy", "generosity", "care",
)


def _culture_key(culture: str) -> str:
    """Normalize obvious dataset aliases before deciding cultures differ."""
    key = " ".join(normalize_text(culture).lower().split())
    for suffix in (" people", " peoples"):
        if key.endswith(suffix):
            key = key[:-len(suffix)].strip()
    return key


def _is_actor_only_l2(entity: CanonicalEntity) -> bool:
    if entity.layer.value != "LAYER_2_SOCIAL":
        return False
    label = normalize_text(entity.canonical_label).lower()
    if any(cue in label for cue in _NORM_VALUE_CUES):
        return False
    return any(marker in label for marker in _ACTOR_GROUP_MARKERS)


def _alignment_eligible(entity: CanonicalEntity) -> bool:
    if entity.layer.value not in _ABSTRACT_LAYERS or len(entity.cultures) != 1:
        return False
    if _is_actor_only_l2(entity):
        return False
    return True


def _candidate_pairs(entities: list[CanonicalEntity],
                     threshold: float) -> list[tuple[int, int, float]]:
    eligible = [
        (i, e) for i, e in enumerate(entities)
        if _alignment_eligible(e)
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
            if _culture_key(ent_i.cultures[0]) == _culture_key(ent_j.cultures[0]):
                continue
            if ent_i.layer != ent_j.layer:
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
    max_pairs: int = 200,
) -> tuple[list[CrossCulturalAlignment], list[CanonicalEntity]]:
    """Find cross-cultural abstraction pivots; optionally LLM-verify."""
    threshold = similarity_threshold or config.ALIGNMENT_SIMILARITY_THRESHOLD

    entities_out = [e.model_copy(deep=True) for e in entities]
    by_id = {e.entity_id: e for e in entities_out}

    pairs = _candidate_pairs(entities, threshold)[:max_pairs]
    if not pairs:
        LOGGER.info("Cross-cultural alignment: no candidate pairs above threshold.")
        return [], entities_out

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
        # Reuse one shared pivot for repeated pairwise matches of the same
        # abstraction. Culture-pair-specific pivot IDs created duplicate
        # "hospitality" pivots and made one entity belong to several equivalent
        # shared nodes.
        pivot_id = stable_id(
            "pivot",
            ent_a.layer.value,
            normalize_node_label(pivot_label),
            prefix="p_",
        )

        cultures = sorted(set(ent_a.cultures) | set(ent_b.cultures))
        aliases = sorted(set(ent_a.aliases) | set(ent_b.aliases) | {ent_a.canonical_label, ent_b.canonical_label})
        instance_ids = sorted(set(ent_a.source_instance_ids) | set(ent_b.source_instance_ids))
        source_instance_cultures = dict(ent_a.source_instance_cultures)
        for instance_id, culture in ent_b.source_instance_cultures.items():
            existing = source_instance_cultures.get(instance_id)
            if existing and existing != culture:
                LOGGER.warning(
                    "Conflicting cultures for source record %s: %s vs %s",
                    instance_id, existing, culture,
                )
                continue
            source_instance_cultures[instance_id] = culture

        if pivot_id not in by_id:
            by_id[pivot_id] = CanonicalEntity(
                entity_id=pivot_id,
                canonical_label=pivot_label,
                layer=ent_a.layer,
                cultures=cultures,
                aliases=aliases,
                source_instance_ids=instance_ids,
                source_instance_cultures=source_instance_cultures,
                is_shared_abstraction=True,
            )
            entities_out.append(by_id[pivot_id])
        else:
            pivot = by_id[pivot_id]
            pivot.is_shared_abstraction = True
            pivot.cultures = sorted(set(pivot.cultures) | set(cultures))
            pivot.aliases = sorted(set(pivot.aliases) | set(aliases))
            pivot.source_instance_ids = sorted(
                set(pivot.source_instance_ids) | set(instance_ids)
            )
            for instance_id, culture in source_instance_cultures.items():
                pivot.source_instance_cultures.setdefault(instance_id, culture)

        by_id[ent_a.entity_id].is_shared_abstraction = True
        by_id[ent_b.entity_id].is_shared_abstraction = True

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
