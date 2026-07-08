"""Entity resolution: merge lexical variants into canonical nodes."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from . import config
from .embedding_utils import cosine_similarity_matrix, embed_texts
from .schemas import CanonicalEntity, CulturalTriple, LayerLabel
from .utils import get_logger, normalize_node_label, normalize_text, stable_id

LOGGER = get_logger("entity_resolver")

_LAYER_GROUPS = {
    LayerLabel.LAYER_1_SURFACE,
    LayerLabel.LAYER_2_SOCIAL,
    LayerLabel.LAYER_3_DEEP_VALUE,
}


def _dominant_endpoint_layer(t: CulturalTriple, endpoint: str) -> LayerLabel:
    return t.subject_layer if endpoint == "subject" else t.object_layer


def _record_culture_map(records: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in records:
        iid = normalize_text(rec.get("instance_id"))
        cg = normalize_text(rec.get("cultural_group"))
        if iid and cg:
            out[iid] = cg
    return out


def _collect_candidates(
    triples: list[CulturalTriple],
    culture_by_instance: dict[str, str],
) -> dict[tuple[str, str], dict]:
    """Key: (normalized_label, layer_value) -> aggregate metadata."""
    candidates: dict[tuple[str, str], dict] = {}
    for t in triples:
        for endpoint, label in (("subject", t.subject), ("object", t.object)):
            if not normalize_text(label):
                continue
            layer = _dominant_endpoint_layer(t, endpoint)
            if layer not in _LAYER_GROUPS:
                continue
            norm = normalize_node_label(label)
            key = (norm, layer.value)
            bucket = candidates.setdefault(key, {
                "label": normalize_text(label),
                "norm": norm,
                "layer": layer,
                "aliases": set(),
                "cultures": set(),
                "instance_ids": set(),
            })
            bucket["aliases"].add(normalize_text(label))
            iid = normalize_text(t.instance_id)
            if iid:
                bucket["instance_ids"].add(iid)
                cg = culture_by_instance.get(iid)
                if cg:
                    bucket["cultures"].add(cg)
    return candidates


def _cluster_keys(keys: list[tuple[str, str]], labels: list[str],
                   threshold: float) -> list[list[int]]:
    if len(keys) <= 1:
        return [[0]] if keys else []

    embeddings = embed_texts(labels, fit=True)
    sim = cosine_similarity_matrix(embeddings)
    n = len(keys)
    visited = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        cluster = []
        visited[i] = True
        while stack:
            cur = stack.pop()
            cluster.append(cur)
            for j in range(n):
                if not visited[j] and sim[cur, j] >= threshold:
                    visited[j] = True
                    stack.append(j)
        clusters.append(cluster)
    return clusters


def resolve_entities(
    triples: list[CulturalTriple],
    records: list[dict],
    *,
    similarity_threshold: Optional[float] = None,
) -> tuple[list[CanonicalEntity], dict[tuple[str, str], str]]:
    """Cluster endpoint labels into canonical entities.

    Returns entities and a mapping (normalized_label, layer_value) -> entity_id.
    """
    threshold = similarity_threshold or config.ER_SIMILARITY_THRESHOLD
    culture_by_instance = _record_culture_map(records)
    candidates = _collect_candidates(triples, culture_by_instance)

    entities: list[CanonicalEntity] = []
    label_layer_to_entity: dict[tuple[str, str], str] = {}

    by_layer: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in candidates:
        by_layer[key[1]].append(key)

    for layer_value, keys in by_layer.items():
        labels = [candidates[k]["label"] for k in keys]
        clusters = _cluster_keys(keys, labels, threshold)
        for cluster in clusters:
            cluster_keys = [keys[i] for i in cluster]
            canonical_label = max(
                (candidates[k]["label"] for k in cluster_keys),
                key=len,
            )
            aliases: set[str] = set()
            cultures: set[str] = set()
            instance_ids: set[str] = set()
            for k in cluster_keys:
                meta = candidates[k]
                aliases.update(meta["aliases"])
                cultures.update(meta["cultures"])
                instance_ids.update(meta["instance_ids"])
            entity_id = stable_id(layer_value, canonical_label, sorted(aliases), prefix="e_")
            entity = CanonicalEntity(
                entity_id=entity_id,
                canonical_label=canonical_label,
                layer=LayerLabel.coerce(layer_value),
                cultures=sorted(cultures),
                aliases=sorted(aliases),
                source_instance_ids=sorted(instance_ids),
            )
            entities.append(entity)
            for k in cluster_keys:
                label_layer_to_entity[k] = entity_id

    LOGGER.info("Entity resolution: %d candidates -> %d canonical entities.",
                len(candidates), len(entities))
    return entities, label_layer_to_entity


def apply_entity_map(
    triples: list[CulturalTriple],
    label_layer_to_entity: dict[tuple[str, str], str],
    entities_by_id: dict[str, CanonicalEntity],
) -> list[CulturalTriple]:
    """Return triples with subject/object rewritten to canonical labels."""
    resolved: list[CulturalTriple] = []
    for t in triples:
        t2 = t.model_copy(deep=True)
        for endpoint, attr in (("subject", "subject_layer"), ("object", "object_layer")):
            label = getattr(t2, endpoint)
            layer = getattr(t2, attr)
            key = (normalize_node_label(label), layer.value)
            eid = label_layer_to_entity.get(key)
            if eid and eid in entities_by_id:
                setattr(t2, endpoint, entities_by_id[eid].canonical_label)
        resolved.append(t2)
    return resolved
