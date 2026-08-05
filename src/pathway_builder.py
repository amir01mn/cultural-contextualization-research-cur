"""Construct flexible cultural evidence pathways from a resolved cultural graph.

A pathway is not forced to be L3 -> L2 -> L1. The ideal FULL pathway still
contains L3, L2, and L1 evidence, but partial paths such as L2 -> L1,
L2 -> L3, or L1 -> L3 are also valid when that is what the extracted graph
contains. This keeps pathway construction faithful to the data instead of
forcing all evidence into one direction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import networkx as nx

from . import config
from .schemas import (
    CanonicalEntity,
    CrossCulturalAlignment,
    CulturalPathway,
    LayerLabel,
    PathwayCompleteness,
    PathwayNodeRef,
)
from .utils import get_logger, normalize_text, stable_id

LOGGER = get_logger("pathway_builder")

_LAYER_RANK = {
    LayerLabel.LAYER_3_DEEP_VALUE: 3,
    LayerLabel.LAYER_2_SOCIAL: 2,
    LayerLabel.LAYER_1_SURFACE: 1,
    LayerLabel.UNKNOWN: 0,
}

_WEAK_PREDICATE_PHRASES = {
    "has cultural topic",
    "has topic",
    "topic",
    "is topic",
    "belongs to topic",
    "has category",
    "category",
    "has cultural group",
    "cultural group",
}

# Exact topic/category labels from the controlled-subset notes. These are useful
# for corpus selection, but they are too generic to become pathway nodes.
_WEAK_TOPIC_LABELS = {
    "community and identity",
    "consumer behavior",
    "cultural and environmental appreciation",
    "cultural exchange",
    "cultural traditions and festivals",
    "drinking and alcohol",
    "family dynamics",
    "family traditions and heritage",
    "finance and economy",
    "food and dining",
    "health and hygiene",
    "household and daily life",
    "lifestyles",
    "miscellaneous",
    "social interactions",
    "social norms and etiquette",
    "work-life balance",
}

# These labels can be valid entities in a graph, but they are too broad to be
# useful as pathway nodes. Keep specific social roles such as guests, hosts,
# restaurant staff, birthday person, customers, and servers.
_WEAK_NODE_LABELS = {
    "american",
    "americans",
    "people",
    "person",
    "individual",
    "individuals",
    "someone",
    "others",
    "other people",
    "community",
    "society",
}


def _layer_rank(layer: LayerLabel | str) -> int:
    if isinstance(layer, LayerLabel):
        return _LAYER_RANK.get(layer, 0)
    return _LAYER_RANK.get(LayerLabel.coerce(layer), 0)


def _completeness_from_layers(layers: set[LayerLabel]) -> PathwayCompleteness:
    has1 = LayerLabel.LAYER_1_SURFACE in layers
    has2 = LayerLabel.LAYER_2_SOCIAL in layers
    has3 = LayerLabel.LAYER_3_DEEP_VALUE in layers
    if has1 and has2 and has3:
        return PathwayCompleteness.FULL
    if (has1 and has2) or (has2 and has3):
        return PathwayCompleteness.SOCIAL
    if has1:
        return PathwayCompleteness.SURFACE
    return PathwayCompleteness.NONE


def _culture_set(node_data: dict) -> set[str]:
    raw = node_data.get("cultures", set())
    if isinstance(raw, str):
        return {c.strip() for c in raw.split(";") if c.strip()}
    return set(raw or [])


def _node_layer(node_data: dict) -> LayerLabel:
    return LayerLabel.coerce(node_data.get("dominant_layer") or node_data.get("layer"))


def _node_label(graph: nx.MultiDiGraph, node_id: str) -> str:
    return normalize_text(graph.nodes[node_id].get("label", node_id))


def _is_weak_pathway_node(graph: nx.MultiDiGraph, node_id: str, culture: str) -> bool:
    """Return True for nodes that should not be used as pathway evidence."""
    label = _node_label(graph, node_id).lower()
    culture_label = normalize_text(culture).lower()

    if not label:
        return True

    if label == culture_label:
        return True

    if label in _WEAK_NODE_LABELS:
        return True

    if label in _WEAK_TOPIC_LABELS:
        return True

    return False


def _is_weak_metadata_edge(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    data: dict,
    culture: str,
) -> bool:
    """Return True for metadata/topic/generic edges that should not become pathways."""
    predicate = normalize_text(data.get("predicate", "")).lower()
    source_label = _node_label(graph, source).lower()
    target_label = _node_label(graph, target).lower()
    culture_label = normalize_text(culture).lower()

    if predicate in _WEAK_PREDICATE_PHRASES:
        return True

    if target_label in _WEAK_TOPIC_LABELS and (
        "topic" in predicate or "category" in predicate or predicate.startswith("has ")
    ):
        return True

    if source_label == culture_label and (
        "topic" in predicate or "category" in predicate or "cultural group" in predicate
    ):
        return True

    # Avoid pathways whose explanation node is only a broad culture/group/actor.
    # These nodes can remain in the graph, but they are not useful retrieval paths.
    if _is_weak_pathway_node(graph, source, culture):
        return True
    if _is_weak_pathway_node(graph, target, culture):
        return True

    return False


def _pathway_refs(graph: nx.MultiDiGraph, node_ids: list[str]) -> list[PathwayNodeRef]:
    return [
        PathwayNodeRef(
            entity_id=n,
            label=graph.nodes[n].get("label", n),
            layer=_node_layer(graph.nodes[n]),
        )
        for n in node_ids
    ]


def _add_pathway(
    *,
    graph: nx.MultiDiGraph,
    pathways: list[CulturalPathway],
    seen_hashes: set[str],
    pivot_ids: set[str],
    culture: str,
    node_ids: list[str],
    triple_ids: set[str],
) -> bool:
    """Add a pathway if it is non-empty, meaningful, and not duplicated."""
    if len(node_ids) < 2:
        return False

    if any(_is_weak_pathway_node(graph, n, culture) for n in node_ids):
        return False

    layers = {_node_layer(graph.nodes[n]) for n in node_ids}
    completeness = _completeness_from_layers(layers)
    if completeness == PathwayCompleteness.NONE:
        return False

    node_key = stable_id(culture, *node_ids)
    if node_key in seen_hashes:
        return False
    seen_hashes.add(node_key)

    shared = any(n in pivot_ids for n in node_ids)
    pivot_id = next((n for n in node_ids if n in pivot_ids), None)
    pathways.append(
        CulturalPathway(
            pathway_id=stable_id(culture, node_key, prefix="pw_"),
            culture=culture,
            nodes=_pathway_refs(graph, node_ids),
            triple_ids=sorted(str(tid) for tid in triple_ids),
            completeness=completeness,
            has_shared_pivot=shared,
            shared_pivot_id=pivot_id,
        )
    )
    return True


def _edge_records_for_culture(
    graph: nx.MultiDiGraph,
    culture: str,
) -> dict[str, list[tuple[str, str, str, dict]]]:
    """Group useful edges by source record for one culture."""
    by_record: dict[str, list[tuple[str, str, str, dict]]] = defaultdict(list)

    for u, v, key, data in graph.edges(keys=True, data=True):
        if culture not in _culture_set(graph.nodes[u]):
            continue
        if culture not in _culture_set(graph.nodes[v]):
            continue
        if _is_weak_metadata_edge(graph, u, v, data, culture):
            continue
        record_id = str(data.get("instance_id") or "unknown_record")
        by_record[record_id].append((u, v, str(key), data))

    return by_record


def _connected_components_for_record(
    edges: list[tuple[str, str, str, dict]],
) -> list[tuple[set[str], set[str]]]:
    """Return weakly connected node components and their triple IDs."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    edge_ids_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    nodes: set[str] = set()

    for u, v, triple_id, _data in edges:
        nodes.update([u, v])
        adjacency[u].add(v)
        adjacency[v].add(u)
        edge_ids_by_pair[tuple(sorted((u, v)))].add(triple_id)

    seen: set[str] = set()
    components: list[tuple[set[str], set[str]]] = []
    for start in sorted(nodes):
        if start in seen:
            continue
        stack = [start]
        comp_nodes: set[str] = set()
        comp_edges: set[str] = set()
        seen.add(start)
        while stack:
            node = stack.pop()
            comp_nodes.add(node)
            for nbr in adjacency[node]:
                comp_edges.update(edge_ids_by_pair[tuple(sorted((node, nbr)))])
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append((comp_nodes, comp_edges))
    return components


def _representative_by_layer(
    graph: nx.MultiDiGraph,
    nodes: set[str],
    layer: LayerLabel,
    culture: str,
) -> Optional[str]:
    candidates = [
        n
        for n in nodes
        if _node_layer(graph.nodes[n]) == layer
        and not _is_weak_pathway_node(graph, n, culture)
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda n: (-int(graph.nodes[n].get("frequency", 1)), _node_label(graph, n).lower()),
    )[0]


def build_pathways(
    graph: nx.MultiDiGraph,
    entities: list[CanonicalEntity],
    alignments: list[CrossCulturalAlignment],
    *,
    max_per_culture: Optional[int] = None,
) -> list[CulturalPathway]:
    """Enumerate flexible, layer-aware cultural evidence pathways.

    The builder keeps meaningful partial paths instead of requiring every path
    to start at L3 and move downward. FULL pathways are built when L1, L2, and
    L3 evidence occurs in the same connected record component. Partial pathways
    such as L2->L1, L2->L3, and L1->L3 are retained when they are the strongest
    evidence available.
    """
    max_per_culture = max_per_culture or config.MAX_PATHWAYS_PER_CULTURE
    pivot_ids = {a.abstraction_node_id for a in alignments}

    cultures: set[str] = set()
    for _nid, data in graph.nodes(data=True):
        cultures.update(_culture_set(data))
    cultures = {c for c in cultures if c}

    pathways: list[CulturalPathway] = []
    seen_hashes: set[str] = set()

    for culture in sorted(cultures):
        count = 0
        by_record = _edge_records_for_culture(graph, culture)

        # First, create the strongest available record-level FULL pathways.
        # This is layer-aware and not dependent on the original edge direction.
        for _record_id, edges in sorted(by_record.items()):
            if count >= max_per_culture:
                break
            for comp_nodes, comp_edge_ids in _connected_components_for_record(edges):
                if count >= max_per_culture:
                    break
                l3 = _representative_by_layer(
                    graph, comp_nodes, LayerLabel.LAYER_3_DEEP_VALUE, culture
                )
                l2 = _representative_by_layer(
                    graph, comp_nodes, LayerLabel.LAYER_2_SOCIAL, culture
                )
                l1 = _representative_by_layer(
                    graph, comp_nodes, LayerLabel.LAYER_1_SURFACE, culture
                )
                if l1 and l2 and l3:
                    if _add_pathway(
                        graph=graph,
                        pathways=pathways,
                        seen_hashes=seen_hashes,
                        pivot_ids=pivot_ids,
                        culture=culture,
                        node_ids=[l3, l2, l1],
                        triple_ids=comp_edge_ids,
                    ):
                        count += 1

        # Then keep meaningful directed partial pathways exactly as extracted.
        # This preserves L2->L1, L2->L3, L1->L3, etc. instead of forcing them
        # into L3->L2->L1 form.
        for _record_id, edges in sorted(by_record.items()):
            if count >= max_per_culture:
                break
            for u, v, triple_id, _data in edges:
                if count >= max_per_culture:
                    break
                if _node_layer(graph.nodes[u]) == LayerLabel.UNKNOWN:
                    continue
                if _node_layer(graph.nodes[v]) == LayerLabel.UNKNOWN:
                    continue
                if _add_pathway(
                    graph=graph,
                    pathways=pathways,
                    seen_hashes=seen_hashes,
                    pivot_ids=pivot_ids,
                    culture=culture,
                    node_ids=[u, v],
                    triple_ids={triple_id},
                ):
                    count += 1

    LOGGER.info("Built %d pathways across %d cultures.", len(pathways), len(cultures))
    return pathways


def pathways_for_culture(pathways: list[CulturalPathway], culture: str) -> list[CulturalPathway]:
    c = normalize_text(culture).lower()
    return [p for p in pathways if normalize_text(p.culture).lower() == c]
