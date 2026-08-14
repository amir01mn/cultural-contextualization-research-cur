"""Construct abstraction→refinement pathways from a resolved cultural graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import networkx as nx

from . import config
from .culture_normalizer import canonical_culture, canonical_cultures
from .schemas import (
    CanonicalEntity,
    CrossCulturalAlignment,
    CulturalPathway,
    CulturalTriple,
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


def _layer_rank(layer: LayerLabel | str) -> int:
    if isinstance(layer, LayerLabel):
        return _LAYER_RANK.get(layer, 0)
    return _LAYER_RANK.get(LayerLabel.coerce(layer), 0)


def _completeness_from_layers(layers: set[LayerLabel]) -> PathwayCompleteness:
    """Classify a pathway by which abstraction layers it actually spans.

    FULL requires a genuine value -> norm -> practice chain (all three layers).
    An L3->L1 jump that skips the social layer is NOT full: the missing L2
    mediation is precisely what the research question is about.
    """
    has1 = LayerLabel.LAYER_1_SURFACE in layers
    has2 = LayerLabel.LAYER_2_SOCIAL in layers
    has3 = LayerLabel.LAYER_3_DEEP_VALUE in layers
    if has1 and has2 and has3:
        return PathwayCompleteness.FULL
    if has2 and (has1 or has3):
        return PathwayCompleteness.SOCIAL
    if has1:
        return PathwayCompleteness.SURFACE
    return PathwayCompleteness.NONE


def _culture_set(node_data: dict) -> set[str]:
    raw = node_data.get("cultures", set())
    if isinstance(raw, str):
        values = [c.strip() for c in raw.split(";") if c.strip()]
    else:
        values = list(raw or [])
    return canonical_cultures(values)


def _node_layer(node_data: dict) -> LayerLabel:
    return LayerLabel.coerce(node_data.get("dominant_layer") or node_data.get("layer"))


def _edge_triple_ids(graph: nx.MultiDiGraph, a: str, b: str) -> set[str]:
    """Triple ids on any edge between a and b, in either direction."""
    ids: set[str] = set()
    for u, v in ((a, b), (b, a)):
        if graph.has_edge(u, v):
            for key in graph[u][v]:
                ids.add(str(key))
    return ids


def _edge_predicate(graph: nx.MultiDiGraph, a: str, b: str) -> str:
    """Best-confidence relation label between a and b, in either direction."""
    best_pred, best_conf = "", -1.0
    for u, v in ((a, b), (b, a)):
        if not graph.has_edge(u, v):
            continue
        for _key, data in graph[u][v].items():
            conf = float(data.get("confidence", 0.5) or 0.5)
            if conf > best_conf:
                best_conf = conf
                best_pred = normalize_text(data.get("predicate")) or ""
    return best_pred


def _edge_generated(graph: nx.MultiDiGraph, a: str, b: str) -> bool:
    """True when every edge between a and b is inferred rather than grounded."""
    flags = []
    for u, v in ((a, b), (b, a)):
        if not graph.has_edge(u, v):
            continue
        for _key, data in graph[u][v].items():
            flags.append(bool(data.get("generated")))
    return bool(flags) and all(flags)


def _edge_quality(graph: nx.MultiDiGraph, a: str, b: str) -> float:
    """Best confidence across edges between a and b; generated edges penalized."""
    best = 0.0
    for u, v in ((a, b), (b, a)):
        if not graph.has_edge(u, v):
            continue
        for _key, data in graph[u][v].items():
            conf = float(data.get("confidence", 0.5) or 0.5)
            if data.get("generated"):
                conf *= 0.5
            best = max(best, conf)
    return best


def _neighbors_by_layer(graph: nx.MultiDiGraph, node: str,
                        culture: str) -> dict[LayerLabel, list[str]]:
    """Group a node's neighbours (either direction) by layer, same culture only."""
    out: dict[LayerLabel, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for nbr in list(graph.successors(node)) + list(graph.predecessors(node)):
        if nbr == node or nbr in seen:
            continue
        seen.add(nbr)
        data = graph.nodes[nbr]
        if culture not in _culture_set(data):
            continue
        out[_node_layer(data)].append(nbr)
    return out


def build_pathways(
    graph: nx.MultiDiGraph,
    entities: list[CanonicalEntity],
    alignments: list[CrossCulturalAlignment],
    *,
    max_per_culture: Optional[int] = None,
) -> list[CulturalPathway]:
    """Assemble abstraction->refinement pathways, pivoting on L2 hub nodes.

    The corpus stores cultural structure as ``L2 -> L1`` (norm to practice) and
    ``L2 -> L3`` (norm to value), because the social norm is the natural subject
    of both statements. A traversal that starts at an L3 node and follows
    outgoing edges downward therefore finds almost nothing: the measured corpus
    holds 57,966 L2->L1 and 22,521 L2->L3 edges but only 19 L3->L2 edges.

    So we pivot on the L2 node instead. An L2 hub with at least one L3 neighbour
    and one L1 neighbour *is* a complete value -> norm -> practice chain; it is
    simply stored with the norm as subject. Edges are treated as undirected for
    discovery and the resulting nodes are ordered L3 -> L2 -> L1 for output.
    """
    max_per_culture = max_per_culture or config.MAX_PATHWAYS_PER_CULTURE
    pivot_ids = {a.abstraction_node_id for a in alignments}

    # Index nodes by culture once, rather than rescanning every node per culture.
    nodes_by_culture: dict[str, list[str]] = defaultdict(list)
    for nid, data in graph.nodes(data=True):
        for culture in _culture_set(data):
            nodes_by_culture[culture].append(nid)

    pathways: list[CulturalPathway] = []
    pathway_scores: list[float] = []
    seen_hashes: set[str] = set()

    for culture in sorted(nodes_by_culture):
        candidates: list[tuple[float, list[str], set[str]]] = []

        for node in nodes_by_culture[culture]:
            if _node_layer(graph.nodes[node]) != LayerLabel.LAYER_2_SOCIAL:
                continue
            nbrs = _neighbors_by_layer(graph, node, culture)
            l3_nodes = nbrs.get(LayerLabel.LAYER_3_DEEP_VALUE, [])
            l1_nodes = nbrs.get(LayerLabel.LAYER_1_SURFACE, [])
            if not l3_nodes and not l1_nodes:
                continue

            # Rank each side by edge quality and keep the strongest few, so one
            # densely connected hub cannot flood the culture's budget.
            l3_ranked = sorted(l3_nodes, key=lambda n: -_edge_quality(graph, node, n))[:4]
            l1_ranked = sorted(l1_nodes, key=lambda n: -_edge_quality(graph, node, n))[:4]

            if l3_ranked and l1_ranked:
                pairs = [(v3, v1) for v3 in l3_ranked for v1 in l1_ranked]
            elif l3_ranked:
                pairs = [(v3, None) for v3 in l3_ranked]
            else:
                pairs = [(None, v1) for v1 in l1_ranked]

            for v3, v1 in pairs:
                path = [n for n in (v3, node, v1) if n]
                triple_ids = set()
                score = 0.0
                if v3:
                    triple_ids |= _edge_triple_ids(graph, node, v3)
                    score += _edge_quality(graph, node, v3)
                if v1:
                    triple_ids |= _edge_triple_ids(graph, node, v1)
                    score += _edge_quality(graph, node, v1)
                if not triple_ids:
                    continue  # require provenance
                if any(n in pivot_ids for n in path):
                    score += 0.5  # prefer cross-cultural pivots
                candidates.append((score, path, triple_ids))

        # Strongest pathways first, then dedupe and cap per culture.
        candidates.sort(key=lambda x: -x[0])
        count = 0
        for cand_score, path, triple_ids in candidates:
            if count >= max_per_culture:
                break
            node_key = stable_id(culture, *path)
            if node_key in seen_hashes:
                continue
            seen_hashes.add(node_key)
            layers = {_node_layer(graph.nodes[n]) for n in path}
            completeness = _completeness_from_layers(layers)
            if completeness == PathwayCompleteness.NONE:
                continue
            refs = [
                PathwayNodeRef(
                    entity_id=n,
                    label=graph.nodes[n].get("label", n),
                    layer=_node_layer(graph.nodes[n]),
                )
                for n in path
            ]
            hop_predicates = [
                _edge_predicate(graph, path[i], path[i + 1])
                for i in range(len(path) - 1)
            ]
            all_generated = all(
                _edge_generated(graph, path[i], path[i + 1])
                for i in range(len(path) - 1)
            )
            pivot_id = next((n for n in path if n in pivot_ids), None)
            pathways.append(CulturalPathway(
                pathway_id=stable_id(culture, node_key, prefix="pw_"),
                culture=culture,
                nodes=refs,
                triple_ids=sorted(triple_ids),
                completeness=completeness,
                has_shared_pivot=pivot_id is not None,
                shared_pivot_id=pivot_id,
                predicates=hop_predicates,
                generated=all_generated,
            ))
            # Prefer complete chains when the global cap has to prune.
            pathway_scores.append(
                cand_score + (1.0 if completeness == PathwayCompleteness.FULL else 0.0)
            )
            count += 1

    max_total = config.MAX_TOTAL_PATHWAYS
    if max_total > 0 and len(pathways) > max_total:
        keep = sorted(range(len(pathways)), key=lambda i: -pathway_scores[i])[:max_total]
        keep.sort()
        LOGGER.info("Pruning %d -> %d pathways (global cap).", len(pathways), max_total)
        pathways = [pathways[i] for i in keep]

    by_completeness: dict[str, int] = defaultdict(int)
    for p in pathways:
        by_completeness[p.completeness.value] += 1
    LOGGER.info("Built %d pathways across %d cultures (%s).",
                len(pathways), len(nodes_by_culture), dict(by_completeness))
    return pathways


def pathways_for_culture(pathways: list[CulturalPathway], culture: str) -> list[CulturalPathway]:
    c = normalize_text(culture).lower()
    return [p for p in pathways if normalize_text(p.culture).lower() == c]
