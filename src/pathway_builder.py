"""Construct abstraction→refinement pathways from a resolved cultural graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import networkx as nx

from . import config
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


def build_pathways(
    graph: nx.MultiDiGraph,
    entities: list[CanonicalEntity],
    alignments: list[CrossCulturalAlignment],
    *,
    max_per_culture: Optional[int] = None,
) -> list[CulturalPathway]:
    """Enumerate L3→L2→L1 coherent pathways per culture."""
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
        start_nodes = []
        for nid, data in graph.nodes(data=True):
            if culture not in _culture_set(data):
                continue
            layer = _node_layer(data)
            if layer == LayerLabel.LAYER_3_DEEP_VALUE:
                start_nodes.append(nid)

        for start in start_nodes:
            stack = [(start, [start], set())]
            while stack and count < max_per_culture:
                node, path, triple_ids = stack.pop()
                node_data = graph.nodes[node]
                path_layers = {_node_layer(graph.nodes[n]) for n in path}

                if len(path) >= 2 or _node_layer(node_data) == LayerLabel.LAYER_1_SURFACE:
                    completeness = _completeness_from_layers(path_layers)
                    if completeness != PathwayCompleteness.NONE:
                        node_key = stable_id(culture, *path)
                        if node_key not in seen_hashes:
                            seen_hashes.add(node_key)
                            refs = [
                                PathwayNodeRef(
                                    entity_id=n,
                                    label=graph.nodes[n].get("label", n),
                                    layer=_node_layer(graph.nodes[n]),
                                )
                                for n in path
                            ]
                            shared = any(n in pivot_ids for n in path)
                            pivot_id = next((n for n in path if n in pivot_ids), None)
                            pathways.append(CulturalPathway(
                                pathway_id=stable_id(culture, node_key, prefix="pw_"),
                                culture=culture,
                                nodes=refs,
                                triple_ids=sorted(triple_ids),
                                completeness=completeness,
                                has_shared_pivot=shared,
                                shared_pivot_id=pivot_id,
                            ))
                            count += 1

                if len(path) >= 4:
                    continue

                cur_rank = _layer_rank(_node_layer(node_data))
                for succ in graph.successors(node):
                    if succ in path:
                        continue
                    if culture not in _culture_set(graph.nodes[succ]):
                        continue
                    succ_rank = _layer_rank(_node_layer(graph.nodes[succ]))
                    if succ_rank > cur_rank:
                        continue
                    edge_triple_ids = set()
                    for _u, _v, key, _data in graph.out_edges(node, keys=True, data=True):
                        if _v == succ:
                            edge_triple_ids.add(str(key))
                    stack.append((succ, path + [succ], triple_ids | edge_triple_ids))

    LOGGER.info("Built %d pathways across %d cultures.", len(pathways), len(cultures))
    return pathways


def pathways_for_culture(pathways: list[CulturalPathway], culture: str) -> list[CulturalPathway]:
    c = normalize_text(culture).lower()
    return [p for p in pathways if normalize_text(p.culture).lower() == c]
