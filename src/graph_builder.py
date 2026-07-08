"""Build and export the layered cultural knowledge graph with networkx.

Nodes carry a soft layer_distribution derived from each triple element's layer.
Edges carry full provenance. Both raw and refined graphs can be exported to
GraphML, JSON, node/edge CSV, and an interactive HTML view (pyvis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import networkx as nx

from . import config
from .schemas import CulturalTriple, LayerLabel
from .utils import get_logger, normalize_node_label, normalize_text, stable_id

LOGGER = get_logger("graph_builder")

LAYER_COLORS = {
    "LAYER_1_SURFACE": "#4C9F70",
    "LAYER_2_SOCIAL": "#3A7CA5",
    "LAYER_3_DEEP_VALUE": "#A24C9F",
    "UNKNOWN": "#999999",
}


def _layer_dist(layer: LayerLabel | str) -> list[float]:
    key = layer.value if isinstance(layer, LayerLabel) else str(layer)
    return list(config.LAYER_DISTRIBUTIONS.get(key, config.LAYER_DISTRIBUTIONS["UNKNOWN"]))


def _dominant_layer(dist: list[float]) -> str:
    labels = ["LAYER_1_SURFACE", "LAYER_2_SOCIAL", "LAYER_3_DEEP_VALUE"]
    idx = max(range(len(dist)), key=lambda i: dist[i])
    return labels[idx]


def _culture_set(data: dict) -> set[str]:
    raw = data.get("cultures", set())
    if isinstance(raw, str):
        return {c.strip() for c in raw.split(";") if c.strip()}
    return set(raw or [])


def _add_node(graph: nx.MultiDiGraph, label: str, layer: LayerLabel,
              instance_id: str, generated: bool) -> str:
    node_id = stable_id(normalize_node_label(label), prefix="n_")
    dist = _layer_dist(layer)
    if graph.has_node(node_id):
        data = graph.nodes[node_id]
        # Accumulate layer distribution and frequency.
        agg = data["layer_distribution"]
        data["layer_distribution"] = [agg[i] + dist[i] for i in range(3)]
        data["frequency"] += 1
        if instance_id:
            data["sources"].add(instance_id)
        data["generated"] = data["generated"] and generated
    else:
        graph.add_node(
            node_id,
            label=normalize_text(label),
            layer_distribution=list(dist),
            sources={instance_id} if instance_id else set(),
            frequency=1,
            generated=generated,
        )
    return node_id


def build_graph(triples: Iterable[CulturalTriple]) -> nx.MultiDiGraph:
    """Construct a MultiDiGraph from triples."""
    graph = nx.MultiDiGraph()
    for t in triples:
        if not t.subject or not t.object:
            continue
        s_id = _add_node(graph, t.subject, t.subject_layer, t.instance_id, t.generated)
        o_id = _add_node(graph, t.object, t.object_layer, t.instance_id, t.generated)
        graph.add_edge(
            s_id,
            o_id,
            key=t.triple_id,
            triple_id=t.triple_id,
            predicate=t.predicate,
            predicate_layer=t.predicate_layer.value,
            dataset_id=t.dataset_id,
            instance_id=t.instance_id,
            source_text=t.source_text,
            confidence=float(t.confidence),
            generated=bool(t.generated),
            evidence=t.evidence or "",
            notes=t.notes or "",
            extraction_method=t.extraction_method,
        )
    LOGGER.info("Built graph: %d nodes, %d edges",
                graph.number_of_nodes(), graph.number_of_edges())
    return graph


def build_graph_resolved(
    triples: Iterable[CulturalTriple],
    entities: list,
    label_layer_to_entity: dict[tuple[str, str], str],
) -> nx.MultiDiGraph:
    """Build graph with canonical entity IDs and culture metadata on nodes."""
    from .entity_resolver import apply_entity_map
    from .schemas import CanonicalEntity
    from .utils import normalize_node_label

    entity_objs = [e if isinstance(e, CanonicalEntity) else CanonicalEntity(**e) for e in entities]
    entities_by_id = {e.entity_id: e for e in entity_objs}
    resolved_triples = apply_entity_map(list(triples), label_layer_to_entity, entities_by_id)

    graph = nx.MultiDiGraph()
    entity_node_ids: dict[str, str] = {}

    for e in entity_objs:
        node_id = e.entity_id
        entity_node_ids[e.entity_id] = node_id
        dist = _layer_dist(e.layer)
        graph.add_node(
            node_id,
            label=e.canonical_label,
            layer=e.layer.value,
            layer_distribution=list(dist),
            cultures=set(e.cultures),
            aliases=set(e.aliases),
            sources=set(e.source_instance_ids),
            frequency=1,
            generated=False,
            canonical_entity_id=e.entity_id,
            is_shared_abstraction=bool(e.is_shared_abstraction),
        )

    for t in resolved_triples:
        if not t.subject or not t.object:
            continue
        s_key = (normalize_node_label(t.subject), t.subject_layer.value)
        o_key = (normalize_node_label(t.object), t.object_layer.value)
        s_id = label_layer_to_entity.get(s_key)
        o_id = label_layer_to_entity.get(o_key)
        if not s_id or not o_id or s_id not in graph or o_id not in graph:
            continue
        graph.nodes[s_id]["frequency"] = graph.nodes[s_id].get("frequency", 1) + 1
        graph.nodes[o_id]["frequency"] = graph.nodes[o_id].get("frequency", 1) + 1
        if t.instance_id:
            graph.nodes[s_id]["sources"].add(t.instance_id)
            graph.nodes[o_id]["sources"].add(t.instance_id)
        graph.add_edge(
            s_id,
            o_id,
            key=t.triple_id,
            triple_id=t.triple_id,
            predicate=t.predicate,
            predicate_layer=t.predicate_layer.value,
            dataset_id=t.dataset_id,
            instance_id=t.instance_id,
            source_text=t.source_text,
            confidence=float(t.confidence),
            generated=bool(t.generated),
            evidence=t.evidence or "",
            notes=t.notes or "",
            extraction_method=t.extraction_method,
        )

    LOGGER.info("Built resolved graph: %d nodes, %d edges",
                graph.number_of_nodes(), graph.number_of_edges())
    return graph


def _normalized_node_view(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a copy with serializable node/edge attributes for export."""
    g = nx.MultiDiGraph()
    for node_id, data in graph.nodes(data=True):
        dist = data.get("layer_distribution", [0.33, 0.33, 0.33])
        total = sum(dist) or 1.0
        norm_dist = [round(x / total, 4) for x in dist]
        g.add_node(
            node_id,
            label=data.get("label", node_id),
            layer_distribution=norm_dist,
            dominant_layer=_dominant_layer(norm_dist),
            sources=";".join(sorted(data.get("sources", set()))),
            frequency=int(data.get("frequency", 1)),
            generated=bool(data.get("generated", False)),
            cultures=";".join(sorted(_culture_set(data))),
            is_shared_abstraction=bool(data.get("is_shared_abstraction", False)),
            canonical_entity_id=data.get("canonical_entity_id", node_id),
        )
    for s, o, key, data in graph.edges(keys=True, data=True):
        g.add_edge(s, o, key=key, **data)
    return g


def export_graphml(graph: nx.MultiDiGraph, path: str | Path) -> None:
    g = _normalized_node_view(graph)
    # GraphML cannot hold list/dict attributes; stringify the distribution.
    for _, data in g.nodes(data=True):
        data["layer_distribution"] = ",".join(str(x) for x in data["layer_distribution"])
    nx.write_graphml(g, str(path))


def export_json(graph: nx.MultiDiGraph, path: str | Path) -> None:
    g = _normalized_node_view(graph)
    from networkx.readwrite import json_graph
    from .utils import write_json

    data = json_graph.node_link_data(g, edges="links")
    write_json(path, data)


def export_csvs(graph: nx.MultiDiGraph, nodes_path: str | Path,
                edges_path: str | Path) -> None:
    import pandas as pd

    g = _normalized_node_view(graph)
    node_rows = []
    for node_id, data in g.nodes(data=True):
        node_rows.append({
            "node_id": node_id,
            "label": data.get("label"),
            "dominant_layer": data.get("dominant_layer"),
            "layer_distribution": data.get("layer_distribution"),
            "frequency": data.get("frequency"),
            "generated": data.get("generated"),
            "sources": data.get("sources"),
        })
    edge_rows = []
    for s, o, key, data in g.edges(keys=True, data=True):
        edge_rows.append({
            "triple_id": key,
            "source": s,
            "target": o,
            "subject_label": g.nodes[s].get("label"),
            "object_label": g.nodes[o].get("label"),
            "predicate": data.get("predicate"),
            "predicate_layer": data.get("predicate_layer"),
            "dataset_id": data.get("dataset_id"),
            "instance_id": data.get("instance_id"),
            "confidence": data.get("confidence"),
            "generated": data.get("generated"),
            "evidence": data.get("evidence"),
            "notes": data.get("notes"),
            "source_text": data.get("source_text"),
        })
    pd.DataFrame(node_rows).to_csv(nodes_path, index=False)
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)


def export_html(graph: nx.MultiDiGraph, path: str | Path) -> bool:
    """Export an interactive pyvis HTML view. Returns False if pyvis missing."""
    try:
        from pyvis.network import Network
    except Exception:  # noqa: BLE001
        LOGGER.warning("pyvis not installed; skipping HTML export for %s", path)
        return False

    g = _normalized_node_view(graph)
    net = Network(height="800px", width="100%", directed=True, notebook=False)
    for node_id, data in g.nodes(data=True):
        layer = data.get("dominant_layer", "UNKNOWN")
        net.add_node(
            node_id,
            label=data.get("label", node_id),
            title=f"{data.get('label')}\nlayer: {layer}\nfreq: {data.get('frequency')}",
            color=LAYER_COLORS.get(layer, "#999999"),
        )
    for s, o, _key, data in g.edges(keys=True, data=True):
        color = "#cccccc" if data.get("generated") else "#555555"
        net.add_edge(s, o, label=data.get("predicate", ""),
                     title=data.get("source_text", ""), color=color)
    try:
        net.write_html(str(path), notebook=False, open_browser=False)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Failed to write pyvis HTML: %s", exc)
        return False
    return True


def export_all(graph: nx.MultiDiGraph, prefix: str,
               outputs_dir: str | Path | None = None) -> dict:
    """Export a graph to all formats with the given filename prefix."""
    outputs_dir = Path(outputs_dir or config.OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "graphml": outputs_dir / f"{prefix}.graphml",
        "json": outputs_dir / f"{prefix}.json",
        "edges_csv": outputs_dir / f"{prefix}_edges.csv",
        "nodes_csv": outputs_dir / f"{prefix}_nodes.csv",
        "html": outputs_dir / f"{prefix}.html",
    }
    export_graphml(graph, paths["graphml"])
    export_json(graph, paths["json"])
    export_csvs(graph, paths["nodes_csv"], paths["edges_csv"])
    export_html(graph, paths["html"])
    LOGGER.info("Exported graph '%s' to %s", prefix, outputs_dir)
    return {k: str(v) for k, v in paths.items()}
