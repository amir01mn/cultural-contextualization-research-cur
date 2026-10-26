"""Record-local pathways over the August triples (S4L, step 2 of the KG v3 plan).

The August builder pivoted on canonical L2 hubs merged across records, so 65%
of its pathways joined triples from different records, it reached 38% of
records, and a 500-per-culture cap cut the largest cultures. Here each pathway
stays inside one CultureBank record, every record with usable triples gets one,
and the pathway carries the record's descriptor text for rendering.

Triples that failed extraction (``llm_failed``) or that carry wording found
only in CultureBank's eval columns (``core.leakage``) are dropped first.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable

from ..core.leakage import triple_uses_eval_text
from ..core.schemas import (
    CulturalPathway,
    CulturalTriple,
    LayerLabel,
    PathwayCompleteness,
    PathwayNodeRef,
)
from ..core.utils import get_logger, normalize_text, stable_id
from .frame_builder import frame_from_row
from .pathway_builder import _completeness_from_layers

LOGGER = get_logger("record_pathways")

L1, L2, L3 = (LayerLabel.LAYER_1_SURFACE, LayerLabel.LAYER_2_SOCIAL,
              LayerLabel.LAYER_3_DEEP_VALUE)


def usable_triples(triples: Iterable[CulturalTriple]) -> tuple[list[CulturalTriple], Counter]:
    kept, dropped = [], Counter()
    for t in triples:
        if t.llm_failed:
            dropped["llm_failed"] += 1
        elif triple_uses_eval_text(t.subject, t.object, t.raw_fields):
            dropped["eval_text"] += 1
        else:
            kept.append(t)
    return kept, dropped


def _key(label: str) -> str:
    return normalize_text(label).lower()


def _record_chain(ts: list[CulturalTriple]) -> tuple[list[tuple[str, LayerLabel]], list[str], list[str]]:
    """Best value -> norm -> practice chain inside one record.

    Returns (nodes, predicates, triple_ids). Prefers an L2 node linked to both
    an L3 and an L1 node; then any two-layer link; then the record's first
    triple.
    """
    adj: dict[tuple[str, LayerLabel], list[tuple[tuple[str, LayerLabel], CulturalTriple]]] = defaultdict(list)
    for t in ts:
        a, b = (t.subject, t.subject_layer), (t.object, t.object_layer)
        adj[a].append((b, t))
        adj[b].append((a, t))

    def conf(t: CulturalTriple) -> float:
        return (0.0 if t.generated else 1.0) + t.confidence

    best = None
    for node, nbrs in adj.items():
        if node[1] != L2:
            continue
        up = sorted(((n, t) for n, t in nbrs if n[1] == L3), key=lambda x: -conf(x[1]))
        down = sorted(((n, t) for n, t in nbrs if n[1] == L1), key=lambda x: -conf(x[1]))
        if up and down:
            score = conf(up[0][1]) + conf(down[0][1])
            if best is None or score > best[0]:
                best = (score, [up[0][0], node, down[0][0]],
                        [up[0][1].predicate, down[0][1].predicate],
                        [up[0][1].triple_id, down[0][1].triple_id])
    if best:
        return best[1], best[2], best[3]
    order = {L3: 0, L2: 1, L1: 2}
    links = [t for t in ts if t.subject_layer != t.object_layer
             and t.subject_layer in order and t.object_layer in order]
    t = max(links, key=conf) if links else max(ts, key=conf)
    a, b = (t.subject, t.subject_layer), (t.object, t.object_layer)
    if order.get(a[1], 3) > order.get(b[1], 3):
        a, b = b, a
    return [a, b], [t.predicate], [t.triple_id]


def build_record_pathways(triples: Iterable[CulturalTriple]) -> tuple[list[CulturalPathway], dict]:
    kept, dropped = usable_triples(triples)
    by_rec: dict[str, list[CulturalTriple]] = defaultdict(list)
    for t in kept:
        by_rec[t.instance_id].append(t)

    # An L3 label used by >= 2 cultures is a shared abstraction (pivot proxy).
    l3_cultures: dict[str, set[str]] = defaultdict(set)
    for t in kept:
        culture = _key((t.raw_fields or {}).get("cultural group")
                       or (t.raw_fields or {}).get("cultural_group") or "")
        for label, layer in ((t.subject, t.subject_layer), (t.object, t.object_layer)):
            if layer == L3:
                l3_cultures[_key(label)].add(culture)

    pathways = []
    for iid, ts in by_rec.items():
        raw = dict(ts[0].raw_fields or {})
        frame = frame_from_row({**raw, "instance_id": iid})
        culture = frame.culture_label if frame else (raw.get("cultural group") or "")
        nodes, preds, tids = _record_chain(ts)
        layers = {layer for _, layer in nodes}
        completeness = _completeness_from_layers(layers)
        if completeness == PathwayCompleteness.NONE:
            completeness = PathwayCompleteness.SURFACE
        pivot = next((lab for lab, layer in nodes
                      if layer == L3 and len(l3_cultures[_key(lab)]) >= 2), None)
        gen = {t.triple_id: t.generated for t in ts}
        pathways.append(CulturalPathway(
            pathway_id=stable_id("record_pathway", iid, prefix="rpw_"),
            culture=culture,
            nodes=[PathwayNodeRef(entity_id=stable_id(lab, layer.value), label=lab, layer=layer)
                   for lab, layer in nodes],
            triple_ids=tids,
            completeness=completeness,
            has_shared_pivot=pivot is not None,
            shared_pivot_id=stable_id(_key(pivot)) if pivot else None,
            predicates=preds,
            generated=all(gen.get(t, False) for t in tids),
            instance_ids=[iid],
            source_text=frame.text if frame else None,
        ))
    stats = {
        "triples_in": len(kept) + sum(dropped.values()),
        "triples_dropped": dict(dropped),
        "records_with_usable_triples": len(by_rec),
        "pathways": len(pathways),
        "completeness": dict(Counter(p.completeness.value for p in pathways)),
        "shared_pivot": sum(p.has_shared_pivot for p in pathways),
    }
    LOGGER.info("Record-local pathways: %s", stats)
    return pathways, stats
