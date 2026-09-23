from src.entity_resolver import resolve_entities
from src.graph_builder import build_graph_resolved
from src.pathway_builder import build_pathways
from src.schemas import (
    CanonicalEntity,
    CulturalTriple,
    LayerLabel,
    PathwayCompleteness,
)


def test_build_pathways_finds_full_path():
    entities = [
        CanonicalEntity(entity_id="l3", canonical_label="respect",
                        layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["Japan"]),
        CanonicalEntity(entity_id="l2", canonical_label="deference norm",
                        layer=LayerLabel.LAYER_2_SOCIAL, cultures=["Japan"]),
        CanonicalEntity(entity_id="l1", canonical_label="remove shoes",
                        layer=LayerLabel.LAYER_1_SURFACE, cultures=["Japan"]),
    ]
    label_map = {
        ("respect", LayerLabel.LAYER_3_DEEP_VALUE.value): "l3",
        ("deference norm", LayerLabel.LAYER_2_SOCIAL.value): "l2",
        ("remove shoes", LayerLabel.LAYER_1_SURFACE.value): "l1",
    }

    triples = [
        CulturalTriple(
            triple_id="t1", subject="respect", predicate="mediates",
            object="deference norm",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_2_SOCIAL,
            instance_id="r1",
        ),
        CulturalTriple(
            triple_id="t2", subject="deference norm", predicate="expressed as",
            object="remove shoes",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="r1",
        ),
    ]
    graph = build_graph_resolved(triples, entities, label_map)
    pathways = build_pathways(graph, entities, [], max_per_culture=50)
    assert pathways
    assert any(p.completeness == PathwayCompleteness.FULL for p in pathways)

def test_pathways_do_not_leak_edges_between_cultures():
    records = [
        {
            "instance_id": "a_japan",
            "cultural_group": "Japan",
        },
        {
            "instance_id": "b_netherlands",
            "cultural_group": "Netherlands",
        },
    ]

    # The two cultures deliberately use the same endpoint concepts.
    # Entity resolution is allowed to merge those concepts into shared
    # canonical nodes, but relation provenance must remain culture-specific.
    triples = [
        CulturalTriple(
            triple_id="t_japan",
            subject="guest",
            predicate="is served",
            object="tea",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="a_japan",
        ),
        CulturalTriple(
            triple_id="t_netherlands",
            subject="guest",
            predicate="is offered",
            object="tea",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="b_netherlands",
        ),
    ]

    entities, label_map = resolve_entities(
        triples,
        records,
        similarity_threshold=0.99,
    )

    graph = build_graph_resolved(
        triples,
        entities,
        label_map,
    )

    pathways = build_pathways(
        graph,
        entities,
        [],
        max_per_culture=50,
    )

    japanese_pathways = [
        p for p in pathways
        if p.culture == "Japan"
    ]

    dutch_pathways = [
        p for p in pathways
        if p.culture == "Netherlands"
    ]

    assert japanese_pathways
    assert dutch_pathways

    # Japanese pathways must use Japanese relation provenance only.
    assert all(
        "t_netherlands" not in p.triple_ids
        for p in japanese_pathways
    )

    # Dutch pathways must use Dutch relation provenance only.
    assert all(
        "t_japan" not in p.triple_ids
        for p in dutch_pathways
    )

    # Make sure each culture actually retained its own evidence.
    assert any(
        "t_japan" in p.triple_ids
        for p in japanese_pathways
    )

    assert any(
        "t_netherlands" in p.triple_ids
        for p in dutch_pathways
    )
