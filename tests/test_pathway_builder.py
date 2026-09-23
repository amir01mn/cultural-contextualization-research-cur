from src.entity_resolver import resolve_entities
from src.graph_builder import build_graph_resolved
from src.pathway_builder import build_pathways
from src.schemas import (
    CanonicalEntity,
    CrossCulturalAlignment,
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


def test_disconnected_same_record_evidence_does_not_make_full_pathway():
    records = [
        {
            "instance_id": "r1",
            "cultural_group": "Japan",
        },
    ]

    # Both triples come from the same source record, but they form two
    # completely disconnected graph components:
    #
    #   hospitality -> generosity
    #
    #   guest -> tea
    #
    # Collectively the record contains L3, L2, and L1 evidence, but there is
    # no connected component containing all three layers.
    triples = [
        CulturalTriple(
            triple_id="t_value",
            subject="hospitality",
            predicate="is associated with",
            object="generosity",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            instance_id="r1",
        ),
        CulturalTriple(
            triple_id="t_practice",
            subject="guest",
            predicate="is offered",
            object="tea",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="r1",
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

    full_pathways = [
        p for p in pathways
        if p.culture == "Japan"
        and p.completeness == PathwayCompleteness.FULL
    ]

    # A FULL pathway should require connected L1/L2/L3 evidence.
    assert not full_pathways

def test_aligned_member_marks_pathway_as_shared():
    entities = [
        CanonicalEntity(
            entity_id="l3", canonical_label="respect",
            layer=LayerLabel.LAYER_3_DEEP_VALUE,
            cultures=["Japan"],
            is_shared_abstraction=True,
        ),
        CanonicalEntity(
            entity_id="l2", canonical_label="guest norm",
            layer=LayerLabel.LAYER_2_SOCIAL, cultures=["Japan"],
        ),
        CanonicalEntity(
            entity_id="l1", canonical_label="remove shoes",
            layer=LayerLabel.LAYER_1_SURFACE, cultures=["Japan"],
        ),
    ]
    label_map = {
        ("respect", LayerLabel.LAYER_3_DEEP_VALUE.value): "l3",
        ("guest norm", LayerLabel.LAYER_2_SOCIAL.value): "l2",
        ("remove shoes", LayerLabel.LAYER_1_SURFACE.value): "l1",
    }
    triples = [
        CulturalTriple(
            triple_id="t1", subject="respect", predicate="supports",
            object="guest norm",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_2_SOCIAL,
            instance_id="r1",
        ),
        CulturalTriple(
            triple_id="t2", subject="guest norm", predicate="expressed as",
            object="remove shoes",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="r1",
        ),
    ]
    alignment = CrossCulturalAlignment(
        alignment_id="a1",
        abstraction_node_id="pivot1",
        culture_a="Japan",
        culture_b="India",
        entity_ids=["l3", "other"],
        verified=True,
    )
    graph = build_graph_resolved(triples, entities, label_map)
    pathways = build_pathways(graph, entities, [alignment], max_per_culture=50)
    full = next(p for p in pathways if p.completeness == PathwayCompleteness.FULL)
    assert full.has_shared_pivot is True
    assert full.shared_pivot_id == "pivot1"


def test_pathway_cap_prefers_grounded_full_evidence_over_record_order():
    records = [
        {"instance_id": "a_generated", "cultural_group": "Japan"},
        {"instance_id": "z_grounded", "cultural_group": "Japan"},
    ]
    triples = [
        CulturalTriple(
            triple_id="ga", subject="generated value", predicate="supports",
            object="generated norm",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_2_SOCIAL,
            instance_id="a_generated", generated=True,
        ),
        CulturalTriple(
            triple_id="gb", subject="generated norm", predicate="expressed as",
            object="generated practice",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="a_generated", generated=True,
        ),
        CulturalTriple(
            triple_id="za", subject="grounded value", predicate="supports",
            object="grounded norm",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_2_SOCIAL,
            instance_id="z_grounded", generated=False,
        ),
        CulturalTriple(
            triple_id="zb", subject="grounded norm", predicate="expressed as",
            object="grounded practice",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="z_grounded", generated=False,
        ),
    ]
    entities, label_map = resolve_entities(triples, records, similarity_threshold=0.99)
    graph = build_graph_resolved(triples, entities, label_map)
    pathways = build_pathways(graph, entities, [], max_per_culture=1)
    assert len(pathways) == 1
    assert set(pathways[0].triple_ids) == {"za", "zb"}
