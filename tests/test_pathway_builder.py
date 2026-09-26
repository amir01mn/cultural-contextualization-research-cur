from src.construction.graph_builder import build_graph_resolved
from src.construction.pathway_builder import build_pathways
from src.core.schemas import CanonicalEntity, LayerLabel, PathwayCompleteness


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
    from src.core.schemas import CulturalTriple

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
