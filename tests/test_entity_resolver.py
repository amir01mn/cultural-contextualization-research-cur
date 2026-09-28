from src.construction.entity_resolver import apply_entity_map, resolve_entities
from src.core.schemas import CulturalTriple, LayerLabel


def test_resolve_entities_merges_similar_labels(sample_record):
    triples = [
        CulturalTriple(
            triple_id="t1", subject="respect for elders", predicate="is", object="value",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            instance_id="test_001",
        ),
        CulturalTriple(
            triple_id="t2", subject="respect for elders", predicate="shown by", object="bowing",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="test_001",
        ),
        CulturalTriple(
            triple_id="t3", subject="respect elders", predicate="is", object="norm",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_2_SOCIAL,
            instance_id="test_001",
        ),
    ]
    entities, label_map = resolve_entities(triples, [sample_record], similarity_threshold=0.5)
    assert len(entities) >= 2
    entities_by_id = {e.entity_id: e for e in entities}
    resolved = apply_entity_map(triples, label_map, entities_by_id)
    assert all(t.subject for t in resolved)


def test_entity_cultures_from_record(sample_record, sample_triples):
    """Entity cultures are stored canonically, not as the raw record string.

    The record says "Japanese"; the canonical key is "japan". Canonicalizing
    here is what lets a benchmark question about "Japan" match corpus entities
    tagged "Japanese", and what lets pathway hops agree on a culture.
    """
    entities, _ = resolve_entities(sample_triples, [sample_record], similarity_threshold=0.99)
    assert entities
    assert any("japan" in e.cultures for e in entities)
    assert not any("Japanese" in e.cultures for e in entities)
