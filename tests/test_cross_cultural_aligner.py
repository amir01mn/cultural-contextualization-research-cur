from src.cross_cultural_aligner import align_cross_cultural
from src.schemas import CanonicalEntity, LayerLabel


def test_align_cross_cultural_embedding_only():
    entities = [
        CanonicalEntity(
            entity_id="e1", canonical_label="respect for elders",
            layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["Japan"],
        ),
        CanonicalEntity(
            entity_id="e2", canonical_label="respect for elders",
            layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["India"],
        ),
    ]
    alignments, updated = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.5)
    assert len(alignments) >= 1
    assert any(e.is_shared_abstraction for e in updated)


def test_no_alignment_same_culture():
    entities = [
        CanonicalEntity(
            entity_id="e1", canonical_label="hospitality",
            layer=LayerLabel.LAYER_2_SOCIAL, cultures=["Japan"],
        ),
        CanonicalEntity(
            entity_id="e2", canonical_label="hospitality norms",
            layer=LayerLabel.LAYER_2_SOCIAL, cultures=["Japan"],
        ),
    ]
    alignments, _ = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.99)
    assert alignments == []
