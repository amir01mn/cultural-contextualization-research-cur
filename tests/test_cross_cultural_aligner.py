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


def test_alignment_preserves_provenance_and_marks_returned_members_shared():
    entities = [
        CanonicalEntity(
            entity_id="e1",
            canonical_label="respect",
            layer=LayerLabel.LAYER_3_DEEP_VALUE,
            cultures=["Japan"],
            source_instance_ids=["jp1"],
            source_instance_cultures={"jp1": "Japan"},
        ),
        CanonicalEntity(
            entity_id="e2",
            canonical_label="respect",
            layer=LayerLabel.LAYER_3_DEEP_VALUE,
            cultures=["India"],
            source_instance_ids=["in1"],
            source_instance_cultures={"in1": "India"},
        ),
    ]
    alignments, updated = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.5
    )
    assert alignments
    by_id = {entity.entity_id: entity for entity in updated}
    alignment = alignments[0]
    pivot = by_id[alignment.abstraction_node_id]
    assert by_id["e1"].is_shared_abstraction is True
    assert by_id["e2"].is_shared_abstraction is True
    assert pivot.source_instance_cultures == {"jp1": "Japan", "in1": "India"}


def test_cross_cultural_alignment_does_not_merge_different_layers():
    entities = [
        CanonicalEntity(
            entity_id="e1", canonical_label="respect",
            layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["Japan"],
        ),
        CanonicalEntity(
            entity_id="e2", canonical_label="respect",
            layer=LayerLabel.LAYER_2_SOCIAL, cultures=["India"],
        ),
    ]
    alignments, _ = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.5
    )
    assert alignments == []


def test_prealigned_multiculture_entity_is_marked_shared():
    entities = [
        CanonicalEntity(
            entity_id="e1", canonical_label="hospitality",
            layer=LayerLabel.LAYER_3_DEEP_VALUE,
            cultures=["Japan", "India"],
        )
    ]
    alignments, updated = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.5
    )
    assert alignments == []
    assert updated[0].is_shared_abstraction is True
