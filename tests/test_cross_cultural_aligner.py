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


def test_multiculture_entity_is_not_implicitly_marked_shared():
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
    assert updated[0].is_shared_abstraction is False



def test_same_culture_aliases_are_not_cross_cultural():
    entities = [
        CanonicalEntity(
            entity_id="e1", canonical_label="politeness",
            layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["Japanese"],
        ),
        CanonicalEntity(
            entity_id="e2", canonical_label="politeness",
            layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["Japanese people"],
        ),
    ]
    alignments, _ = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.5
    )
    assert alignments == []


def test_actor_only_l2_labels_are_not_shared_abstraction_candidates():
    entities = [
        CanonicalEntity(
            entity_id="e1", canonical_label="customers",
            layer=LayerLabel.LAYER_2_SOCIAL, cultures=["Korean"],
        ),
        CanonicalEntity(
            entity_id="e2", canonical_label="customers",
            layer=LayerLabel.LAYER_2_SOCIAL, cultures=["Spaniards"],
        ),
    ]
    alignments, _ = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.5
    )
    assert alignments == []


def test_norm_like_l2_labels_remain_alignment_candidates():
    entities = [
        CanonicalEntity(
            entity_id="e1", canonical_label="guest etiquette",
            layer=LayerLabel.LAYER_2_SOCIAL, cultures=["Japan"],
        ),
        CanonicalEntity(
            entity_id="e2", canonical_label="guest etiquette",
            layer=LayerLabel.LAYER_2_SOCIAL, cultures=["India"],
        ),
    ]
    alignments, _ = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.5
    )
    assert alignments


def test_same_concept_across_three_cultures_reuses_one_pivot():
    entities = [
        CanonicalEntity(
            entity_id="e1", canonical_label="hospitality",
            layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["Brazil"],
        ),
        CanonicalEntity(
            entity_id="e2", canonical_label="hospitality",
            layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["Pakistan"],
        ),
        CanonicalEntity(
            entity_id="e3", canonical_label="hospitality",
            layer=LayerLabel.LAYER_3_DEEP_VALUE, cultures=["Russia"],
        ),
    ]
    alignments, updated = align_cross_cultural(
        entities, backend=None, verify=False, similarity_threshold=0.5
    )
    assert len(alignments) == 3
    pivot_ids = {alignment.abstraction_node_id for alignment in alignments}
    assert len(pivot_ids) == 1
    pivot_id = next(iter(pivot_ids))
    pivot = next(entity for entity in updated if entity.entity_id == pivot_id)
    assert set(pivot.cultures) == {"Brazil", "Pakistan", "Russia"}
