from src.schemas import (
    CulturalRecord,
    CulturalTriple,
    EvaluationExample,
    LayerLabel,
)


def test_layer_label_coerce_valid():
    assert LayerLabel.coerce("LAYER_1_SURFACE") == LayerLabel.LAYER_1_SURFACE
    assert LayerLabel.coerce("layer_2_social") == LayerLabel.LAYER_2_SOCIAL


def test_layer_label_coerce_aliases_and_invalid():
    assert LayerLabel.coerce("L3") == LayerLabel.LAYER_3_DEEP_VALUE
    assert LayerLabel.coerce("surface") == LayerLabel.LAYER_1_SURFACE
    assert LayerLabel.coerce("garbage") == LayerLabel.UNKNOWN
    assert LayerLabel.coerce(None) == LayerLabel.UNKNOWN


def test_cultural_record_missing_fields_default_none():
    rec = CulturalRecord(instance_id="x")
    assert rec.cultural_group is None
    assert rec.dataset_id == "culturebank"
    assert rec.raw == {}


def test_cultural_triple_confidence_clamped_and_layers_coerced():
    t = CulturalTriple(
        triple_id="t",
        subject="a",
        predicate="b",
        object="c",
        subject_layer="bogus",
        confidence=5.0,
    )
    assert t.confidence == 1.0
    assert t.subject_layer == LayerLabel.UNKNOWN


def test_cultural_triple_preserves_provenance():
    t = CulturalTriple(
        triple_id="t",
        subject="a",
        predicate="b",
        object="c",
        source_text="src",
        dataset_id="culturebank",
        instance_id="i1",
        raw_fields={"k": "v"},
        extraction_method="llm_based",
        generated=True,
    )
    assert t.dataset_id == "culturebank"
    assert t.instance_id == "i1"
    assert t.source_text == "src"
    assert t.raw_fields == {"k": "v"}
    assert t.extraction_method == "llm_based"
    assert t.generated is True


def test_evaluation_example_target_layers_coercion():
    ex = EvaluationExample(
        example_id="e",
        source_instance_id="i",
        question="q?",
        target_layers=["LAYER_2_SOCIAL", "bogus"],
    )
    assert ex.target_layers[0] == LayerLabel.LAYER_2_SOCIAL
    assert ex.target_layers[1] == LayerLabel.UNKNOWN


def test_text_blob_includes_components():
    t = CulturalTriple(triple_id="t", subject="guest", predicate="removes",
                       object="shoes", evidence="actor_behavior")
    blob = t.text_blob()
    assert "guest" in blob and "shoes" in blob and "actor_behavior" in blob
