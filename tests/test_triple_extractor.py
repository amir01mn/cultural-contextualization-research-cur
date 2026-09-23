from src import triple_extractor
from src.local_llm import LLMUnavailableError
from src.schemas import CulturalTriple


def test_schema_fallback_preserves_provenance_and_marks_inferred_bridges(sample_record):
    triples = triple_extractor.extract_triples(
        sample_record,
        mode="schema_fallback",
    )

    assert len(triples) > 0
    assert all(isinstance(t, CulturalTriple) for t in triples)

    # Every fallback triple must preserve its source provenance.
    for t in triples:
        assert t.extraction_method == "schema_fallback"
        assert t.instance_id == "test_001"
        assert t.source_text

    # Direct field-derived relationships should remain grounded.
    grounded = [t for t in triples if t.predicate != "may express"]
    assert grounded
    assert all(t.generated is False for t in grounded)

    # Value bridges inferred from the record wording must be marked as
    # generated rather than presented as directly observed facts.
    value_bridges = [t for t in triples if t.predicate == "may express"]
    assert value_bridges
    assert all(t.generated is True for t in value_bridges)
    assert all(t.confidence < 0.5 for t in value_bridges)
    assert all(
        "Cautious value bridge" in (t.notes or "")
        for t in value_bridges
    )


def test_llm_mode_unavailable_without_fallback_raises(monkeypatch, sample_record):
    def boom(*args, **kwargs):
        raise LLMUnavailableError("no ollama")

    monkeypatch.setattr(triple_extractor.local_llm, "generate_json", boom)
    try:
        triple_extractor.extract_triples(sample_record, mode="llm_based",
                                         allow_fallback=False)
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError:
        pass


def test_llm_mode_unavailable_with_fallback_uses_schema(monkeypatch, sample_record):
    def boom(*args, **kwargs):
        raise LLMUnavailableError("no ollama")

    monkeypatch.setattr(triple_extractor.local_llm, "generate_json", boom)
    triples = triple_extractor.extract_triples(sample_record, mode="llm_based",
                                               allow_fallback=True)
    assert len(triples) > 0
    assert all(t.extraction_method == "schema_fallback" for t in triples)


def test_llm_mode_parses_llm_json(monkeypatch, sample_record):
    fake = {
        "record_id": "test_001",
        "triples": [
            {
                "subject": "guest",
                "predicate": "removes",
                "object": "shoes",
                "subject_layer": "LAYER_2_SOCIAL",
                "predicate_layer": "LAYER_1_SURFACE",
                "object_layer": "LAYER_1_SURFACE",
                "confidence": 0.9,
                "generated": False,
                "evidence": "actor_behavior",
                "notes": None,
            }
        ],
    }
    monkeypatch.setattr(triple_extractor.local_llm, "generate_json", lambda *a, **k: fake)
    triples = triple_extractor.extract_triples(sample_record, mode="llm_based")
    assert len(triples) == 1
    assert triples[0].extraction_method == "llm_based"
    assert triples[0].subject == "guest"
