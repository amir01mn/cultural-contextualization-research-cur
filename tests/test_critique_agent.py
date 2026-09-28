from src.construction import critique_agent
from src.llm.local_llm import LLMUnavailableError
from src.core.schemas import CulturalTriple, LayerLabel


def test_rule_based_critique_flags_missing_layers(sample_record):
    # Only a surface-level triple -> should flag missing L2/L3 mediation.
    triples = [
        CulturalTriple(
            triple_id="t_only",
            subject="guest",
            predicate="removes",
            object="shoes",
            subject_layer=LayerLabel.LAYER_1_SURFACE,
            predicate_layer=LayerLabel.LAYER_1_SURFACE,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="test_001",
            source_text="src",
        )
    ]
    result = critique_agent.critique_record_rule_based(sample_record, triples)
    issue_types = {i.issue_type for i in result.issues}
    assert "missing_layer_2_social" in issue_types
    assert "missing_layer_3_value" in issue_types
    assert result.method == "rule_based"


def test_suggested_triples_are_generated_and_low_confidence(sample_record):
    triples = [
        CulturalTriple(
            triple_id="t_only",
            subject="guest",
            predicate="removes",
            object="shoes",
            subject_layer=LayerLabel.LAYER_1_SURFACE,
            predicate_layer=LayerLabel.LAYER_1_SURFACE,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="test_001",
            source_text="src",
        )
    ]
    result = critique_agent.critique_record_rule_based(sample_record, triples)
    for s in result.suggested_triples:
        assert s.generated is True
        assert s.confidence < 0.5


def test_critique_llm_unavailable_without_fallback_raises(monkeypatch, sample_record, sample_triples):
    monkeypatch.setattr(
        critique_agent.local_llm, "generate_json",
        lambda *a, **k: (_ for _ in ()).throw(LLMUnavailableError("no ollama")),
    )
    try:
        critique_agent.critique_record(sample_record, sample_triples,
                                       mode="llm_based", allow_fallback=False)
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError:
        pass


def test_critique_llm_unavailable_with_fallback(monkeypatch, sample_record, sample_triples):
    monkeypatch.setattr(
        critique_agent.local_llm, "generate_json",
        lambda *a, **k: (_ for _ in ()).throw(LLMUnavailableError("no ollama")),
    )
    result = critique_agent.critique_record(sample_record, sample_triples,
                                            mode="llm_based", allow_fallback=True)
    assert result.method == "rule_based"


def test_build_refined_triples_combines_and_keeps_originals(sample_record, sample_triples):
    records_by_id = {"test_001": sample_record}
    triples_by_record = {"test_001": sample_triples}
    critiques = [
        critique_agent.critique_record_rule_based(sample_record, sample_triples)
    ]
    refined = critique_agent.build_refined_triples(records_by_id, triples_by_record, critiques)
    # All original triples are retained.
    original_ids = {t.triple_id for t in sample_triples}
    refined_ids = {t.triple_id for t in refined}
    assert original_ids.issubset(refined_ids)
    assert len(refined) >= len(sample_triples)
