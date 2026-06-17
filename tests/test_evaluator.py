from src import evaluator
from src.evaluator import (
    SYSTEM_BASELINE,
    SYSTEM_RAW,
    SYSTEM_REFINED,
    compute_verdict,
    score_answer,
    summarize_results,
)
from src.schemas import (
    EvaluationExample,
    EvaluationResult,
    LayerLabel,
    SystemAnswer,
)


def _example() -> EvaluationExample:
    return EvaluationExample(
        example_id="e1",
        source_instance_id="test_001",
        question="What should a guest do when entering a Japanese home?",
        cultural_group="Japanese",
        expected_keywords=["shoes", "slippers", "respect"],
        avoid_keywords=["primitive", "weird"],
        target_layers=[LayerLabel.LAYER_2_SOCIAL, LayerLabel.LAYER_3_DEEP_VALUE],
    )


def test_score_answer_rewards_coverage_and_layers(sample_triples):
    ex = _example()
    good = ("In this context, a guest should remove their shoes and accept "
            "slippers. This reflects social expectations of the host-guest "
            "relationship and a deeper value of respect and harmony.")
    score, breakdown = score_answer(good, ex, sample_triples)
    assert breakdown["keyword_coverage"] > 0
    assert breakdown["layer_coverage"] > 0
    assert score > 0


def test_score_answer_penalizes_generated_dominance(sample_triples):
    ex = _example()
    # All retrieved triples generated -> penalty applied.
    gen_only = [t.model_copy(update={"generated": True}) for t in sample_triples]
    _score, breakdown = score_answer("respect and shoes", ex, gen_only)
    assert breakdown["generated_dominance_penalty"] < 0
    assert breakdown["used_generated_ratio"] == 1.0


def test_score_answer_bad_keyword_zeroes_component(sample_triples):
    ex = _example()
    _score, breakdown = score_answer("this is weird and primitive", ex, sample_triples)
    assert breakdown["avoids_bad_keywords"] == 0.0


def _make_result(b, r, rf, gen_ratio=0.0) -> EvaluationResult:
    return EvaluationResult(
        example_id="x",
        question="q",
        answers={
            SYSTEM_BASELINE: SystemAnswer(system=SYSTEM_BASELINE, score=b),
            SYSTEM_RAW: SystemAnswer(system=SYSTEM_RAW, score=r),
            SYSTEM_REFINED: SystemAnswer(system=SYSTEM_REFINED, score=rf,
                                         used_generated_ratio=gen_ratio,
                                         answer="respect and social roles"),
        },
    )


def test_summarize_and_pass_verdict():
    results = [_make_result(50, 60, 65, gen_ratio=0.1) for _ in range(5)]
    summary = summarize_results(results)
    assert summary["raw_kg_improvement_over_baseline"] == 10.0
    assert summary["refined_kg_improvement_over_raw"] == 5.0
    assert summary["verdict"] == "PASS"


def test_verdict_fail_when_raw_not_better():
    summary = {
        "raw_kg_improvement_over_baseline": 0.0,
        "refined_kg_improvement_over_raw": 2.0,
        "generated_support_ratio_percent": 10.0,
    }
    assert compute_verdict(summary) == "FAIL"


def test_verdict_partial_when_generated_too_high():
    summary = {
        "raw_kg_improvement_over_baseline": 8.0,
        "refined_kg_improvement_over_raw": 4.0,
        "generated_support_ratio_percent": 40.0,
    }
    assert compute_verdict(summary) == "PARTIAL PASS"


def test_generate_scenario_fallback_builds_question(sample_record):
    ex = evaluator.generate_scenario_fallback(sample_record)
    assert ex.question
    assert ex.cultural_group == "Japanese"
    assert len(ex.expected_keywords) > 0
