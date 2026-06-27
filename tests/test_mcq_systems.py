from src import mcq_systems
from src.mcq_systems import (
    SYSTEM_S0,
    SYSTEM_S2,
    SYSTEM_S3,
    RetrievalContext,
    build_prompt,
    parse_easy_answer,
    parse_hard_answer,
    run_system,
)


def test_parse_easy_answer_from_answer_line():
    assert parse_easy_answer("Reasoning...\nANSWER: B", {"A": "x", "B": "y"}) == "B"
    assert parse_easy_answer("ANSWER: c", {"A": "x", "B": "y", "C": "z"}) == "C"


def test_parse_easy_answer_fallback_letter():
    assert parse_easy_answer("I think the answer is D", {"C": "x", "D": "y"}) == "D"
    assert parse_easy_answer("no letters here", {"A": "x"}) is None


def test_parse_hard_answer():
    assert parse_hard_answer("ANSWER: TRUE") is True
    assert parse_hard_answer("ANSWER: false") is False
    assert parse_hard_answer("It is true.") is True
    assert parse_hard_answer("unclear") is None


def test_s0_prompt_has_no_evidence(mcq_easy_question):
    sp, up, ids = build_prompt(SYSTEM_S0, mcq_easy_question, None, top_k=5)
    assert ids == []
    assert "Retrieved" not in up
    assert "ANSWER:" in up


def test_s3_layered_prompt_includes_layer_labels(mcq_easy_question, sample_triples):
    ctx = RetrievalContext(triples=sample_triples)
    sp, up, ids = build_prompt(SYSTEM_S3, mcq_easy_question, ctx, top_k=5)
    # Layered evidence should reference abstraction layers.
    assert "LAYER_" in up
    assert len(ids) >= 1


def test_s2_flat_prompt_excludes_layer_labels(mcq_easy_question, sample_triples):
    ctx = RetrievalContext(triples=sample_triples)
    sp, up, ids = build_prompt(SYSTEM_S2, mcq_easy_question, ctx, top_k=5)
    assert "LAYER_" not in up
    assert len(ids) >= 1


def test_run_system_easy_scoring(mcq_easy_question, fake_backend_factory):
    backend = fake_backend_factory(lambda prompt, sysp: "ANSWER: B")
    answers = run_system(SYSTEM_S0, [mcq_easy_question], backend, None, top_k=5)
    assert len(answers) == 1
    assert answers[0].predicted == "B"
    assert answers[0].correct is True


def test_run_system_hard_scoring(mcq_hard_group, fake_backend_factory):
    # Respond TRUE for the first option (gold True) and FALSE for the second (gold False).
    def responder(prompt, sysp):
        return "ANSWER: TRUE" if "Touch the elder" in prompt else "ANSWER: FALSE"

    backend = fake_backend_factory(responder)
    answers = run_system(SYSTEM_S0, mcq_hard_group, backend, None, top_k=5)
    assert all(a.correct for a in answers)
