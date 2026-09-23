from src import mcq_systems
from src.entity_resolver import resolve_entities
from src.graph_builder import build_graph_resolved
from src.pathway_builder import build_pathways
from src.mcq_systems import (
    SYSTEM_S0,
    SYSTEM_S2,
    SYSTEM_S3,
    SYSTEM_S4,
    SYSTEM_S4_TRIPLE,
    RetrievalContext,
    build_prompt,
    parse_easy_answer,
    parse_hard_answer,
    run_system,
)
from src.schemas import (
    CulturalPathway,
    CulturalTriple,
    LayerLabel,
    PathwayCompleteness,
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
    sp, up, ids, _meta = build_prompt(SYSTEM_S0, mcq_easy_question, None, top_k=5)
    assert ids == []
    assert "Retrieved" not in up
    assert "ANSWER:" in up


def test_s3_layered_prompt_includes_layer_labels(mcq_easy_question, sample_triples):
    ctx = RetrievalContext(triples=sample_triples)
    sp, up, ids, _meta = build_prompt(SYSTEM_S3, mcq_easy_question, ctx, top_k=5)
    # Layered evidence should reference abstraction layers.
    assert "LAYER_" in up
    assert len(ids) >= 1


def test_s2_flat_prompt_excludes_layer_labels(mcq_easy_question, sample_triples):
    ctx = RetrievalContext(triples=sample_triples)
    sp, up, ids, _meta = build_prompt(SYSTEM_S2, mcq_easy_question, ctx, top_k=5)
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


def test_s4_prompt_preserves_pathway_predicates(mcq_easy_question):
    records = [
        {
            "instance_id": "r1",
            "cultural_group": "Japan",
        },
    ]

    triples = [
        CulturalTriple(
            triple_id="t1",
            subject="respect",
            predicate="mediates",
            object="deference norm",
            subject_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_2_SOCIAL,
            instance_id="r1",
        ),
        CulturalTriple(
            triple_id="t2",
            subject="deference norm",
            predicate="expressed as",
            object="remove shoes",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            instance_id="r1",
        ),
    ]

    entities, label_map = resolve_entities(
        triples,
        records,
        similarity_threshold=0.99,
    )
    graph = build_graph_resolved(triples, entities, label_map)
    pathways = build_pathways(graph, entities, [], max_per_culture=50)

    full_pathway = next(
        p for p in pathways
        if p.culture == "Japan"
        and p.completeness == PathwayCompleteness.FULL
    )

    class FixedPathwayRetriever:
        def __init__(self, pathway):
            self.pathways = [pathway]

        def retrieve(self, query, top_k=5, *, question_culture=None):
            return [(self.pathways[0], 1.0)]

    ctx = RetrievalContext(
        pathway_retriever=FixedPathwayRetriever(full_pathway),
    )

    _sp, user_prompt, ids, _meta = build_prompt(
        SYSTEM_S4,
        mcq_easy_question,
        ctx,
        top_k=1,
    )

    assert ids == [full_pathway.pathway_id]

    # S4 is meant to expose connected relational evidence, not just an
    # ordered list of layer-labelled concepts. The predicates supporting the
    # pathway must therefore survive into the evidence shown to the LLM.
    assert "mediates" in user_prompt
    assert "expressed as" in user_prompt



def test_s4_pathway_and_triple_ablation_use_same_requested_retrieval_budget(
    mcq_easy_question,
):
    class RecordingPathwayRetriever:
        def __init__(self):
            self.pathways = []
            self.requested_top_k = []

        def retrieve(self, query, top_k=5, *, question_culture=None):
            self.requested_top_k.append(top_k)
            return []

    class RecordingTripleRetriever:
        def __init__(self):
            self.requested_top_k = []

        def retrieve(self, query, top_k=5, *, question_culture=None):
            self.requested_top_k.append(top_k)
            return []

    pathway_retriever = RecordingPathwayRetriever()
    triple_retriever = RecordingTripleRetriever()

    ctx = RetrievalContext(pathway_retriever=pathway_retriever)
    ctx.refined_triple_retriever = triple_retriever

    requested_top_k = 4

    build_prompt(
        SYSTEM_S4,
        mcq_easy_question,
        ctx,
        top_k=requested_top_k,
    )
    build_prompt(
        SYSTEM_S4_TRIPLE,
        mcq_easy_question,
        ctx,
        top_k=requested_top_k,
    )

    # The pathway condition and its triple ablation must enter their
    # retrievers with the same requested top-k. Each retriever may then apply
    # the same internal candidate multiplier before reranking.
    assert pathway_retriever.requested_top_k == [requested_top_k]
    assert triple_retriever.requested_top_k == [requested_top_k]



def test_s4_postranking_does_not_apply_culture_penalty_twice():
    pathway = CulturalPathway(
        pathway_id="pw_mismatch",
        culture="American",
        completeness=PathwayCompleteness.FULL,
    )

    # PathwayRetriever.retrieve() has already applied the culture mismatch
    # penalty before _rank_pathways() receives this score.
    already_reranked_score = mcq_systems.culture_match_score(
        pathway.culture,
        "Japan",
    )

    ranked = mcq_systems._rank_pathways(
        [(pathway, already_reranked_score)],
        question_culture="Japan",
    )

    assert len(ranked) == 1

    # FULL has weight 1.0, so a second culture penalty must not change the
    # score at this post-ranking stage.
    assert ranked[0][1] == already_reranked_score



def test_s4_postranking_preserves_retriever_scores_without_pathway_bonuses():
    full_pathway = CulturalPathway(
        pathway_id="pw_full",
        culture="Japan",
        completeness=PathwayCompleteness.FULL,
        has_shared_pivot=True,
    )
    social_pathway = CulturalPathway(
        pathway_id="pw_social",
        culture="Japan",
        completeness=PathwayCompleteness.SOCIAL,
        has_shared_pivot=False,
    )

    # These scores represent the output of PathwayRetriever after semantic and
    # culture reranking. The post-ranking stage should not inject additional
    # completeness or shared-pivot bonuses into the retrieval-unit ablation.
    hits = [
        (social_pathway, 0.90),
        (full_pathway, 0.80),
    ]

    ranked = mcq_systems._rank_pathways(
        hits,
        question_culture="Japan",
    )

    assert [pw.pathway_id for pw, _score in ranked] == [
        "pw_social",
        "pw_full",
    ]
    assert [score for _pw, score in ranked] == [0.90, 0.80]
