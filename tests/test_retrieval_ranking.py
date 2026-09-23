from src.retrieval_ranking import culture_match_score, rerank_hits
from src.schemas import CulturalPathway, LayerLabel, PathwayCompleteness, PathwayNodeRef


def test_culture_match_aliases():
    assert culture_match_score("American", "USA") == 1.0
    assert culture_match_score("Japanese", "Japan") == 1.0
    assert culture_match_score("American", "Japan") < 0.5


def test_rerank_prefers_culture_match():
    hits = [
        ("wrong", 0.9),
        ("right", 0.5),
    ]
    ranked = rerank_hits(
        hits,
        top_k=1,
        question_culture="Japan",
        get_culture=lambda x: "American" if x == "wrong" else "Japanese",
    )
    assert ranked[0][0] == "right"


def test_rerank_penalizes_generated():
    hits = [("a", 1.0), ("b", 1.0)]
    ranked = rerank_hits(
        hits,
        top_k=2,
        question_culture="Japan",
        get_culture=lambda _x: "Japan",
        get_generated=lambda x: x == "b",
    )
    assert ranked[0][0] == "a"
    assert ranked[0][1] > ranked[1][1]


def test_pathway_shared_pivot_cross_culture():
    pw = CulturalPathway(
        pathway_id="x",
        culture="American",
        completeness=PathwayCompleteness.SOCIAL,
        has_shared_pivot=True,
        nodes=[PathwayNodeRef(entity_id="1", label="respect", layer=LayerLabel.LAYER_3_DEEP_VALUE)],
    )
    hits = [(pw, 0.8)]
    ranked = rerank_hits(
        hits,
        top_k=1,
        question_culture="Japan",
        get_culture=lambda p: p.culture,
        allow_cross_culture_shared=lambda p: p.has_shared_pivot,
    )
    assert ranked


def test_rerank_scales_penalty_with_generated_ratio():
    hits = [("grounded", 1.0), ("mixed", 1.0), ("generated", 1.0)]
    ratios = {"grounded": 0.0, "mixed": 0.5, "generated": 1.0}
    ranked = rerank_hits(
        hits,
        top_k=3,
        question_culture="Japan",
        get_culture=lambda _x: "Japan",
        get_generated_ratio=lambda x: ratios[x],
    )
    scores = {item: score for item, score in ranked}
    assert scores["grounded"] > scores["mixed"] > scores["generated"]
