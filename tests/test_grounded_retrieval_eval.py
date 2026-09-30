import importlib.util
from pathlib import Path

from src.schemas import (
    CulturalPathway,
    CulturalTriple,
    LayerLabel,
    PathwayCompleteness,
    PathwayEdgeRef,
)

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "16_eval_grounded_retrieval.py"
)
SPEC = importlib.util.spec_from_file_location("grounded_retrieval_eval", MODULE_PATH)
gr = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gr)


def _triple(tid: str, iid: str) -> CulturalTriple:
    return CulturalTriple(
        triple_id=tid,
        subject="s",
        predicate="p",
        object="o",
        instance_id=iid,
    )


def test_evaluate_ranked_units_hit_recall_and_rr():
    q = {
        "question_id": "easy_x",
        "fact_group_id": "fact_x",
        "support_instance_ids": ["r1", "r2"],
    }
    ranked = [
        (_triple("t0", "other"), 0.9),
        (_triple("t1", "r1"), 0.8),
        (_triple("t2", "r2"), 0.7),
    ]

    rows = gr.evaluate_ranked_units(
        method="raw_triples",
        question=q,
        ranked_units=ranked,
        get_instance_ids=gr.triple_instance_ids,
        ks=[1, 2, 3],
    )

    assert rows[0]["hit"] is False
    assert rows[0]["known_support_recall"] == 0.0
    assert rows[1]["hit"] is True
    assert rows[1]["known_support_recall"] == 0.5
    assert rows[1]["first_relevant_rank"] == 2
    assert rows[1]["rr_at_k"] == 0.5
    assert rows[2]["known_support_recall"] == 1.0


def test_pathway_relevance_uses_edge_instance_provenance():
    pw = CulturalPathway(
        pathway_id="pw1",
        culture="Japan",
        completeness=PathwayCompleteness.SURFACE,
        edges=[
            PathwayEdgeRef(
                triple_id="t1",
                source_entity_id="a",
                target_entity_id="b",
                source_label="a",
                target_label="b",
                predicate="p",
                source_layer=LayerLabel.LAYER_2_SOCIAL,
                target_layer=LayerLabel.LAYER_1_SURFACE,
                instance_id="support_record",
                cultural_group="Japan",
            )
        ],
        triple_ids=["t1"],
    )
    assert gr.pathway_instance_ids(pw) == {"support_record"}


def test_fact_group_macro_gives_duplicate_fact_one_vote():
    rows = [
        {
            "method": "raw_text",
            "k": 1,
            "fact_group_id": "fact_dup",
            "hit": True,
            "known_support_recall": 1.0,
            "rr_at_k": 1.0,
        },
        {
            "method": "raw_text",
            "k": 1,
            "fact_group_id": "fact_dup",
            "hit": True,
            "known_support_recall": 1.0,
            "rr_at_k": 1.0,
        },
        {
            "method": "raw_text",
            "k": 1,
            "fact_group_id": "fact_other",
            "hit": False,
            "known_support_recall": 0.0,
            "rr_at_k": 0.0,
        },
    ]

    question = gr.aggregate(rows, by_fact_group=False)[0]
    fact = gr.aggregate(rows, by_fact_group=True)[0]

    assert question["hit_at_k"] == 2 / 3
    assert fact["hit_at_k"] == 0.5
    assert question["n_units"] == 3
    assert fact["n_units"] == 2
