"""Offline tests for pilot-only train QA-to-source retrieval comparisons."""
import numpy as np
import pytest

from scripts.build_kazculture_holdout_index import verify_sources
from scripts.probe_kazculture_holdout_retrieval import (
    choose_questions, top_distinct_source_ids, score_ranks,
)


def sample(p):
    return {
        "passage_id":p,"text":f"Пассаж {p}",
        "dataset":"issai/KazCulture","split":"train",
        "sources":["source"],"categories":["customs"],
        "train_row_ids":[1],"culture":"Kazakh",
    }


def frame(p, version):
    return {
        **sample(p),"schema_version":version,
        "label_status":"candidate","triples":[],
        "pathways":[],"norm_candidates":[],
    }


def test_exact_source_match_and_train_only():
    p=[sample(f"p{i}") for i in range(150)]
    f=[frame(f"p{i}","kazculture-kg-nested-norms-v2.2")
       for i in range(150)]
    verify_sources(p,f,"v22")
    f[-1]["text"]="mismatched"
    with pytest.raises(ValueError,match="mismatch"):
        verify_sources(p,f,"v22")


def test_question_selection_uses_only_indexed_source_and_cap():
    qa=[
        {"passage_id":f"p{i%5}","question":f"question {i}","answer":"NOT EMBEDDED"}
        for i in range(50)
    ]+[{"passage_id":"test","question":"test question","answer":"FORBIDDEN"}]
    rows=choose_questions(qa,{"p0","p1","p2"},max_questions=100,
                          per_passage=2,seed=42)
    assert len(rows)==6
    assert {r["passage_id"] for r in rows}=={"p0","p1","p2"}
    assert all("test" not in r["question"] for r in rows)
    assert rows==choose_questions(qa,{"p0","p1","p2"},max_questions=100,
                                  per_passage=2,seed=42)


def test_rank_deduplicates_triples_within_passages():
    units=[
        {"system":"S2","unit_id":"S2:1","passage_id":"p1",
         "dataset":"KazCulture","indirect":False},
        {"system":"S2","unit_id":"S2:2","passage_id":"p1",
         "dataset":"KazCulture","indirect":False},
        {"system":"S2","unit_id":"S2:3","passage_id":"p2",
         "dataset":"KazCulture","indirect":False},
        {"system":"S1","unit_id":"S1:1","passage_id":"p1",
         "dataset":"KazCulture","indirect":False},
    ]
    scores=np.array([0.9,0.8,0.7,0.6],dtype=np.float32)
    assert top_distinct_source_ids(units,scores,system="S2")==["p1","p2"]
    assert top_distinct_source_ids(units,scores,system="S1")==["p1"]
    summary=score_ranks(units,np.asarray([scores],dtype=np.float32),
                        ["p2"],systems=("S2",))
    assert summary["S2"]["source_recall_at_1_all"]==0.0
    assert summary["S2"]["source_recall_at_3_all"]==1.0
    assert summary["S2"]["mrr_at_8_all"]==0.5


def test_conditional_coverage_distinct_from_global_recall():
    units=[{"system":"S2","unit_id":"S2:1","passage_id":"p1",
            "dataset":"KazCulture","indirect":False}]
    scores=np.array([[0.5],[0.8]],dtype=np.float32)
    result=score_ranks(units,scores,["p1","p2"],systems=("S2",))["S2"]
    assert result["n_questions_with_gold_source_available"]==1
    assert result["source_recall_at_5_all"]==0.5
    assert result["source_recall_at_5_conditional"]==1.0
    assert result["gold_source_availability"]==0.5
