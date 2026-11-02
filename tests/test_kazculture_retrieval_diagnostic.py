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



def test_index_loader_preserves_unicode_line_separator_inside_json(tmp_path):
    """U+2028 in source text must not be mistaken for a JSONL record boundary."""
    import json
    import numpy as np
    from src.retrieval.kazculture_index import MultilingualIndex, MODEL_NAME

    folder=tmp_path/"index"
    folder.mkdir()
    unit={
        "unit_id":"S1:p1","passage_id":"p1","system":"S1",
        "dataset":"KazCulture","culture":"Kazakh","source":["source"],
        "category":["customs"],"values":[],"indirect":False,
        "text":"before\u2028after",
    }
    (folder/"units.jsonl").write_text(
        json.dumps(unit,ensure_ascii=False)+"\n",encoding="utf-8"
    )
    np.save(folder/"embeddings.npy",np.asarray([[1.0,0.0]],dtype=np.float32))
    (folder/"metadata.json").write_text(json.dumps({
        "embedding_model":MODEL_NAME,"n_units":1,"dimension":2
    }),encoding="utf-8")

    ix=MultilingualIndex(folder)
    assert len(ix.units)==1
    assert ix.units[0]["text"]=="before\u2028after"



def test_source_first_enrichment_preserves_s1_and_measures_attachability():
    from scripts.probe_kazculture_source_enrichment import summarize_source_first
    units=[
        {"system":"S1","passage_id":"p1","dataset":"KazCulture","indirect":False},
        {"system":"S1","passage_id":"p2","dataset":"KazCulture","indirect":False},
        {"system":"S2","passage_id":"p1","dataset":"KazCulture","indirect":False},
        {"system":"S4R","passage_id":"p2","dataset":"KazCulture","indirect":False},
    ]
    rankings=[["p1","p2"],["p1","p2"]]
    gold=["p1","p2"]
    out=summarize_source_first(units,rankings,gold,systems=("S2","S4R"))
    assert out["s1_recall_at_1"]==0.5
    assert out["s1_recall_at_5"]==1.0
    assert out["structured"]["S2"]["gold_structured_availability_all"]==0.5
    assert out["structured"]["S2"]["gold_enrichment_rate_conditional_on_s1_hit_at_5"]==0.5
    assert out["structured"]["S4R"]["gold_enrichment_rate_conditional_on_s1_hit_at_5"]==0.5



def test_overlap_qwen_diagnostic_selects_by_source_not_answer():
    from types import SimpleNamespace
    from scripts.eval_kazculture_holdout_enriched_qa import select_overlap
    from src.loaders.kazculture_corpus import passage_id

    questions=[SimpleNamespace(question_id="q1"),SimpleNamespace(question_id="q2")]
    rows=[
        {"item_description":"Source A","answer_label":"D"},
        {"item_description":"Source B","answer_label":"A"},
    ]
    selected=select_overlap(questions,rows,{passage_id("Source B")})
    assert [(q.question_id,pid) for q,pid in selected]==[
        ("q2",passage_id("Source B"))
    ]


def test_source_first_qwen_condition_freezes_s1_and_attaches_other_index():
    from scripts.eval_kazculture_holdout_enriched_qa import source_first_evidence

    source={
        "unit_id":"S1:p1","system":"S1","passage_id":"p1",
        "dataset":"KazCulture","indirect":False,"source":["book"],
        "text":"raw passage","culture":"Kazakh","category":["customs"],"values":[],
    }
    fact={
        "unit_id":"S2:t1","system":"S2","passage_id":"p1",
        "dataset":"KazCulture","indirect":False,"source":["book"],
        "text":"fact relation","culture":"Kazakh","category":["customs"],"values":[],
    }
    class S1Index:
        units=[source]
        def search(self,query,**kwargs):
            assert kwargs["system"]=="S1"
            return [{**source,"similarity":0.9}]
    class StructuredIndex:
        units=[fact]

    evidence,hits,structured=source_first_evidence(
        S1Index(),"question",StructuredIndex(),("S2",)
    )
    assert [h["passage_id"] for h in hits]==["p1"]
    assert [u["unit_id"] for u in structured]==["S2:t1"]
    assert "raw passage" in evidence and "fact relation" in evidence



def test_same_source_structured_units_have_explicit_nonretrieval_score():
    from scripts.eval_kazculture_holdout_enriched_qa import same_source_units

    class Index:
        units=[{
            "unit_id":"S2:t1","system":"S2","passage_id":"p1",
            "dataset":"KazCulture","indirect":False,"source":["book"],
            "text":"fact","culture":"Kazakh","category":["customs"],"values":[],
        }]

    rows=same_source_units(Index(),"p1","S2")
    assert len(rows)==1
    assert rows[0]["similarity"] is None
    assert rows[0]["attachment_mode"]=="same_source_provenance"
