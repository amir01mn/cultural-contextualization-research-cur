"""Offline tests for frozen nested-norm v2.2 on blind held-out source material."""
import json
import pytest
from src.construction.kazculture_kg_v22 import (
    aggregate_v22,SCHEMA_VERSION,SYSTEM_V22,
)
from scripts.build_kazculture_heldout_audit import rows_for_review
from scripts.compare_kazculture_kg_holdout import compare


def passage(text,uid="p0"):
    return {
        "passage_id":uid,"text":text,"dataset":"issai/KazCulture",
        "split":"train","sources":["Kazakh folklore encyclopedia"],
        "categories":["Customs"],"culture":"Kazakh","train_row_ids":[0],
    }


def parse(psg, items):
    seg={**psg,"_chunk_start":0,"_chunk_end":len(psg["text"])}
    raw=json.dumps({"triples":items},ensure_ascii=False)
    return aggregate_v22(psg,[seg],[raw],model="fake")


def fact(quote,predicate="PRACTICED_BY",**extra):
    return {
        "subject":"Үлкендерге құрмет","predicate":predicate,
        "object":"жастар","evidence":quote,"norm":None,**extra,
    }


def custom(quote,values=None):
    return {
        "behavior":"үлкендерге құрмет көрсету",
        "norm":"үлкендерге құрмет көрсетіледі",
        "norm_type":"elder respect custom",
        "rule_kind":"explicit_custom","evidence":quote,
        "values":[] if values is None else values,
    }


def test_fact_without_norm_is_flat_only():
    p=passage("Садақтың кермесі малдың сіңірінен жасалады.")
    output=parse(p,[{
        "subject":"Садақтың кермесі","predicate":"MADE_FROM",
        "object":"малдың сіңірі","evidence":p["text"],"norm":None,
    }])
    assert output["schema_version"]==SCHEMA_VERSION
    assert len(output["triples"])==1 and output["pathways"]==[]
    assert output["norm_candidates"]==[]
    assert output["triples"][0]["knowledge_kind"]=="factual"


def test_norm_only_is_not_forced_into_schwartz_l3():
    p=passage("Үлкендерге құрмет көрсету керек.")
    row=fact(p["text"])
    row["norm"]=custom(p["text"])
    out=parse(p,[row])
    assert len(out["triples"])==1
    assert len(out["norm_candidates"])==1
    assert out["pathways"]==[]
    assert out["triples"][0]["knowledge_kind"]=="norm_only"
    assert out["triples"][0]["pathway"] is None


def test_explicit_value_stays_provisional():
    p=passage("Үлкендерге құрмет көрсету керек.")
    row=fact(p["text"])
    row["norm"]=custom(p["text"],["tradition"])
    out=parse(p,[row])
    assert len(out["pathways"])==1
    assert out["pathways"][0]["values"]==["tradition"]
    assert out["pathways"][0]["review_status"]=="unreviewed"
    assert out["pathways"][0]["provisional"]


def test_descriptive_material_fact_must_not_generate_norm():
    p=passage("Оқшантай теріден жасалады.")
    out=parse(p,[{
        "subject":"Оқшантай","predicate":"MADE_FROM","object":"тері",
        "evidence":p["text"],"norm":custom(p["text"],["tradition"]),
    }])
    assert len(out["triples"])==1
    assert out["pathways"]==[] and out["norm_candidates"]==[]
    assert out["rejections"]["norm_on_descriptive_predicate"]==1


def test_unsupported_source_quote_discards_attached_norm_together():
    p=passage("Үлкендерге құрмет көрсету керек.")
    out=parse(p,[fact("completely fabricated quote",
                           norm=custom(p["text"],["tradition"]))])
    assert not out["triples"] and not out["pathways"]
    assert out["rejections"]["unverified_source_quote"]==1
    assert "orphan_pathway" not in out["rejections"]


def test_invalid_norm_quote_does_not_discard_valid_triple():
    p=passage("Үлкендерге құрмет көрсету керек.")
    out=parse(p,[fact(p["text"],norm=custom("fake norm quote",["tradition"]))])
    assert len(out["triples"])==1 and not out["norm_candidates"]
    assert out["rejections"]["unsupported_norm_quote_or_schema"]==1


def test_false_source_title_becomes_metadata_rejection():
    p=passage("Садақтың кермесі малдың сіңірінен жасалады.")
    out=parse(p,[{
        "subject":"Kazakh folklore encyclopedia",
        "predicate":"IS_A","object":"садақ","evidence":p["text"],
    }])
    assert not out["triples"]
    assert out["rejections"]["metadata_subject"]==1


def test_max_three_without_cross_chunk_orphans():
    p=passage("Үлкендерге құрмет көрсету керек.")
    items=[]
    for i in range(5):
        t=fact(p["text"],subject=f"жастар {i}")
        t["norm"]=custom(p["text"],["tradition"])
        items.append(t)
    out=parse(p,items)
    assert len(out["triples"])==3
    assert len(out["pathways"])==3
    assert out["rejections"]["over_max_triples"]==2


def test_heldout_rejects_unmatched_sources_and_blinds_versions():
    v21=[]
    v22=[]
    for i in range(150):
        p=passage(f"Үлкендерге құрмет көрсету керек. {i}",f"p{i}")
        r21=parse(p,[fact(p["text"])])
        r21["schema_version"]="kazculture-kg-factual-normative-v2.1"
        r22=parse(p,[fact(p["text"])])
        v21.append(r21)
        v22.append(r22)
    c=compare(v21,v22)
    assert c["matched_source_passages"]==150
    rows,key=rows_for_review(v21,v22,per_version=20)
    assert len(rows)==40 and len(key)==40
    assert all("model_version" not in row for row in rows)
    assert all(row["factual_triple_correct_0_or_1"]=="" for row in rows)
    v22[-1]["text"]="Different source"
    with pytest.raises(ValueError,match="Source text differs"):
        compare(v21,v22)


def test_prompt_explicitly_disallows_standalone_path_array():
    assert "no triple_index" in SYSTEM_V22
    assert "values=[]" in SYSTEM_V22
