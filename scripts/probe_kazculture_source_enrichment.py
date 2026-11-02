#!/usr/bin/env python3
"""Source-first enrichment diagnostic for held-out KazCulture V2.1/V2.2.

Raw S1 passages remain the retrieval backbone. Structured evidence is never
allowed to remove or rerank an S1 source in this diagnostic. We measure how
often S2/S3/S4F/S4R evidence is available to enrich sources already retrieved
by S1. This is train-only self-source analysis, not QA accuracy or unseen-
passage generalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

from src.loaders.kazculture_corpus import read_rows
from src.retrieval.kazculture_index import MultilingualIndex
from scripts.probe_kazculture_holdout_retrieval import (
    choose_questions, top_distinct_source_ids,
)

STRUCTURED=("S2","S3","S4F","S4R")


def summarize_source_first(units, s1_rankings, gold_ids, *, systems=STRUCTURED):
    if len(s1_rankings)!=len(gold_ids):
        raise ValueError("Ranking/gold mismatch")
    n=len(gold_ids)
    if not n:
        raise ValueError("No questions")
    hit1=[bool(r) and r[0]==g for r,g in zip(s1_rankings,gold_ids)]
    hit5=[g in r[:5] for r,g in zip(s1_rankings,gold_ids)]
    out={
        "n_questions":n,
        "s1_recall_at_1":sum(hit1)/n,
        "s1_recall_at_5":sum(hit5)/n,
        "structured":{},
    }
    for system in systems:
        available={u["passage_id"] for u in units
                   if u["system"]==system and u["dataset"]=="KazCulture"
                   and not u["indirect"]}
        gold_available=[g in available for g in gold_ids]
        top1_enrichable=[bool(r) and r[0] in available for r in s1_rankings]
        correct_top1=sum(a and b for a,b in zip(hit1,gold_available))
        s1_hit5_n=sum(hit5)
        enriched_hit5=sum(a and b for a,b in zip(hit5,gold_available))
        out["structured"][system]={
            "indexed_source_count":len(available),
            "gold_structured_availability_all":sum(gold_available)/n,
            "top1_retrieved_source_enrichment_rate":sum(top1_enrichable)/n,
            "correct_top1_with_structured_evidence_rate_all":correct_top1/n,
            "correct_top1_enrichment_rate_conditional":(
                correct_top1/sum(hit1) if any(hit1) else None
            ),
            "s1_hit_at_5_and_gold_structured_rate_all":enriched_hit5/n,
            "gold_enrichment_rate_conditional_on_s1_hit_at_5":(
                enriched_hit5/s1_hit5_n if s1_hit5_n else None
            ),
        }
    return out


def _manifest(directory:Path, version:str):
    d=json.loads((directory/"diagnostic_manifest.json").read_text(encoding="utf-8"))
    if d.get("status")!="exploratory_unreviewed_do_not_report_as_human_validated":
        raise ValueError("Not a diagnostic-only index")
    if d.get("method")!=version or d.get("source_count")!=150:
        raise ValueError("Version/source metadata mismatch")
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--index_v21",type=Path,required=True)
    p.add_argument("--index_v22",type=Path,required=True)
    p.add_argument("--train_dev_qa",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max_questions",type=int,default=300)
    p.add_argument("--per_passage",type=int,default=3)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",default="cpu")
    p.add_argument("--batch_size",type=int,default=16)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("Diagnostic report exists; use a new versioned output")

    i21=MultilingualIndex(a.index_v21,device=a.device)
    i22=MultilingualIndex(a.index_v22,device=a.device)
    m21=_manifest(a.index_v21,"v21")
    m22=_manifest(a.index_v22,"v22")
    if m21["source_input_sha256"]!=m22["source_input_sha256"]:
        raise ValueError("Indexes use different source passages")

    s1_sources={u["passage_id"] for u in i21.units if u["system"]=="S1"}
    if len(s1_sources)!=150 or s1_sources!={u["passage_id"] for u in i22.units
                                           if u["system"]=="S1"}:
        raise ValueError("S1 source sets differ")

    qa=read_rows(a.train_dev_qa)
    selected=choose_questions(
        qa,s1_sources,max_questions=a.max_questions,
        per_passage=a.per_passage,seed=a.seed,
    )
    if len(selected)<30:
        raise ValueError("Insufficient source-linked train questions")
    questions=[r["question"] for r in selected]
    gold=[r["passage_id"] for r in selected]

    model=i21._load_model()
    vec=np.asarray(model.encode(
        questions,batch_size=a.batch_size,show_progress_bar=False,
        normalize_embeddings=True,convert_to_numpy=True
    ),dtype=np.float32)
    scores=np.asarray(vec @ np.asarray(i21.emb,dtype=np.float32).T,dtype=np.float32)
    rankings=[
        top_distinct_source_ids(i21.units,scores[i],system="S1",k=8)
        for i in range(len(questions))
    ]

    results={
        "v21":summarize_source_first(i21.units,rankings,gold),
        "v22":summarize_source_first(i22.units,rankings,gold),
    }
    report={
        "protocol":"train-only source-first enrichment diagnostic v1",
        "selected_queries":len(selected),
        "unique_gold_passages":len(set(gold)),
        "source_corpus_size":len(s1_sources),
        "question_selection":{
            "seed":a.seed,"max_questions":a.max_questions,
            "per_passage":a.per_passage,
        },
        "design":"S1 ranking frozen; structured KG evidence only attaches to retrieved source IDs",
        "systems":results,
        "status":"exploratory_candidate_KG_unreviewed",
        "interpretation":[
            "S1 source ranking is unchanged by structured evidence.",
            "Metrics measure enrichment availability, not QA improvement.",
            "All queries are train QA linked to indexed train sources.",
            "Candidate triples/norms/values are not independently human validated.",
        ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "selected_queries":len(selected),
        "s1_R5":results["v21"]["s1_recall_at_5"],
        "v21_S2_gold_enrichment_on_S1_R5":
            results["v21"]["structured"]["S2"]["gold_enrichment_rate_conditional_on_s1_hit_at_5"],
        "v22_S2_gold_enrichment_on_S1_R5":
            results["v22"]["structured"]["S2"]["gold_enrichment_rate_conditional_on_s1_hit_at_5"],
        "v21_S4R_gold_enrichment_on_S1_R5":
            results["v21"]["structured"]["S4R"]["gold_enrichment_rate_conditional_on_s1_hit_at_5"],
        "v22_S4R_gold_enrichment_on_S1_R5":
            results["v22"]["structured"]["S4R"]["gold_enrichment_rate_conditional_on_s1_hit_at_5"],
        "saved":str(a.out),
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
