#!/usr/bin/env python3
"""Train-only, source-grounded retrieval diagnostic for V2.1 vs V2.2.

Uses the *same* sampled Kazakh training questions and BGE-M3 embeddings for
both graph versions; gold is the original TRAIN source passage ID.
Do NOT interpret self-source recall as unseen-document generalization or
QA answer accuracy. Gold answers/labels never enter embeddings/queries.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

from src.loaders.kazculture_corpus import read_rows
from src.retrieval.kazculture_index import MultilingualIndex

SYSTEMS=("S1","S2","S3","S4F","S4R")
KS=(1,3,5,8)


def choose_questions(qa, source_ids, *, max_questions=300,
                     per_passage=3, seed=42):
    if max_questions<1 or per_passage<1:
        raise ValueError("Positive question limits required")
    grouped=defaultdict(list)
    for row in qa:
        if row.get("passage_id") not in source_ids:
            continue
        if not isinstance(row.get("question"),str) or not row["question"].strip():
            continue
        grouped[row["passage_id"]].append(row)
    rng=random.Random(seed)
    selected=[]
    for source_id in sorted(grouped):
        sampled=grouped[source_id][:]
        rng.shuffle(sampled)
        selected.extend(sampled[:per_passage])
    rng.shuffle(selected)
    return selected[:max_questions]


def top_distinct_source_ids(units, similarities, *, system, k=8):
    """Align ranking with MultilingualIndex.search: score, unit ID, unique source."""
    if system not in SYSTEMS:
        raise ValueError("Unknown evidence system")
    indices=[
        i for i,u in enumerate(units) if u["system"]==system
        and u["dataset"]=="KazCulture" and not u["indirect"]
        and np.isfinite(similarities[i])
    ]
    indices.sort(key=lambda i:(-float(similarities[i]),units[i]["unit_id"]))
    ranked=[]
    seen=set()
    for i in indices:
        pid=units[i]["passage_id"]
        if pid not in seen:
            seen.add(pid)
            ranked.append(pid)
            if len(ranked)>=k:
                break
    return ranked


def score_ranks(units, score_matrix, gold_ids, *, systems=SYSTEMS):
    if len(score_matrix)!=len(gold_ids):
        raise ValueError("Query count mismatch")
    if score_matrix.ndim!=2 or score_matrix.shape[1]!=len(units):
        raise ValueError("Incorrect similarity matrix shape")
    n=len(gold_ids)
    stats={}
    for system in systems:
        available={u["passage_id"] for u in units if u["system"]==system
                   and u["dataset"]=="KazCulture" and not u["indirect"]}
        supported=[i for i,gold in enumerate(gold_ids) if gold in available]
        ranks=[]
        for i,gold in enumerate(gold_ids):
            matches=top_distinct_source_ids(units,score_matrix[i],system=system,k=8)
            ranks.append(matches.index(gold)+1 if gold in matches else None)
        within=lambda k:sum(rank is not None and rank<=k for rank in ranks)
        result={
            "n_questions":n,
            "n_sources_with_indexed_evidence":len(available),
            "n_questions_with_gold_source_available":len(supported),
            "gold_source_availability":len(supported)/n if n else None,
            "mrr_at_8_all":sum(1/rank for rank in ranks if rank is not None)/n if n else None,
        }
        for k in KS:
            result[f"source_recall_at_{k}_all"]=within(k)/n if n else None
            result[f"source_recall_at_{k}_conditional"]=(
                within(k)/len(supported) if supported else None
            )
        stats[system]=result
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--index_v21",type=Path,required=True)
    parser.add_argument("--index_v22",type=Path,required=True)
    parser.add_argument("--train_dev_qa",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--max_questions",type=int,default=300)
    parser.add_argument("--per_passage",type=int,default=3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--batch_size",type=int,default=16)
    a=parser.parse_args()
    if a.out.exists():
        raise FileExistsError("Diagnostic report exists; never overwrite frozen results")

    i21=MultilingualIndex(a.index_v21,device=a.device)
    i22=MultilingualIndex(a.index_v22,device=a.device)
    def manifest(directory, version):
        path=directory/"diagnostic_manifest.json"
        d=json.loads(path.read_text(encoding="utf-8"))
        if d.get("status")!="exploratory_unreviewed_do_not_report_as_human_validated":
            raise ValueError("Not a diagnostic-only index")
        if d.get("method")!=version:
            raise ValueError("Version/index metadata mismatch")
        if d.get("source_count")!=150:
            raise ValueError("Expected 150 held-out sources")
        return d
    m21=manifest(a.index_v21,"v21")
    m22=manifest(a.index_v22,"v22")
    if m21["source_input_sha256"]!=m22["source_input_sha256"]:
        raise ValueError("V2.1 and V2.2 indexes come from different source files")
    sources21={u["passage_id"] for u in i21.units if u["system"]=="S1"}
    sources22={u["passage_id"] for u in i22.units if u["system"]=="S1"}
    if len(sources21)!=150 or sources21!=sources22:
        raise ValueError("Different held-out source ID sets")

    qa=read_rows(a.train_dev_qa)
    selected=choose_questions(qa,sources21,max_questions=a.max_questions,
                              per_passage=a.per_passage,seed=a.seed)
    if len(selected)<30:
        raise ValueError(f"Only {len(selected)} valid train-source questions")
    questions=[r["question"] for r in selected]
    gold_ids=[r["passage_id"] for r in selected]

    # Reuse same normalized query embedding for both methods; all evidence
    # was built separately and no answers enter the retrieval computation.
    model=i21._load_model()
    vec=np.asarray(model.encode(
        questions,batch_size=a.batch_size,show_progress_bar=False,
        normalize_embeddings=True,convert_to_numpy=True),dtype=np.float32)
    if len(vec)!=len(questions) or not np.isfinite(vec).all():
        raise ValueError("Non-finite or incomplete query vectors")
    results={}
    for method,index in (("v21",i21),("v22",i22)):
        scores=np.asarray(vec @ np.asarray(index.emb,dtype=np.float32).T,
                          dtype=np.float32)
        results[method]=score_ranks(index.units,scores,gold_ids)
    compare={}
    for system in SYSTEMS:
        compare[system]={
            "v22_minus_v21_source_recall_at_5_all":(
                results["v22"][system]["source_recall_at_5_all"]
                -results["v21"][system]["source_recall_at_5_all"]
            ),
            "v22_minus_v21_source_recall_at_5_conditional":(
                results["v22"][system]["source_recall_at_5_conditional"]
                -results["v21"][system]["source_recall_at_5_conditional"]
            ) if results["v22"][system]["source_recall_at_5_conditional"] is not None
                and results["v21"][system]["source_recall_at_5_conditional"] is not None
                else None,
        }
    report={
        "protocol":"train-only self-source recall diagnostic v1",
        "selected_queries":len(selected),
        "unique_gold_passages":len(set(gold_ids)),
        "source_corpus_size":len(sources21),
        "question_selection":{"seed":a.seed,"max_questions":a.max_questions,
                              "per_passage":a.per_passage},
        "systems":results,
        "comparison":compare,
        "status":"exploratory_candidate_KG_unreviewed",
        "interpretation":[
            "This tests retrieving an indexed source given its own train QA question.",
            "This does NOT test held-out unseen passages or generative QA accuracy.",
            "Value/norm labels and graph triples remain unreviewed candidates.",
            "Do not tune thresholds or decide final KG model using test answers.",
        ],
    }
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                     encoding="utf-8")
    print(json.dumps({
        "selected_queries":len(selected),
        "unique_gold_passages":len(set(gold_ids)),
        "S1_v21_R5":results["v21"]["S1"]["source_recall_at_5_all"],
        "S1_v22_R5":results["v22"]["S1"]["source_recall_at_5_all"],
        "S2_v21_R5":results["v21"]["S2"]["source_recall_at_5_all"],
        "S2_v22_R5":results["v22"]["S2"]["source_recall_at_5_all"],
        "saved":str(a.out),
    },ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
