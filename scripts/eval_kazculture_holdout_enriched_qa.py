#!/usr/bin/env python3
"""Exploratory source-first Qwen QA on test rows overlapping the 150 holdout sources.

This is NOT the final KazCulture benchmark. It selects test questions solely by
exact source-passage overlap with the frozen 150-source diagnostic corpus, then
compares the frozen S0 predictions with S1 raw-source evidence and optional
same-source S2/S4R enrichment. Test answers never affect retrieval or condition
selection. Results estimate behavior on an overlap-only subset, not unseen-
source generalization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

from src.loaders.kazculture_loader import load_test
from src.loaders.kazculture_corpus import read_rows, passage_id
from src.retrieval.kazculture_index import MultilingualIndex, render_evidence
from src.evaluation.kazculture_rag import (
    V2EvidenceScorer, evidence_prompt, paired_statistics, prediction,
)


CONDITIONS=(
    "S1",
    "V21_S1_S2","V21_S1_S4R","V21_S1_S2_S4R",
    "V22_S1_S2","V22_S1_S4R","V22_S1_S2_S4R",
)


def _load_jsonl(path:Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_overlap(questions,test_rows,source_ids):
    if len(questions)!=len(test_rows):
        raise ValueError("Question/test-row alignment mismatch")
    selected=[]
    for q,row in zip(questions,test_rows):
        pid=passage_id(row.get("item_description") or "")
        if pid in source_ids:
            selected.append((q,pid))
    return selected


def same_source_units(index,passage_id_value,system):
    return sorted([
        u for u in index.units
        if u["dataset"]=="KazCulture" and not u["indirect"]
        and u["passage_id"]==passage_id_value and u["system"]==system
    ],key=lambda u:u["unit_id"])


def source_first_evidence(index,question,structured_systems=()):
    hits=index.search(question,system="S1",k=1,min_similarity=0.0)
    if not hits:
        return "",[],[]
    source=hits[0]
    structured=[]
    for system in structured_systems:
        structured.extend(same_source_units(index,source["passage_id"],system))
    evidence=render_evidence([source],max_chars=100000)
    if structured:
        evidence += "\n\n[OPTIONAL SAME-SOURCE STRUCTURED EVIDENCE]\n"
        evidence += render_evidence(structured,max_chars=100000)
    return evidence,[source],structured


def _condition(index21,index22,condition,q):
    if condition=="S1":
        return source_first_evidence(index21,q.question,())
    version,rest=condition.split("_",1)
    index=index21 if version=="V21" else index22
    systems=[]
    if "S2" in rest:
        systems.append("S2")
    if "S4R" in rest:
        systems.append("S4R")
    return source_first_evidence(index,q.question,systems)


def _trim_structured_to_context(scorer,q,evidence,source_hits,structured):
    rendered=evidence_prompt(scorer.tokenizer,q,evidence)
    n=len(scorer.tokenizer(rendered,add_special_tokens=False)["input_ids"])
    if n<=scorer.max_context_tokens:
        return evidence,structured,n,False
    # Structured evidence is optional. Drop it before ever changing the S1 source.
    if structured:
        source_only=render_evidence(source_hits,max_chars=100000)
        rendered=evidence_prompt(scorer.tokenizer,q,source_only)
        n=len(scorer.tokenizer(rendered,add_special_tokens=False)["input_ids"])
        if n<=scorer.max_context_tokens:
            return source_only,[],n,True
    raise ValueError(
        f"S1 source alone exceeds Qwen context for {q.question_id}; "
        "do not silently truncate a scientific evaluation source"
    )


def main():
    p=argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--test_file",type=Path,required=True)
    p.add_argument("--baseline_predictions",type=Path,required=True)
    p.add_argument("--index_v21",type=Path,required=True)
    p.add_argument("--index_v22",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen3-32B")
    p.add_argument("--model_revision",default="9216db5781bf21249d130ec9da846c4624c16137")
    p.add_argument("--conditions",default=",".join(CONDITIONS))
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args()

    conditions=[x.strip() for x in a.conditions.split(",") if x.strip()]
    if not conditions or set(conditions)-set(CONDITIONS) or len(set(conditions))!=len(conditions):
        raise ValueError(f"Invalid/duplicate conditions: {conditions}")

    questions,_=load_test(local_file=str(a.test_file))
    test_rows=read_rows(a.test_file)
    i21=MultilingualIndex(a.index_v21,device="cpu")
    i22=MultilingualIndex(a.index_v22,device="cpu")
    sources21={u["passage_id"] for u in i21.units if u["system"]=="S1"}
    sources22={u["passage_id"] for u in i22.units if u["system"]=="S1"}
    if len(sources21)!=150 or sources21!=sources22:
        raise ValueError("Expected identical 150-source S1 diagnostic indexes")

    chosen=select_overlap(questions,test_rows,sources21)
    if not chosen:
        raise ValueError("No test questions overlap the 150 held-out train sources")
    selected_questions=[q for q,_ in chosen]
    source_by_qid={q.question_id:pid for q,pid in chosen}

    availability={}
    for version,index in (("v21",i21),("v22",i22)):
        availability[version]={}
        for system in ("S2","S4R"):
            eligible={u["passage_id"] for u in index.units if u["system"]==system}
            availability[version][system]=sum(pid in eligible for _,pid in chosen)

    info={
        "overlap_questions":len(chosen),
        "unique_overlap_sources":len(set(source_by_qid.values())),
        "availability_question_counts":availability,
        "selection":"exact source passage ID only; no answer labels used",
    }
    if a.preflight:
        print(json.dumps(info,ensure_ascii=False,indent=2))
        return

    baseline=_load_jsonl(a.baseline_predictions)
    base_by_id={r["question_id"]:r for r in baseline}
    if len(base_by_id)!=len(baseline):
        raise ValueError("Duplicate baseline prediction IDs")
    if any(q.question_id not in base_by_id for q in selected_questions):
        raise ValueError("Frozen S0 baseline missing overlap question IDs")
    base=[base_by_id[q.question_id] for q in selected_questions]
    if any(r["answer_key"]!=q.answer_key for r,q in zip(base,selected_questions)):
        raise ValueError("Baseline/test gold mismatch")

    a.out.mkdir(parents=True,exist_ok=True)
    settings={
        **info,"conditions":conditions,"model":a.model,
        "model_revision":a.model_revision,
        "baseline":str(a.baseline_predictions.resolve()),
        "test_file":str(a.test_file.resolve()),
        "index_v21":str(a.index_v21.resolve()),
        "index_v22":str(a.index_v22.resolve()),
        "status":"exploratory_overlap_only_not_final_benchmark",
    }
    settings_path=a.out/"settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8"))!=settings:
            raise ValueError("Settings changed; use a new output directory")
    else:
        settings_path.write_text(json.dumps(settings,ensure_ascii=False,indent=2)+"\n",
                                 encoding="utf-8")

    scorer=None
    reports={}
    for condition in conditions:
        path=a.out/f"{condition}.jsonl"
        completed={}
        if path.exists():
            if not a.resume:
                raise FileExistsError(f"{path} exists; pass --resume")
            for row in _load_jsonl(path):
                completed[row["question_id"]]=row
        remaining=[q for q in selected_questions if q.question_id not in completed]
        if remaining and scorer is None:
            scorer=V2EvidenceScorer(a.model,model_revision=a.model_revision)
        if remaining:
            with path.open("a",encoding="utf-8") as out:
                for j,q in enumerate(remaining,1):
                    evidence,source_hits,structured=_condition(i21,i22,condition,q)
                    evidence,used_structured,tokens,dropped=_trim_structured_to_context(
                        scorer,q,evidence,source_hits,structured
                    )
                    scores=scorer.score_with_evidence(q,evidence)
                    row=prediction(q,scores,source_hits+used_structured,evidence,condition)
                    row.update({
                        "gold_source_passage_id":source_by_qid[q.question_id],
                        "top1_source_correct":bool(source_hits and source_hits[0]["passage_id"]==
                                                   source_by_qid[q.question_id]),
                        "structured_units_requested":len(structured),
                        "structured_units_used":len(used_structured),
                        "structured_dropped_for_context":dropped,
                        "prompt_tokens":tokens,
                    })
                    out.write(json.dumps(row,ensure_ascii=False)+"\n")
                    out.flush()
                    completed[q.question_id]=row
                    if j%10==0:
                        print(f"{condition}: {len(completed)}/{len(selected_questions)}",flush=True)
        ordered=[completed[q.question_id] for q in selected_questions]
        report=paired_statistics(base,ordered)
        report.update({
            "condition":condition,
            "top1_source_recall":sum(r["top1_source_correct"] for r in ordered)/len(ordered),
            "questions_with_structured_used":sum(r["structured_units_used"]>0 for r in ordered),
            "structured_dropped_for_context_n":sum(r["structured_dropped_for_context"]
                                                   for r in ordered),
        })
        reports[condition]=report
        (a.out/f"{condition}_report.json").write_text(
            json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
        )
        print(json.dumps({
            "condition":condition,
            "accuracy":report["system_accuracy"],
            "delta_pp":report["difference_pp"],
            "structured_q":report["questions_with_structured_used"],
        }),flush=True)

    final={"S0_subset_accuracy":sum(bool(x["correct"]) for x in base)/len(base),
           "n":len(base),"reports":reports,"settings":settings}
    (a.out/"comparison.json").write_text(
        json.dumps(final,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
    )
    print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
