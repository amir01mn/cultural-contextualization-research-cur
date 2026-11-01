#!/usr/bin/env python3
"""Compare held-out v2.1 vs v2.2 extraction on exactly identical sources.

Coverage and parse reliability are purely mechanical. Human-reviewed triple
and norm precision MUST be assessed separately; more paths is not better.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.loaders.kazculture_corpus import read_rows


def describe(rows):
    n=len(rows)
    triples=sum(len(r.get("triples",[])) for r in rows)
    paths=sum(len(r.get("pathways",[])) for r in rows)
    return {
        "passages":n, "covered_passages":sum(bool(r.get("triples")) for r in rows),
        "passage_coverage":sum(bool(r.get("triples")) for r in rows)/n,
        "candidate_triples":triples, "proposed_schwartz_pathways":paths,
        "norm_only_candidates":sum(len(r.get("norm_candidates",[])) for r in rows)-paths
             if all("norm_candidates" in r for r in rows) else None,
        "generation_length_truncations":sum(x=="length" for r in rows
                                            for x in r.get("finish_reasons",[])),
        "invalid_json_segments":sum(x=="invalid_json" for r in rows
                                    for x in r.get("chunk_status",[])),
        "invalid_quotes":sum(r.get("rejections",{}).get("unverified_source_quote",0)
                             for r in rows),
        "invalid_predicates":sum(r.get("rejections",{}).get("invalid_predicate",0)
                                for r in rows),
        "orphan_pathways":sum(r.get("rejections",{}).get("orphan_pathway",0)
                             for r in rows),
    }


def compare(a,b):
    if not a or not b:
        raise ValueError("Both matched pilot inputs must have rows")
    m1={r["passage_id"]:r for r in a}
    m2={r["passage_id"]:r for r in b}
    if len(m1)!=len(a) or len(m2)!=len(b):
        raise ValueError("Duplicate passage IDs")
    if set(m1)!=set(m2):
        raise ValueError("Must compare identical 150 passage IDs")
    for k in m1:
        if m1[k]["text"]!=m2[k]["text"]:
            raise ValueError(f"Source text differs for {k}")
        if m1[k].get("split")!="train" or m2[k].get("split")!="train":
            raise ValueError("Not train-only source")
    if len(m1)!=150:
        raise ValueError(f"Expected 150 held-out sources, got {len(m1)}")
    return {
        "matched_source_passages":len(m1),
        "v21":describe(a),
        "v22":describe(b),
        "limitation":"No human-reviewed accuracy, no QA-performance claims.",
        "no_test_questions_used":True,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--v21",type=Path,required=True)
    p.add_argument("--v22",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=compare(read_rows(a.v21),read_rows(a.v22))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
