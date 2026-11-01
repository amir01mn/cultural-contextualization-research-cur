#!/usr/bin/env python3
"""Create version-blinded manual candidate audits for frozen held-out pilots.

Reviewers label factual accuracy, whether a norm is warranted, and whether
model-proposed norm and value are supported. The audit excludes version labels
and hides a matching key in a separate file; never auto-populates human labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from src.loaders.kazculture_corpus import read_rows
from scripts.compare_kazculture_kg_holdout import compare


def rows_for_review(v21, v22, *, per_version=50, seed=42):
    # Validate exact same source data, 150 held-out passages.
    compare(v21,v22)
    rng=random.Random(seed)
    collected=[]
    key={}
    for system,frames in (("v21",v21),("v22",v22)):
        available=[(f,t) for f in frames for t in f.get("triples",[])]
        if len(available)<per_version:
            raise ValueError(f"{system}: only {len(available)} triples, need {per_version}")
        sampled=rng.sample(available,per_version)
        for frame,t in sampled:
            norm = t.get("norm_candidate") or t.get("pathway") or {}
            path = t.get("pathway") or {}
            original = f"{system}:{t['triple_id']}"
            blind_id = "review_" + hashlib.sha256((original + f":{seed}").encode()).hexdigest()[:12]
            if blind_id in key:
                raise ValueError("Audit ID collision")
            key[blind_id]={
                "model_version":system,
                "triple_id":t["triple_id"],
                "passage_id":frame["passage_id"],
            }
            collected.append({
                "audit_id":blind_id,
                "passage":frame["text"],
                "subject":t["subject"],
                "predicate":t["predicate"],
                "object":t["object"],
                "factual_source_quote":t["evidence"],
                "norm_proposed_0_or_1":int(bool(norm)),
                "norm_type_provisional":norm.get("norm_type",""),
                "norm_provisional":norm.get("norm",""),
                "norm_evidence_quote":norm.get("evidence",""),
                "rule_kind_provisional":norm.get("rule_kind",""),
                "schwartz_values_provisional":"; ".join(path.get("values",[])),
                "factual_triple_correct_0_or_1":"",
                "norm_warranted_by_source_0_or_1":"",
                "proposed_norm_correct_0_or_1_if_present":"",
                "proposed_values_correct_0_or_1_if_present":"",
                "reviewer_notes":"",
            })
    rng.shuffle(collected)
    return collected,key


def main():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--v21",required=True,type=Path)
    p.add_argument("--v22",required=True,type=Path)
    p.add_argument("--audit",required=True,type=Path)
    p.add_argument("--mapping",required=True,type=Path)
    p.add_argument("--sample_per_version",type=int,default=50)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.sample_per_version<1:
        raise ValueError("sample_per_version must be positive")
    if args.audit==args.mapping:
        raise ValueError("Review and mapping files must be separate")
    if args.audit.exists() or args.mapping.exists():
        raise FileExistsError("Existing audit/mapping: do not overwrite reviewer work")
    rows,key=rows_for_review(
        read_rows(args.v21),read_rows(args.v22),
        per_version=args.sample_per_version,seed=args.seed,
    )
    args.audit.parent.mkdir(parents=True,exist_ok=True)
    args.mapping.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create, never overwrite annotations.
    with args.mapping.open("x",encoding="utf-8") as out:
        json.dump({"seed":args.seed,"sample_size":len(rows),"keys":key},
                  out,ensure_ascii=False,indent=2)
        out.write("\n")
    with args.audit.open("x",encoding="utf-8",newline="") as out:
        w=csv.DictWriter(out,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({
        "rows_to_human_review":len(rows),
        "versions_hidden_from_audit":True,
        "audit_csv":str(args.audit),
        "unblinding_mapping_keep_separate":str(args.mapping),
        "human_score_columns_blank":True,
        "note":"This samples proposed triples only; does not measure missed source facts.",
    },indent=2))


if __name__=="__main__":
    main()
