#!/usr/bin/env python3
"""Score independently human-reviewed, version-blinded KG held-out audits.

Quality rates are descriptive precision estimates on sampled proposed triples
and pathways, never coverage of ALL source facts. A missing proposed pathway
does not count as a wrong proposed pathway. No automatic human labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def evaluate(rows, mapping, *, expected_per_version=50,
             min_triple_precision=0.8, min_pathway_precision=0.8,
             min_proposed_values=8):
    if len({r["audit_id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate blinded audit IDs")
    mapping_keys = mapping["keys"]
    if set(r["audit_id"] for r in rows)!=set(mapping_keys):
        raise ValueError("Audit/mapping ID mismatch: do not mix different samples")
    groups=defaultdict(list)
    for r in rows:
        version=mapping_keys[r["audit_id"]]["model_version"]
        groups[version].append(r)
        for field in ("factual_triple_correct_0_or_1",
                      "norm_warranted_by_source_0_or_1"):
            if r[field] not in ("0","1"):
                raise ValueError(f"Human review missing {field}: {r['audit_id']}")
        norm_exists=r["norm_proposed_0_or_1"]=="1"
        if r["norm_proposed_0_or_1"] not in ("0","1"):
            raise ValueError("Invalid proposed norm flag")
        if norm_exists:
            if r["proposed_norm_correct_0_or_1_if_present"] not in ("0","1"):
                raise ValueError(f"Missing proposed-norm human score: {r['audit_id']}")
        elif r["proposed_norm_correct_0_or_1_if_present"]!="":
            raise ValueError("Cannot grade a non-proposed norm")
        has_values=bool(r["schwartz_values_provisional"].strip())
        if has_values:
            if not norm_exists or r["proposed_values_correct_0_or_1_if_present"] not in ("0","1"):
                raise ValueError(f"Missing Schwartz human score: {r['audit_id']}")
        elif r["proposed_values_correct_0_or_1_if_present"]!="":
            raise ValueError("Cannot grade absent Schwartz values")
    results={}
    for name,rr in groups.items():
        if len(rr)!=expected_per_version:
            raise ValueError(f"{name}: expected {expected_per_version} reviewed rows")
        n=len(rr)
        norms=[r for r in rr if r["norm_proposed_0_or_1"]=="1"]
        values=[r for r in rr if r["schwartz_values_provisional"].strip()]
        # Precision requires correctness of the foundational triple, in addition
        # to the inferred norm and its Schwartz-value link.
        correct_triple=sum(r["factual_triple_correct_0_or_1"]=="1" for r in rr)
        correct_norm=sum(
            r["factual_triple_correct_0_or_1"]=="1"
            and r["norm_warranted_by_source_0_or_1"]=="1"
            and r["proposed_norm_correct_0_or_1_if_present"]=="1"
            for r in norms
        )
        correct_values=sum(
            r["factual_triple_correct_0_or_1"]=="1"
            and r["norm_warranted_by_source_0_or_1"]=="1"
            and r["proposed_norm_correct_0_or_1_if_present"]=="1"
            and r["proposed_values_correct_0_or_1_if_present"]=="1"
            for r in values
        )
        triple_precision=correct_triple/n
        val_precision=correct_values/len(values) if values else None
        results[name]={
            "reviewed_triples":n,
            "triple_precision":triple_precision,
            "human_normatively_eligible_count":sum(
                r["norm_warranted_by_source_0_or_1"]=="1" for r in rr),
            "proposed_norms_n":len(norms),
            "norm_precision_conditional_on_foundational_triple":
                correct_norm/len(norms) if norms else None,
            "proposed_values_n":len(values),
            "complete_pathway_precision":val_precision,
            "gate":("pass" if triple_precision>=min_triple_precision
                    and len(values)>=min_proposed_values
                    and val_precision is not None
                    and val_precision>=min_pathway_precision
                    else "not_enough_evidence_or_quality"),
        }
    if set(results)!={"v21","v22"}:
        raise ValueError("Expected blinded review from both v21 and v22")
    return {
        "source":"completed independent manual human annotations",
        "by_version":results,
        "limitations":[
            "Precision estimates for sampled generated candidates only",
            "Not source-level factual recall or cultural path recall",
            "Thresholds are project screening gates, not calibrated statistical conclusions",
            "Version-specific styles may remain recognizable despite hidden IDs",
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--audit",required=True,type=Path)
    p.add_argument("--mapping",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--expected_per_version",type=int,default=50)
    args=p.parse_args()
    with args.audit.open(encoding="utf-8",newline="") as f:
        rows=list(csv.DictReader(f))
    mapping=json.loads(args.mapping.read_text(encoding="utf-8"))
    result=evaluate(rows,mapping,expected_per_version=args.expected_per_version)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",
                        encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
