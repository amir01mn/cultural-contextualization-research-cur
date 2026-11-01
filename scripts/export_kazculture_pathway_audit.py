#!/usr/bin/env python3
"""Export every proposed KazCulture V2/V2.1 cultural pathway for manual audit.

Separate from random triple audit: for small samples we inspect *all* proposed
pathways. Official review labels intentionally blank and never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.loaders.kazculture_corpus import read_rows


def pathway_audit_rows(frames: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for f in frames:
        for triple in f.get("triples", []):
            path = triple.get("pathway")
            if not path:
                continue
            tid = triple["triple_id"]
            if tid in seen:
                raise ValueError(f"Duplicate path/triple ID: {tid}")
            seen.add(tid)
            result.append({
                "triple_id": tid,
                "passage_id": f["passage_id"],
                "category": "; ".join(f.get("categories", [])),
                "source": "; ".join(f.get("sources", [])),
                "passage": f["text"],
                "subject": triple["subject"],
                "predicate": triple["predicate"],
                "object": triple["object"],
                "triple_quote": triple["evidence"],
                "rule_kind_provisional": path.get("rule_kind", ""),
                "norm_type_provisional": path.get("norm_type", ""),
                "norm_provisional": path.get("norm", ""),
                "values_provisional": "; ".join(path.get("values", [])),
                "norm_source_quote": path.get("evidence", ""),
                "triple_correct_0_or_1": "",
                "normatively_eligible_0_or_1": "",
                "pathway_correct_0_or_1": "",
                "reviewer_notes": "",
            })
    return result


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--frames", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    rows = pathway_audit_rows(read_rows(args.frames))
    if not rows:
        print("No proposed pathways: no audit file written.")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Never clobber an existing human annotation.
    with args.out.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "proposed_pathways": len(rows),
        "unique_source_passages": len({r["passage_id"] for r in rows}),
        "categories": dict(Counter(r["category"] for r in rows)),
        "review_columns_are_blank": True,
        "meaning": "Human judgment still required; quote match alone is not correctness.",
        "output_file": str(args.out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
