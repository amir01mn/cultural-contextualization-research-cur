#!/usr/bin/env python3
"""Compare KG v1, v2 and v2.1 on EXACTLY matched source passages.

Candidate quantity is descriptive. Semantic accuracy and reliability remain
unknown until an independent human review, and unmatched samples are rejected.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.loaders.kazculture_corpus import read_rows


def metrics(rows: list[dict], version: str) -> dict:
    n = len(rows)
    triples = sum(len(x.get("triples", [])) for x in rows)
    paths = sum(
        len(x.get("pathways", [])) for x in rows
    ) if version != "v1" else None
    result = {
        "n_passages": n,
        "passages_with_triples": sum(bool(x.get("triples")) for x in rows),
        "coverage": sum(bool(x.get("triples")) for x in rows) / n,
        "candidate_triples": triples,
        "triples_per_passage": triples / n,
        "proposed_pathways": paths,
        "pathways_per_triple": paths / triples if paths is not None and triples else None,
        "invalid_json_segments": (
            sum(s == "invalid_json" for row in rows
                for s in row.get("chunk_status", []))
            if version != "v1" else None
        ),
        "length_truncations": (
            sum(s == "length" for row in rows
                for s in row.get("finish_reasons", []))
            if version != "v1" else None
        ),
    }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--v2", type=Path, required=True)
    p.add_argument("--v21", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    catalogs = {}
    for key, file in [("v1", a.v1), ("v2", a.v2), ("v21", a.v21)]:
        rows = read_rows(file)
        mapping = {x["passage_id"]: x for x in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"{key} has duplicate passage IDs")
        catalogs[key] = mapping
    selected = list(catalogs["v21"])
    if not selected or any(not set(selected) <= set(catalogs[name])
                           for name in ("v1", "v2")):
        raise ValueError("V2.1 must be a nonempty same-source subset of v1 and v2")
    output = {
        "matched_sources": len(selected),
        "v1": metrics([catalogs["v1"][i] for i in selected], "v1"),
        "v2": metrics([catalogs["v2"][i] for i in selected], "v2"),
        "v21": metrics([catalogs["v21"][i] for i in selected], "v21"),
        "measurement_scope": (
            "Counts and extraction failures only. Does NOT measure correctness; "
            "V1 pathways were not proposed by the same formal schema. "
            "Samples revised using v2 audit observations are not blind validation."
        ),
    }
    output["v21_vs_v2"] = {
        "candidate_triple_difference": (
            output["v21"]["candidate_triples"] -
            output["v2"]["candidate_triples"]
        ),
        "coverage_difference_pp": round(
            100 * (output["v21"]["coverage"] - output["v2"]["coverage"]), 2
        ),
        "length_truncations_difference": (
            output["v21"]["length_truncations"] -
            output["v2"]["length_truncations"]
        ),
        "invalid_json_difference": (
            output["v21"]["invalid_json_segments"] -
            output["v2"]["invalid_json_segments"]
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
