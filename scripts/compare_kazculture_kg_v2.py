#!/usr/bin/env python3
"""Matched-source extraction coverage comparison: original KG pilot versus v2.

This compares source IDs, candidate-triple counts, parsed-pathway coverage and
failure modes. It CANNOT establish factual correctness without blinded human
review; v1 and v2 output differ and neither is labeled ground truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.loaders.kazculture_corpus import read_rows


def stats(rows: list[dict], *, version: str) -> dict:
    total = sum(len(row.get("triples") or []) for row in rows)
    if version == "v2":
        paths = sum(len(row.get("pathways") or []) for row in rows)
        truncated = sum(x == "length" for row in rows
                        for x in row.get("finish_reasons", []))
        invalid_json = sum(x == "invalid_json" for row in rows
                           for x in row.get("chunk_status", []))
    else:
        # Old v1 did not separate valid pathways from factual triples.
        # Counting triples with a norm AND norm type AND values is an
        # approximate descriptive proxy, not an equivalent v2 pathway count.
        paths = sum(
            bool(t.get("chunk_norm", row.get("norm"))) and
            bool(t.get("chunk_norm_type", row.get("norm_type"))) and
            bool(t.get("chunk_values", row.get("values")))
            for row in rows for t in row.get("triples", [])
        )
        truncated = None
        invalid_json = None
    return {
        "n_passages": len(rows),
        "frames_with_any_triple": sum(bool(row.get("triples")) for row in rows),
        "candidate_triples": total,
        "candidate_triples_per_passage": total / len(rows),
        "source_coverage": sum(bool(row.get("triples")) for row in rows) / len(rows),
        "pathway_count_or_v1_proxy": paths,
        "pathway_fraction_of_triples_or_proxy": paths / total if total else None,
        "generation_length_truncations": truncated,
        "invalid_json_segments": invalid_json,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--old_frames", type=Path, required=True)
    p.add_argument("--new_frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    old_rows = read_rows(args.old_frames)
    new_rows = read_rows(args.new_frames)
    old = {row["passage_id"]: row for row in old_rows}
    if len(old) != len(old_rows):
        raise ValueError("Duplicate v1 source passage ID")
    new = {row["passage_id"]: row for row in new_rows}
    if len(new) != len(new_rows):
        raise ValueError("Duplicate v2 source passage ID")
    if not new or not set(new) <= set(old):
        raise ValueError("V2 pilot must be a nonempty subset of original pilot source IDs")
    matched_ids = [row["passage_id"] for row in new_rows]
    matched_old = [old[pid] for pid in matched_ids]
    report = {
        "matched_source_passages": len(matched_ids),
        "v1": stats(matched_old, version="v1"),
        "v2": stats(new_rows, version="v2"),
        "interpretation": (
            "Matched source IDs, not a human-labeled accuracy comparison. "
            "V1 norm/value proxy is not equivalent to a verified V2 pathway. "
            "Both versions' extracted triples are unreviewed."
        ),
    }
    report["changes"] = {
        "candidate_triple_count": (
            report["v2"]["candidate_triples"] - report["v1"]["candidate_triples"]
        ),
        "covered_passage_count": (
            report["v2"]["frames_with_any_triple"] -
            report["v1"]["frames_with_any_triple"]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
