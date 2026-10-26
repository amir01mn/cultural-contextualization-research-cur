#!/usr/bin/env python3
"""Build frozen multilingual KazCulture retrieval index after human pilot audit."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.construction.kazculture_kg import pathways, validate_audit
from src.loaders.kazculture_corpus import read_rows, write_jsonl, validate_no_test_content
from src.retrieval.kazculture_index import MultilingualIndex, make_units


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--passages", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--audit_csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--culturebank_frames", type=Path, default=None,
                   help="Explicit accessible CultureBank v3/full/frames.jsonl path")
    p.add_argument("--overlap_file", type=Path, default=None)
    p.add_argument("--strict_passage_disjoint", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    audit = validate_audit(args.audit_csv)
    passages = read_rows(args.passages)
    frames = read_rows(args.frames)
    by_pid = {p["passage_id"]: p for p in passages}
    frame_ids = [f["passage_id"] for f in frames]
    if len(frame_ids) != len(set(frame_ids)) or set(frame_ids) != set(by_pid):
        raise ValueError("Frame checkpoint must match training passage source IDs exactly")
    if any(f.get("split") != "train" or f.get("dataset") != "issai/KazCulture"
           for f in frames):
        raise ValueError("Non-train KazCulture frame detected")

    excluded = set()
    if args.strict_passage_disjoint:
        if args.overlap_file is None:
            raise ValueError("Strict mode requires --overlap_file")
        overlap = json.loads(args.overlap_file.read_text(encoding="utf-8"))
        excluded = set(overlap["exclude_train_passage_ids"])
        if not excluded <= set(by_pid):
            raise ValueError("Overlap list not based on this training corpus")
        passages = [p for p in passages if p["passage_id"] not in excluded]
        frames = [f for f in frames if f["passage_id"] not in excluded]
        validate_no_test_content(passages, excluded)

    culturebank = []
    if args.culturebank_frames:
        culturebank = read_rows(args.culturebank_frames)
        # Only CultureBank source-backed frames. No CulturalBench eval fields.
        for frame in culturebank:
            if any(key.startswith("eval_") for key in frame):
                raise ValueError("CultureBank eval field detected")
            if frame.get("label_status") == "ok" and not frame.get("frame_id"):
                raise ValueError("Missing CultureBank source frame provenance")
    units = make_units(passages, frames, culturebank)
    # Retrying after a walltime must finish graph manifests without re-embedding.
    # Verify the exact unit payload before reusing any completed index.
    existing_meta = args.out / "metadata.json"
    if existing_meta.exists():
        import hashlib
        expected = hashlib.sha256("".join(
            json.dumps(u, ensure_ascii=False, sort_keys=True) + "\\n"
            for u in units
        ).encode("utf-8")).hexdigest()
        stats = json.loads(existing_meta.read_text(encoding="utf-8"))
        if stats.get("units_sha256") != expected:
            raise ValueError("Completed index belongs to another corpus; refusing reuse")
    else:
        stats = MultilingualIndex.create(args.out, units, device=args.device,
                                         batch_size=args.batch_size)
    write_jsonl(args.out / "graph_frames.jsonl", frames)
    write_jsonl(args.out / "graph_triples.jsonl", [
        triple for frame in frames for triple in frame["triples"]
    ])
    write_jsonl(args.out / "graph_pathways.jsonl", [
        pathway for frame in frames for pathway in pathways(frame)
    ])

    # Align only through shared L3 value labels, never assert two source-specific
    # norms are equivalent without independent verification.
    kaz_values = Counter(v for frame in frames for v in frame.get("values", []))
    cb_values = Counter(v for frame in culturebank if frame.get("label_status") == "ok"
                        for v in frame.get("values", []))
    shared = sorted(set(kaz_values) & set(cb_values))
    alignment = [{
        "pivot": value, "layer": "L3_deep_value", "alignment_status": "provisional",
        "kazculture_frame_count": kaz_values[value],
        "culturebank_frame_count": cb_values[value],
        "claim": "shared abstract value tag only; not a shared cultural practice",
    } for value in shared]
    write_jsonl(args.out / "culturebank_alignment.jsonl", alignment)
    manifest = {
        **stats, "passages": len(passages), "frames": len(frames),
        "triples": sum(len(frame["triples"]) for frame in frames),
        "pathways": sum(len(pathways(frame)) for frame in frames),
        "strict_passage_disjoint": args.strict_passage_disjoint,
        "excluded_test_overlap_passages": len(excluded),
        "manual_audit": audit,
        "culturebank_frame_count": len(culturebank),
        "shared_l3_value_pivots": shared,
        "status": "unreviewed_full_graph_provisional_abstractions",
    }
    (args.out / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
