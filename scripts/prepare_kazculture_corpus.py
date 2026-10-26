#!/usr/bin/env python3
"""Stage immutable train-only KazCulture passages and passage-overlap manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_corpus import (
    corpus_from_train, deterministic_pilot, read_rows,
    test_overlap_report, write_jsonl,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--train_file", type=Path, default=None,
                   help="Existing train Parquet or JSONL; omitted downloads gated HF train split")
    p.add_argument("--test_file", type=Path, required=True,
                   help="Read ONLY to detect shared source passages; never included in KG records")
    p.add_argument("--out", type=Path, default=Path("data/kazculture_pipeline/corpus"))
    p.add_argument("--pilot_size", type=int, default=750)
    p.add_argument("--dataset_revision", default=None)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.train_file:
        rows = read_rows(args.train_file)
        origin = str(args.train_file.resolve())
    else:
        from datasets import load_dataset
        dataset = load_dataset("issai/KazCulture", split="train", token=True,
                               revision=args.dataset_revision)
        rows = list(dataset)
        origin = f"issai/KazCulture train ({dataset._fingerprint})"
        # Keep staged immutable train copy for subsequent runs (not Git-tracked).
        dataset.to_parquet(str(args.out / "train.parquet"))

    passages, dev_qa, stats = corpus_from_train(rows)
    if len(rows) != 14803:
        raise ValueError(f"Expected KazCulture train size 14803, found {len(rows)}")
    test = read_rows(args.test_file)
    if len(test) != 1334:
        raise ValueError(f"Expected KazCulture test size 1334, found {len(test)}")
    overlap = test_overlap_report(passages, test)
    pilot = deterministic_pilot(passages, args.pilot_size)
    write_jsonl(args.out / "passages.jsonl", passages)
    write_jsonl(args.out / "pilot_passages.jsonl", pilot)
    # Q/A pairs are stored separately and never passed to the KG builder.
    write_jsonl(args.out / "train_dev_qa.jsonl", dev_qa)
    (args.out / "test_overlap.json").write_text(
        json.dumps(overlap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        **stats, "pilot_passages": len(pilot),
        "train_source": origin, "test_file": str(args.test_file),
        "test_overlap": {k: v for k, v in overlap.items()
                         if k != "exclude_train_passage_ids"},
        "notes": "Never ingest test passage/QA into KG. Dev QA stored separately from corpus."
    }
    (args.out / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
