#!/usr/bin/env python3
"""Frozen Qwen3-32B V2 retrieval evaluation for KazCulture S1-S4R."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_loader import load_test
from src.loaders.kazculture_corpus import read_rows, passage_id
from src.retrieval.kazculture_index import MultilingualIndex
from src.evaluation.kazculture_rag import (
    V2EvidenceScorer, category_report, paired_statistics, prediction,
    retrieval_recall, selected_evidence,
)


def load_saved(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--test_file", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--baseline_predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--systems", default="S1,S2,S3,S4F,S4R",
                   help="Comma-separated: S1,S2,S3,S4F,S4R,S4R_combined")
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--model_revision", default="9216db5781bf21249d130ec9da846c4624c16137")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--min_similarity", type=float, default=0.0)
    p.add_argument("--config", type=Path,
                   help="Frozen train-development threshold JSON; overrides --min_similarity")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retrieval_only", action="store_true",
                   help="CPU-only retrieval inspection; never reads test labels in retrieval")
    args = p.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    allowed = {"S1", "S2", "S3", "S4F", "S4R", "S4R_combined"}
    if not systems or len(systems) != len(set(systems)) or set(systems) - allowed:
        raise ValueError(f"Invalid/duplicate systems: {systems}")
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("source") != "train_dev_only":
            raise ValueError("Threshold config must originate from training development")
        args.min_similarity = float(config["min_similarity"])
        args.top_k = int(config["top_k"])
    if not (0 <= args.min_similarity <= 1):
        raise ValueError("Expected cosine threshold between 0 and 1")

    questions, qmeta = load_test(local_file=str(args.test_file), limit=args.limit)
    test_rows = read_rows(args.test_file)
    by_id = {q.question_id: q for q in questions}
    baseline = load_saved(args.baseline_predictions)
    base_by_id = {r["question_id"]: r for r in baseline}
    if len(base_by_id) != len(baseline) or not set(by_id) <= set(base_by_id):
        raise ValueError("S0 V2 baseline predictions must cover exactly these test IDs")
    baseline = [base_by_id[q.question_id] for q in questions]
    if any(r["answer_key"] != by_id[r["question_id"]].answer_key
           or r.get("system") not in (None, "S0") for r in baseline):
        raise ValueError("V2 baseline labels/IDs differ from requested test data")

    index = MultilingualIndex(args.index, device="cpu")
    manifest = json.loads((args.index / "build_manifest.json").read_text(encoding="utf-8"))
    if "S4R_combined" in systems and manifest["culturebank_frame_count"] == 0:
        raise ValueError("Combined system requires a CultureBank source frame corpus")
    excluded = set()
    if manifest.get("strict_passage_disjoint"):
        # Independently recompute the overlap mask from the current test rows.
        all_test_ids = {passage_id(r.get("item_description") or "") for r in test_rows}
        index_ids = {u["passage_id"] for u in index.units
                     if u["dataset"] == "KazCulture"}
        if all_test_ids & index_ids:
            raise ValueError("Strict test-overlap source leakage detected in index")
        excluded = all_test_ids

    args.out.mkdir(parents=True, exist_ok=True)
    settings = {
        "systems": systems, "index": str(args.index.resolve()),
        "test_file": str(args.test_file.resolve()), "n_questions": len(questions),
        "model": args.model, "model_revision": args.model_revision,
        "baseline": str(args.baseline_predictions.resolve()),
        "top_k": args.top_k, "min_similarity": args.min_similarity,
        "strict_passage_disjoint": manifest.get("strict_passage_disjoint", False),
        "prompt": "Qwen3 V2, same MCQ and scoring; only added evidence changes",
    }
    settings_file = args.out / "settings.json"
    if settings_file.exists():
        if json.loads(settings_file.read_text(encoding="utf-8")) != settings:
            raise ValueError("Experiment settings changed; use a new --out directory")
    else:
        settings_file.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    scorer = None
    results = {}
    test_source_ids = {
        questions[i].question_id: passage_id(test_rows[i].get("item_description") or "")
        for i in range(len(questions))
    }
    eligible = {u["passage_id"] for u in index.units if u["dataset"] == "KazCulture"}
    for system in systems:
        outfile = args.out / f"{system}.jsonl"
        completed = {}
        if outfile.exists():
            if not args.resume:
                raise FileExistsError(f"{outfile} exists: pass --resume")
            for row in load_saved(outfile):
                qid = row["question_id"]
                if qid not in by_id or qid in completed or row["system"] != system or (
                    row["answer_key"] != by_id[qid].answer_key
                ):
                    raise ValueError(f"Corrupted checkpoint for {system}")
                completed[qid] = row
        remaining = [q for q in questions if q.question_id not in completed]
        if args.retrieval_only:
            for q in remaining[:10]:
                _, hits = selected_evidence(
                    index, q, system, top_k=args.top_k,
                    min_similarity=args.min_similarity,
                    exclude_passage_ids=excluded,
                )
                print(json.dumps({
                    "system": system, "question_id": q.question_id,
                    "hits": [{"source": h["source"], "passage_id": h["passage_id"],
                              "score": h["similarity"], "dataset": h["dataset"]}
                             for h in hits]
                }, ensure_ascii=False))
            continue

        if remaining and scorer is None:
            scorer = V2EvidenceScorer(
                args.model, model_revision=args.model_revision
            )
        if remaining:
            with outfile.open("a", encoding="utf-8") as stream:
                for i, q in enumerate(remaining, 1):
                    evidence, hits = selected_evidence(
                        index, q, system, top_k=args.top_k,
                        min_similarity=args.min_similarity,
                        exclude_passage_ids=excluded,
                    )
                    scores = scorer.score_with_evidence(q, evidence)
                    row = prediction(q, scores, hits, evidence, system)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed[q.question_id] = row
                    if i % 25 == 0:
                        print(f"{system}: {len(completed)}/{len(questions)}", flush=True)
        ordered = [completed[q.question_id] for q in questions]
        paired = paired_statistics(baseline, ordered)
        paired["category_accuracy"] = category_report(ordered)
        paired["retrieval"] = retrieval_recall(ordered, test_source_ids, eligible)
        paired["no_evidence_count"] = sum(not r["retrieved_unit_ids"] for r in ordered)
        paired["mean_retrieved"] = (
            sum(len(r["retrieved_unit_ids"]) for r in ordered) / len(ordered)
        )
        results[system] = paired
        (args.out / f"{system}_report.json").write_text(
            json.dumps(paired, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"system": system,
                          "accuracy": paired["system_accuracy"],
                          "delta_pp": paired["difference_pp"]}), flush=True)
    if not args.retrieval_only:
        (args.out / "comparison.json").write_text(
            json.dumps({"S0": {
                "accuracy": sum(bool(r["correct"]) for r in baseline) / len(baseline),
                "n": len(baseline),
            }, "systems": results, "settings": settings},
                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
