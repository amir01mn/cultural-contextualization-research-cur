#!/usr/bin/env python3
"""Phase 1: reproduce Qwen3-32B closed-book KazCulture accuracy.

    python scripts/eval_kazculture_phase1.py --test_file /path/test.parquet
    python scripts/eval_kazculture_phase1.py --test_file test.jsonl --dry_run
    python scripts/eval_kazculture_phase1.py --test_file test.parquet --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.kazculture_phase1 import (  # noqa: E402
    PROMPT_VERSION, evaluate_batch, save_summary,
)
from src.loaders.kazculture_loader import load_test  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--test_file", default=None)
    p.add_argument("--dataset_revision", default=None)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--model_revision", default=None)
    p.add_argument("--output", default="data/outputs/kazculture_phase1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--expected_size", type=int, default=1334)
    p.add_argument("--max_context_tokens", type=int, default=4096)
    p.add_argument("--dry_run", action="store_true",
                   help="Validate full dataset and prompt examples without loading a model")
    p.add_argument("--resume", action="store_true",
                   help="Reuse completed predictions after checking question IDs and hashes")
    args = p.parse_args(argv)

    questions, source_meta = load_test(
        local_file=args.test_file, revision=args.dataset_revision,
        limit=args.limit, expected_size=args.expected_size)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        **source_meta, "model": args.model,
        "model_revision": args.model_revision,
        "prompt_version": PROMPT_VERSION,
        "non_thinking": True, "scoring": "single-next-token-log_probability",
        "no_retrieval": True, "temperature": None,
        "max_context_tokens": args.max_context_tokens,
        "subset_not_comparable": len(questions) != args.expected_size,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        from src.evaluation.kazculture_phase1 import question_text
        print(json.dumps({
            "status": "dry_run_passed", "n": len(questions),
            "example_prompt": question_text(questions[0]) if questions else "",
            "metadata": metadata,
        }, ensure_ascii=False, indent=2))
        return 0

    from src.evaluation.kazculture_hf_scorer import HFQwenLetterScorer
    from src.evaluation.kazculture_phase1 import evaluate_batch, prompt_hash

    predictions_file = out / "predictions.jsonl"
    completed = {}
    if args.resume and predictions_file.exists():
        for line in predictions_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                completed[r["question_id"]] = r
        allowable = {q.question_id: q for q in questions}
        if any(qid not in allowable
               or r.get("prompt_sha256") != prompt_hash(allowable[qid])
               or r.get("answer_key") != allowable[qid].answer_key
               for qid, r in completed.items()):
            raise ValueError("Checkpoint records differ from current data; refuse resume")
        prior_meta = json.loads((out / "checkpoint_metadata.json").read_text())
        if prior_meta != metadata:
            raise ValueError("Checkpoint model, dataset, or prompt mismatch")
    elif predictions_file.exists():
        raise FileExistsError(
            f"{predictions_file} exists; use a new output directory or --resume")

    (out / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    remaining = [q for q in questions if q.question_id not in completed]
    if remaining:
        scorer = HFQwenLetterScorer(
            args.model, model_revision=args.model_revision,
            max_context_tokens=args.max_context_tokens)
        with predictions_file.open("a", encoding="utf-8") as f:
            for i, q in enumerate(remaining, 1):
                item = evaluate_batch([q], scorer)[0]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                completed[q.question_id] = item
                if i % 25 == 0:
                    print(f"Predictions completed: {len(completed)}/{len(questions)}", flush=True)
    ordered = [completed[q.question_id] for q in questions]
    summary = save_summary(out, ordered, metadata)
    print(json.dumps({
        "status": "complete", "accuracy_pct": summary["accuracy_pct"],
        "n": summary["n"], "published_baseline_pct": 39.51,
        "full_comparable": not metadata["subset_not_comparable"],
        "output": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
