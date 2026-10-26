#!/usr/bin/env python3
"""Phase 1 V2: assistant-prefix Qwen3 MCQ scoring with isolated checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_loader import load_test
from src.evaluation.kazculture_phase1 import evaluate_batch, save_summary
from src.evaluation.kazculture_hf_scorer_v2 import (
    HFQwenLetterScorerV2, PROTOCOL_VERSION,
    candidate_token_ids, prompt_sha256, render_prompt,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--test_file", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--model_revision", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--max_context_tokens", type=int, default=4096)
    p.add_argument("--check_tokenization", action="store_true",
                   help="CPU-only validation of the actual cached tokenizer")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    questions, dataset_metadata = load_test(
        local_file=args.test_file, limit=args.limit,
    )
    metadata = {
        **dataset_metadata, "model": args.model,
        "model_revision": args.model_revision,
        "prompt_version": PROTOCOL_VERSION,
        "non_thinking": True,
        "scoring": "assistant-prefix-space-letter-next-token-logprob",
        "no_retrieval": True,
        "temperature": None,
        "max_context_tokens": args.max_context_tokens,
        "subset_not_comparable": len(questions) != 1334,
    }
    if args.check_tokenization:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, revision=args.model_revision, trust_remote_code=False
        )
        sample = questions[0]
        rendered = render_prompt(tokenizer, sample)
        tokens = candidate_token_ids(tokenizer, rendered)
        print(json.dumps({
            "status": "tokenization_passed", "protocol": PROTOCOL_VERSION,
            "n": len(questions),
            "candidate_ids": tokens,
            "prompt_suffix": rendered[-100:],
            "non_thinking": True,
        }, ensure_ascii=False, indent=2))
        return 0

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata_file = out / "checkpoint_metadata.json"
    predictions_path = out / "predictions.jsonl"
    completed = {}
    if predictions_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{predictions_path} exists; pass --resume or use a new output dir")
        if not metadata_file.exists():
            raise ValueError("Missing checkpoint metadata; refusing unsafe resume")
        if json.loads(metadata_file.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Checkpoint metadata differs; refusing unsafe resume")
        by_id = {q.question_id: q for q in questions}
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            key = item["question_id"]
            if (key not in by_id or key in completed
                    or item.get("prompt_sha256") != prompt_sha256(by_id[key])
                    or item.get("answer_key") != by_id[key].answer_key):
                raise ValueError("Duplicate or mismatched record; refusing resume")
            completed[key] = item
    else:
        metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    remaining = [q for q in questions if q.question_id not in completed]
    if remaining:
        scorer = HFQwenLetterScorerV2(
            args.model, model_revision=args.model_revision,
            max_context_tokens=args.max_context_tokens
        )
        with predictions_path.open("a", encoding="utf-8") as output:
            for index, q in enumerate(remaining, 1):
                prediction = evaluate_batch([q], scorer)[0]
                prediction["prompt_sha256"] = prompt_sha256(q)
                output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                output.flush()
                completed[q.question_id] = prediction
                if index % 25 == 0:
                    print(f"V2: {len(completed)}/{len(questions)}", flush=True)

    ordered = [completed[q.question_id] for q in questions]
    result = save_summary(out, ordered, metadata)
    print(json.dumps({
        "status": "complete", "protocol": PROTOCOL_VERSION,
        "n": result["n"], "accuracy_pct": result["accuracy_pct"],
        "full_comparable": not metadata["subset_not_comparable"],
        "output": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
