#!/usr/bin/env python3
"""Checkpointed GPU extraction of KazCulture source-grounded KG candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.construction.kazculture_kg import (
    EXTRACTION_SYSTEM, audit_rows, extract_frame, extraction_user, write_audit_csv,
)
from src.loaders.kazculture_corpus import read_rows


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vllm_generate(passages: list[dict], *, model, tokenizer, system, max_tokens: int):
    from vllm import SamplingParams
    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": extraction_user(p)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False
    ) for p in passages]
    outputs = model.generate(
        prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=42)
    )
    return [o.outputs[0].text if o.outputs else "" for o in outputs]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--passages", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--model_revision", default="9216db5781bf21249d130ec9da846c4624c16137")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--audit_count", type=int, default=60)
    p.add_argument("--audit_csv", type=Path)
    p.add_argument("--audit_only", action="store_true",
                   help="Rebuild audit sample from checkpoint without model")
    args = p.parse_args()

    passages = read_rows(args.passages)
    if not passages or any(p.get("split") != "train" or "answer" in p or "question" in p
                           or "answer_label" in p or "item_description" in p for p in passages):
        raise ValueError("Input must be sanitized, train-only passage corpus")
    ids = [p["passage_id"] for p in passages]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate passage IDs in build input")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata_file = args.out.with_suffix(".manifest.json")
    metadata = {
        "input_sha256": fingerprint(args.passages), "n_input_passages": len(passages),
        "model": args.model, "model_revision": args.model_revision,
        "tp": args.tp, "max_tokens": args.max_tokens,
        "prompt_sha256": hashlib.sha256(EXTRACTION_SYSTEM.encode()).hexdigest(),
        "provenance": "KazCulture train item_description only",
    }

    if metadata_file.exists():
        if json.loads(metadata_file.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Extraction manifest changed; use a new --out directory")
    elif args.out.exists():
        raise ValueError("Checkpoint exists without metadata")
    else:
        metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    done = {}
    if args.out.exists():
        for row in read_rows(args.out):
            pid = row["passage_id"]
            if pid in done or pid not in set(ids):
                raise ValueError("Corrupt or duplicate checkpoint passage ID")
            done[pid] = row

    remaining = [p for p in passages if p["passage_id"] not in done]
    if args.audit_only and remaining:
        raise ValueError(f"Cannot audit: {len(remaining)} passages not extracted")
    if remaining and not args.audit_only:
        from vllm import LLM
        model = LLM(model=args.model, revision=args.model_revision,
                    tensor_parallel_size=args.tp, dtype="bfloat16",
                    max_model_len=4096, gpu_memory_utilization=0.85,
                    trust_remote_code=False)
        tokenizer = model.get_tokenizer()
        with args.out.open("a", encoding="utf-8") as output:
            for start in range(0, len(remaining), args.batch_size):
                batch = remaining[start:start + args.batch_size]
                generated = vllm_generate(
                    batch, model=model, tokenizer=tokenizer,
                    system=EXTRACTION_SYSTEM, max_tokens=args.max_tokens
                )
                if len(batch) != len(generated):
                    raise RuntimeError("Batch output mismatch; no silent fallback")
                for row, raw in zip(batch, generated):
                    frame = extract_frame(row, raw, model=args.model)
                    output.write(json.dumps(frame, ensure_ascii=False) + "\n")
                    done[frame["passage_id"]] = frame
                output.flush()
                print(f"Extracted {len(done)}/{len(passages)}", flush=True)

    frames = [done[pid] for pid in ids]
    successes = sum(f["label_status"] == "candidate" for f in frames)
    triples = sum(len(f["triples"]) for f in frames)
    report = {"passages": len(frames), "candidate_frames": successes,
              "candidate_triples": triples, "candidate_rate": successes / len(frames)}
    if successes / len(frames) < 0.5:
        raise RuntimeError(f"Extraction coverage below 50% gate: {report}")
    if args.audit_csv:
        sample = audit_rows(frames, args.audit_count)
        write_audit_csv(args.audit_csv, sample)
        report["audit_csv_requires_manual_review"] = str(args.audit_csv)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
