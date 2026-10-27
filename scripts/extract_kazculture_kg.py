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
from src.construction.kazculture_chunking import (
    CHUNKING_VERSION, aggregate_chunks, prompt_tokens, split_passage,
    render_extraction_prompt,
)


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vllm_generate(passages: list[dict], *, model, tokenizer, system, max_tokens: int):
    from vllm import SamplingParams
    # Every segment must have passed the tokenizer budget check before vLLM.
    prompts = [render_extraction_prompt(tokenizer, p, system=system) for p in passages]
    if any(len(tokenizer.encode(prompt, add_special_tokens=False)) >
           (4096 - max_tokens - 64) for prompt in prompts):
        raise ValueError("Unsafe overlength prompt: refusing vLLM request")
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
    p.add_argument("--preflight_only", action="store_true",
                   help="Check passage prompt lengths and chunk coverage using tokenizer, no GPU")
    args = p.parse_args()

    passages = read_rows(args.passages)
    if not passages or any(p.get("split") != "train" or "answer" in p or "question" in p
                           or "answer_label" in p or "item_description" in p for p in passages):
        raise ValueError("Input must be sanitized, train-only passage corpus")
    ids = [p["passage_id"] for p in passages]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate passage IDs in build input")
    if args.preflight_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.model, revision=args.model_revision, trust_remote_code=False,
        )
        stats = {"n_passages": len(passages), "max_model_len": 4096,
                 "reserved_output_tokens": args.max_tokens,
                 "safety_tokens": 64, "long_passages": 0,
                 "total_segments": 0, "max_segments_per_passage": 0,
                 "max_segment_input_tokens": 0}
        for psg in passages:
            count = prompt_tokens(tok, psg)
            parts = split_passage(tok, psg, max_tokens=args.max_tokens)
            stats["long_passages"] += int(len(parts) > 1)
            stats["total_segments"] += len(parts)
            stats["max_segments_per_passage"] = max(
                stats["max_segments_per_passage"], len(parts))
            stats["max_segment_input_tokens"] = max(
                stats["max_segment_input_tokens"],
                *(prompt_tokens(tok, part) for part in parts),
            )
        print(json.dumps({"status": "preflight_passed", **stats}, indent=2))
        return 0
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
    by_id = {p["passage_id"]: p for p in passages}
    for pid, previous in done.items():
        if previous.get("text") != by_id[pid]["text"]:
            raise ValueError("Checkpoint source text changed; refusing unsafe resume")
    # Existing 264 finished records keep their original model inputs and
    # outputs. The system prompt and sampling have not changed. Only inputs
    # that would overflow the 4096-token window are split into verbatim chunks.
    protocol_path = args.out.with_suffix(".chunking.json")
    protocol = {"chunking_version": CHUNKING_VERSION, "context_tokens": 4096,
                "output_tokens": args.max_tokens, "safety_tokens": 64,
                "overlap_chars": 160, "source_quote_validation": "model_visible_chunk",
                "checkpoint_compatibility": "old short-passage records retained"}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Chunking protocol changed for this checkpoint")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
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
                # Token-length validated segmentation of every original source.
                nested = [split_passage(tokenizer, row, max_tokens=args.max_tokens)
                          for row in batch]
                flat = [part for parts in nested for part in parts]
                generated = []
                # Bound generation requests independently from passage count:
                # one long passage can produce multiple source-aligned chunks.
                for off in range(0, len(flat), args.batch_size):
                    generated.extend(vllm_generate(
                        flat[off:off + args.batch_size],
                        model=model, tokenizer=tokenizer,
                        system=EXTRACTION_SYSTEM, max_tokens=args.max_tokens,
                    ))
                if len(flat) != len(generated):
                    raise RuntimeError("Chunk generation count mismatch")
                offset = 0
                for row, chunks in zip(batch, nested):
                    outputs = generated[offset:offset + len(chunks)]
                    offset += len(chunks)
                    frame = aggregate_chunks(row, chunks, outputs, model=args.model)
                    output.write(json.dumps(frame, ensure_ascii=False) + "\n")
                    done[frame["passage_id"]] = frame
                output.flush()
                print(
                    f"Extracted {len(done)}/{len(passages)} "
                    f"({len(flat)} model-visible segments in batch)", flush=True
                )

    frames = [done[pid] for pid in ids]
    successes = sum(f["label_status"] == "candidate" for f in frames)
    triples = sum(len(f["triples"]) for f in frames)
    report = {"passages": len(frames), "candidate_frames": successes,
              "candidate_triples": triples, "candidate_rate": successes / len(frames)}
    if successes / len(frames) < 0.5:
        raise RuntimeError(f"Extraction coverage below 50% gate: {report}")
    if args.audit_csv:
        # NEVER overwrite annotations a human may already have entered.
        if args.audit_csv.exists():
            report["existing_manual_audit_preserved"] = str(args.audit_csv)
        else:
            sample = audit_rows(frames, args.audit_count)
            write_audit_csv(args.audit_csv, sample)
            report["audit_csv_requires_manual_review"] = str(args.audit_csv)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
