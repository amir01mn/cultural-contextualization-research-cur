#!/usr/bin/env python3
"""Versioned KazCulture v2.2 frozen held-out pilot, nested norms and optional values.

Run on untouched source indices 150:300 from the original 750 pilot,
for a matched comparison. Never overwrites prior pilot artifacts.
All outputs use a new explicit versioned file stem. GPU writes a checkpoint
after each completed batch; CPU-only preflight verifies model input budgets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.construction.kazculture_chunking import (
    CHUNKING_VERSION, render_extraction_prompt, split_passage, prompt_tokens,
)
from src.construction.kazculture_kg_v22 import (
    SCHEMA_VERSION, SYSTEM_V22, aggregate_v22,
)
from src.construction.kazculture_kg_v2 import (
    extraction_user_v2, make_v2_audit, save_v2_audit,
)
from src.loaders.kazculture_corpus import read_rows

MAX_CONTEXT = 4096
SAFETY = 64


def _fingerprint(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _segments(tokenizer, passage: dict, *, max_tokens: int) -> list[dict]:
    return split_passage(
        tokenizer, passage, max_model_len=MAX_CONTEXT,
        max_tokens=max_tokens, safety_tokens=SAFETY,
        system=SYSTEM_V22, user_builder=extraction_user_v2,
    )


def _prompts(tokenizer, segments: list[dict], *, max_tokens: int) -> list[str]:
    rendered = [
        render_extraction_prompt(tokenizer, p, system=SYSTEM_V22,
                                 user_builder=extraction_user_v2)
        for p in segments
    ]
    upper = MAX_CONTEXT - max_tokens - SAFETY
    if any(len(tokenizer.encode(x, add_special_tokens=False)) > upper for x in rendered):
        raise ValueError("v2 prompt exceeds verified pre-generation context budget")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit_csv", type=Path, required=True)
    parser.add_argument("--start", type=int, default=150)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--audit_count", type=int, default=60)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--model_revision",
                        default="9216db5781bf21249d130ec9da846c4624c16137")
    parser.add_argument("--max_tokens", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()

    if not 0 < args.max_tokens < MAX_CONTEXT - SAFETY:
        raise ValueError("Unsupported output-token allocation")
    if args.start < 0 or args.limit < 1 or args.batch_size < 1 or args.audit_count < 1:
        raise ValueError("limit, batch_size, audit_count must be positive")
    if args.out == args.passages or args.out == args.audit_csv:
        raise ValueError("Input and output paths must differ")
    if not args.out.parent.name.startswith("kg_v22"):
        raise ValueError("V2.2 output must be inside a dedicated kg_v22 directory")
    if not args.audit_csv.parent.name.startswith("kg_v22"):
        raise ValueError("V2.2 audit must be inside a dedicated kg_v22 directory")

    rows = read_rows(args.passages)
    if args.start + args.limit > len(rows):
        raise ValueError(f"Requested passage slice [{args.start}:{args.start + args.limit}], only {len(rows)} available")
    passages = rows[args.start:args.start + args.limit]
    if any(
        p.get("split") != "train"
        or "question" in p or "answer" in p or "answer_label" in p
        or "item_description" in p for p in passages
    ):
        raise ValueError("Unsafe input: source passages must be train-only")
    ids = [p["passage_id"] for p in passages]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate passage IDs in sample")

    metadata = {
        "protocol": SCHEMA_VERSION, "source_sha256": _fingerprint(args.passages),
        "selected_passage_ids_sha256": hashlib.sha256(
            "\n".join(ids).encode("utf-8")).hexdigest(),
        "passage_start": args.start, "passage_limit": len(passages), "model": args.model,
        "model_revision": args.model_revision, "max_tokens": args.max_tokens,
        "batch_size": args.batch_size, "tp": args.tp,
        "context": MAX_CONTEXT, "safety_tokens": SAFETY,
        "chunking_policy": CHUNKING_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_V22.encode("utf-8")).hexdigest(),
        "audit_count": args.audit_count,
    }
    if args.preflight_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            args.model, revision=args.model_revision, trust_remote_code=False)
        report = {"passages": len(passages), "segments": 0,
                  "chunked_passages": 0, "largest_input_tokens": 0}
        for p in passages:
            parts = _segments(tok, p, max_tokens=args.max_tokens)
            report["segments"] += len(parts)
            report["chunked_passages"] += int(len(parts) > 1)
            report["largest_input_tokens"] = max(
                report["largest_input_tokens"],
                *(prompt_tokens(tok, x, system=SYSTEM_V22,
                                user_builder=extraction_user_v2) for x in parts))
        print(json.dumps({"status": "preflight_passed", **report}, indent=2))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.out.with_suffix(".manifest.json")
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Manifest differs; choose a different output path")
    elif args.out.exists():
        raise FileExistsError("Checkpoint exists without matching manifest")
    else:
        manifest.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    completed = {}
    if args.out.exists():
        for entry in read_rows(args.out):
            pid = entry["passage_id"]
            if pid in completed or pid not in set(ids):
                raise ValueError("Corrupt/duplicate checkpoint records")
            source = passages[ids.index(pid)]
            if entry.get("text") != source["text"] or entry.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Checkpoint passage or schema mismatch")
            completed[pid] = entry

    pending = [p for p in passages if p["passage_id"] not in completed]
    if args.audit_only and pending:
        raise ValueError(f"Cannot audit incomplete extraction: {len(pending)} missing")
    if pending and not args.audit_only:
        from vllm import LLM, SamplingParams

        model = LLM(
            model=args.model, revision=args.model_revision, dtype="bfloat16",
            tensor_parallel_size=args.tp, max_model_len=MAX_CONTEXT,
            gpu_memory_utilization=0.85, trust_remote_code=False,
        )
        tokenizer = model.get_tokenizer()
        generation = SamplingParams(
            temperature=0.0, max_tokens=args.max_tokens, seed=42,
        )
        with args.out.open("a", encoding="utf-8") as handle:
            for offset in range(0, len(pending), args.batch_size):
                group = pending[offset:offset + args.batch_size]
                pieces = [_segments(tokenizer, p, max_tokens=args.max_tokens)
                          for p in group]
                flat = [p for chunks in pieces for p in chunks]
                raw_outputs = []
                reasons = []
                for start in range(0, len(flat), args.batch_size):
                    block = flat[start:start + args.batch_size]
                    outputs = model.generate(
                        _prompts(tokenizer, block, max_tokens=args.max_tokens),
                        generation,
                    )
                    if len(outputs) != len(block):
                        raise RuntimeError("GPU output count mismatch")
                    for result in outputs:
                        option = result.outputs[0] if result.outputs else None
                        raw_outputs.append(option.text if option else "")
                        reasons.append(option.finish_reason if option else "missing")
                pos = 0
                for p, segments in zip(group, pieces):
                    stop = pos + len(segments)
                    frame = aggregate_v22(
                        p, segments, raw_outputs[pos:stop], model=args.model,
                    )
                    frame["finish_reasons"] = reasons[pos:stop]
                    # Save enough original model output for debugging failed
                    # JSON without confusing it with verified KG evidence.
                    frame["raw_failure_excerpt"] = [
                        raw_outputs[j][-900:]
                        for j in range(pos, stop)
                        if frame["chunk_status"][j - pos] == "invalid_json"
                    ]
                    handle.write(json.dumps(frame, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed[p["passage_id"]] = frame
                    pos = stop
                print(f"V2.2 extracted {len(completed)}/{len(passages)} passages",
                      flush=True)

    ordered = [completed[pid] for pid in ids]
    triple_count = sum(len(frame["triples"]) for frame in ordered)
    norm_count = sum(len(frame["pathways"]) for frame in ordered)
    rejections = Counter()
    for frame in ordered:
        rejections.update(frame.get("rejections", {}))
    summary = {
        "schema_version": SCHEMA_VERSION, "passages": len(ordered),
        "frames_with_triples": sum(bool(f["triples"]) for f in ordered),
        "candidate_triples": triple_count, "candidate_pathways": norm_count,
        "pathway_fraction_of_triples": norm_count / triple_count if triple_count else None,
        "json_parse_fail_segments": sum(
            x == "invalid_json" for f in ordered for x in f["chunk_status"]),
        "generation_length_truncations": sum(
            x == "length" for f in ordered for x in f.get("finish_reasons", [])),
        "rejection_reasons": dict(rejections),
        "norm_only_candidates": sum(len(f.get("norm_candidates", [])) for f in ordered) - norm_count,
        "norm_candidates_all": sum(len(f.get("norm_candidates", [])) for f in ordered),
        "status": "candidate_extractions_only_requires_independent_human_audit",
    }
    report_path = args.out.with_suffix(".summary.json")
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.audit_csv.exists():
        summary["manual_audit_preserved"] = str(args.audit_csv)
    elif triple_count >= args.audit_count:
        save_v2_audit(args.audit_csv, make_v2_audit(
            ordered, n=args.audit_count))
        summary["audit_requires_manual_review"] = str(args.audit_csv)
    else:
        summary["audit_not_created"] = (
            f"Only {triple_count} triples; requires {args.audit_count}"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
