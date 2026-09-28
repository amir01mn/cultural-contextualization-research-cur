"""Exhaustively audit CulturalBench coverage over a large CultureBank corpus.

Unlike scripts/10_audit_culturalbench_coverage.py, this script never truncates
matched target-culture records. It partitions all matching records into bounded
prompt chunks, judges every chunk, and aggregates the chunk labels.

This is intended for the full CultureBank coverage diagnostic, where a single
culture may have too many records to fit in one LLM context window.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.llm_backend import get_backend  # noqa: E402
from src.retrieval_ranking import culture_match_score, passage_culture  # noqa: E402
from src.schemas import MCQQuestion  # noqa: E402
from src.utils import ensure_dir, normalize_text, read_jsonl, truncate, write_jsonl  # noqa: E402

BASE_PATH = Path(__file__).resolve().parent / "10_audit_culturalbench_coverage.py"
SPEC = importlib.util.spec_from_file_location("coverage_audit_base", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(base)

LABEL_PRIORITY = {"ABSENT": 0, "PARTIAL": 1, "COVERED": 2}


def chunks(items: list[dict], size: int) -> list[list[dict]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")
    return [items[i:i + size] for i in range(0, len(items), size)]


def load_passages_from_records(path: Path) -> list[dict]:
    return [base.record_to_passage(row) for row in read_jsonl(path)]


def validate_chunk_audit(
    q: MCQQuestion,
    candidates: list[dict],
    raw_output: str,
) -> dict:
    candidate_by_id = {
        str(doc.get("instance_id") or doc.get("passage_id") or "").strip(): doc
        for doc in candidates
    }

    parse_error: Optional[str] = None
    try:
        audit = base.parse_audit_output(raw_output)
    except Exception as exc:  # noqa: BLE001
        audit = {
            "label": "ABSENT",
            "supporting_instance_ids": [],
            "rationale": "Audit output could not be parsed; manual review required.",
            "confidence": 0.0,
        }
        parse_error = str(exc)

    valid_support_ids: list[str] = []
    invalid_support_ids: list[str] = []
    cross_culture_support_ids: list[str] = []

    for instance_id in audit["supporting_instance_ids"]:
        doc = candidate_by_id.get(instance_id)
        if doc is None:
            invalid_support_ids.append(instance_id)
            continue
        if culture_match_score(passage_culture(doc), q.cultural_group) < 0.9:
            cross_culture_support_ids.append(instance_id)
            continue
        valid_support_ids.append(instance_id)

    # COVERED and PARTIAL must both point to at least one directly relevant
    # target-culture record. Otherwise the safest corpus-coverage label is ABSENT.
    if audit["label"] in {"COVERED", "PARTIAL"} and not valid_support_ids:
        audit["label"] = "ABSENT"
        audit["confidence"] = min(float(audit["confidence"]), 0.5)
        audit["rationale"] = (
            audit["rationale"]
            + " No valid target-culture supporting record was cited; downgraded to ABSENT."
        ).strip()

    return {
        "label": audit["label"],
        "supporting_instance_ids": valid_support_ids,
        "invalid_support_instance_ids": invalid_support_ids,
        "cross_culture_support_instance_ids": cross_culture_support_ids,
        "rationale": audit["rationale"],
        "confidence": float(audit["confidence"]),
        "parse_error": parse_error,
        "raw_judge_output": truncate(raw_output, 3000),
    }


def aggregate_question(
    q: MCQQuestion,
    culture_docs: list[dict],
    chunk_results: list[dict],
) -> dict:
    if not culture_docs:
        return {
            "question_id": q.question_id,
            "group_id": q.group_id,
            "variant": base._variant_name(q),
            "cultural_group": q.cultural_group,
            "question": q.question,
            "gold_description": base.gold_description(q),
            "label": "ABSENT",
            "supporting_instance_ids": [],
            "rationale": "No target-culture CultureBank records were found under the validated culture matcher.",
            "confidence": 1.0,
            "target_culture_record_count": 0,
            "chunks_audited": 0,
            "audited_all_target_culture_records": True,
            "chunk_results": [],
        }

    best_priority = max(LABEL_PRIORITY[r["label"]] for r in chunk_results)
    selected = [r for r in chunk_results if LABEL_PRIORITY[r["label"]] == best_priority]
    label = selected[0]["label"]

    support_ids = []
    seen = set()
    for result in selected:
        for instance_id in result["supporting_instance_ids"]:
            if instance_id not in seen:
                seen.add(instance_id)
                support_ids.append(instance_id)

    best = max(selected, key=lambda r: r["confidence"])
    rationale = best["rationale"]
    if len(selected) > 1:
        rationale = (
            f"{rationale} "
            f"({len(selected)} audited chunks received the aggregate label {label}.)"
        )

    return {
        "question_id": q.question_id,
        "group_id": q.group_id,
        "variant": base._variant_name(q),
        "cultural_group": q.cultural_group,
        "question": q.question,
        "gold_description": base.gold_description(q),
        "label": label,
        "supporting_instance_ids": support_ids,
        "rationale": rationale,
        "confidence": best["confidence"],
        "target_culture_record_count": len(culture_docs),
        "chunks_audited": len(chunk_results),
        "audited_all_target_culture_records": True,
        "chunk_results": chunk_results,
    }


def write_markdown(path: Path, rows: list[dict], run_meta: dict) -> None:
    ensure_dir(path.parent)
    counts = Counter(row["label"] for row in rows)
    easy = Counter(row["label"] for row in rows if row["variant"] == "easy")
    hard = Counter(row["label"] for row in rows if row["variant"] == "hard")

    lines = [
        "# Exhaustive CulturalBench Corpus Coverage Audit",
        "",
        "Every target-culture record in the supplied CultureBank corpus was audited in bounded chunks. "
        "No target-culture record cap was applied.",
        "",
        f"- Model judge: {run_meta['model']}",
        f"- Corpus records/passages: {run_meta['num_passages']}",
        f"- Corpus source: {run_meta['corpus_source']}",
        f"- Questions: {len(rows)}",
        f"- Records per audit chunk: {run_meta['chunk_size']}",
        "",
        "## Coverage summary",
        "",
        "| Split | COVERED | PARTIAL | ABSENT |",
        "|---|---:|---:|---:|",
        f"| All | {counts['COVERED']} | {counts['PARTIAL']} | {counts['ABSENT']} |",
        f"| Easy | {easy['COVERED']} | {easy['PARTIAL']} | {easy['ABSENT']} |",
        f"| Hard | {hard['COVERED']} | {hard['PARTIAL']} | {hard['ABSENT']} |",
        "",
        "## Per-question audit",
        "",
        "| Question | Variant | Culture | Label | Target-culture records | Chunks | Support IDs | Confidence | Rationale |",
        "|---|---|---|---|---:|---:|---|---:|---|",
    ]

    for row in rows:
        culture = (row.get("cultural_group") or "unknown").replace("|", "\\|")
        rationale = row["rationale"].replace("|", "\\|")
        support = ", ".join(row["supporting_instance_ids"]) or "—"
        lines.append(
            f"| {row['question_id']} | {row['variant']} | {culture} | "
            f"{row['label']} | {row['target_culture_record_count']} | "
            f"{row['chunks_audited']} | {support} | {row['confidence']:.2f} | "
            f"{rationale} |"
        )

    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- COVERED requires at least one cited target-culture record that directly supports the gold target.",
        "- PARTIAL requires at least one cited target-culture record about the same practice/event/norm but missing a decisive detail.",
        "- ABSENT means the exhaustive target-culture audit found no such supporting record; it does not mean the benchmark claim is culturally false.",
        "- This remains an automated evidence audit. Supporting records should be manually checked before publication.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Exhaustively audit CulturalBench coverage over a large CultureBank corpus."
    )
    p.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument(
        "--questions_jsonl",
        default="data/controlled/food_hospitality_culturalbench_questions.jsonl",
    )
    p.add_argument("--records_jsonl", required=True)
    p.add_argument("--chunk_size", type=int, default=24)
    p.add_argument("--max_passage_chars", type=int, default=500)
    p.add_argument(
        "--output_jsonl",
        default="data/outputs/coverage_full_culturebank_exhaustive.jsonl",
    )
    p.add_argument(
        "--output_md",
        default="data/outputs/coverage_full_culturebank_exhaustive.md",
    )
    args = p.parse_args(argv)

    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0")

    questions = base.load_questions(Path(args.questions_jsonl))
    passages = load_passages_from_records(Path(args.records_jsonl))

    backend = get_backend(args.provider, model=args.model)
    status = backend.status()
    if not status.get("available"):
        raise RuntimeError(f"Backend unavailable: {status.get('message')}")
    if hasattr(backend, "warmup"):
        backend.warmup()

    all_prompts: list[str] = []
    prompt_meta: list[tuple[int, int, MCQQuestion, list[dict]]] = []
    culture_docs_by_question: list[list[dict]] = []

    for q_idx, q in enumerate(questions):
        culture_docs = [
            doc for doc in passages
            if culture_match_score(passage_culture(doc), q.cultural_group) >= 0.9
        ]
        culture_docs_by_question.append(culture_docs)

        for chunk_idx, candidate_chunk in enumerate(chunks(culture_docs, args.chunk_size)):
            prompt = base.build_audit_prompt(
                q,
                candidate_chunk,
                max_passage_chars=args.max_passage_chars,
            )
            all_prompts.append(prompt)
            prompt_meta.append((q_idx, chunk_idx, q, candidate_chunk))

    outputs = (
        backend.generate(all_prompts, system_prompt=base.COVERAGE_SYSTEM)
        if all_prompts
        else []
    )

    per_question_chunks: list[list[dict]] = [[] for _ in questions]
    for meta, output in zip(prompt_meta, outputs):
        q_idx, chunk_idx, q, candidate_chunk = meta
        result = validate_chunk_audit(q, candidate_chunk, output)
        result["chunk_index"] = chunk_idx
        result["candidate_instance_ids"] = [
            str(doc.get("instance_id") or doc.get("passage_id") or "")
            for doc in candidate_chunk
        ]
        per_question_chunks[q_idx].append(result)

    rows = [
        aggregate_question(q, culture_docs_by_question[i], per_question_chunks[i])
        for i, q in enumerate(questions)
    ]

    write_jsonl(args.output_jsonl, rows)
    run_meta = {
        "provider": args.provider,
        "model": backend.model,
        "num_passages": len(passages),
        "corpus_source": args.records_jsonl,
        "chunk_size": args.chunk_size,
        "max_passage_chars": args.max_passage_chars,
    }
    write_markdown(Path(args.output_md), rows, run_meta)

    counts = Counter(row["label"] for row in rows)
    total_chunks = sum(row["chunks_audited"] for row in rows)
    print("\n===== EXHAUSTIVE CULTURALBENCH COVERAGE AUDIT =====\n")
    print(f"Corpus passages: {len(passages)}")
    print(f"Questions: {len(rows)}")
    print(f"Total target-culture chunks audited: {total_chunks}")
    print(
        f"COVERED={counts['COVERED']}  "
        f"PARTIAL={counts['PARTIAL']}  "
        f"ABSENT={counts['ABSENT']}"
    )
    print(f"JSONL: {args.output_jsonl}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
