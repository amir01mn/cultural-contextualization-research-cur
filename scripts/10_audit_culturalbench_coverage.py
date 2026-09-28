"""Audit whether the Scale corpus contains evidence for controlled CulturalBench items.

This is a corpus-coverage diagnostic, not an answer-generation condition. The
judge receives the gold benchmark target and CultureBank passages and must decide
whether those passages contain enough target-culture evidence to support the
gold answer/judgment.

Labels:
- COVERED: corpus evidence directly supports the gold target.
- PARTIAL: related target-culture evidence exists but is insufficient.
- ABSENT: no inspected target-culture evidence supports the gold target.

The audit uses all passages whose cultural group matches the benchmark culture
under the validated culture-alias matcher. A small semantic fallback is also
included to catch plausible culture-label variants that are not covered by the
alias table. Cross-cultural passages may not be used as supporting evidence.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.llm_backend import get_backend  # noqa: E402
from src.retrieval_ranking import culture_match_score, passage_culture  # noqa: E402
from src.retriever import TextRetriever  # noqa: E402
from src.schemas import MCQQuestion, MCQVariant  # noqa: E402
from src.utils import ensure_dir, get_logger, normalize_text, read_jsonl, truncate, write_jsonl  # noqa: E402

LOGGER = get_logger("10_audit_culturalbench_coverage")

VALID_LABELS = {"COVERED", "PARTIAL", "ABSENT"}

COVERAGE_SYSTEM = """You are auditing evidence coverage in a cultural dataset.

Use ONLY the supplied CultureBank candidate passages as evidence. Do not use
your own cultural knowledge, stereotypes, assumptions, or the wording of the
benchmark answer itself as evidence.

Your task is not to answer the benchmark question. The gold answer/judgment is
provided only so you can determine whether the dataset contains evidence that
would support it.

Coverage labels:
- COVERED: one or more target-culture passages directly state or clearly entail
  enough information to support the gold answer/judgment.
- PARTIAL: a target-culture passage discusses the SAME practice, event,
  situation, or norm asked about, but a decisive detail needed to establish the
  gold answer/judgment is missing.
- ABSENT: the exact practice/event/norm needed for the gold target is not
  discussed by the supplied target-culture passages. Merely sharing the same
  culture, broad topic, or value is ABSENT rather than PARTIAL.

Important rules:
1. Only passages about the target culture can count as supporting evidence.
2. For a FALSE hard-item judgment, absence of the candidate practice is NOT
   evidence that it is false. COVERED requires affirmative contrary evidence or
   a passage that directly refutes the candidate.
3. Do not infer a specific practice from a broad value or vaguely related norm.
4. PARTIAL requires direct topical overlap with the exact practice/event/norm.
   A passage about cultural identity, food in general, or an unrelated social
   norm is ABSENT for a question about a different practice.
5. supporting_instance_ids must contain only exact IDs from the supplied
   passages that actually support COVERED or directly relevant PARTIAL evidence.
6. If evidence discusses the exact practice but is ambiguous about the decisive
   detail, choose PARTIAL rather than COVERED.
7. Choose a confidence value from 0 to 1 that reflects your actual confidence;
   do not mechanically copy the example value.
8. Return JSON only, with no markdown.

Return exactly:
{
  "label": "COVERED | PARTIAL | ABSENT",
  "supporting_instance_ids": ["..."],
  "rationale": "brief evidence-grounded explanation",
  "confidence": 0.85
}
"""


def _variant_name(q: MCQQuestion) -> str:
    value = q.variant.value if hasattr(q.variant, "value") else q.variant
    return str(value).lower()


def load_questions(path: Path) -> list[MCQQuestion]:
    return [MCQQuestion(**row) for row in read_jsonl(path)]


def record_to_passage(rec: dict) -> dict:
    """Create the same raw-text passage view used by the corpus builder."""
    parts = []
    for key in (
        "cultural_group", "context", "actor", "recipient", "relation",
        "actor_behavior", "recipient_behavior", "topic", "source_text",
    ):
        value = normalize_text(rec.get(key))
        if value:
            parts.append(value)
    text = ". ".join(dict.fromkeys(parts))
    return {
        "passage_id": rec.get("instance_id", ""),
        "instance_id": rec.get("instance_id", ""),
        "dataset_id": rec.get("dataset_id", "culturebank"),
        "cultural_group": rec.get("cultural_group"),
        "topic": rec.get("topic"),
        "text": text or normalize_text(rec.get("source_text")),
    }


def gold_description(q: MCQQuestion) -> str:
    if q.variant == MCQVariant.EASY:
        answer_text = q.options.get(q.answer_key or "", "")
        options = "\n".join(f"{k}. {v}" for k, v in q.options.items())
        return (
            f"Options:\n{options}\n"
            f"Gold answer: {q.answer_key}. {answer_text}"
        )

    gold = "TRUE" if q.answer_bool else "FALSE"
    return (
        f"Candidate answer: {q.option_text or ''}\n"
        f"Gold judgment for that candidate: {gold}"
    )


def audit_query(q: MCQQuestion) -> str:
    parts = [q.question, q.cultural_group or ""]
    if q.variant == MCQVariant.EASY:
        parts.append(q.options.get(q.answer_key or "", ""))
    elif q.option_text:
        parts.append(q.option_text)
    return " ".join(p for p in parts if p)


def select_candidates(
    q: MCQQuestion,
    passages: list[dict],
    retriever: TextRetriever,
    *,
    max_culture_records: int = 80,
    semantic_fallback_k: int = 10,
) -> tuple[list[dict], int, bool]:
    """Return broad audit candidates and target-culture record count.

    All culture-matched records are included when possible. If a target culture
    has more than max_culture_records, semantic retrieval is used to cap them.
    A small all-corpus semantic fallback catches possible unmatched label
    variants, but cross-cultural fallback records cannot count as support.
    """
    culture_docs = [
        doc for doc in passages
        if culture_match_score(passage_culture(doc), q.cultural_group) >= 0.9
    ]
    culture_count = len(culture_docs)
    culture_truncated = culture_count > max_culture_records

    query = audit_query(q)
    if culture_truncated:
        culture_retriever = TextRetriever(culture_docs)
        culture_docs = [
            doc for doc, _score in culture_retriever.retrieve(
                query,
                top_k=max_culture_records,
                question_culture=q.cultural_group,
            )
        ]

    fallback = [
        doc for doc, _score in retriever.retrieve(
            query,
            top_k=semantic_fallback_k,
            question_culture=q.cultural_group,
        )
    ]

    merged: list[dict] = []
    seen: set[str] = set()
    for doc in culture_docs + fallback:
        instance_id = str(doc.get("instance_id") or doc.get("passage_id") or "").strip()
        key = instance_id or normalize_text(doc.get("text"))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(doc)

    return merged, culture_count, culture_truncated


def format_candidates(
    q: MCQQuestion,
    candidates: list[dict],
    *,
    max_passage_chars: int = 500,
) -> str:
    blocks = []
    for doc in candidates:
        instance_id = str(doc.get("instance_id") or doc.get("passage_id") or "")
        culture = normalize_text(doc.get("cultural_group"))
        topic = normalize_text(doc.get("topic"))
        text = truncate(doc.get("text", ""), max_passage_chars)
        match = culture_match_score(culture, q.cultural_group) >= 0.9
        blocks.append(
            f"[{instance_id}] target_culture_match={str(match).lower()} | "
            f"culture={culture or 'unknown'} | topic={topic or 'unknown'}\n{text}"
        )
    return "\n\n".join(blocks) if blocks else "(no candidate passages)"


def build_audit_prompt(q: MCQQuestion, candidates: list[dict], max_passage_chars: int) -> str:
    return f"""Target culture: {q.cultural_group or 'unknown'}

Benchmark question:
{q.question}

{gold_description(q)}

CultureBank candidate passages:
{format_candidates(q, candidates, max_passage_chars=max_passage_chars)}

Determine whether the supplied target-culture passages contain sufficient
dataset evidence for the gold answer/judgment. Remember that the benchmark
question and gold answer are not themselves evidence.
"""


def parse_audit_output(text: str) -> dict:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\s*\`\`\`(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\`\`\`\s*$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found in coverage-audit output")
    obj = json.loads(cleaned[start:end + 1])

    label = str(obj.get("label", "")).strip().upper()
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid coverage label: {label!r}")

    ids = obj.get("supporting_instance_ids") or []
    if not isinstance(ids, list):
        ids = []
    ids = [str(x).strip() for x in ids if str(x).strip()]

    try:
        confidence = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "label": label,
        "supporting_instance_ids": ids,
        "rationale": normalize_text(obj.get("rationale")),
        "confidence": confidence,
    }


def write_markdown(path: Path, rows: list[dict], run_meta: dict) -> None:
    ensure_dir(path.parent)
    counts = Counter(row["label"] for row in rows)
    easy = Counter(row["label"] for row in rows if row["variant"] == "easy")
    hard = Counter(row["label"] for row in rows if row["variant"] == "hard")

    lines = [
        "# CulturalBench Corpus Coverage Audit",
        "",
        "This audit measures whether the Scale corpus contains target-culture "
        "evidence supporting each controlled CulturalBench gold answer. It does "
        "not measure retrieval or answer accuracy.",
        "",
        f"- Model judge: {run_meta['model']}",
        f"- Corpus passages: {run_meta['num_passages']}",
        f"- Corpus source: {run_meta.get('corpus_source', 'unknown')}",
        f"- Questions: {len(rows)}",
        f"- Semantic fallback k: {run_meta['semantic_fallback_k']}",
        f"- Max target-culture records per question: {run_meta['max_culture_records']}",
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
        "| Question | Variant | Culture | Label | Target-culture records | Support IDs | Confidence | Rationale |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for row in rows:
        rationale = row["rationale"].replace("|", "\\|")
        support = ", ".join(row["supporting_instance_ids"]) or "—"
        culture = (row.get("cultural_group") or "unknown").replace("|", "\\|")
        lines.append(
            f"| {row['question_id']} | {row['variant']} | "
            f"{culture} | "
            f"{row['label']} | {row['target_culture_record_count']} | "
            f"{support} | {row['confidence']:.2f} | {rationale} |"
        )

    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- COVERED means sufficient supporting evidence was found in the inspected target-culture corpus records.",
        "- PARTIAL means related evidence exists but does not establish the benchmark gold target.",
        "- ABSENT means no supporting evidence was found in the inspected records; it is not evidence that the benchmark answer is culturally false.",
        "- This is an automated evidence audit and should be manually checked before publication, especially PARTIAL items and low-confidence labels.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Audit CultureBank coverage for controlled CulturalBench items.")
    p.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument(
        "--questions_jsonl",
        default="data/controlled/food_hospitality_culturalbench_questions.jsonl",
    )
    p.add_argument("--passages", default="data/processed/corpus_passages.jsonl")
    p.add_argument(
        "--records_jsonl",
        default=None,
        help=(
            "Optional normalized CultureBank records JSONL. When supplied, "
            "passages are constructed deterministically in memory using the "
            "same record fields as the corpus builder."
        ),
    )
    p.add_argument("--semantic_fallback_k", type=int, default=10)
    p.add_argument("--max_culture_records", type=int, default=80)
    p.add_argument("--max_passage_chars", type=int, default=500)
    p.add_argument("--output_jsonl", default="data/outputs/coverage_audit.jsonl")
    p.add_argument("--output_md", default="data/outputs/coverage_audit.md")
    args = p.parse_args(argv)

    questions_path = Path(args.questions_jsonl)
    passages_path = Path(args.passages)
    if not questions_path.exists():
        raise FileNotFoundError(questions_path)

    questions = load_questions(questions_path)
    if args.records_jsonl:
        records_path = Path(args.records_jsonl)
        if not records_path.exists():
            raise FileNotFoundError(records_path)
        passages = [record_to_passage(row) for row in read_jsonl(records_path)]
        corpus_source = str(records_path)
    else:
        if not passages_path.exists():
            raise FileNotFoundError(passages_path)
        passages = list(read_jsonl(passages_path))
        corpus_source = str(passages_path)

    retriever = TextRetriever(passages)

    backend = get_backend(args.provider, model=args.model)
    status = backend.status()
    LOGGER.info("Coverage judge backend: %s", status.get("message", ""))
    if not status.get("available"):
        LOGGER.error("Backend unavailable: %s", status.get("message"))
        sys.exit(2)
    if hasattr(backend, "warmup"):
        backend.warmup()

    prompts = []
    metadata = []
    for q in questions:
        candidates, culture_count, culture_truncated = select_candidates(
            q,
            passages,
            retriever,
            max_culture_records=args.max_culture_records,
            semantic_fallback_k=args.semantic_fallback_k,
        )
        prompts.append(build_audit_prompt(q, candidates, args.max_passage_chars))
        metadata.append((q, candidates, culture_count, culture_truncated))

    outputs = backend.generate(prompts, system_prompt=COVERAGE_SYSTEM)

    rows = []
    for (q, candidates, culture_count, culture_truncated), output in zip(metadata, outputs):
        candidate_ids = {
            str(doc.get("instance_id") or doc.get("passage_id") or "").strip()
            for doc in candidates
        }
        candidate_by_id = {
            str(doc.get("instance_id") or doc.get("passage_id") or "").strip(): doc
            for doc in candidates
        }

        parse_error: Optional[str] = None
        try:
            audit = parse_audit_output(output)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not parse audit for %s: %s", q.question_id, exc)
            audit = {
                "label": "PARTIAL",
                "supporting_instance_ids": [],
                "rationale": "Audit output could not be parsed; manual review required.",
                "confidence": 0.0,
            }
            parse_error = str(exc)

        valid_support_ids = []
        invalid_support_ids = []
        cross_culture_support_ids = []
        for instance_id in audit["supporting_instance_ids"]:
            if instance_id not in candidate_ids:
                invalid_support_ids.append(instance_id)
                continue
            doc = candidate_by_id[instance_id]
            if culture_match_score(passage_culture(doc), q.cultural_group) < 0.9:
                cross_culture_support_ids.append(instance_id)
                continue
            valid_support_ids.append(instance_id)

        # A COVERED label must cite at least one valid target-culture record.
        if audit["label"] == "COVERED" and not valid_support_ids:
            audit["label"] = "PARTIAL"
            audit["confidence"] = min(audit["confidence"], 0.5)
            audit["rationale"] = (
                audit["rationale"]
                + " No valid target-culture supporting record was cited; downgraded to PARTIAL."
            ).strip()

        rows.append({
            "question_id": q.question_id,
            "group_id": q.group_id,
            "variant": _variant_name(q),
            "cultural_group": q.cultural_group,
            "question": q.question,
            "gold_description": gold_description(q),
            "label": audit["label"],
            "supporting_instance_ids": valid_support_ids,
            "invalid_support_instance_ids": invalid_support_ids,
            "cross_culture_support_instance_ids": cross_culture_support_ids,
            "rationale": audit["rationale"],
            "confidence": audit["confidence"],
            "target_culture_record_count": culture_count,
            "target_culture_records_truncated": culture_truncated,
            "candidate_count": len(candidates),
            "candidate_instance_ids": sorted(candidate_ids),
            "parse_error": parse_error,
            "raw_judge_output": truncate(output, 3000),
        })

    write_jsonl(args.output_jsonl, rows)
    run_meta = {
        "provider": args.provider,
        "model": backend.model,
        "num_passages": len(passages),
        "corpus_source": corpus_source,
        "semantic_fallback_k": args.semantic_fallback_k,
        "max_culture_records": args.max_culture_records,
        "max_passage_chars": args.max_passage_chars,
    }
    write_markdown(Path(args.output_md), rows, run_meta)

    counts = Counter(row["label"] for row in rows)
    print("\n===== CULTURALBENCH COVERAGE AUDIT =====\n")
    print(f"Corpus passages: {len(passages)}")
    print(f"Questions: {len(rows)}")
    print(
        f"COVERED={counts['COVERED']}  "
        f"PARTIAL={counts['PARTIAL']}  "
        f"ABSENT={counts['ABSENT']}"
    )
    print(f"JSONL: {args.output_jsonl}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
