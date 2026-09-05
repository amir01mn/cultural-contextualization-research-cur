"""Step 12: diagnose S4 pathway retrieval culture matching.

This script checks whether the S4 pathway retriever returns evidence from the
same target culture as each selected CulturalBench question. It is intended to
run after scripts/06_build_retrieval_corpus.py has produced a pathway index.

Example:
    python scripts/12_s4_retrieval_diagnostics.py \
      --questions_jsonl data/controlled/food_hospitality_culturalbench_questions.jsonl \
      --pathway_index data/experiments/food_hospitality_multiculture_keep_only_full77/pathway_index \
      --output data/experiments/food_hospitality_multiculture_keep_only_full77/retrieval_diagnostics_s4.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pathway_retriever import PathwayRetriever  # noqa: E402
from src.retrieval_ranking import culture_match_score  # noqa: E402


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def make_query(question_row: dict) -> str:
    """Create the retrieval query from question text and options."""
    parts = [question_row.get("question", "")]
    options = question_row.get("options") or {}
    if isinstance(options, dict):
        for key, value in options.items():
            parts.append(f"{key}. {value}")
    elif question_row.get("option_text"):
        parts.append(str(question_row["option_text"]))
    return "\n".join(p for p in parts if p)


def run_diagnostics(
    questions_jsonl: Path,
    pathway_index: Path,
    output: Path,
    top_k: int,
) -> list[dict]:
    retriever = PathwayRetriever(pathway_index)
    rows: list[dict] = []

    with questions_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            question = json.loads(line)
            question_id = question.get("question_id")
            variant = question.get("variant")
            target_culture = (
                question.get("cultural_group")
                or (question.get("raw") or {}).get("country")
                or ""
            )
            query = make_query(question)

            hits = retriever.retrieve(
                query,
                top_k=top_k,
                question_culture=target_culture,
            )

            hit_cultures = [hit[0].culture for hit in hits]
            hit_completeness = [_enum_value(hit[0].completeness) for hit in hits]
            hit_scores = [hit[1] for hit in hits]
            match_flags = [
                culture_match_score(hit[0].culture, target_culture) >= 0.9
                for hit in hits
            ]

            first_match_rank = ""
            for rank, matched in enumerate(match_flags, 1):
                if matched:
                    first_match_rank = rank
                    break

            rows.append(
                {
                    "question_id": question_id,
                    "variant": variant,
                    "target_culture": target_culture,
                    "answer_key": question.get("answer_key"),
                    "top1_culture": hit_cultures[0] if hit_cultures else "",
                    "top1_completeness": hit_completeness[0] if hit_completeness else "",
                    "top1_score": hit_scores[0] if hit_scores else "",
                    "match_at_1": bool(match_flags[:1] and match_flags[0]),
                    "match_at_3": any(match_flags[:3]),
                    f"match_at_{top_k}": any(match_flags[:top_k]),
                    "first_match_rank": first_match_rank,
                    f"full_at_{top_k}": "FULL" in hit_completeness,
                    "top_cultures": " | ".join(hit_cultures),
                    "top_completeness": " | ".join(hit_completeness),
                    "top_scores": " | ".join(f"{score:.4f}" for score in hit_scores),
                    "question": question.get("question", ""),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def pct(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "0.0%"


def print_summary(rows: list[dict], top_k: int) -> None:
    n = len(rows)
    m1 = sum(bool(row["match_at_1"]) for row in rows)
    m3 = sum(bool(row["match_at_3"]) for row in rows)
    mk = sum(bool(row[f"match_at_{top_k}"]) for row in rows)
    fk = sum(bool(row[f"full_at_{top_k}"]) for row in rows)

    print("Questions:", n)
    print("match@1:", m1, "/", n, pct(m1, n))
    print("match@3:", m3, "/", n, pct(m3, n))
    print(f"match@{top_k}:", mk, "/", n, pct(mk, n))
    print(f"FULL@{top_k}:", fk, "/", n, pct(fk, n))

    print("\nTop-1 retrieved culture counts:")
    for culture, count in Counter(row["top1_culture"] for row in rows).most_common():
        print(f"  {culture}: {count}")

    print(f"\nTarget cultures with no match in top {top_k}:")
    misses = [row for row in rows if not row[f"match_at_{top_k}"]]
    if not misses:
        print("  none")
    for row in misses:
        print(row["question_id"], "|", row["target_culture"], "| top:", row["top_cultures"])

    print("\nCoverage by target culture:")
    by_culture: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_culture[row["target_culture"]].append(row)
    for culture in sorted(by_culture):
        group = by_culture[culture]
        matched = sum(bool(row[f"match_at_{top_k}"]) for row in group)
        print(f"  {culture}: {matched}/{len(group)} match@{top_k} ({pct(matched, len(group))})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose S4 pathway retrieval.")
    parser.add_argument(
        "--questions_jsonl",
        type=Path,
        default=Path("data/controlled/food_hospitality_culturalbench_questions.jsonl"),
    )
    parser.add_argument(
        "--pathway_index",
        type=Path,
        default=Path("data/processed/pathway_index"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/retrieval_diagnostics_s4.csv"),
    )
    parser.add_argument("--top_k", type=int, default=8)
    args = parser.parse_args(argv)

    rows = run_diagnostics(
        questions_jsonl=args.questions_jsonl,
        pathway_index=args.pathway_index,
        output=args.output,
        top_k=args.top_k,
    )
    print("Wrote:", args.output)
    print_summary(rows, top_k=args.top_k)


if __name__ == "__main__":
    main()
