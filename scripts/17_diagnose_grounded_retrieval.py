"""Diagnose why grounded support is missed by each retrieval representation.

Reports, per grounded question and method:
- whether any known support record is represented anywhere in that corpus view
- rank of first known support when retrieving deeply
- which support instance_ids are absent from the representation entirely

This separates construction loss from ranking failure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pathway_retriever import PathwayRetriever  # noqa: E402
from src.retriever import TextRetriever, TripleRetriever  # noqa: E402
from src.schemas import CulturalTriple  # noqa: E402
from src.utils import normalize_text, read_jsonl  # noqa: E402


def make_query(question: dict) -> str:
    parts = [question.get("question", "")]
    if question.get("cultural_group"):
        parts.append(str(question["cultural_group"]))
    options = question.get("options") or {}
    if isinstance(options, dict):
        parts.extend(str(v) for v in options.values())
    elif question.get("option_text"):
        parts.append(str(question["option_text"]))
    return normalize_text(" ".join(p for p in parts if p))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions_jsonl", required=True)
    p.add_argument("--passages", default="data/processed/corpus_passages.jsonl")
    p.add_argument("--raw_triples", default="data/processed/corpus_triples.jsonl")
    p.add_argument("--refined_triples", default="data/processed/corpus_triples_refined.jsonl")
    p.add_argument("--pathway_index", default="data/processed/pathway_index")
    p.add_argument("--deep_k", type=int, default=100)
    p.add_argument(
        "--output",
        default="/scratch/divc/cultural_kg_outputs/grounded_retrieval1000_v1_diagnostics.csv",
    )
    args = p.parse_args(argv)

    questions = list(read_jsonl(Path(args.questions_jsonl)))
    passages = list(read_jsonl(Path(args.passages)))
    raw_triples = [CulturalTriple(**r) for r in read_jsonl(Path(args.raw_triples))]
    refined_triples = [CulturalTriple(**r) for r in read_jsonl(Path(args.refined_triples))]
    pathways = PathwayRetriever(Path(args.pathway_index))

    passage_ids = {
        str(r.get("instance_id") or r.get("passage_id") or "")
        for r in passages
        if (r.get("instance_id") or r.get("passage_id"))
    }
    raw_by_instance = {}
    for t in raw_triples:
        if t.instance_id:
            raw_by_instance.setdefault(str(t.instance_id), []).append(t.triple_id)
    refined_by_instance = {}
    for t in refined_triples:
        if t.instance_id:
            refined_by_instance.setdefault(str(t.instance_id), []).append(t.triple_id)
    pathway_by_instance = {}
    for pw in pathways.pathways:
        for edge in pw.edges:
            if edge.instance_id:
                pathway_by_instance.setdefault(str(edge.instance_id), []).append(pw.pathway_id)

    retrievers = {
        "raw_text": TextRetriever(passages),
        "raw_triples": TripleRetriever(raw_triples),
        "refined_triples": TripleRetriever(refined_triples),
        "pathways": pathways,
    }

    rows = []
    summary = {m: {"represented_questions": 0, "retrieved_questions": 0} for m in retrievers}

    for q in questions:
        qid = str(q.get("question_id"))
        support = {str(x) for x in q.get("support_instance_ids", [])}
        query = make_query(q)
        culture = q.get("cultural_group")

        represented = {
            "raw_text": support & passage_ids,
            "raw_triples": support & set(raw_by_instance),
            "refined_triples": support & set(refined_by_instance),
            "pathways": support & set(pathway_by_instance),
        }

        hits = {
            "raw_text": retrievers["raw_text"].retrieve(
                query, top_k=args.deep_k, question_culture=culture
            ),
            "raw_triples": retrievers["raw_triples"].retrieve(
                query, top_k=args.deep_k, question_culture=culture
            ),
            "refined_triples": retrievers["refined_triples"].retrieve(
                query, top_k=args.deep_k, question_culture=culture
            ),
            "pathways": retrievers["pathways"].retrieve(
                query, top_k=args.deep_k, question_culture=culture
            ),
        }

        for method, method_hits in hits.items():
            first_rank = 0
            found_ids = set()
            for rank, (unit, _score) in enumerate(method_hits, 1):
                if method == "raw_text":
                    ids = {
                        str(unit.get("instance_id") or unit.get("passage_id") or "")
                    }
                elif method in {"raw_triples", "refined_triples"}:
                    ids = {str(unit.instance_id)} if unit.instance_id else set()
                else:
                    ids = {
                        str(edge.instance_id)
                        for edge in unit.edges
                        if edge.instance_id
                    }
                overlap = ids & support
                if overlap:
                    found_ids.update(overlap)
                    if not first_rank:
                        first_rank = rank

            represented_ids = represented[method]
            if represented_ids:
                summary[method]["represented_questions"] += 1
            if first_rank:
                summary[method]["retrieved_questions"] += 1

            rows.append(
                {
                    "question_id": qid,
                    "fact_group_id": q.get("fact_group_id"),
                    "culture": culture,
                    "method": method,
                    "support_total": len(support),
                    "represented_support_count": len(represented_ids),
                    "represented_support_ids": "|".join(sorted(represented_ids)),
                    "missing_from_representation": "|".join(sorted(support - represented_ids)),
                    "first_support_rank_at_deep_k": first_rank,
                    "retrieved_support_count_at_deep_k": len(found_ids),
                    "retrieved_support_ids_at_deep_k": "|".join(sorted(found_ids)),
                }
            )

    out = Path(args.output)
    _write_csv(out, rows)

    print("Questions:", len(questions))
    print("Deep k:", args.deep_k)
    for method in ("raw_text", "raw_triples", "refined_triples", "pathways"):
        s = summary[method]
        print(
            f"{method:16s} represented={s['represented_questions']}/{len(questions)} "
            f"retrieved@{args.deep_k}={s['retrieved_questions']}/{len(questions)}"
        )

    print("\nQuestions where pathway known support is absent from pathway representation:")
    misses = [
        r for r in rows
        if r["method"] == "pathways" and r["represented_support_count"] == 0
    ]
    if not misses:
        print("  none")
    for r in misses:
        print(
            f"  {r['question_id']} | {r['culture']} | "
            f"missing={r['missing_from_representation']}"
        )

    print("\nQuestions where pathway support exists but was not retrieved at deep_k:")
    ranking_misses = [
        r for r in rows
        if r["method"] == "pathways"
        and r["represented_support_count"] > 0
        and r["first_support_rank_at_deep_k"] == 0
    ]
    if not ranking_misses:
        print("  none")
    for r in ranking_misses:
        print(
            f"  {r['question_id']} | {r['culture']} | "
            f"represented={r['represented_support_ids']}"
        )

    print("\nWrote:", out)


if __name__ == "__main__":
    main()
