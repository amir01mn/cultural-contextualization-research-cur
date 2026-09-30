"""Evaluate retrieval against grounded CulturalBench-Easy support records.

This is retrieval-only evaluation. It does not call the answering LLM.

The grounded question JSONL must contain support_instance_ids and
fact_group_id fields, as produced by
scripts/15_build_grounded_easy_retrieval_set.py.

Methods:
  raw_text        CultureBank passages (S1)
  raw_triples     raw triples (same retrieval ranking used by S2 and S3)
  refined_triples critique/refined triple retrieval (S4 triple ablation)
  pathways        abstraction/refinement pathways (S4)

A retrieved unit is relevant when its provenance contains at least one approved
CultureBank support instance_id for the question. Because support labels come
from a high-precision discovery/review process rather than exhaustive human
annotation, known-support recall is a lower-bound diagnostic, not total recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pathway_retriever import PathwayRetriever  # noqa: E402
from src.retriever import TextRetriever, TripleRetriever  # noqa: E402
from src.schemas import CulturalTriple  # noqa: E402
from src.utils import normalize_text, read_jsonl  # noqa: E402


def make_query(question: dict) -> str:
    """Match the production CulturalBench retrieval query."""
    parts = [question.get("question", "")]
    if question.get("cultural_group"):
        parts.append(str(question["cultural_group"]))
    options = question.get("options") or {}
    if isinstance(options, dict):
        parts.extend(str(v) for v in options.values())
    elif question.get("option_text"):
        parts.append(str(question["option_text"]))
    return normalize_text(" ".join(p for p in parts if p))


def triple_instance_ids(triple: CulturalTriple) -> set[str]:
    return {str(triple.instance_id)} if triple.instance_id else set()


def pathway_instance_ids(pathway) -> set[str]:
    return {
        str(edge.instance_id)
        for edge in pathway.edges
        if getattr(edge, "instance_id", None)
    }


def passage_instance_ids(passage: dict) -> set[str]:
    iid = passage.get("instance_id") or passage.get("passage_id")
    return {str(iid)} if iid else set()


def evaluate_ranked_units(
    *,
    method: str,
    question: dict,
    ranked_units: list[tuple[object, float]],
    get_instance_ids,
    ks: list[int],
) -> list[dict]:
    support = {str(x) for x in question.get("support_instance_ids", []) if x}
    if not support:
        raise ValueError(
            f"Question {question.get('question_id')} has no support_instance_ids"
        )

    unit_support_ids: list[set[str]] = []
    unit_ids: list[str] = []
    scores: list[float] = []

    for unit, score in ranked_units:
        ids = set(get_instance_ids(unit))
        unit_support_ids.append(ids.intersection(support))
        scores.append(float(score))
        if method == "raw_text":
            unit_ids.append(
                str(unit.get("passage_id") or unit.get("instance_id") or "")
            )
        elif method in {"raw_triples", "refined_triples"}:
            unit_ids.append(str(unit.triple_id))
        elif method == "pathways":
            unit_ids.append(str(unit.pathway_id))
        else:
            unit_ids.append("")

    first_rank = 0
    for rank, found in enumerate(unit_support_ids, 1):
        if found:
            first_rank = rank
            break

    rows = []
    for k in ks:
        covered = set().union(*unit_support_ids[:k]) if unit_support_ids[:k] else set()
        rows.append(
            {
                "method": method,
                "question_id": question.get("question_id"),
                "fact_group_id": question.get("fact_group_id")
                or f"fact_{question.get('question_id')}",
                "cultural_group": question.get("cultural_group"),
                "k": k,
                "hit": bool(covered),
                "known_support_recall": len(covered) / len(support),
                "known_support_found": len(covered),
                "known_support_total": len(support),
                "first_relevant_rank": first_rank if first_rank and first_rank <= k else 0,
                "rr_at_k": (1.0 / first_rank) if first_rank and first_rank <= k else 0.0,
                "retrieved_unit_ids": "|".join(unit_ids[:k]),
                "retrieved_scores": "|".join(f"{s:.8f}" for s in scores[:k]),
                "retrieved_support_instance_ids": "|".join(sorted(covered)),
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: list[dict], *, by_fact_group: bool) -> list[dict]:
    """Macro-average question metrics, optionally giving each fact group equal weight."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["k"]))].append(row)

    out: list[dict] = []
    for (method, k), group in sorted(grouped.items()):
        if not by_fact_group:
            hit_values = [float(r["hit"]) for r in group]
            recall_values = [float(r["known_support_recall"]) for r in group]
            rr_values = [float(r["rr_at_k"]) for r in group]
            n_units = len(group)
        else:
            by_fact: dict[str, list[dict]] = defaultdict(list)
            for r in group:
                by_fact[str(r["fact_group_id"])].append(r)
            hit_values = []
            recall_values = []
            rr_values = []
            for fact_rows in by_fact.values():
                hit_values.append(_mean([float(r["hit"]) for r in fact_rows]))
                recall_values.append(
                    _mean([float(r["known_support_recall"]) for r in fact_rows])
                )
                rr_values.append(_mean([float(r["rr_at_k"]) for r in fact_rows]))
            n_units = len(by_fact)

        out.append(
            {
                "aggregation": "fact_group_macro" if by_fact_group else "question_macro",
                "method": method,
                "k": k,
                "n_units": n_units,
                "hit_at_k": _mean(hit_values),
                "known_support_recall_at_k": _mean(recall_values),
                "mrr_at_k": _mean(rr_values),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Evaluate retrieval against grounded CulturalBench support records."
    )
    p.add_argument("--questions_jsonl", required=True)
    p.add_argument("--passages", default="data/processed/corpus_passages.jsonl")
    p.add_argument("--raw_triples", default="data/processed/corpus_triples.jsonl")
    p.add_argument(
        "--refined_triples",
        default="data/processed/corpus_triples_refined.jsonl",
    )
    p.add_argument("--pathway_index", default="data/processed/pathway_index")
    p.add_argument("--ks", default="1,3,5,8,12")
    p.add_argument("--output_prefix", default="data/outputs/grounded_retrieval_v1")
    args = p.parse_args(argv)

    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})
    if not ks or min(ks) <= 0:
        raise ValueError("--ks must contain positive integers")
    max_k = max(ks)

    questions = list(read_jsonl(Path(args.questions_jsonl)))
    passages = list(read_jsonl(Path(args.passages)))
    raw_triples = [
        CulturalTriple(**row) for row in read_jsonl(Path(args.raw_triples))
    ]
    refined_triples = [
        CulturalTriple(**row) for row in read_jsonl(Path(args.refined_triples))
    ]

    passage_retriever = TextRetriever(passages)
    raw_retriever = TripleRetriever(raw_triples)
    refined_retriever = TripleRetriever(refined_triples)
    pathway_retriever = PathwayRetriever(Path(args.pathway_index))

    print("Questions:", len(questions))
    print("Passages:", len(passages))
    print("Raw triples:", len(raw_triples))
    print("Refined triples:", len(refined_triples))
    print("Pathways:", len(pathway_retriever.pathways))
    print("k values:", ks)
    print(
        "NOTE: raw_triples is the shared retrieval ranking for S2/S3; "
        "S2 vs S3 differs in evidence formatting, not retrieval ranking."
    )

    detail: list[dict] = []
    for q in questions:
        query = make_query(q)
        culture = q.get("cultural_group")

        method_hits = {
            "raw_text": passage_retriever.retrieve(
                query, top_k=max_k, question_culture=culture
            ),
            "raw_triples": raw_retriever.retrieve(
                query, top_k=max_k, question_culture=culture
            ),
            "refined_triples": refined_retriever.retrieve(
                query, top_k=max_k, question_culture=culture
            ),
            "pathways": pathway_retriever.retrieve(
                query, top_k=max_k, question_culture=culture
            ),
        }
        getters = {
            "raw_text": passage_instance_ids,
            "raw_triples": triple_instance_ids,
            "refined_triples": triple_instance_ids,
            "pathways": pathway_instance_ids,
        }

        for method, hits in method_hits.items():
            detail.extend(
                evaluate_ranked_units(
                    method=method,
                    question=q,
                    ranked_units=hits,
                    get_instance_ids=getters[method],
                    ks=ks,
                )
            )

    summary = aggregate(detail, by_fact_group=False)
    summary.extend(aggregate(detail, by_fact_group=True))

    prefix = Path(args.output_prefix)
    detail_path = Path(str(prefix) + "_detail.csv")
    summary_path = Path(str(prefix) + "_summary.csv")
    json_path = Path(str(prefix) + "_summary.json")
    write_csv(detail_path, detail)
    write_csv(summary_path, summary)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "question_count": len(questions),
                "fact_group_count": len(
                    {
                        q.get("fact_group_id") or f"fact_{q.get('question_id')}"
                        for q in questions
                    }
                ),
                "ks": ks,
                "corpus": {
                    "passages": len(passages),
                    "raw_triples": len(raw_triples),
                    "refined_triples": len(refined_triples),
                    "pathways": len(pathway_retriever.pathways),
                },
                "method_note": (
                    "raw_triples is the shared retrieval ranking for current S2/S3; "
                    "refined_triples is the S4 triple-retrieval ablation."
                ),
                "qrel_note": (
                    "Known support labels are high-precision lower-bound qrels, not "
                    "an exhaustive set of all possibly relevant CultureBank records."
                ),
                "summary": summary,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== GROUNDED RETRIEVAL SUMMARY =====\n")
    for aggregation in ("question_macro", "fact_group_macro"):
        print(aggregation)
        for row in summary:
            if row["aggregation"] != aggregation:
                continue
            print(
                f"  {row['method']:16s} k={row['k']:2d} "
                f"hit={row['hit_at_k']:.3f} "
                f"known_recall={row['known_support_recall_at_k']:.3f} "
                f"mrr={row['mrr_at_k']:.3f}"
            )
        print()

    print("Detail CSV:", detail_path)
    print("Summary CSV:", summary_path)
    print("Summary JSON:", json_path)


if __name__ == "__main__":
    main()
