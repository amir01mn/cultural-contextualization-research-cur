#!/usr/bin/env python3
"""Analyze completed KazCulture S1 full vs exact-passage-disjoint strict runs."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.loaders.kazculture_loader import load_test
from src.loaders.kazculture_corpus import read_rows, passage_id


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_id(rows: list[dict], name: str) -> dict[str, dict]:
    out = {}
    for row in rows:
        qid = row["question_id"]
        if qid in out:
            raise ValueError(f"Duplicate {name} question_id: {qid}")
        out[qid] = row
    return out


def accuracy(rows: list[dict]) -> float | None:
    return sum(bool(r["correct"]) for r in rows) / len(rows) if rows else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_file", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--strict", type=Path, required=True)
    p.add_argument("--overlap_file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases_out", type=Path, required=True)
    a = p.parse_args()

    questions, _ = load_test(local_file=str(a.test_file))
    test_rows = read_rows(a.test_file)
    if len(questions) != len(test_rows):
        raise ValueError("Question/test-row alignment mismatch")

    base = by_id(load_jsonl(a.baseline), "baseline")
    full = by_id(load_jsonl(a.full), "full")
    strict = by_id(load_jsonl(a.strict), "strict")
    qids = [q.question_id for q in questions]
    expected = set(qids)
    for name, rows in (("baseline", base), ("full", full), ("strict", strict)):
        if set(rows) != expected:
            raise ValueError(f"{name} IDs do not exactly match the 1,334 test questions")

    overlap = json.loads(a.overlap_file.read_text(encoding="utf-8"))
    overlap_train_ids = set(overlap["exclude_train_passage_ids"])

    q_by_id = {q.question_id: q for q in questions}
    source_by_qid = {
        q.question_id: passage_id(row.get("item_description") or "")
        for q, row in zip(questions, test_rows)
    }

    buckets: Counter[str] = Counter()
    category = defaultdict(lambda: Counter(n=0, full_correct=0, strict_correct=0))
    cases = []
    exact_hit_rows = []
    exact_miss_rows = []
    nonoverlap_rows = []

    for qid in qids:
        q = q_by_id[qid]
        b, f, s = base[qid], full[qid], strict[qid]
        if b["answer_key"] != q.answer_key or f["answer_key"] != q.answer_key or s["answer_key"] != q.answer_key:
            raise ValueError(f"Answer-key mismatch for {qid}")

        fc = bool(f["correct"])
        sc = bool(s["correct"])
        if fc and sc:
            bucket = "both_correct"
        elif fc and not sc:
            bucket = "full_only"
        elif not fc and sc:
            bucket = "strict_only"
        else:
            bucket = "both_wrong"
        buckets[bucket] += 1

        cat = category[q.category]
        cat["n"] += 1
        cat["full_correct"] += int(fc)
        cat["strict_correct"] += int(sc)

        gold_source = source_by_qid[qid]
        overlap_eligible = gold_source in overlap_train_ids
        exact_hit = overlap_eligible and gold_source in set(f.get("retrieved_passage_ids", []))
        row_for_conditional = {"correct": fc}
        if exact_hit:
            exact_hit_rows.append(row_for_conditional)
        elif overlap_eligible:
            exact_miss_rows.append(row_for_conditional)
        else:
            nonoverlap_rows.append(row_for_conditional)

        cases.append({
            "question_id": qid,
            "category": q.category,
            "question": q.question,
            "options": q.options,
            "answer_key": q.answer_key,
            "baseline_prediction": b["prediction"],
            "baseline_correct": bool(b["correct"]),
            "full_prediction": f["prediction"],
            "full_correct": fc,
            "strict_prediction": s["prediction"],
            "strict_correct": sc,
            "full_vs_strict_bucket": bucket,
            "gold_source_passage_id": gold_source,
            "exact_train_source_overlap": overlap_eligible,
            "full_exact_source_retrieved": exact_hit,
            "full_retrieved_passage_ids": f.get("retrieved_passage_ids", []),
            "full_retrieval_scores": f.get("retrieval_scores", []),
            "strict_retrieved_passage_ids": s.get("retrieved_passage_ids", []),
            "strict_retrieval_scores": s.get("retrieval_scores", []),
        })

    category_summary = {}
    for name, c in sorted(category.items()):
        category_summary[name] = {
            "n": c["n"],
            "full_accuracy": c["full_correct"] / c["n"],
            "strict_accuracy": c["strict_correct"] / c["n"],
            "full_minus_strict_pp": round(
                100 * (c["full_correct"] - c["strict_correct"]) / c["n"], 4
            ),
        }

    summary = {
        "n": len(qids),
        "baseline_accuracy": accuracy(list(base.values())),
        "full_accuracy": accuracy(list(full.values())),
        "strict_accuracy": accuracy(list(strict.values())),
        "full_minus_strict_pp": round(
            100 * (accuracy(list(full.values())) - accuracy(list(strict.values()))), 4
        ),
        "paired_full_vs_strict": dict(buckets),
        "overlap_analysis": {
            "exact_train_source_overlap_n": len(exact_hit_rows) + len(exact_miss_rows),
            "full_exact_source_retrieved_n": len(exact_hit_rows),
            "full_exact_source_missed_n": len(exact_miss_rows),
            "full_accuracy_when_exact_source_retrieved": accuracy(exact_hit_rows),
            "full_accuracy_when_exact_source_not_retrieved_but_available": accuracy(exact_miss_rows),
            "nonoverlap_test_questions_n": len(nonoverlap_rows),
            "full_accuracy_on_nonoverlap_test_questions": accuracy(nonoverlap_rows),
        },
        "category_full_vs_strict": category_summary,
        "recommended_case_sets": {
            "full_only": "Questions most dependent on exact-source availability.",
            "both_wrong": "Highest-priority candidates for S2/S4 enrichment analysis.",
            "strict_only": "Cases where full retrieval introduced evidence that hurt relative to strict.",
        },
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    a.cases_out.parent.mkdir(parents=True, exist_ok=True)
    with a.cases_out.open("w", encoding="utf-8") as f:
        for row in cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
