#!/usr/bin/env python3
"""Audit saved KazCulture predictions; no model/GPU access needed.

Reports gold vs predicted letter counts, confusion matrix, mean candidate
log-probs, and mean candidate margins. Does not change the frozen run.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CHOICES = ("A", "B", "C", "D")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--predictions", default="data/outputs/kazculture_phase1/predictions.jsonl")
    args = parser.parse_args()

    source = Path(args.predictions)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No prediction records")
    ids = [r["question_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in predictions")
    for record in rows:
        if record.get("answer_key") not in CHOICES or record.get("prediction") not in CHOICES:
            raise ValueError(f"Invalid gold/prediction label: {record['question_id']}")
        scores = record.get("letter_logprobs")
        if not isinstance(scores, dict) or set(scores) != set(CHOICES):
            raise ValueError(f"Missing candidate log probabilities: {record['question_id']}")
        if any(not math.isfinite(float(scores[c])) for c in CHOICES):
            raise ValueError(f"Non-finite candidate log probability: {record['question_id']}")

    gold = Counter(r["answer_key"] for r in rows)
    pred = Counter(r["prediction"] for r in rows)
    matrix = {g: {p: 0 for p in CHOICES} for g in CHOICES}
    by_gold = defaultdict(list)
    for r in rows:
        matrix[r["answer_key"]][r["prediction"]] += 1
        by_gold[r["answer_key"]].append(r)

    logprob_means = {c: sum(float(r["letter_logprobs"][c]) for r in rows) / len(rows)
                     for c in CHOICES}
    # Positive means that the selected candidate beat the second best.
    margins = []
    for r in rows:
        values = sorted([float(r["letter_logprobs"][c]) for c in CHOICES], reverse=True)
        margins.append(values[0] - values[1])
    report = {
        "n": len(rows),
        "accuracy_pct": round(100 * sum(r["answer_key"] == r["prediction"] for r in rows) / len(rows), 4),
        "gold_label_counts": {c: gold[c] for c in CHOICES},
        "prediction_counts": {c: pred[c] for c in CHOICES},
        "always_letter_accuracy_pct": {c: round(100 * gold[c] / len(rows), 4) for c in CHOICES},
        "confusion_matrix_rows_gold_columns_prediction": matrix,
        "accuracy_by_gold_label": {
            c: round(sum(r["answer_key"] == r["prediction"] for r in by_gold[c]) / len(by_gold[c]), 4)
            if by_gold[c] else None for c in CHOICES
        },
        "mean_next_token_log_probability_per_letter": {
            c: round(logprob_means[c], 5) for c in CHOICES
        },
        "top2_log_probability_margin_mean": round(sum(margins) / len(margins), 5),
        "top2_log_probability_margin_max": round(max(margins), 5),
        "first_three_example_letter_logprobs": [
            {"question_id": r["question_id"], "gold": r["answer_key"],
             "pred": r["prediction"], "logprobs": r["letter_logprobs"]}
            for r in rows[:3]
        ],
        "interpretation_note": (
            "Next-token label biases may reflect model prompt or tokenizer behavior. "
            "These diagnostics alone cannot identify the paper's exact evaluation setup."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
