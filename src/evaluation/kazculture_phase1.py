"""Closed-book KazCulture MCQ probability scoring utilities.

The evaluator never places item_description, free-text answer, or gold label in
the model prompt. Only the prediction reducer sees gold labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from ..loaders.kazculture_loader import CHOICES, KazCultureQuestion

# Fixed prior to seeing test results. Paper's exact open-model prompt template
# was not released with the dataset; this is a documented reproduction protocol.
PROMPT_VERSION = "kazculture-closedbook-kz-v1"


def question_text(q: KazCultureQuestion) -> str:
    """Kazakh MCQ, no passage or reference answer included."""
    return ("Сұрақ: " + q.question + "\n"
            + "\n".join(f"{label}. {q.options[label]}" for label in CHOICES)
            + "\nДұрыс жауаптың тек әрпін таңдаңыз (A, B, C немесе D).")


def prompt_hash(q: KazCultureQuestion) -> str:
    return hashlib.sha256(question_text(q).encode("utf-8")).hexdigest()


def choose_highest(scores: Mapping[str, float]) -> str:
    """Argmax of conditional log probabilities with documented A/B/C/D tie order."""
    if set(scores) != set(CHOICES):
        raise ValueError(f"Expected exactly A,B,C,D; received {set(scores)}")
    if any(not math.isfinite(float(scores[k])) for k in CHOICES):
        raise ValueError("Non-finite candidate log probability")
    return max(CHOICES, key=lambda k: float(scores[k]))


class LetterScorer(Protocol):
    def score(self, questions: Sequence[KazCultureQuestion]) -> list[dict[str, float]]:
        ...


def evaluate_batch(questions: Sequence[KazCultureQuestion], scorer: LetterScorer) -> list[dict]:
    """Pair letter scores with gold labels outside the model input."""
    scores = scorer.score(questions)
    if len(scores) != len(questions):
        raise ValueError("Scorer returned the wrong number of predictions")
    return [{
        "question_id": q.question_id,
        "category": q.category,
        "prompt_sha256": prompt_hash(q),
        "prediction": choose_highest(s),
        "answer_key": q.answer_key,
        "correct": choose_highest(s) == q.answer_key,
        "letter_logprobs": {k: float(s[k]) for k in CHOICES},
    } for q, s in zip(questions, scores)]


def wilson_95(successes: int, total: int) -> list[float]:
    """95% Wilson score interval for binomial accuracy."""
    if not total:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    rad = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, mid - rad), min(1.0, mid + rad)]


def summarize(records: Sequence[Mapping]) -> dict:
    if not records:
        raise ValueError("No predictions to summarize")
    ids = [r["question_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation IDs")
    n = len(records)
    correct = sum(bool(r["correct"]) for r in records)
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for record in records:
        groups[str(record.get("category") or "Unknown")].append(record)
    categories = {}
    for category, items in sorted(groups.items()):
        k = sum(bool(r["correct"]) for r in items)
        categories[category] = {"n": len(items), "correct": k, "accuracy": k / len(items)}
    return {
        "n": n, "correct": correct, "accuracy": correct / n,
        "accuracy_pct": round(100 * correct / n, 4),
        "wilson_95": wilson_95(correct, n),
        "category_accuracy": categories,
        "prediction_counts": dict(Counter(r["prediction"] for r in records)),
        "published_baseline_pct": 39.51,
        "published_two_stage_finetuned_pct": 64.54,
        "difference_from_published_baseline_pp": round(100 * correct / n - 39.51, 4),
    }


def save_summary(out_dir: Path, records: Sequence[Mapping], metadata: Mapping) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = summarize(records)
    result["metadata"] = dict(metadata)
    (out_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
