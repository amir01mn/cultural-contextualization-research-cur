"""H2 mechanism analysis: accuracy by pathway completeness and shared-node availability."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .mcq_systems import KG_SYSTEMS, SYSTEM_S4, SYSTEM_S4_NO_SHARED, SYSTEM_S4_TRIPLE
from ..core.schemas import MCQQuestion, MCQSystemAnswer, MCQVariant, PathwayCompleteness
from ..core.utils import get_logger

LOGGER = get_logger("mechanism_analysis")

MECHANISM_SYSTEMS = KG_SYSTEMS | {SYSTEM_S4, SYSTEM_S4_TRIPLE, SYSTEM_S4_NO_SHARED}
COMPLETENESS_BUCKETS = [
    PathwayCompleteness.FULL.value,
    PathwayCompleteness.SOCIAL.value,
    PathwayCompleteness.SURFACE.value,
    PathwayCompleteness.NONE.value,
]


def _unit_correctness(questions: list[MCQQuestion],
                      answers: list[MCQSystemAnswer]) -> list[tuple[str, int]]:
    """Return (unit_id, correct) pairs aligned with eval scoring units."""
    units: list[tuple[str, int]] = []
    hard_groups: dict[str, list[int]] = defaultdict(list)

    for i, q in enumerate(questions):
        ans = answers[i] if i < len(answers) else None
        if q.variant == MCQVariant.EASY:
            correct = 1 if ans and ans.correct else 0
            units.append((q.group_id, correct))
        else:
            hard_groups[q.group_id].append(i)

    for group_id, idxs in hard_groups.items():
        all_correct = all(
            (idx < len(answers) and answers[idx] is not None and answers[idx].correct)
            for idx in idxs
        )
        units.append((group_id, 1 if all_correct else 0))
    return units


def _tag_for_unit(questions: list[MCQQuestion],
                  answers: list[MCQSystemAnswer],
                  unit_id: str) -> tuple[str, bool, float]:
    """Aggregate mechanism tags for one scoring unit (easy item or hard group)."""
    idxs = [i for i, q in enumerate(questions) if q.group_id == unit_id]
    if not idxs:
        return PathwayCompleteness.NONE.value, False, 0.0

    completeness_rank = {
        PathwayCompleteness.FULL.value: 4,
        PathwayCompleteness.SOCIAL.value: 3,
        PathwayCompleteness.SURFACE.value: 2,
        PathwayCompleteness.NONE.value: 1,
    }
    best_bucket = PathwayCompleteness.NONE.value
    best_score = 0.0
    shared = False

    for i in idxs:
        ans = answers[i] if i < len(answers) else None
        if not ans:
            continue
        bucket = ans.completeness or PathwayCompleteness.NONE.value
        score = ans.completeness_score if ans.completeness_score is not None else 0.0
        if completeness_rank.get(bucket, 0) > completeness_rank.get(best_bucket, 0):
            best_bucket = bucket
        best_score = max(best_score, score)
        if ans.shared_node_available:
            shared = True

    return best_bucket, shared, best_score


def summarize_mechanisms(questions: list[MCQQuestion],
                         answers_by_system: dict[str, list[MCQSystemAnswer]]) -> dict:
    """Compute H2 tables: accuracy by completeness bucket and shared-node availability."""
    systems = [s for s in answers_by_system if s in MECHANISM_SYSTEMS]
    out: dict = {"systems": systems, "by_completeness": {}, "by_shared_node": {}}

    for system in systems:
        answers = answers_by_system[system]
        units = _unit_correctness(questions, answers)

        by_comp: dict[str, dict] = {b: {"n": 0, "correct": 0} for b in COMPLETENESS_BUCKETS}
        by_shared: dict[str, dict] = {
            "shared": {"n": 0, "correct": 0},
            "not_shared": {"n": 0, "correct": 0},
        }

        for unit_id, correct in units:
            bucket, shared, _score = _tag_for_unit(questions, answers, unit_id)
            if bucket not in by_comp:
                bucket = PathwayCompleteness.NONE.value
            by_comp[bucket]["n"] += 1
            by_comp[bucket]["correct"] += correct

            key = "shared" if shared else "not_shared"
            by_shared[key]["n"] += 1
            by_shared[key]["correct"] += correct

        out["by_completeness"][system] = {
            b: {
                "n": stats["n"],
                "accuracy": round(stats["correct"] / stats["n"], 4) if stats["n"] else None,
            }
            for b, stats in by_comp.items()
        }
        out["by_shared_node"][system] = {
            k: {
                "n": stats["n"],
                "accuracy": round(stats["correct"] / stats["n"], 4) if stats["n"] else None,
            }
            for k, stats in by_shared.items()
        }

    return out
