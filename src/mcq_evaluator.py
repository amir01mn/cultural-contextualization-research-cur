"""CulturalBench scoring: accuracy, bootstrap CIs, McNemar tests, outputs.

Scoring units:
  - Easy: each question is one unit; correct iff the chosen letter matches gold.
  - Hard: each underlying question (group) is one unit; correct iff ALL of its
    True/False item judgments are correct (CulturalBench-Hard convention).
  - Combined: all Easy questions + all Hard groups pooled.

We report per-system accuracy with 95% paired-bootstrap confidence intervals and
McNemar significance tests between systems (the paired test for MCQ accuracy).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from . import config
from .mcq_systems import ALL_SYSTEMS
from .schemas import MCQItemResult, MCQQuestion, MCQSystemAnswer, MCQVariant
from .utils import ensure_dir, get_logger, write_json

LOGGER = get_logger("mcq_evaluator")


# ===========================================================================
# Assemble item results and scoring units
# ===========================================================================

def build_item_results(questions: list[MCQQuestion],
                       answers_by_system: dict[str, list[MCQSystemAnswer]]) -> list[MCQItemResult]:
    results = []
    for i, q in enumerate(questions):
        answers = {}
        for system, alist in answers_by_system.items():
            if i < len(alist) and alist[i] is not None:
                answers[system] = alist[i]
        results.append(MCQItemResult(
            question_id=q.question_id,
            group_id=q.group_id,
            variant=q.variant,
            cultural_group=q.cultural_group,
            answers=answers,
        ))
    return results


def compute_units(questions: list[MCQQuestion],
                  answers_by_system: dict[str, list[MCQSystemAnswer]]):
    """Return (units_meta, correctness_by_system).

    units_meta: list of {unit_id, variant, cultural_group} (one per Easy
    question and one per Hard group).
    correctness_by_system: dict system -> list[int] aligned with units_meta.
    """
    systems = list(answers_by_system.keys())

    # Index questions by position for group aggregation.
    easy_units = []  # (unit_id, culture)
    hard_groups: dict[str, dict] = {}  # group_id -> {culture, idxs}
    for i, q in enumerate(questions):
        if q.variant == MCQVariant.EASY:
            easy_units.append((q.group_id, q.cultural_group, i))
        else:
            g = hard_groups.setdefault(q.group_id, {"culture": q.cultural_group, "idxs": []})
            g["idxs"].append(i)

    units_meta = []
    correctness_by_system = {s: [] for s in systems}

    # Easy units.
    for unit_id, culture, i in easy_units:
        units_meta.append({"unit_id": unit_id, "variant": "easy", "cultural_group": culture})
        for s in systems:
            alist = answers_by_system[s]
            correct = bool(alist[i].correct) if i < len(alist) and alist[i] else False
            correctness_by_system[s].append(1 if correct else 0)

    # Hard units (group correct iff all items correct).
    for group_id, info in hard_groups.items():
        units_meta.append({"unit_id": group_id, "variant": "hard",
                           "cultural_group": info["culture"]})
        for s in systems:
            alist = answers_by_system[s]
            all_correct = all(
                (idx < len(alist) and alist[idx] is not None and alist[idx].correct)
                for idx in info["idxs"]
            )
            correctness_by_system[s].append(1 if all_correct else 0)

    return units_meta, correctness_by_system


# ===========================================================================
# Statistics
# ===========================================================================

def _accuracy(vec: list[int]) -> float:
    return sum(vec) / len(vec) if vec else 0.0


def paired_bootstrap(correctness_by_system: dict[str, list[int]],
                     n_boot: int = 10000,
                     seed: int = 42) -> dict:
    """Per-system accuracy 95% CIs using shared (paired) resample indices."""
    import numpy as np

    systems = list(correctness_by_system.keys())
    arrays = {s: np.asarray(correctness_by_system[s], dtype=float) for s in systems}
    n = len(next(iter(arrays.values()))) if arrays else 0
    out = {s: {"accuracy": round(_accuracy(correctness_by_system[s]), 4),
               "ci_low": 0.0, "ci_high": 0.0} for s in systems}
    if n == 0:
        return out
    rng = np.random.default_rng(seed)
    boot = {s: np.empty(n_boot) for s in systems}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        for s in systems:
            boot[s][b] = arrays[s][idx].mean()
    for s in systems:
        lo, hi = np.percentile(boot[s], [2.5, 97.5])
        out[s]["ci_low"] = round(float(lo), 4)
        out[s]["ci_high"] = round(float(hi), 4)
    return out


def mcnemar(vec_a: list[int], vec_b: list[int]) -> dict:
    """McNemar paired test between two systems' per-unit correctness."""
    b = sum(1 for a, bb in zip(vec_a, vec_b) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(vec_a, vec_b) if a == 0 and bb == 1)
    n = b + c
    result = {"b_a_only": b, "c_b_only": c, "n_discordant": n,
              "delta_accuracy": round(_accuracy(vec_a) - _accuracy(vec_b), 4)}
    if n == 0:
        result["p_value"] = 1.0
        result["method"] = "no_discordant_pairs"
        return result
    try:
        from scipy.stats import binomtest

        p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
        result["p_value"] = round(float(p), 6)
        result["method"] = "exact_binomial"
    except Exception:  # noqa: BLE001
        # Normal approximation with continuity correction.
        stat = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
        # Two-sided p from chi-square(1) survival via erfc.
        z = math.sqrt(stat)
        p = math.erfc(z / math.sqrt(2))
        result["p_value"] = round(float(p), 6)
        result["method"] = "normal_approx_cc"
    return result


# ===========================================================================
# Summary
# ===========================================================================

def summarize(questions: list[MCQQuestion],
              answers_by_system: dict[str, list[MCQSystemAnswer]],
              *,
              n_boot: int = 10000,
              seed: int = 42) -> dict:
    units_meta, correctness = compute_units(questions, answers_by_system)
    systems = list(answers_by_system.keys())

    def subset(variant: Optional[str]) -> dict[str, list[int]]:
        idxs = [i for i, m in enumerate(units_meta)
                if variant is None or m["variant"] == variant]
        return {s: [correctness[s][i] for i in idxs] for s in systems}

    combined = subset(None)
    easy = subset("easy")
    hard = subset("hard")

    summary = {
        "systems": systems,
        "num_units_total": len(units_meta),
        "num_units_easy": sum(1 for m in units_meta if m["variant"] == "easy"),
        "num_units_hard": sum(1 for m in units_meta if m["variant"] == "hard"),
        "accuracy": {
            "combined": paired_bootstrap(combined, n_boot, seed),
            "easy": paired_bootstrap(easy, n_boot, seed) if any(easy.values()) else {},
            "hard": paired_bootstrap(hard, n_boot, seed) if any(hard.values()) else {},
        },
    }

    # Pairwise McNemar (combined units) for all ordered system pairs of interest.
    pairwise = {}
    for i, a in enumerate(systems):
        for bsys in systems[i + 1:]:
            key = f"{a}__vs__{bsys}"
            pairwise[key] = mcnemar(combined[a], combined[bsys])
    summary["mcnemar_combined"] = pairwise

    # Per-culture accuracy (combined units).
    per_culture: dict[str, dict] = defaultdict(lambda: {"n": 0})
    for i, m in enumerate(units_meta):
        culture = m["cultural_group"] or "unknown"
        entry = per_culture[culture]
        entry["n"] += 1
        for s in systems:
            entry.setdefault(s, 0)
            entry[s] += correctness[s][i]
    for culture, entry in per_culture.items():
        for s in systems:
            entry[s] = round(entry.get(s, 0) / entry["n"], 4) if entry["n"] else 0.0
    summary["per_culture_accuracy"] = dict(per_culture)

    return summary, units_meta, correctness


# ===========================================================================
# Persistence
# ===========================================================================

def save_outputs(questions: list[MCQQuestion],
                 answers_by_system: dict[str, list[MCQSystemAnswer]],
                 summary: dict,
                 units_meta: list[dict],
                 correctness: dict[str, list[int]],
                 *,
                 outputs_dir=None,
                 run_meta: Optional[dict] = None) -> dict:
    import pandas as pd

    outputs_dir = outputs_dir or config.OUTPUTS_DIR
    ensure_dir(outputs_dir)
    paths = {
        "results": outputs_dir / "culturalbench_results.json",
        "csv": outputs_dir / "culturalbench_results.csv",
        "summary_md": outputs_dir / "culturalbench_summary.md",
        "plot": outputs_dir / "culturalbench_accuracy.png",
    }

    item_results = build_item_results(questions, answers_by_system)
    write_json(paths["results"], {
        "run_meta": run_meta or {},
        "summary": summary,
        "results": [r.model_dump() for r in item_results],
    })

    # Flat CSV: one row per unit per system.
    rows = []
    for i, m in enumerate(units_meta):
        for s in summary["systems"]:
            rows.append({
                "unit_id": m["unit_id"],
                "variant": m["variant"],
                "cultural_group": m["cultural_group"],
                "system": s,
                "correct": correctness[s][i],
            })
    pd.DataFrame(rows).to_csv(paths["csv"], index=False)

    _write_summary_md(paths["summary_md"], summary, run_meta or {})
    _save_plot(paths["plot"], summary)
    return {k: str(v) for k, v in paths.items()}


def _fmt_acc(entry: dict) -> str:
    return f"{entry['accuracy']*100:.1f}% [{entry['ci_low']*100:.1f}, {entry['ci_high']*100:.1f}]"


def _write_summary_md(path, summary: dict, run_meta: dict) -> None:
    systems = summary["systems"]
    combined = summary["accuracy"]["combined"]
    easy = summary["accuracy"].get("easy", {})
    hard = summary["accuracy"].get("hard", {})

    lines = [
        "# CulturalBench Evaluation Summary (Phase 1: baselines S0-S3)",
        "",
        f"- LLM provider: {run_meta.get('provider', config.LLM_PROVIDER)}",
        f"- Model: {run_meta.get('model', config.active_model())}",
        f"- Retrieval top-k: {run_meta.get('top_k', config.RETRIEVAL_TOP_K)}",
        f"- Corpus triples: {run_meta.get('corpus_triples', 'n/a')}, "
        f"passages: {run_meta.get('corpus_passages', 'n/a')}",
        f"- Units: {summary['num_units_total']} "
        f"(easy={summary['num_units_easy']}, hard={summary['num_units_hard']})",
        "",
        "## Accuracy (95% paired-bootstrap CI)",
        "",
        "| System | Combined | Easy | Hard |",
        "|---|---|---|---|",
    ]
    labels = {
        "S0_no_kg": "S0 No-KG",
        "S1_raw_text_rag": "S1 Raw-text RAG",
        "S2_flat_kg": "S2 Flat KG",
        "S3_layered_kg": "S3 Layered KG",
    }
    for s in systems:
        c = combined.get(s, {})
        e = easy.get(s, {})
        h = hard.get(s, {})
        lines.append(
            f"| {labels.get(s, s)} | {_fmt_acc(c) if c else 'n/a'} | "
            f"{_fmt_acc(e) if e else 'n/a'} | {_fmt_acc(h) if h else 'n/a'} |"
        )

    lines += ["", "## McNemar pairwise tests (combined units)", "",
              "| Comparison | delta acc | discordant (b/c) | p-value | method |",
              "|---|---|---|---|---|"]
    for key, m in summary.get("mcnemar_combined", {}).items():
        comp = key.replace("__vs__", " vs ")
        lines.append(
            f"| {comp} | {m['delta_accuracy']*100:+.1f}% | "
            f"{m['b_a_only']}/{m['c_b_only']} | {m.get('p_value')} | {m.get('method')} |"
        )

    lines += [
        "",
        "## Notes",
        "- This is Phase 1 (baselines only). S2/S3 use a raw KG without entity",
        "  resolution; the S4 abstraction->refinement system is a later phase.",
        "- Interpretation per docs/RESEARCH_DESIGN.md: we expect KG-augmented",
        "  systems (S2/S3) to improve over S0/S1 if structure helps.",
    ]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_plot(path, summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib unavailable; skipping plot: %s", exc)
        return

    systems = summary["systems"]
    combined = summary["accuracy"]["combined"]
    labels = [s.split("_", 1)[0] for s in systems]
    accs = [combined.get(s, {}).get("accuracy", 0.0) * 100 for s in systems]
    los = [combined.get(s, {}).get("ci_low", 0.0) * 100 for s in systems]
    his = [combined.get(s, {}).get("ci_high", 0.0) * 100 for s in systems]
    yerr = [[a - lo for a, lo in zip(accs, los)], [hi - a for a, hi in zip(accs, his)]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, accs, yerr=yerr, capsize=6,
                  color=["#999999", "#4C9F70", "#3A7CA5", "#A24C9F"][:len(systems)])
    ax.set_ylabel("CulturalBench accuracy (%) - combined")
    ax.set_title("CulturalBench accuracy by system (95% CI)")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
