"""Compare the current evaluation against the archived baseline run (pipeline step 4 of 4).

Produces the before/after table for the write-up: corpus health, pathway
structure, and per-system CulturalBench accuracy, plus the diagnostics that
explain *why* the systems moved (evidence-withholding rate, error recovery vs
corruption, and the oracle ceiling a perfect router would reach).

    python scripts/compare_runs.py
    python scripts/compare_runs.py --baseline data/outputs/broken_run_baseline
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import config  # noqa: E402
from src.core.utils import get_logger  # noqa: E402

LOGGER = get_logger("compare_runs")

BASELINE_SYSTEM = "S0_no_kg"


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not read %s (%s)", path, exc)
        return None


def _accuracy_table(results: dict) -> dict[str, dict]:
    if not results:
        return {}
    return results.get("summary", {}).get("accuracy", {}).get("combined", {})


def _correct(record: dict, system: str) -> bool | None:
    ans = (record.get("answers") or {}).get(system)
    if not ans:
        return None
    return bool(ans.get("correct"))


def _routing_diagnostics(results: dict) -> dict:
    """Error recovery vs corruption, and the oracle ceiling.

    These are what decide whether augmentation is a net gain: the KG can know
    things the model does not and still lose overall, because there are far more
    correct answers available to corrupt than wrong ones to fix.
    """
    if not results:
        return {}
    records = results.get("results") or []
    systems = [s for s in results.get("summary", {}).get("systems", [])
               if s != BASELINE_SYSTEM]
    out: dict = {"systems": {}}

    base_wrong = [r for r in records if _correct(r, BASELINE_SYSTEM) is False]
    base_right = [r for r in records if _correct(r, BASELINE_SYSTEM) is True]

    for system in systems:
        recovered = sum(1 for r in base_wrong if _correct(r, system) is True)
        broken = sum(1 for r in base_right if _correct(r, system) is False)
        withheld = sum(
            1 for r in records
            if not ((r.get("answers") or {}).get(system, {}) or {}).get("retrieved_ids")
        )
        oracle = sum(
            1 for r in records
            if _correct(r, BASELINE_SYSTEM) is True or _correct(r, system) is True
        )
        out["systems"][system] = {
            "recovered": recovered,
            "recoverable": len(base_wrong),
            "recovery_rate": recovered / len(base_wrong) if base_wrong else 0.0,
            "broken": broken,
            "breakable": len(base_right),
            "breakage_rate": broken / len(base_right) if base_right else 0.0,
            "net": recovered - broken,
            "evidence_withheld": withheld,
            "total_units": len(records),
            "oracle_accuracy": oracle / len(records) if records else 0.0,
        }
    out["baseline_wrong"] = len(base_wrong)
    out["baseline_right"] = len(base_right)
    return out


def _pathway_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    completeness = Counter()
    cultures = Counter()
    shared = 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            completeness[row.get("completeness", "NONE")] += 1
            cultures[row.get("culture", "")] += 1
            if row.get("has_shared_pivot"):
                shared += 1
    return {
        "total": total,
        "completeness": dict(completeness),
        "cultures": len(cultures),
        "shared_pivot": shared,
    }


def build_report(baseline_dir: Path, current_results: Path,
                 current_pathways: Path) -> str:
    base = _load_json(baseline_dir / "culturalbench_results.json")
    cur = _load_json(current_results)

    lines = ["# Before / after comparison", ""]

    if not cur:
        lines.append(f"No current results at `{current_results}`. Run scripts/eval_culturalbench.py first.")
        return "\n".join(lines) + "\n"

    base_acc = _accuracy_table(base)
    cur_acc = _accuracy_table(cur)
    base_meta = (base or {}).get("run_meta", {})
    cur_meta = cur.get("run_meta", {})

    lines += [
        "## Run configuration",
        "",
        "| | baseline (broken) | current |",
        "|---|---|---|",
        f"| model | {base_meta.get('model', 'n/a')} | {cur_meta.get('model', 'n/a')} |",
        f"| scored units | {(base or {}).get('summary', {}).get('num_units_total', 'n/a')} "
        f"| {cur.get('summary', {}).get('num_units_total', 'n/a')} |",
        f"| corpus triples | {base_meta.get('corpus_triples', 'n/a')} "
        f"| {cur_meta.get('corpus_triples', 'n/a')} |",
        f"| pathways | {base_meta.get('corpus_pathways', 'n/a')} "
        f"| {cur_meta.get('corpus_pathways', 'n/a')} |",
        "",
        "> Unit counts differ if CulturalBench-Hard grouping was corrected: Hard is",
        "> scored per question (all True/False items correct), not per item.",
        "",
        "## Combined accuracy",
        "",
        "| System | baseline | current | delta |",
        "|---|---|---|---|",
    ]

    for system in cur.get("summary", {}).get("systems", []):
        c = cur_acc.get(system, {})
        b = base_acc.get(system, {})
        c_txt = f"{c['accuracy'] * 100:.1f}% [{c['ci_low'] * 100:.1f}, {c['ci_high'] * 100:.1f}]" if c else "-"
        b_txt = f"{b['accuracy'] * 100:.1f}%" if b else "-"
        delta = f"{(c['accuracy'] - b['accuracy']) * 100:+.1f}" if (b and c) else "-"
        lines.append(f"| {system} | {b_txt} | {c_txt} | {delta} |")

    s0 = cur_acc.get(BASELINE_SYSTEM, {}).get("accuracy")
    if s0 is not None:
        lines += ["", "## Did any system beat the closed-book baseline?", ""]
        winners = [
            (s, a["accuracy"]) for s, a in cur_acc.items()
            if s != BASELINE_SYSTEM and a.get("accuracy", 0) > s0
        ]
        if winners:
            for s, acc in sorted(winners, key=lambda x: -x[1]):
                lines.append(f"- **{s}: {acc * 100:.1f}%** vs S0 {s0 * 100:.1f}% "
                             f"(+{(acc - s0) * 100:.1f})")
        else:
            lines.append(f"- No. S0 remains best at {s0 * 100:.1f}%.")

        mcnemar = cur.get("summary", {}).get("mcnemar_combined", {})
        if mcnemar:
            lines += ["", "| Comparison | delta | p-value |", "|---|---|---|"]
            for key, m in mcnemar.items():
                if BASELINE_SYSTEM in key:
                    lines.append(
                        f"| {key.replace('__vs__', ' vs ')} "
                        f"| {m['delta_accuracy'] * 100:+.1f}% | {m.get('p_value')} |")

    diag = _routing_diagnostics(cur)
    if diag.get("systems"):
        lines += [
            "",
            "## Routing diagnostics",
            "",
            f"Baseline is wrong on {diag['baseline_wrong']} units and right on "
            f"{diag['baseline_right']}.",
            "",
            "| System | recovers S0 errors | corrupts S0 correct | net | evidence withheld | oracle(S0 or X) |",
            "|---|---|---|---|---|---|",
        ]
        for system, d in diag["systems"].items():
            lines.append(
                f"| {system} | {d['recovered']}/{d['recoverable']} "
                f"({d['recovery_rate'] * 100:.1f}%) | {d['broken']}/{d['breakable']} "
                f"({d['breakage_rate'] * 100:.1f}%) | {d['net']:+d} | "
                f"{d['evidence_withheld']}/{d['total_units']} | "
                f"{d['oracle_accuracy'] * 100:.1f}% |"
            )
        lines += [
            "",
            "*Oracle* is the accuracy a perfect router would reach by choosing",
            "per question between closed-book and that system. It bounds what",
            "better gating alone could deliver on this corpus.",
        ]

    cur_pw = _pathway_stats(current_pathways)
    base_pw = _pathway_stats(baseline_dir / "corpus_pathways.jsonl")
    if cur_pw or base_pw:
        lines += [
            "",
            "## Pathway structure",
            "",
            "| | baseline | current |",
            "|---|---|---|",
            f"| total | {base_pw.get('total', 'n/a')} | {cur_pw.get('total', 'n/a')} |",
            f"| FULL | {base_pw.get('completeness', {}).get('FULL', 0)} "
            f"| {cur_pw.get('completeness', {}).get('FULL', 0)} |",
            f"| SOCIAL | {base_pw.get('completeness', {}).get('SOCIAL', 0)} "
            f"| {cur_pw.get('completeness', {}).get('SOCIAL', 0)} |",
            f"| SURFACE | {base_pw.get('completeness', {}).get('SURFACE', 0)} "
            f"| {cur_pw.get('completeness', {}).get('SURFACE', 0)} |",
            f"| with shared pivot | {base_pw.get('shared_pivot', 0)} "
            f"| {cur_pw.get('shared_pivot', 0)} |",
            f"| distinct cultures | {base_pw.get('cultures', 'n/a')} "
            f"| {cur_pw.get('cultures', 'n/a')} |",
        ]

    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Compare current run against baseline.")
    p.add_argument("--baseline", default=str(config.OUTPUTS_DIR / "broken_run_baseline"))
    p.add_argument("--results", default=str(config.OUTPUTS_DIR / "culturalbench_results.json"))
    p.add_argument("--pathways", default=str(config.PATHWAYS_PATH))
    p.add_argument("--output", default=str(config.OUTPUTS_DIR / "before_after_comparison.md"))
    args = p.parse_args(argv)

    report = build_report(Path(args.baseline), Path(args.results), Path(args.pathways))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report, encoding="utf-8")
    print(report)
    LOGGER.info("Wrote comparison -> %s", args.output)


if __name__ == "__main__":
    main()
