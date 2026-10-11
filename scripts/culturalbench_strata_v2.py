#!/usr/bin/env python
"""CulturalBench (secondary): re-score the August per-item results by coverage.

Pre-registered in docs/EVAL_V2_PLAN.md: no new GPU run. Each CulturalBench
question's country is mapped to a canonical culture; the number of CultureBank
descriptors (``corpus_passages.jsonl`` rows) with that canonical culture fixes
the stratum: ``<20``, ``20-99``, ``>=100``. Per system and stratum we report
Easy accuracy (per question), Hard accuracy (official group metric: a question
counts only if all of its True/False items are correct; ``compute_units``), and
McNemar vs S0 within the stratum.

Reads (read-only): data/outputs/culturalbench_results.json,
data/processed/corpus_passages.jsonl. Writes data/outputs/eval_v2/culturalbench_strata/.

    python scripts/culturalbench_strata_v2.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core import config  # noqa: E402
from src.core.culture_normalizer import canonical_culture  # noqa: E402
from src.core.schemas import MCQVariant  # noqa: E402
from src.core.utils import get_logger, read_json, read_jsonl, write_json  # noqa: E402

LOGGER = get_logger("culturalbench_strata_v2")

STRATA = ("<20", "20-99", ">=100")
BASELINE = "S0_no_kg"


def stratum_for(count: int) -> str:
    if count < 20:
        return "<20"
    if count < 100:
        return "20-99"
    return ">=100"


def descriptor_counts(groups: list[str]) -> Counter:
    """CultureBank descriptors per canonical culture."""
    return Counter(c for c in (canonical_culture(g) for g in groups) if c)


def assign_strata(results: list[dict], counts: Counter) -> dict[str, dict]:
    """Per CulturalBench country -> {canonical, n_descriptors, stratum}."""
    out = {}
    for r in results:
        g = r.get("cultural_group") or ""
        if g not in out:
            canon = canonical_culture(g)
            n = counts.get(canon, 0) if canon else 0
            out[g] = {"canonical": canon, "n_descriptors": n, "stratum": stratum_for(n)}
    return out


def units_from_results(results: list[dict]):
    """Rebuild (questions, answers_by_system) from the saved per-item results
    and score them with the official unit logic of ``mcq_evaluator.compute_units``."""
    from src.evaluation.mcq_evaluator import compute_units

    systems = sorted({s for r in results for s in r["answers"]})
    questions, answers = [], {s: [] for s in systems}
    for r in results:
        questions.append(SimpleNamespace(question_id=r["question_id"], group_id=r["group_id"],
                                         variant=MCQVariant(r["variant"]),
                                         cultural_group=r["cultural_group"]))
        for s in systems:
            a = r["answers"].get(s)
            answers[s].append(SimpleNamespace(correct=bool(a["correct"])) if a else None)
    return compute_units(questions, answers)


def stratified_table(units_meta: list[dict], correctness: dict[str, list[int]],
                     strata_by_group: dict[str, dict], *, n_boot: int = 10000) -> dict:
    from src.evaluation.mcq_evaluator import mcnemar, paired_bootstrap

    cells: dict[tuple, list[int]] = defaultdict(list)
    for i, u in enumerate(units_meta):
        st = strata_by_group[u["cultural_group"]]["stratum"]
        for key in ((u["variant"], st), ("combined", st), (u["variant"], "all"),
                    ("combined", "all")):
            cells[key].append(i)
    table = {}
    for (variant, st), idxs in sorted(cells.items()):
        sub = {s: [v[i] for i in idxs] for s, v in correctness.items()}
        boot = paired_bootstrap(sub, n_boot=n_boot)
        entry = {"n_units": len(idxs),
                 "n_countries": len({units_meta[i]["cultural_group"] for i in idxs}),
                 "systems": {}}
        for s in sub:
            e = dict(boot[s])
            if s != BASELINE and BASELINE in sub:
                m = mcnemar(sub[s], sub[BASELINE])
                e["vs_S0"] = {"delta": m["delta_accuracy"], "b_sys_only": m["b_a_only"],
                              "c_s0_only": m["c_b_only"], "p_value": m["p_value"]}
            entry["systems"][s] = e
        table[f"{variant}|{st}"] = entry
    return table


def to_markdown(summary: dict) -> str:
    L = ["# CulturalBench re-scored by CultureBank coverage (secondary; Eval v2)", "",
         "Source: August per-item results `data/outputs/culturalbench_results.json` "
         "(Qwen2.5-7B-Instruct, top-k 8). No new GPU run. Strata = CultureBank "
         "descriptors per canonical country (`corpus_passages.jsonl`): <20, 20-99, >=100. "
         "Hard = official group metric (all True/False items correct). McNemar vs S0 is "
         "per stratum, unadjusted (secondary analysis).", "",
         "## Strata", "", "| Stratum | Countries | Easy units | Hard units |", "|---|---|---|---|"]
    t = summary["table"]
    for st in STRATA:
        L.append(f"| {st} | {summary['countries_per_stratum'].get(st, 0)} | "
                 f"{t.get(f'easy|{st}', {}).get('n_units', 0)} | "
                 f"{t.get(f'hard|{st}', {}).get('n_units', 0)} |")
    systems = list(t["combined|all"]["systems"])
    for variant in ("easy", "hard", "combined"):
        L += ["", f"## {variant.capitalize()} accuracy by stratum "
                  "(Δ vs S0, McNemar p)", "",
              "| System | " + " | ".join(STRATA + ("all",)) + " |",
              "|---|" + "---|" * (len(STRATA) + 1)]
        for s in systems:
            row = []
            for st in STRATA + ("all",):
                e = t.get(f"{variant}|{st}", {}).get("systems", {}).get(s)
                if not e:
                    row.append("—")
                    continue
                cell = f"{e['accuracy']:.1%}"
                if "vs_S0" in e:
                    cell += f" ({e['vs_S0']['delta']:+.1%}, p={e['vs_S0']['p_value']:.3g})"
                row.append(cell)
            L.append(f"| {s} | " + " | ".join(row) + " |")
    L += ["", "## Country → stratum", "",
          "| Country | Canonical | CultureBank descriptors | Stratum |", "|---|---|---|---|"]
    for g, v in sorted(summary["countries"].items(), key=lambda kv: -kv[1]["n_descriptors"]):
        L.append(f"| {g} | {v['canonical']} | {v['n_descriptors']} | {v['stratum']} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=config.OUTPUTS_DIR / "culturalbench_results.json")
    ap.add_argument("--passages", type=Path, default=config.PROCESSED_DIR / "corpus_passages.jsonl")
    ap.add_argument("--out_dir", type=Path,
                    default=config.OUTPUTS_DIR / "eval_v2" / "culturalbench_strata")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    results = read_json(args.results)["results"]
    counts = descriptor_counts([p.get("cultural_group") or "" for p in read_jsonl(args.passages)])
    strata = assign_strata(results, counts)
    units_meta, correctness = units_from_results(results)
    LOGGER.info("%d units (%d countries)", len(units_meta), len(strata))
    table = stratified_table(units_meta, correctness, strata, n_boot=args.n_boot)
    summary = {
        "source_results": str(args.results),
        "strata_rule": "CultureBank descriptors per canonical culture: <20, 20-99, >=100",
        "countries_per_stratum": dict(Counter(v["stratum"] for v in strata.values())),
        "countries": strata,
        "table": table,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "summary.json", summary)
    (args.out_dir / "summary.md").write_text(to_markdown(summary))
    LOGGER.info("Wrote %s", args.out_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
