#!/usr/bin/env python3
"""KazCulture v2 test analysis, exactly as in docs/KAZCULTURE_PREREGISTRATION_V2.md §4-§5.

    python scripts/analyze_kazculture_final_v2.py \
      --eval_dir data/kazculture_pipeline/evaluation/final_v2 \
      --olmo_dir data/kazculture_pipeline/evaluation/final_v2_olmo \
      --near_dedup_list data/kazculture_pipeline/audit/near_duplicates_v1/near_dedup_list.json \
      --out data/kazculture_pipeline/evaluation/analysis_v2
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.intrinsic_eval import holm  # noqa: E402
from src.evaluation.kazculture_final import compare_paired, read_jsonl_checkpoint  # noqa: E402

PRIMARY = ("strict/S1", "strict/S1-OG")
SECONDARY = [("strict/S1-OG", "strict/S1-OG+L"), ("strict/S1-OGpo", "strict/S4R-OGF"),
             ("full/S1", "full/S1-OG"), ("full/S1-OG", "full/S1-OG+L")]
DESCRIPTIVE = [("strict/S1", "strict/S1-OGpo"), ("strict/S1", "strict/S1-OG+L"),
               ("strict/S1", "strict/S4R-OGF"), ("closed_book/S0", "strict/S1-OG")]
MIN_EFFECT = 2.0


def load(d: Path, key: str):
    p = d / f"{key}.jsonl"
    rows = read_jsonl_checkpoint(p) if p.exists() else []
    return rows if len(rows) == 1334 else None


def cmp(d, a, b, ids=None):
    ra, rb = load(d, a), load(d, b)
    if ra is None or rb is None:
        return None
    r = compare_paired(ra, rb, question_ids=ids)
    r["baseline"], r["system"] = a, b
    return r


def fmt(r, holm_p=None):
    lo, hi = r["bootstrap_95_ci_pp"]
    h = "" if holm_p is None else f"{holm_p:.3g}"
    return (f"| {r['system']} vs {r['baseline']} | {r['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] "
            f"| +{r['rescued']}/−{r['harmed']} | {r['mcnemar_exact_p']:.3g} | {h} |")


HEAD = "| comparison | Δ pp | 95% CI | rescued/harmed | McNemar p | Holm p |\n|---|---|---|---|---|---|"


def sig(r, p=None):
    p = r["mcnemar_exact_p"] if p is None else p
    lo, hi = r["bootstrap_95_ci_pp"]
    return p < 0.05 and (lo > 0 or hi < 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, required=True)
    ap.add_argument("--olmo_dir", type=Path)
    ap.add_argument("--near_dedup_list", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    d = a.eval_dir
    rep = {"generated_utc": datetime.now(timezone.utc).isoformat()}
    lines = ["# KazCulture v2 test analysis (preregistered)", "",
             f"Generated {rep['generated_utc']}. Plan: `docs/KAZCULTURE_PREREGISTRATION_V2.md`.", ""]

    acc = []
    for key in ["closed_book/S0", "strict/S1", "strict/S1-OG", "strict/S1-OG+L", "strict/S1-OGpo",
                "strict/S4R-OGF", "full/S1", "full/S1-OG", "full/S1-OG+L"]:
        rows = load(d, key)
        if rows:
            c = sum(r["correct"] for r in rows)
            ch = sum(r.get("rendered_chars", 0) for r in rows) / len(rows)
            acc.append((key, c, 100 * c / len(rows), ch))
    rep["accuracy"] = acc
    lines += ["## Accuracy", "", "| run | correct /1334 | acc % | mean evidence chars |",
              "|---|---|---|---|"]
    lines += [f"| {k} | {c} | {p:.2f} | {ch:.0f} |" for k, c, p, ch in acc] + [""]

    prim = cmp(d, *PRIMARY)
    rep["primary"] = prim
    lines += ["## Primary", "", HEAD]
    if prim:
        lines.append(fmt(prim))
    sec = {f"{b} vs {a_}": cmp(d, a_, b) for a_, b in SECONDARY}
    sec = {k: v for k, v in sec.items() if v}
    hp = holm({k: v["mcnemar_exact_p"] for k, v in sec.items()}) if sec else {}
    rep["secondary"] = {k: {**v, "holm_p": hp[k]} for k, v in sec.items()}
    lines += ["", "## Secondary (Holm family)", "", HEAD] + [fmt(v, hp[k]) for k, v in sec.items()]
    desc = [r for r in (cmp(d, a_, b) for a_, b in DESCRIPTIVE) if r]
    rep["descriptive"] = desc
    lines += ["", "## Descriptive", "", HEAD] + [fmt(r) for r in desc]

    nd = None
    if a.near_dedup_list and a.near_dedup_list.exists():
        flagged = set(json.loads(a.near_dedup_list.read_text())["flagged_question_ids"])
        s1 = load(d, "strict/S1")
        if s1:
            keep = {r["question_id"] for r in s1} - flagged
            nd = cmp(d, *PRIMARY, ids=keep)
            rep["near_dedup_primary"] = nd
            lines += ["", f"## Strict near-dedup subset ({len(flagged)} flagged questions removed)",
                      "", HEAD] + ([fmt(nd)] if nd else [])
    ol = None
    if a.olmo_dir:
        ol = cmp(a.olmo_dir, "strict/S1", "strict/S1-OG")
        rep["olmo"] = ol
        lines += ["", "## OLMo-2 replication", "", HEAD] + ([fmt(ol)] if ol else [])

    if prim:
        cats = prim["category_delta"]
        lines += ["", "## Primary by category", "", "| category | n | Δ pp | net |", "|---|---|---|---|"]
        lines += [f"| {k} | {v['n']} | {v['delta_pp']:+.1f} | {v['net']:+d} |" for k, v in cats.items()]

    crit = {}
    if prim:
        crit["1_primary_significant_ge_2pp"] = sig(prim) and prim["delta_pp"] >= MIN_EFFECT
    if nd and ol:
        crit["2_neardedup_and_olmo_same_direction"] = nd["delta_pp"] > 0 and ol["delta_pp"] > 0
    crit["3_test_touched_once"] = "attested in git history (prereg 6e72f21 before run)"
    layer_keys = ["strict/S1-OG+L vs strict/S1-OG", "strict/S4R-OGF vs strict/S1-OGpo"]
    if all(k in rep["secondary"] for k in layer_keys):
        helps = any(rep["secondary"][k]["delta_pp"] > 0 and sig(rep["secondary"][k],
                                                                rep["secondary"][k]["holm_p"])
                    for k in layer_keys)
        crit["4_layers"] = "layers help" if helps else "layers neutral/negative on KazCulture"
    rep["criteria"] = crit
    lines += ["", "## §5 criteria", ""] + [f"- **{k}**: {v}" for k, v in crit.items()]

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "final_analysis_v2.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    (a.out / "final_analysis_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
