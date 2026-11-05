#!/usr/bin/env python3
"""Pre-registered analysis of the final KazCulture runs (preregistration §5-§6).

Reads the per-question JSONL written by scripts/eval_kazculture_final.py and
reports: the primary comparison (strict S4R vs S1), the Holm-corrected
secondary family, the two controls (S4R vs CTRL-LEN / CTRL-SHUF, Holm over 2),
descriptive comparisons vs S0, per-category deltas, the OLMo replication, an
optional strict-near-dedup subset, and an automatic checklist of the five §6
criteria (met / not met / pending). Statistics: exact two-sided McNemar and a
paired bootstrap 95% CI (10,000 resamples, seed 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.intrinsic_eval import holm
from src.evaluation.kazculture_final import (
    CLOSED_BOOK, compare_paired, file_sha256, read_jsonl_checkpoint,
)

ALPHA = 0.05
MIN_EFFECT_PP = 2.0
PRIMARY = (("strict", "S4R"), ("strict", "S1"))
SECONDARY = [
    (("strict", "S4R-noabs"), ("strict", "S1")),
    (("strict", "S1+L"), ("strict", "S1")),
    (("full", "S4R"), ("full", "S1")),
    (("full", "S4R-noabs"), ("full", "S1")),
    (("full", "S1+L"), ("full", "S1")),
]
CONTROLS = [
    (("strict", "S4R"), ("strict", "CTRL-LEN")),
    (("strict", "S4R"), ("strict", "CTRL-SHUF")),
]
DESCRIPTIVE_VS_S0 = [("strict", c) for c in ("S1", "S4R", "S4R-noabs", "S1+L",
                                             "CTRL-LEN", "CTRL-SHUF")] + \
                    [("full", c) for c in ("S1", "S4R", "S4R-noabs", "S1+L")]
CATEGORY_MIN_N = 10
CATEGORY_MIN_POSITIVE = 5
PREREG_CATEGORY_COUNT = 7


def label(key) -> str:
    return f"{key[0]}/{key[1]}"


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def tests_hash() -> str:
    """Hash of the test-suite files that guard this harness (criterion 3 audit)."""
    h = hashlib.sha256()
    for path in sorted((ROOT / "tests").glob("test_kazculture*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


class Runs:
    def __init__(self, folder: Path | None, n_expected: int | None):
        self.folder = folder
        self.n_expected = n_expected
        self.cache: dict = {}

    def get(self, key) -> list[dict] | None:
        if self.folder is None:
            return None
        if key not in self.cache:
            setting, cond = key
            sub = CLOSED_BOOK if cond == "S0" else setting
            path = self.folder / sub / f"{cond}.jsonl"
            rows = read_jsonl_checkpoint(path, repair=False) if path.exists() else None
            if rows is not None and self.n_expected is not None and len(rows) != self.n_expected:
                rows = None  # incomplete run: never analyzed
            self.cache[key] = rows
        return self.cache[key]

    def status(self, key) -> str:
        if self.folder is None:
            return "missing"
        setting, cond = key
        path = self.folder / (CLOSED_BOOK if cond == "S0" else setting) / f"{cond}.jsonl"
        if not path.exists():
            return "missing"
        return "complete" if self.get(key) is not None else "incomplete"


def run_comparisons(runs: Runs, pairs, *, n_boot: int, seed: int,
                    question_ids: set | None = None) -> dict:
    out = {}
    for sys_key, base_key in pairs:
        name = f"{label(sys_key)} vs {label(base_key)}"
        s, b = runs.get(sys_key), runs.get(base_key)
        if s is None or b is None:
            out[name] = {"status": "pending",
                         "inputs": {label(sys_key): runs.status(sys_key),
                                    label(base_key): runs.status(base_key)}}
            continue
        out[name] = {"status": "computed",
                     **compare_paired(b, s, n_boot=n_boot, seed=seed, question_ids=question_ids)}
    return out


def add_holm(family: dict) -> None:
    computed = {k: v["mcnemar_exact_p"] for k, v in family.items() if v["status"] == "computed"}
    adj = holm(computed) if computed else {}
    for k, v in family.items():
        if k in adj:
            v["holm_p"] = adj[k]
            v["holm_family_size"] = len(computed)


def significant(res: dict, p_key: str = "mcnemar_exact_p") -> bool:
    lo, hi = res["bootstrap_95_ci_pp"]
    return res[p_key] < ALPHA and (lo > 0 or hi < 0)


def frame_audit_status(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"status": "pending", "reason": "no frame-audit scored JSON "
                "(preregistration §7: no human reviewer available)",
                "path": str(path) if path else None}
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("criterion_met", "passed", "pass", "accepted"):
        if isinstance(data.get(key), bool):
            return {"status": "met" if data[key] else "not met", "source_key": key,
                    "path": str(path)}
    if data.get("reviewer_type") in ("ai", "model", "llm"):
        return {"status": "not met", "reason": "AI screening is not a substitute (§7)",
                "path": str(path)}
    return {"status": "pending", "reason": "scored JSON lacks a boolean verdict key "
            "(criterion_met/passed); needs manual reading", "path": str(path),
            "keys": sorted(data)[:30]}


def checklist(primary: dict, controls: dict, olmo: dict, *, audit: dict,
              integrity: dict, attest: bool) -> dict:
    items = {}
    p = primary
    if p["status"] != "computed":
        items["1_primary_significant_and_ge_2pp"] = {"status": "pending"}
    else:
        ok = significant(p) and p["delta_pp"] >= MIN_EFFECT_PP
        items["1_primary_significant_and_ge_2pp"] = {
            "status": "met" if ok else "not met", "delta_pp": p["delta_pp"],
            "mcnemar_exact_p": p["mcnemar_exact_p"], "ci_pp": p["bootstrap_95_ci_pp"]}
    if any(v["status"] != "computed" for v in controls.values()):
        items["2_beats_ctrl_len_and_ctrl_shuf"] = {"status": "pending"}
    else:
        ok = all(v["delta_pp"] > 0 and v["holm_p"] < ALPHA for v in controls.values())
        items["2_beats_ctrl_len_and_ctrl_shuf"] = {
            "status": "met" if ok else "not met",
            "detail": {k: {"delta_pp": v["delta_pp"], "holm_p": v["holm_p"]}
                       for k, v in controls.items()}}
    items["3_test_touched_once_tuning_on_train"] = {
        "status": "met" if attest else "pending (attestation required)",
        **integrity}
    items["4_frame_labels_human_validated"] = audit
    if p["status"] != "computed":
        items["5_not_concentrated_and_replicates"] = {"status": "pending"}
    else:
        elig = {k: v for k, v in p["category_delta"].items() if v["n"] >= CATEGORY_MIN_N}
        positive = sorted(k for k, v in elig.items() if v["delta_pp"] > 0)
        cat_ok = len(positive) >= CATEGORY_MIN_POSITIVE
        o = olmo.get("strict/S4R vs strict/S1", {"status": "pending"})
        if o["status"] != "computed":
            status = "pending" if cat_ok else "not met"
            rep = None
        else:
            rep = o["delta_pp"] > 0
            status = "met" if (cat_ok and rep) else "not met"
        items["5_not_concentrated_and_replicates"] = {
            "status": status,
            "categories_n_ge_10": len(elig), "positive_categories": positive,
            "required_positive": CATEGORY_MIN_POSITIVE,
            "note": (f"preregistration text says {PREREG_CATEGORY_COUNT} categories with n>=10; "
                     f"the data has {len(elig)}. The rule is applied as written "
                     f"(>= {CATEGORY_MIN_POSITIVE} positive) over all eligible categories."
                     if len(elig) != PREREG_CATEGORY_COUNT else None),
            "olmo_strict_s4r_minus_s1_pp": None if o["status"] != "computed" else o["delta_pp"],
            "olmo_direction_replicates": rep}
    return items


def render_markdown(report: dict) -> str:
    lines = ["# KazCulture final analysis", "",
             f"Generated {report['generated_utc']} at commit `{report['integrity']['git_commit']}`.",
             "", "## Accuracy", "", "| run | n | correct | acc % |", "|---|---|---|---|"]
    for k, v in report["accuracy"].items():
        lines.append(f"| {k} | {v['n']} | {v['correct']} | {100 * v['accuracy']:.2f} |")

    def table(title, fam, holm_col=True):
        out = ["", f"## {title}", "", "| comparison | Δ pp | 95% CI | McNemar p | Holm p |",
               "|---|---|---|---|---|"]
        for k, v in fam.items():
            if v["status"] != "computed":
                out.append(f"| {k} | pending | | | |")
                continue
            lo, hi = v["bootstrap_95_ci_pp"]
            out.append(f"| {k} | {v['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                       f"{v['mcnemar_exact_p']:.3g} | {v.get('holm_p', '') if holm_col else ''} |")
        return out

    lines += table("Primary", {"strict/S4R vs strict/S1": report["primary"]}, holm_col=False)
    lines += table("Secondary (Holm)", report["secondary"])
    lines += table("Controls (Holm over 2)", report["controls"])
    lines += table("Versus S0 (descriptive)", report["vs_s0"], holm_col=False)
    lines += table("OLMo-2 replication", report["olmo"], holm_col=False)
    if report.get("near_dedup_subset"):
        lines += table("Strict-near-dedup subset (descriptive)",
                       report["near_dedup_subset"]["comparisons"], holm_col=False)
    lines += ["", "## §6 checklist", ""]
    for k, v in report["checklist"].items():
        lines.append(f"- **{k}**: {v['status']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--eval_dir", type=Path, required=True)
    p.add_argument("--olmo_dir", type=Path)
    p.add_argument("--test_file", type=Path, required=True)
    p.add_argument("--frame_audit_json", type=Path)
    p.add_argument("--near_dedup_list", type=Path,
                   help="audit_kazculture_near_duplicates.py flagged list JSON")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attest_test_once", action="store_true",
                   help="Assert each test condition was run once with no post-hoc changes")
    a = p.parse_args(argv)

    settings = json.loads((a.eval_dir / "settings.json").read_text(encoding="utf-8"))
    n_exp = settings.get("n_questions")
    test_sha = file_sha256(a.test_file)
    runs = Runs(a.eval_dir, n_exp)
    olmo_runs = Runs(a.olmo_dir, n_exp) if a.olmo_dir else Runs(None, n_exp)
    kw = dict(n_boot=a.n_boot, seed=a.seed)

    accuracy = {}
    for key in [("strict", "S0")] + DESCRIPTIVE_VS_S0:
        rows = runs.get(key)
        if rows is not None:
            name = "closed_book/S0" if key[1] == "S0" else label(key)
            accuracy[name] = {"n": len(rows), "correct": sum(bool(r["correct"]) for r in rows),
                              "accuracy": sum(bool(r["correct"]) for r in rows) / len(rows)}
    primary = run_comparisons(runs, [PRIMARY], **kw)
    primary = next(iter(primary.values()))
    secondary = run_comparisons(runs, SECONDARY, **kw)
    add_holm(secondary)
    controls = run_comparisons(runs, CONTROLS, **kw)
    add_holm(controls)
    vs_s0 = run_comparisons(runs, [(k, ("strict", "S0")) for k in DESCRIPTIVE_VS_S0], **kw)
    olmo = run_comparisons(olmo_runs, [PRIMARY, (("strict", "S1"), ("strict", "S0")),
                                       (("strict", "S4R"), ("strict", "S0"))], **kw)

    near = None
    if a.near_dedup_list and a.near_dedup_list.exists():
        nd = json.loads(a.near_dedup_list.read_text(encoding="utf-8"))
        flagged = set(nd["flagged_question_ids"])
        base = runs.get(("strict", "S1"))
        keep = {r["question_id"] for r in base} - flagged if base else None
        near = {"flagged_questions": len(flagged), "kept_questions": len(keep) if keep else None,
                "criteria": nd.get("criteria"),
                "comparisons": run_comparisons(runs, [PRIMARY] + CONTROLS, question_ids=keep, **kw)
                if keep else {}}

    integrity = {
        "git_commit": git_head(), "test_file_sha256": test_sha,
        "eval_settings_test_file_sha256": settings.get("test_file_sha256"),
        "test_file_hash_matches_eval": settings.get("test_file_sha256") == test_sha,
        "kazculture_tests_sha256": tests_hash(),
        "eval_runs": read_jsonl_checkpoint(a.eval_dir / "runs.jsonl", repair=False)
        if (a.eval_dir / "runs.jsonl").exists() else [],
    }
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "eval_dir": str(a.eval_dir), "olmo_dir": str(a.olmo_dir) if a.olmo_dir else None,
        "statistics": {"mcnemar": "exact two-sided", "bootstrap": kw, "alpha": ALPHA,
                       "min_effect_pp": MIN_EFFECT_PP},
        "accuracy": accuracy, "primary": primary, "secondary": secondary,
        "controls": controls, "vs_s0": vs_s0, "olmo": olmo, "near_dedup_subset": near,
        "integrity": integrity,
    }
    report["checklist"] = checklist(primary, controls, olmo,
                                    audit=frame_audit_status(a.frame_audit_json),
                                    integrity={k: integrity[k] for k in (
                                        "git_commit", "test_file_sha256",
                                        "test_file_hash_matches_eval", "kazculture_tests_sha256")},
                                    attest=a.attest_test_once)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "final_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (a.out / "final_analysis.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in report["checklist"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
