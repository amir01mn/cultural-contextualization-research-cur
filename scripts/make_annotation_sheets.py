"""Human-label sheets for validating the phi-4 judge (docs/EVAL_V3_PLAN.md).

    judge_open.csv     150 open-ended answers, stratified by system and by the
                       judge's verdict; annotators fill `human_entails` (yes/no)
    stereotype.csv     100 audited triples (half judge-flagged as stereotyping);
                       annotators fill `human_stereotyping` (yes/no)

The judge's own label is kept in a separate key file so annotators label blind.

    python scripts/make_annotation_sheets.py \\
        --open_dir data/outputs/eval_v3/culturebank \\
        --triple_audit data/outputs/eval_v2/intrinsic/triple_audit.jsonl \\
        --out_dir data/annotation/v3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_sheet(open_dir: Path, out_dir: Path, n: int, seed: int) -> int:
    answers = {(str(a["qid"]), a["system"]): a for a in _read_jsonl(open_dir / "open_answers.jsonl")}
    verdicts = _read_jsonl(open_dir / "open_judge.jsonl")
    strata = defaultdict(list)
    for v in verdicts:
        strata[(v["system"], bool(v.get("entails")))].append(v)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(strata)))
    picked = []
    for key in sorted(strata, key=str):
        pool = strata[key]
        picked += rng.sample(pool, min(per, len(pool)))
    rng.shuffle(picked)
    picked = picked[:n]
    with (out_dir / "judge_open.csv").open("w", newline="", encoding="utf-8") as f, \
            (out_dir / "judge_open_KEY.csv").open("w", newline="", encoding="utf-8") as k:
        w, wk = csv.writer(f), csv.writer(k)
        w.writerow(["item", "cultural_group", "reference_description", "question", "answer",
                    "human_entails"])
        wk.writerow(["item", "qid", "system", "judge_entails"])
        for i, v in enumerate(picked, 1):
            a = answers[(str(v["qid"]), v["system"])]
            w.writerow([i, a.get("cultural_group", ""), a.get("gold_desc", ""),
                        a.get("question", ""), a.get("answer", ""), ""])
            wk.writerow([i, v["qid"], v["system"], v.get("entails")])
    return len(picked)


def stereotype_sheet(audit_path: Path, out_dir: Path, n: int, seed: int,
                     triples_path: Path | None = None) -> int:
    rows = _read_jsonl(audit_path)
    if triples_path and triples_path.exists():
        text = {t["triple_id"]: t for t in _read_jsonl(triples_path)}
        for r in rows:
            t = text.get(r["triple_id"], {})
            r.setdefault("triple", " | ".join(str(t.get(x, "")) for x in
                                              ("subject", "predicate", "object")))
            r.setdefault("source_text", t.get("source_text", ""))
    flagged = [r for r in rows if r.get("stereotyping") is True]
    clean = [r for r in rows if r.get("stereotyping") is False]
    rng = random.Random(seed)
    picked = rng.sample(flagged, min(n // 2, len(flagged))) + \
        rng.sample(clean, min(n - min(n // 2, len(flagged)), len(clean)))
    rng.shuffle(picked)
    with (out_dir / "stereotype.csv").open("w", newline="", encoding="utf-8") as f, \
            (out_dir / "stereotype_KEY.csv").open("w", newline="", encoding="utf-8") as k:
        w, wk = csv.writer(f), csv.writer(k)
        w.writerow(["item", "culture", "triple", "source_text", "human_stereotyping",
                    "notes"])
        wk.writerow(["item", "triple_id", "judge_stereotyping", "generated"])
        for i, r in enumerate(picked, 1):
            triple = r.get("triple") or " | ".join(
                str(r.get(x, "")) for x in ("subject", "predicate", "object"))
            w.writerow([i, r.get("culture", ""), triple, r.get("source_text", ""), "", ""])
            wk.writerow([i, r.get("triple_id", ""), r.get("stereotyping"), r.get("generated")])
    return len(picked)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--open_dir", type=Path, default=None)
    p.add_argument("--triple_audit", type=Path, default=None)
    p.add_argument("--triples", type=Path, default=None,
                   help="the audit's input triples (for text), e.g. data/eval_v2/intrinsic/triples.jsonl")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--n_open", type=int, default=150)
    p.add_argument("--n_triples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.open_dir:
        print("judge_open.csv rows:", judge_sheet(args.open_dir, args.out_dir, args.n_open, args.seed))
    if args.triple_audit:
        print("stereotype.csv rows:",
              stereotype_sheet(args.triple_audit, args.out_dir, args.n_triples, args.seed,
                               args.triples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
