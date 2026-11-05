#!/usr/bin/env python3
"""Blinded 100-frame human review of KazCulture frames v3k labels, and its scorer.

    build   seeded sample (seed 0) of labelled frames -> review CSV + separate key
    score   reviewed CSV + key -> rates; refuses any blank or invalid human field

The CSV shows only a review id, the passage text and the proposed labels: no
passage id, model, or status column. Frames whose label is "none" (the model
found no norm) are included at their natural rate, with empty proposed fields.

Reviewer fields (all must be filled; nothing is pre-filled):
    norm_warranted_0_or_1   1 = the passage states or implies a social norm
    norm_correct_0_or_1     1 = the proposed norm and norm type are supported by
                            the passage (NA when nothing was proposed)
    values_correct_0_or_1   1 = the Schwartz values fit the norm (NA when none)
    notes                   free text (may be empty)

    python scripts/build_kazculture_frame_audit.py build
    python scripts/build_kazculture_frame_audit.py score --reviewed <csv>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "kazculture_pipeline" / "frames_v3k"
COLUMNS = ["review_id", "passage_text", "proposed_norm", "proposed_norm_type",
           "proposed_values", "norm_warranted_0_or_1", "norm_correct_0_or_1",
           "values_correct_0_or_1", "notes"]
HUMAN = ("norm_warranted_0_or_1", "norm_correct_0_or_1", "values_correct_0_or_1")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build(frames: list[dict], norm_types: dict[str, str], *, n: int = 100,
          seed: int = 0) -> tuple[list[dict], dict]:
    """Review rows and the blinding key for a seeded sample of labelled frames."""
    pool = sorted((f for f in frames if f.get("label_status") in ("ok", "none")),
                  key=lambda f: f["frame_id"])
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} labelled frames")
    rng = np.random.default_rng(seed)
    picked = [pool[int(i)] for i in rng.choice(len(pool), size=n, replace=False)]
    rows, key = [], {}
    for k, f in enumerate(picked, 1):          # rng order, so ids reveal nothing
        rid = f"F{k:03d}"
        ok = f["label_status"] == "ok"
        rows.append({
            "review_id": rid, "passage_text": f["text"],
            "proposed_norm": (f.get("norm") or "") if ok else "",
            "proposed_norm_type": (norm_types.get(f.get("norm_type_id"))
                                   or f.get("norm_type_label") or "") if ok else "",
            "proposed_values": "; ".join(f.get("values") or []) if ok else "",
            **{h: "" for h in HUMAN}, "notes": ""})
        key[rid] = {"passage_id": f["frame_id"], "label_status": f["label_status"],
                    "category": f.get("topic", "")}
    return rows, {"seed": seed, "n": n, "keys": key}


def write_csv(path: Path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(xs: list[int]) -> dict:
    n = len(xs)
    if not n:
        return {"n": 0, "rate": None, "wilson_95": None}
    k, z = sum(xs), 1.96
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return {"n": n, "k": k, "rate": round(p, 4),
            "wilson_95": [round(mid - half, 4), round(mid + half, 4)]}


def score(rows: list[dict], key: dict) -> dict:
    """Rates over a fully reviewed sheet; raises on any blank or invalid field."""
    keys = key["keys"]
    ids = [r["review_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate review ids")
    if set(ids) != set(keys):
        raise ValueError("review sheet and key do not match (different samples?)")
    warranted, correct, values, none_right = [], [], [], []
    for r in rows:
        status = keys[r["review_id"]]["label_status"]
        cells = {h: (r.get(h) or "").strip().upper() for h in HUMAN}
        if cells["norm_warranted_0_or_1"] not in ("0", "1"):
            raise ValueError(f"{r['review_id']}: norm_warranted_0_or_1 must be 0 or 1")
        allowed = ("0", "1") if status == "ok" else ("NA",)
        for h in ("norm_correct_0_or_1", "values_correct_0_or_1"):
            if cells[h] not in allowed:
                raise ValueError(f"{r['review_id']}: {h} must be one of {allowed} "
                                 f"(label {status!r}); blanks are refused")
        w = int(cells["norm_warranted_0_or_1"])
        if status == "ok":
            warranted.append(w)
            correct.append(int(cells["norm_correct_0_or_1"]))
            values.append(int(cells["values_correct_0_or_1"]))
        else:
            none_right.append(1 - w)
    return {
        "n_reviewed": len(rows),
        "proposed_norm_warranted": _rate(warranted),
        "proposed_norm_correct": _rate(correct),
        "proposed_values_correct": _rate(values),
        "none_label_correct": _rate(none_right),
        "note": "Human review only; precision of proposed labels on a seeded sample.",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--frames_dir", type=Path, default=FRAMES_DIR)
    b.add_argument("--n", type=int, default=100)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out_dir", type=Path, default=None)
    s = sub.add_parser("score")
    s.add_argument("--reviewed", type=Path, required=True)
    s.add_argument("--key", type=Path, default=FRAMES_DIR / "audit" / "frame_review_100_key.json")
    s.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    if a.cmd == "build":
        out_dir = a.out_dir or a.frames_dir / "audit"
        csv_path = out_dir / f"frame_review_{a.n}.csv"
        key_path = out_dir / f"frame_review_{a.n}_key.json"
        if csv_path.exists() or key_path.exists():
            raise FileExistsError(f"{csv_path} exists; the review sample is frozen")
        frames = _read_jsonl(a.frames_dir / "frames.jsonl")
        nts = {n["norm_type_id"]: n["label"]
               for n in _read_jsonl(a.frames_dir / "full" / "norm_types.jsonl")}
        rows, key = build(frames, nts, n=a.n, seed=a.seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        key["csv_sha256"] = write_csv(csv_path, rows)
        key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {csv_path} and {key_path} (keep the key away from the reviewer)")
        return 0

    key = json.loads(a.key.read_text(encoding="utf-8"))
    with a.reviewed.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    res = score(rows, key)
    text = json.dumps(res, indent=2)
    if a.out:
        a.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
