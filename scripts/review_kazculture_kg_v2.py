#!/usr/bin/env python3
"""Summarize independently completed human audit of KazCulture KG v2.

Triple correctness is assessed on every sampled factual edge. Proposed
pathways have a separate precision denominator; human normative eligibility
independently measures missed opportunities. Never auto-fill human judgments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.construction.kazculture_kg_v2 import summarize_v2_audit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--audit_csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min_rows", type=int, default=40)
    p.add_argument("--minimum_correctness", type=float, default=0.8)
    p.add_argument("--min_proposed_pathways", type=int, default=8)
    args = p.parse_args()

    result = summarize_v2_audit(
        args.audit_csv, min_rows=args.min_rows,
        min_correctness=args.minimum_correctness,
        min_proposed_paths=args.min_proposed_pathways,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["pilot_gate"] != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
