#!/usr/bin/env python3
"""Paired comparison of KazCulture-only vs combined KG Qwen3 predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluation.kazculture_rag import paired_statistics, category_report


def read(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--kazculture_only", type=Path, required=True)
    p.add_argument("--combined", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    original = read(a.kazculture_only)
    combined = read(a.combined)
    pairs = paired_statistics(original, combined)
    pairs["kazculture_only_category_accuracy"] = category_report(original)
    pairs["combined_category_accuracy"] = category_report(combined)
    pairs["source_comparison"] = "KazCulture-only S4R versus L3-aligned S4R_combined"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(pairs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
