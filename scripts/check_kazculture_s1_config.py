#!/usr/bin/env python3
"""Assert a (re)tuned S1 config equals the documented frozen one.

Expected: top_k=8, min_similarity=0.45, budget 5,000 chars, train-dev
self-source recall 285/400. Exit 0 on exact match, 3 on any mismatch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.kazculture_final import check_documented_s1_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("config", type=Path)
    p.add_argument("--report", type=Path, help="Write the check JSON here")
    a = p.parse_args(argv)
    rep = check_documented_s1_config(json.loads(a.config.read_text(encoding="utf-8")))
    rep["config"] = str(a.config)
    text = json.dumps(rep, indent=2)
    if a.report:
        a.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if rep["exact"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
