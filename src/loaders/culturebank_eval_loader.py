"""CultureBank rows with their KG ``instance_id`` (eval v2, Evaluation 1).

The August corpus build loaded CultureBank through ``culturebank_loader`` from
the HF splits in the order ``tiktok`` then ``reddit`` and, because the CSVs carry
no id column, assigned ``instance_id = stable_id("culturebank", i, prefix="cb_")``
where ``i`` is the row's position in that concatenation. Passages keep
``passage_id == instance_id``; triples keep ``instance_id`` plus the full CSV row
in ``raw_fields``. This module reproduces that id for every CSV row, including
rows that produced no triples.

Depends only on ``src.core``.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Iterable, Optional

from ..core.utils import normalize_text, stable_id

SPLIT_ORDER = ("tiktok", "reddit")
DATASET_ID = "culturebank"


def instance_id_for_row(index: int, dataset_id: str = DATASET_ID) -> str:
    """The corpus ``instance_id`` of the ``index``-th CultureBank row."""
    return stable_id(dataset_id, index, prefix="cb_")


def find_culturebank_csvs(hf_home: str | Path) -> list[Path]:
    """CSV paths in corpus-build order (tiktok, reddit) from an HF cache."""
    base = Path(hf_home) / "hub" / "datasets--SALT-NLP--CultureBank" / "snapshots"
    paths: list[Path] = []
    for split in SPLIT_ORDER:
        found = sorted(glob.glob(str(base / "*" / split / "*.csv")))
        if not found:
            raise FileNotFoundError(f"No CultureBank {split} CSV under {base}")
        paths.append(Path(found[0]))
    return paths


def _to_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rows_with_instance_ids(csv_paths: Iterable[str | Path]) -> list[dict]:
    """Read the CSVs in order; add ``row_index``, ``instance_id`` and ``split``.

    Values are kept as strings (``agreement`` also as ``agreement_f``) and
    whitespace-normalized, matching ``normalize_text`` used by the corpus build.
    """
    rows: list[dict] = []
    for path in csv_paths:
        split = Path(path).parent.name
        with open(path, newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                idx = len(rows)
                row = {k: normalize_text(v) for k, v in raw.items()}
                row["row_index"] = idx
                row["instance_id"] = instance_id_for_row(idx)
                row["split"] = split
                row["agreement_f"] = _to_float(raw.get("agreement"))
                rows.append(row)
    return rows


def read_jsonl_items(path: str | Path) -> list[dict]:
    """Read prepared eval items (one JSON object per line)."""
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
