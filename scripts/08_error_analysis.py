"""Create a manual error-analysis table from CulturalBench outputs.

This helper is designed for the current main-branch system set:
    S0_no_kg
    S1_raw_text_rag
    S2_flat_kg
    S3_layered_kg
    S4_abstraction_refinement_kg
    S4_triple_retrieval
    S4_no_shared_pivot

It reads data/outputs/culturalbench_results.json from scripts/07_eval_culturalbench.py
and writes compact CSV/Markdown tables of disagreement cases to inspect manually.

Example:
    python scripts/08_error_analysis.py \
        --results data/outputs/culturalbench_results.json \
        --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SYSTEMS = [
    "S0_no_kg",
    "S1_raw_text_rag",
    "S2_flat_kg",
    "S3_layered_kg",
    "S4_abstraction_refinement_kg",
    "S4_triple_retrieval",
    "S4_no_shared_pivot",
]

KG_SYSTEMS = [
    "S2_flat_kg",
    "S3_layered_kg",
    "S4_abstraction_refinement_kg",
    "S4_triple_retrieval",
    "S4_no_shared_pivot",
]


def _load_results(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Expected 'results' list in {path}")
    return results


def _answer(row: dict[str, Any], system: str) -> dict[str, Any] | None:
    answers = row.get("answers") or {}
    return answers.get(system)


def _correct(row: dict[str, Any], system: str) -> bool | None:
    ans = _answer(row, system)
    if ans is None:
        return None
    value = ans.get("correct")
    if value is None:
        return None
    return bool(value)


def _predicted(row: dict[str, Any], system: str) -> str:
    ans = _answer(row, system)
    if ans is None:
        return "n/a"
    pred = ans.get("predicted")
    return "n/a" if pred is None else str(pred)


def _retrieved_ids(row: dict[str, Any], system: str) -> str:
    ans = _answer(row, system)
    if ans is None:
        return ""
    ids = ans.get("retrieved_ids") or []
    return ";".join(str(x) for x in ids)


def _metadata(row: dict[str, Any], system: str, key: str) -> str:
    ans = _answer(row, system)
    if ans is None:
        return ""
    value = ans.get(key)
    return "" if value is None else str(value)


def _pattern(row: dict[str, Any]) -> str:
    s0 = _correct(row, "S0_no_kg")
    s1 = _correct(row, "S1_raw_text_rag")
    s2 = _correct(row, "S2_flat_kg")
    s3 = _correct(row, "S3_layered_kg")
    s4 = _correct(row, "S4_abstraction_refinement_kg")
    s4_triple = _correct(row, "S4_triple_retrieval")
    s4_no_shared = _correct(row, "S4_no_shared_pivot")

    kg_vals = [v for v in (s2, s3, s4, s4_triple, s4_no_shared) if v is not None]
    if s0 is True and any(v is False for v in kg_vals):
        return "S0 correct, retrieval/KG wrong"
    if s0 is False and any(v is True for v in kg_vals):
        return "S0 wrong, retrieval/KG correct"
    if s4 is True and s3 is False:
        return "S4 correct, S3 wrong"
    if s4 is False and s3 is True:
        return "S4 wrong, S3 correct"
    if s4 is True and (s4_triple is False or s4_no_shared is False):
        return "S4 correct, ablation wrong"
    if s4 is False and (s4_triple is True or s4_no_shared is True):
        return "S4 wrong, ablation correct"
    if s1 is True and any(v is False for v in kg_vals):
        return "S1 correct, KG/pathway wrong"
    if s1 is False and any(v is True for v in kg_vals):
        return "S1 wrong, KG/pathway correct"
    return "other"


def _interesting(row: dict[str, Any]) -> bool:
    vals = [_correct(row, s) for s in SYSTEMS]
    known = [v for v in vals if v is not None]
    return len(set(known)) > 1


def _compact_row(row: dict[str, Any]) -> dict[str, str]:
    out = {
        "unit_id": str(row.get("question_id") or row.get("unit_id") or ""),
        "group_id": str(row.get("group_id") or ""),
        "variant": str(row.get("variant") or ""),
        "cultural_group": str(row.get("cultural_group") or ""),
        "pattern": _pattern(row),
        "manual_error_label": "",
        "manual_notes": "",
    }
    for system in SYSTEMS:
        status = _correct(row, system)
        out[f"{system}_correct"] = "n/a" if status is None else str(int(status))
        out[f"{system}_predicted"] = _predicted(row, system)
        out[f"{system}_retrieved_ids"] = _retrieved_ids(row, system)
        out[f"{system}_completeness"] = _metadata(row, system, "completeness")
        out[f"{system}_shared_node_available"] = _metadata(row, system, "shared_node_available")
    return out


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{
            "unit_id": "",
            "group_id": "",
            "variant": "",
            "cultural_group": "",
            "pattern": "no_disagreements_found",
            "manual_error_label": "",
            "manual_notes": "",
        }]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CulturalBench Manual Error Analysis",
        "",
        "Use this table to manually inspect why systems disagree before making strong claims.",
        "",
        "| Unit | Variant | Culture | Pattern | S0 | S1 | S2 | S3 | S4 | S4-triple | S4-no-shared | Manual notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {unit_id} | {variant} | {culture} | {pattern} | "
            "{s0} | {s1} | {s2} | {s3} | {s4} | {s4t} | {s4ns} | {notes} |".format(
                unit_id=row.get("unit_id", ""),
                variant=row.get("variant", ""),
                culture=row.get("cultural_group", ""),
                pattern=row.get("pattern", ""),
                s0=row.get("S0_no_kg_correct", ""),
                s1=row.get("S1_raw_text_rag_correct", ""),
                s2=row.get("S2_flat_kg_correct", ""),
                s3=row.get("S3_layered_kg_correct", ""),
                s4=row.get("S4_abstraction_refinement_kg_correct", ""),
                s4t=row.get("S4_triple_retrieval_correct", ""),
                s4ns=row.get("S4_no_shared_pivot_correct", ""),
                notes=row.get("manual_notes", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CulturalBench error-analysis table.")
    parser.add_argument("--results", default="data/outputs/culturalbench_results.json")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--csv", default="data/outputs/culturalbench_error_analysis.csv")
    parser.add_argument("--md", default="data/outputs/culturalbench_error_analysis.md")
    args = parser.parse_args()

    results = _load_results(Path(args.results))
    disagreement_rows = [_compact_row(row) for row in results if _interesting(row)]
    selected = disagreement_rows[: args.limit]

    _write_csv(selected, Path(args.csv))
    _write_md(selected, Path(args.md))

    print(f"Selected {len(selected)} disagreement cases from {len(results)} result rows.")
    print(f"CSV: {args.csv}")
    print(f"Markdown: {args.md}")


if __name__ == "__main__":
    main()
