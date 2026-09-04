"""Create keep-only and keep+maybe CultureBank subset files from the multiculture audit CSV.

Run from the repository root after completing manual audit:

    python scripts/11_make_keep_only_multiculture_subset.py

The script prefers the original JSONL records file so raw fields are preserved. If a
record is missing there, it reconstructs a normalized record from audit columns.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

BASE = Path("data/controlled")
AUDIT_PATH = BASE / "food_hospitality_multiculture_audit.csv"
RECORDS_PATH = BASE / "food_hospitality_multiculture_records.jsonl"
KEEP_ONLY_OUT = BASE / "food_hospitality_multiculture_records_keep_only.jsonl"
KEEP_PLUS_MAYBE_OUT = BASE / "food_hospitality_multiculture_records_keep_plus_maybe.jsonl"
COUNTS_OUT = BASE / "food_hospitality_multiculture_keep_only_counts.csv"
MAYBE_REVIEW_OUT = BASE / "food_hospitality_multiculture_maybe_review.csv"

TARGET_CULTURES = [
    "south korea", "iran", "bangladesh", "spain", "indonesia", "zimbabwe",
    "netherlands", "new zealand", "turkey", "china", "pakistan",
    "south africa", "japan", "philippines", "russia", "brazil",
]

def split_semis(value: Any) -> list[str]:
    return [x.strip() for x in str(value or "").split(";") if x.strip()]

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

def reconstruct_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "dataset_id": "culturebank",
        "instance_id": row.get("instance_id") or "",
        "cultural_group": row.get("cultural_group") or None,
        "context": row.get("context") or None,
        "goal": None,
        "actor": row.get("actor") or None,
        "recipient": row.get("recipient") or None,
        "relation": row.get("relation") or None,
        "actor_behavior": row.get("actor_behavior") or None,
        "recipient_behavior": row.get("recipient_behavior") or None,
        "topic": row.get("topic") or None,
        "agreement": None,
        "source_text": row.get("source_text") or None,
        "raw": {"reconstructed_from": str(AUDIT_PATH)},
        "controlled_subset": {},
    }

def apply_audit(record: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    record = dict(record)
    meta = dict(record.get("controlled_subset") or {})
    meta.update({
        "topic": "food_hospitality",
        "selection_score": int(float(row.get("selection_score") or 0)),
        "primary_terms": split_semis(row.get("primary_terms")),
        "social_terms": split_semis(row.get("social_terms")),
        "value_terms": split_semis(row.get("value_terms")),
        "selection_reason": row.get("selection_reason") or "",
        "matched_target_cultures": split_semis(row.get("matched_target_cultures")),
        "cleaned_target_cultures": split_semis(row.get("cleaned_target_cultures")),
        "manual_audit_status": (row.get("manual_audit_status") or "").strip().lower(),
        "manual_audit_notes": row.get("manual_audit_notes") or "",
    })
    record["controlled_subset"] = meta
    return record

def main() -> None:
    with AUDIT_PATH.open(encoding="utf-8-sig", newline="") as f:
        audit_rows = list(csv.DictReader(f))

    original = {str(r.get("instance_id")): r for r in read_jsonl(RECORDS_PATH)}

    keep_rows = []
    keep_plus_maybe_rows = []
    maybe_csv_rows = []
    keep_counts: Counter[str] = Counter()
    maybe_counts: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()

    for row in audit_rows:
        status = (row.get("manual_audit_status") or "").strip().lower()
        cleaned = split_semis(row.get("cleaned_target_cultures"))
        for culture in split_semis(row.get("matched_target_cultures")):
            original_counts[culture] += 1
        if status == "maybe":
            maybe_csv_rows.append(row)
        if status not in {"keep", "maybe"} or not cleaned:
            continue
        record = original.get(str(row.get("instance_id"))) or reconstruct_record(row)
        record = apply_audit(record, row)
        keep_plus_maybe_rows.append(record)
        if status == "keep":
            keep_rows.append(record)
            for culture in cleaned:
                keep_counts[culture] += 1
        elif status == "maybe":
            for culture in cleaned:
                maybe_counts[culture] += 1

    write_jsonl(KEEP_ONLY_OUT, keep_rows)
    write_jsonl(KEEP_PLUS_MAYBE_OUT, keep_plus_maybe_rows)

    if maybe_csv_rows:
        with MAYBE_REVIEW_OUT.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=audit_rows[0].keys())
            writer.writeheader()
            writer.writerows(maybe_csv_rows)

    with COUNTS_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "culture", "original_candidates", "keep", "maybe", "keep_plus_maybe", "recommended_use"
        ])
        writer.writeheader()
        for culture in TARGET_CULTURES:
            k = keep_counts[culture]
            m = maybe_counts[culture]
            km = k + m
            if k >= 5:
                rec = "usable keep-only coverage"
            elif k > 0:
                rec = "low keep-only coverage; usable with caution"
            elif km > 0:
                rec = "maybe-only coverage; manually inspect before use"
            else:
                rec = "no usable coverage from current run"
            writer.writerow({
                "culture": culture,
                "original_candidates": original_counts[culture],
                "keep": k,
                "maybe": m,
                "keep_plus_maybe": km,
                "recommended_use": rec,
            })

    print(f"Wrote keep-only records: {len(keep_rows)} -> {KEEP_ONLY_OUT}")
    print(f"Wrote keep+maybe records: {len(keep_plus_maybe_rows)} -> {KEEP_PLUS_MAYBE_OUT}")
    print(f"Wrote maybe review CSV: {len(maybe_csv_rows)} -> {MAYBE_REVIEW_OUT}")
    print(f"Wrote counts: {COUNTS_OUT}")

if __name__ == "__main__":
    main()
