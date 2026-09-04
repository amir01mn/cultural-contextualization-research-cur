"""Step 10: build culture-matched CultureBank subsets for selected CulturalBench questions.

This script extends the controlled food/hospitality workflow. Step 9 created a
topic-controlled subset. Step 10 makes that subset culture-aware by selecting
CultureBank records that match the cultures appearing in the selected
CulturalBench food/hospitality question file.

Purpose:
    - identify the target cultures in the selected CulturalBench questions
    - scan CultureBank for food/hospitality records from those cultures
    - write one combined multi-culture subset
    - write per-culture JSONL files and an audit CSV
    - report cultures with weak or zero CultureBank coverage

Example:
    python scripts/10_select_culture_matched_subset.py \
      --questions_jsonl data/controlled/food_hospitality_culturalbench_questions.jsonl \
      --scan_limit 50000 \
      --per_culture_max 50

Outputs:
    data/controlled/food_hospitality_multiculture_records.jsonl
    data/controlled/food_hospitality_multiculture_audit.csv
    data/controlled/food_hospitality_multiculture_selection_notes.md
    data/controlled/food_hospitality_multiculture_by_culture/*.jsonl
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import load_culturebank  # noqa: E402
from src.utils import ensure_dir, normalize_text, read_jsonl, write_jsonl  # noqa: E402


CONTROLLED_DIR = config.DATA_DIR / "controlled"
DEFAULT_QUESTIONS = CONTROLLED_DIR / "food_hospitality_culturalbench_questions.jsonl"
DEFAULT_RECORDS_OUT = CONTROLLED_DIR / "food_hospitality_multiculture_records.jsonl"
DEFAULT_AUDIT_OUT = CONTROLLED_DIR / "food_hospitality_multiculture_audit.csv"
DEFAULT_NOTES_OUT = CONTROLLED_DIR / "food_hospitality_multiculture_selection_notes.md"
DEFAULT_BY_CULTURE_DIR = CONTROLLED_DIR / "food_hospitality_multiculture_by_culture"

# Reuse the Step 9 food/hospitality scoring logic so Step 10 remains comparable
# with the earlier controlled subset.
_SELECTOR_PATH = Path(__file__).with_name("09_select_controlled_subset.py")
_SPEC = importlib.util.spec_from_file_location("controlled_subset_selector", _SELECTOR_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load selector module from {_SELECTOR_PATH}")
_SELECTOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SELECTOR)

_text_for_record = _SELECTOR._text_for_record
_score_text = _SELECTOR._score_text
_is_selected = _SELECTOR._is_selected
_selection_reason = _SELECTOR._selection_reason


# These aliases map CulturalBench country labels to likely CultureBank cultural
# group labels. They are deliberately conservative and used only as retrieval
# filters, not as claims that these labels are interchangeable.
CULTURE_ALIASES = {
    "bangladesh": ["bangladesh", "bangladeshi", "bengali"],
    "india": ["india", "indian"],
    "indonesia": ["indonesia", "indonesian"],
    "iran": ["iran", "iranian", "persian"],
    "japan": ["japan", "japanese"],
    "netherlands": ["netherlands", "dutch", "holland"],
    "new zealand": ["new zealand", "new zealander", "māori", "maori", "kiwi"],
    "russia": ["russia", "russian"],
    "south korea": ["south korea", "south korean", "korea", "korean"],
    "spain": ["spain", "spanish"],
    "turkey": ["turkey", "turkish"],
    "united states": ["united states", "american", "usa", "u.s.", "u.s.a."],
    "zimbabwe": ["zimbabwe", "zimbabwean", "shona", "mashona"],
}


def _norm(value: Any) -> str:
    return normalize_text(value).lower()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _norm(value)).strip("_")
    return slug or "unknown"


def _canonical_culture(value: Any) -> str:
    s = _norm(value)
    if not s:
        return ""
    # Normalize aliases back to the selected CulturalBench culture label where possible.
    for canonical, aliases in CULTURE_ALIASES.items():
        if s == canonical or s in aliases:
            return canonical
    return s


def _aliases_for(culture: str) -> list[str]:
    canonical = _canonical_culture(culture)
    aliases = {canonical}
    aliases.update(CULTURE_ALIASES.get(canonical, []))
    aliases.add(culture)
    return sorted(a for a in aliases if a)


def _contains_alias(text: str, alias: str) -> bool:
    alias = _norm(alias)
    if not alias:
        return False
    if " " in alias or "." in alias or "-" in alias:
        return alias in text
    return re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text) is not None


def _question_culture(question: dict[str, Any]) -> str:
    for key in ("cultural_group", "culture", "country", "region"):
        if question.get(key):
            return str(question[key])
    raw = question.get("raw") or {}
    if isinstance(raw, dict):
        for key in ("country", "Country", "cultural_group", "culture", "nation", "region"):
            if raw.get(key):
                return str(raw[key])
    return ""


def _load_target_cultures(args: argparse.Namespace) -> list[str]:
    explicit = [c.strip() for c in args.target_cultures.split(",") if c.strip()]
    cultures = []
    if explicit:
        cultures.extend(explicit)
    if args.questions_jsonl:
        questions = list(read_jsonl(args.questions_jsonl))
        cultures.extend(_question_culture(q) for q in questions)
    canonical = []
    seen = set()
    for culture in cultures:
        c = _canonical_culture(culture)
        if c and c not in seen:
            canonical.append(c)
            seen.add(c)
    return canonical


def _culture_fields_text(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("cultural_group", "region", "country", "culture"):
        if record.get(key):
            parts.append(str(record[key]))
    raw = record.get("raw") or {}
    if isinstance(raw, dict):
        for key in (
            "cultural group", "cultural_group", "culture", "country",
            "Country", "region", "Region", "nation",
        ):
            if raw.get(key):
                parts.append(str(raw[key]))
    return _norm(" ".join(parts))


def _full_record_text(record: dict[str, Any]) -> str:
    return _norm(_text_for_record(record))


def _matched_target_cultures(
    record: dict[str, Any],
    target_cultures: list[str],
    match_full_record_text: bool = False,
) -> list[str]:
    culture_text = _culture_fields_text(record)
    full_text = _full_record_text(record) if match_full_record_text else ""
    matched = []
    for target in target_cultures:
        aliases = _aliases_for(target)
        if any(_contains_alias(culture_text, alias) for alias in aliases):
            matched.append(target)
        elif match_full_record_text and any(_contains_alias(full_text, alias) for alias in aliases):
            matched.append(target)
    return matched


def _enrich_record(record: dict[str, Any], meta: dict[str, Any],
                   matched_cultures: list[str], args: argparse.Namespace) -> dict[str, Any]:
    enriched = dict(record)
    existing = dict(enriched.get("controlled_subset") or {})
    existing.update({
        "topic": args.topic,
        "selection_score": meta["score"],
        "primary_terms": meta["primary_terms"],
        "social_terms": meta["social_terms"],
        "value_terms": meta["value_terms"],
        "selection_reason": _selection_reason(meta),
        "matched_target_cultures": matched_cultures,
        "source_question_file": str(args.questions_jsonl) if args.questions_jsonl else "",
        "manual_audit_status": "unreviewed",
        "manual_audit_notes": "",
    })
    enriched["controlled_subset"] = existing
    return enriched


def _select_records(
    records: list[dict[str, Any]],
    target_cultures: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], Counter]:
    candidates_by_culture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned_culture_counter: Counter = Counter()

    for record in records:
        group = _canonical_culture(record.get("cultural_group") or "unknown")
        scanned_culture_counter[group or "unknown"] += 1

        meta = _score_text(_text_for_record(record))
        if not _is_selected(meta, args.min_score, args.require_social):
            continue

        matched = _matched_target_cultures(
            record,
            target_cultures,
            match_full_record_text=args.match_full_record_text,
        )
        if not matched:
            continue

        enriched = _enrich_record(record, meta, matched, args)
        for culture in matched:
            candidates_by_culture[culture].append(enriched)

    selected_by_culture: dict[str, list[dict[str, Any]]] = {}
    for culture, rows in candidates_by_culture.items():
        rows.sort(
            key=lambda r: (
                r["controlled_subset"]["selection_score"],
                len(r["controlled_subset"]["social_terms"]),
                len(r["controlled_subset"]["primary_terms"]),
            ),
            reverse=True,
        )
        selected_by_culture[culture] = rows[: args.per_culture_max]

    # Combine selected rows without duplicating a CultureBank instance that matched
    # more than one target culture.
    combined_by_id: dict[str, dict[str, Any]] = {}
    for culture in target_cultures:
        for row in selected_by_culture.get(culture, []):
            key = str(row.get("instance_id") or json.dumps(row, sort_keys=True, default=str))
            if key not in combined_by_id:
                combined_by_id[key] = row
            else:
                existing_meta = combined_by_id[key].setdefault("controlled_subset", {})
                merged = set(existing_meta.get("matched_target_cultures", []))
                merged.update(row.get("controlled_subset", {}).get("matched_target_cultures", []))
                existing_meta["matched_target_cultures"] = sorted(merged)

    combined = list(combined_by_id.values())
    combined.sort(
        key=lambda r: (
            ",".join(r.get("controlled_subset", {}).get("matched_target_cultures", [])),
            -int(r.get("controlled_subset", {}).get("selection_score", 0)),
            str(r.get("instance_id", "")),
        )
    )
    return combined, selected_by_culture, scanned_culture_counter


def _write_audit_csv(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fields = [
        "instance_id", "cultural_group", "matched_target_cultures", "topic",
        "context", "selection_score", "primary_terms", "social_terms",
        "value_terms", "selection_reason", "manual_audit_status",
        "manual_audit_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            meta = record.get("controlled_subset", {})
            writer.writerow({
                "instance_id": record.get("instance_id", ""),
                "cultural_group": record.get("cultural_group", ""),
                "matched_target_cultures": ";".join(meta.get("matched_target_cultures", [])),
                "topic": record.get("topic", ""),
                "context": record.get("context", ""),
                "selection_score": meta.get("selection_score", ""),
                "primary_terms": ";".join(meta.get("primary_terms", [])),
                "social_terms": ";".join(meta.get("social_terms", [])),
                "value_terms": ";".join(meta.get("value_terms", [])),
                "selection_reason": meta.get("selection_reason", ""),
                "manual_audit_status": meta.get("manual_audit_status", "unreviewed"),
                "manual_audit_notes": meta.get("manual_audit_notes", ""),
            })


def _write_by_culture(selected_by_culture: dict[str, list[dict[str, Any]]],
                      out_dir: Path) -> None:
    ensure_dir(out_dir)
    for culture, rows in selected_by_culture.items():
        write_jsonl(out_dir / f"{_slug(culture)}.jsonl", rows)


def _write_notes(
    selected: list[dict[str, Any]],
    selected_by_culture: dict[str, list[dict[str, Any]]],
    scanned_cultures: Counter,
    target_cultures: list[str],
    args: argparse.Namespace,
    path: Path,
) -> None:
    ensure_dir(path.parent)
    records_by_culture = Counter()
    for record in selected:
        for culture in record.get("controlled_subset", {}).get("matched_target_cultures", []):
            records_by_culture[culture] += 1

    no_coverage = [c for c in target_cultures if records_by_culture[c] == 0]
    low_coverage = [
        c for c in target_cultures
        if 0 < records_by_culture[c] < args.min_report_coverage
    ]

    lines = [
        "# Culture-Matched Food/Hospitality Subset Notes",
        "",
        "## Purpose",
        "",
        "This file summarizes the CultureBank food/hospitality records selected for the cultures that appear in the selected CulturalBench food/hospitality questions.",
        "The goal is to make the next S0-S4 evaluation culture-matched, rather than relying on an American-only KG corpus.",
        "",
        "## Settings",
        "",
        f"- Topic: `{args.topic}`",
        f"- Questions file: `{args.questions_jsonl}`",
        f"- CultureBank scan limit: {args.scan_limit}",
        f"- Per-culture maximum: {args.per_culture_max}",
        f"- Minimum record score: {args.min_score}",
        f"- Require social/hospitality term: {args.require_social}",
        f"- Match full record text for culture aliases: {args.match_full_record_text}",
        f"- Combined selected records: {len(selected)}",
        "",
        "## Target cultures from selected CulturalBench questions",
        "",
        "| Target culture | Selected CultureBank records | Status |",
        "|---|---:|---|",
    ]
    for culture in target_cultures:
        count = records_by_culture[culture]
        if count == 0:
            status = "no CultureBank coverage found"
        elif count < args.min_report_coverage:
            status = "low coverage; inspect manually"
        else:
            status = "usable candidate coverage"
        lines.append(f"| {culture} | {count} | {status} |")

    lines += [
        "",
        "## Cultures with no or low selected coverage",
        "",
    ]
    if no_coverage:
        lines.append("- No selected records: " + ", ".join(no_coverage))
    else:
        lines.append("- No selected records: none")
    if low_coverage:
        lines.append("- Low selected coverage: " + ", ".join(low_coverage))
    else:
        lines.append("- Low selected coverage: none")
    lines += [
        "",
        "## Most common CultureBank cultural_group values scanned",
        "",
        "| CultureBank cultural_group | Scanned count |",
        "|---|---:|",
    ]
    for culture, count in scanned_cultures.most_common(30):
        lines.append(f"| {culture} | {count} |")

    lines += [
        "",
        "## Manual audit instructions",
        "",
        "Open the audit CSV and set `manual_audit_status` to one of:",
        "",
        "- `keep`: culturally meaningful food/hospitality evidence for the matched target culture",
        "- `maybe`: relevant but weak, ambiguous, or missing enough social/value context",
        "- `remove`: incidental food/hospitality mention, wrong culture, or not useful for cultural reasoning",
        "",
        "During audit, check whether each row supports L1 surface practice, L2 social norm/role, L3 deep value, and a plausible pathway.",
        "No outside cultural knowledge should be added during this audit; use the normalized fields and source text only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select culture-matched CultureBank food/hospitality records."
    )
    parser.add_argument("--topic", default="food_hospitality", choices=["food_hospitality"])
    parser.add_argument("--dataset", default=config.DATASET_NAME)
    parser.add_argument("--split", default=config.DEFAULT_SPLIT)
    parser.add_argument("--scan_limit", type=int, default=50000)
    parser.add_argument("--per_culture_max", type=int, default=50)
    parser.add_argument("--min_score", type=int, default=3)
    parser.add_argument("--require_social", action="store_true", default=True)
    parser.add_argument(
        "--input_records",
        default=None,
        help="Optional normalized CultureBank JSONL to filter instead of loading HF.",
    )
    parser.add_argument(
        "--questions_jsonl",
        default=str(DEFAULT_QUESTIONS),
        help="Selected CulturalBench question JSONL used to derive target cultures.",
    )
    parser.add_argument(
        "--target_cultures",
        default="",
        help="Optional comma-separated target cultures. Combined with --questions_jsonl.",
    )
    parser.add_argument(
        "--match_full_record_text",
        action="store_true",
        default=False,
        help="Also match culture aliases in full record text, not only culture fields.",
    )
    parser.add_argument("--min_report_coverage", type=int, default=5)
    parser.add_argument("--records_output", default=str(DEFAULT_RECORDS_OUT))
    parser.add_argument("--audit_output", default=str(DEFAULT_AUDIT_OUT))
    parser.add_argument("--notes_output", default=str(DEFAULT_NOTES_OUT))
    parser.add_argument("--by_culture_dir", default=str(DEFAULT_BY_CULTURE_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    target_cultures = _load_target_cultures(args)
    if not target_cultures:
        raise RuntimeError(
            "No target cultures found. Provide --questions_jsonl or --target_cultures."
        )

    if args.input_records:
        records = list(read_jsonl(args.input_records))
    else:
        records = load_culturebank(args.dataset, args.split, args.scan_limit)

    selected, selected_by_culture, scanned_cultures = _select_records(
        records, target_cultures, args
    )

    write_jsonl(args.records_output, selected)
    _write_audit_csv(selected, Path(args.audit_output))
    _write_by_culture(selected_by_culture, Path(args.by_culture_dir))
    _write_notes(
        selected,
        selected_by_culture,
        scanned_cultures,
        target_cultures,
        args,
        Path(args.notes_output),
    )

    print(f"Target cultures: {', '.join(target_cultures)}")
    print(f"Scanned CultureBank records: {len(records)}")
    print(f"Selected combined CultureBank records: {len(selected)}")
    print(f"Records JSONL: {args.records_output}")
    print(f"Audit CSV: {args.audit_output}")
    print(f"Notes: {args.notes_output}")
    print(f"By-culture output directory: {args.by_culture_dir}")
    print("Per-culture selected counts:")
    for culture in target_cultures:
        print(f"  {culture}: {len(selected_by_culture.get(culture, []))}")


if __name__ == "__main__":
    main()
