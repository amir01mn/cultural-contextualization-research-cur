"""Step 9: select a controlled CultureBank topic subset before broad scaling.

This script implements Step 6 from the meeting action plan. It creates a focused
food/hospitality subset so we can debug extraction quality, node canonicalization,
pathway construction, and S4 retrieval behavior before scaling to thousands of
random CultureBank rows.

Examples:
    # Select up to 100 food/hospitality CultureBank records from a 2000-row scan.
    python scripts/09_select_controlled_subset.py --scan_limit 2000 --max_records 100

    # Also collect matching CulturalBench questions for a future controlled eval.
    python scripts/09_select_controlled_subset.py --include_culturalbench --cb_limit 250

Outputs:
    data/controlled/food_hospitality_records.jsonl
    data/controlled/food_hospitality_audit.csv
    data/controlled/food_hospitality_selection_notes.md
    data/controlled/food_hospitality_culturalbench_questions.jsonl  (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.culturalbench_loader import load_culturalbench  # noqa: E402
from src.data_loader import load_culturebank  # noqa: E402
from src.utils import ensure_dir, normalize_text, read_jsonl, write_jsonl  # noqa: E402


CONTROLLED_DIR = config.DATA_DIR / "controlled"

PRIMARY_TERMS = [
    "food", "foods", "meal", "meals", "eat", "eating", "dish", "dishes",
    "drink", "drinks", "tea", "coffee", "cafe", "restaurant", "dinner",
    "lunch", "breakfast", "snack", "dessert", "bread", "rice", "injera",
    "feast", "banquet", "cooking", "cook", "serve food", "served food",
]

SOCIAL_TERMS = [
    "hospitality", "hospitable", "guest", "guests", "host", "hosts",
    "visitor", "visitors", "welcome", "welcoming", "offer", "offering",
    "serve", "serving", "share", "sharing", "refuse", "refusing", "accept",
    "accepting", "invite", "invitation", "family meal", "gathering",
    "celebration", "ceremony", "polite", "impolite", "etiquette", "respect",
]

VALUE_TERMS = [
    "respect", "care", "friendship", "belonging", "generosity", "gratitude",
    "honor", "honour", "family", "community", "solidarity", "purity", "modesty",
]

# These often indicate weak topic matches unless paired with social/cultural terms.
WEAK_ONLY_TERMS = ["restaurant", "cafe", "cooking", "cook"]

SEARCH_FIELDS = [
    "cultural_group", "context", "goal", "actor", "recipient", "relation",
    "actor_behavior", "recipient_behavior", "topic", "source_text", "raw",
]


def _text_for_record(record: dict[str, Any]) -> str:
    parts = []
    for field in SEARCH_FIELDS:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            parts.append(json.dumps(value, ensure_ascii=False, default=str))
        else:
            parts.append(str(value))
    return normalize_text(" ".join(parts)).lower()


def _question_to_dict(question: Any) -> dict[str, Any]:
    if hasattr(question, "model_dump"):
        return question.model_dump()
    return dict(question)


def _text_for_question(question: Any) -> str:
    q = _question_to_dict(question)
    parts = [
        q.get("question"), q.get("cultural_group"), q.get("region"),
        q.get("option_text"), q.get("raw"),
    ]
    options = q.get("options") or {}
    if isinstance(options, dict):
        parts.extend(options.values())
    return normalize_text(" ".join(
        json.dumps(p, ensure_ascii=False, default=str)
        if isinstance(p, (dict, list, tuple)) else str(p)
        for p in parts if p
    )).lower()


def _term_matches(text: str, terms: list[str]) -> list[str]:
    matches = []
    for term in terms:
        t = term.lower().strip()
        if not t:
            continue
        if " " in t or "-" in t:
            if t in text:
                matches.append(term)
        elif re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text):
            matches.append(term)
    return sorted(set(matches))


def _score_text(text: str) -> dict[str, Any]:
    primary = _term_matches(text, PRIMARY_TERMS)
    social = _term_matches(text, SOCIAL_TERMS)
    value = _term_matches(text, VALUE_TERMS)
    weak_only = _term_matches(text, WEAK_ONLY_TERMS)

    score = (2 * len(primary)) + len(social) + len(value)
    weak_food_only = bool(weak_only) and not social and len(primary) == len(weak_only)
    return {
        "score": score,
        "primary_terms": primary,
        "social_terms": social,
        "value_terms": value,
        "weak_food_only": weak_food_only,
        "has_primary": bool(primary),
        "has_social": bool(social),
    }


def _is_selected(meta: dict[str, Any], min_score: int, require_social: bool) -> bool:
    if meta["weak_food_only"]:
        return False
    if meta["score"] < min_score:
        return False
    if not meta["has_primary"]:
        return False
    if require_social and not meta["has_social"]:
        return False
    return True


def _selection_reason(meta: dict[str, Any]) -> str:
    bits = []
    if meta["primary_terms"]:
        bits.append("primary=" + ", ".join(meta["primary_terms"][:5]))
    if meta["social_terms"]:
        bits.append("social=" + ", ".join(meta["social_terms"][:5]))
    if meta["value_terms"]:
        bits.append("value=" + ", ".join(meta["value_terms"][:5]))
    return "; ".join(bits)


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_records:
        return list(read_jsonl(args.input_records))
    return load_culturebank(args.dataset, args.split, args.scan_limit)


def _select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        meta = _score_text(_text_for_record(record))
        if not _is_selected(meta, args.min_score, args.require_social):
            continue
        enriched = dict(record)
        enriched["controlled_subset"] = {
            "topic": args.topic,
            "selection_score": meta["score"],
            "primary_terms": meta["primary_terms"],
            "social_terms": meta["social_terms"],
            "value_terms": meta["value_terms"],
            "selection_reason": _selection_reason(meta),
            "manual_audit_status": "unreviewed",
            "manual_audit_notes": "",
        }
        selected.append(enriched)

    selected.sort(
        key=lambda r: (
            r["controlled_subset"]["selection_score"],
            len(r["controlled_subset"]["social_terms"]),
            len(r["controlled_subset"]["primary_terms"]),
        ),
        reverse=True,
    )
    return selected[: args.max_records]


def _write_audit_csv(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fields = [
        "instance_id", "cultural_group", "topic", "context", "selection_score",
        "primary_terms", "social_terms", "value_terms", "selection_reason",
        "manual_audit_status", "manual_audit_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            meta = record.get("controlled_subset", {})
            writer.writerow({
                "instance_id": record.get("instance_id", ""),
                "cultural_group": record.get("cultural_group", ""),
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


def _select_culturalbench_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = load_culturalbench(
        variants=tuple(v.strip() for v in args.cb_variants.split(",") if v.strip()),
        limit=args.cb_limit,
        allow_fallback=args.allow_fallback,
    )
    selected = []
    for question in questions:
        meta = _score_text(_text_for_question(question))
        if not _is_selected(meta, args.cb_min_score, args.require_social):
            continue
        q = _question_to_dict(question)
        q["controlled_subset"] = {
            "topic": args.topic,
            "selection_score": meta["score"],
            "primary_terms": meta["primary_terms"],
            "social_terms": meta["social_terms"],
            "value_terms": meta["value_terms"],
            "selection_reason": _selection_reason(meta),
        }
        selected.append(q)
    return selected


def _write_notes(records: list[dict[str, Any]], cb_questions: list[dict[str, Any]],
                 args: argparse.Namespace, path: Path) -> None:
    ensure_dir(path.parent)
    cultures = Counter(r.get("cultural_group") or "unknown" for r in records)
    topics = Counter(r.get("topic") or "unknown" for r in records)
    lines = [
        "# Controlled Subset Selection Notes",
        "",
        f"Topic: `{args.topic}`",
        "",
        "## Purpose",
        "",
        "This subset supports controlled scalability before broad KG expansion. Use it to debug extraction, canonicalization, pathway construction, and S4 retrieval behavior on one interpretable cultural topic.",
        "",
        "## Selection settings",
        "",
        f"- CultureBank scan limit: {args.scan_limit}",
        f"- Max selected records: {args.max_records}",
        f"- Minimum record score: {args.min_score}",
        f"- Require social/hospitality term: {args.require_social}",
        f"- Selected CultureBank records: {len(records)}",
        f"- Selected CulturalBench questions: {len(cb_questions)}",
        "",
        "## CultureBank culture distribution",
        "",
        "| Cultural group | Count |",
        "|---|---:|",
    ]
    for culture, count in cultures.most_common(20):
        lines.append(f"| {culture} | {count} |")
    lines += [
        "",
        "## CultureBank topic distribution",
        "",
        "| Topic | Count |",
        "|---|---:|",
    ]
    for topic, count in topics.most_common(20):
        lines.append(f"| {topic} | {count} |")
    lines += [
        "",
        "## Manual audit instructions",
        "",
        "Open `food_hospitality_audit.csv` and set `manual_audit_status` to one of:",
        "",
        "- `keep`: clearly about food/hospitality as a cultural practice, norm, or value",
        "- `maybe`: relevant but weak, ambiguous, or missing social/value context",
        "- `remove`: only mentions food/hospitality without cultural reasoning value",
        "",
        "For kept rows, note whether they support L1 surface practice, L2 social norm, L3 deep value, and possible L3 -> L2 -> L1 pathways.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a controlled food/hospitality subset.")
    parser.add_argument("--topic", default="food_hospitality", choices=["food_hospitality"])
    parser.add_argument("--dataset", default=config.DATASET_NAME)
    parser.add_argument("--split", default=config.DEFAULT_SPLIT)
    parser.add_argument("--scan_limit", type=int, default=2000)
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--min_score", type=int, default=3)
    parser.add_argument("--require_social", action="store_true", default=True)
    parser.add_argument("--input_records", default=None,
                        help="Optional normalized CultureBank JSONL to filter instead of loading HF.")
    parser.add_argument("--include_culturalbench", action="store_true", default=False)
    parser.add_argument("--allow_fallback", action="store_true", default=False,
                        help="Allow tiny CulturalBench fallback when --include_culturalbench is used.")
    parser.add_argument("--cb_limit", type=int, default=250)
    parser.add_argument("--cb_variants", default="easy,hard")
    parser.add_argument("--cb_min_score", type=int, default=2)
    parser.add_argument("--records_output", default=str(CONTROLLED_DIR / "food_hospitality_records.jsonl"))
    parser.add_argument("--audit_output", default=str(CONTROLLED_DIR / "food_hospitality_audit.csv"))
    parser.add_argument("--notes_output", default=str(CONTROLLED_DIR / "food_hospitality_selection_notes.md"))
    parser.add_argument("--cb_output", default=str(CONTROLLED_DIR / "food_hospitality_culturalbench_questions.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args)
    selected = _select_records(records, args)

    write_jsonl(args.records_output, selected)
    _write_audit_csv(selected, Path(args.audit_output))

    cb_questions: list[dict[str, Any]] = []
    if args.include_culturalbench:
        cb_questions = _select_culturalbench_questions(args)
        write_jsonl(args.cb_output, cb_questions)

    _write_notes(selected, cb_questions, args, Path(args.notes_output))

    print(f"Scanned CultureBank records: {len(records)}")
    print(f"Selected CultureBank records: {len(selected)}")
    print(f"Records JSONL: {args.records_output}")
    print(f"Audit CSV: {args.audit_output}")
    print(f"Notes: {args.notes_output}")
    if args.include_culturalbench:
        print(f"Selected CulturalBench questions: {len(cb_questions)}")
        print(f"CulturalBench JSONL: {args.cb_output}")


if __name__ == "__main__":
    main()
