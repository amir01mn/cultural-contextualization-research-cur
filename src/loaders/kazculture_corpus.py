"""KazCulture training passages with immutable passage provenance.

Never feed train QA answers, gold labels, or test fields to a KG builder.
Deduplicate source passages, not question/answer records. Source-level IDs
permit a strict leakage audit and held-out-passage evaluation.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def normalized_passage(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def passage_id(text: str) -> str:
    return "kzp_" + hashlib.sha256(normalized_passage(text).encode("utf-8")).hexdigest()[:20]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
    raise ValueError("Input must be .parquet or .jsonl")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def corpus_from_train(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Return deduplicated passages and train-only QA->passage references.

    QA is stored in a separate dev file; it NEVER enters passage or graph
    records. A passage's original source/category and row ordinals remain.
    """
    by_pid: dict[str, dict] = {}
    dev_rows: list[dict] = []
    for i, row in enumerate(rows):
        if row.get("set") not in (None, "train"):
            raise ValueError(f"Non-train row at offset {i}")
        passage = str(row.get("item_description") or "").strip()
        if not passage:
            raise ValueError(f"Missing training passage at row {i}")
        pid = passage_id(passage)
        source = str(row.get("source") or "unspecified")
        category = str(row.get("category_english") or "Unknown")
        if pid not in by_pid:
            by_pid[pid] = {
                "passage_id": pid, "text": passage,
                "dataset": "issai/KazCulture", "split": "train",
                "sources": [], "categories": [], "train_row_ids": [],
                "culture": "Kazakh",
            }
        unit = by_pid[pid]
        if source not in unit["sources"]:
            unit["sources"].append(source)
        if category not in unit["categories"]:
            unit["categories"].append(category)
        unit["train_row_ids"].append(i)
        question, answer = str(row.get("question") or ""), str(row.get("answer") or "")
        if question.strip() and answer.strip():
            dev_rows.append({
                "train_row_id": i, "question": question, "answer": answer,
                "passage_id": pid, "category": category,
            })
    passages = sorted(by_pid.values(), key=lambda p: p["passage_id"])
    return passages, dev_rows, {
        "train_rows": len(rows), "unique_passages": len(passages),
        "duplicate_passage_rows": len(rows) - len(passages),
        "train_qa_count": len(dev_rows),
    }


def deterministic_pilot(passages: list[dict], limit: int, seed: int = 42) -> list[dict]:
    """Select stable, category-distributed pilot, with no test-set usage."""
    if limit <= 0 or limit > len(passages):
        raise ValueError("Pilot limit must be within corpus size")
    by_cat = defaultdict(list)
    for p in passages:
        by_cat[p["categories"][0] if p["categories"] else "Unknown"].append(p)
    rng = np.random.default_rng(seed)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda p: p["passage_id"])
        rng.shuffle(by_cat[cat])
    result, position = [], {k: 0 for k in by_cat}
    categories = sorted(by_cat)
    while len(result) < limit:
        progressed = False
        for cat in categories:
            i = position[cat]
            if i < len(by_cat[cat]):
                result.append(by_cat[cat][i])
                position[cat] = i + 1
                progressed = True
                if len(result) == limit:
                    break
        if not progressed:
            break
    return result


def test_overlap_report(passages: list[dict], test_rows: list[dict]) -> dict:
    """Use test passages ONLY for detecting leakage, never building the KG."""
    train_ids = {p["passage_id"] for p in passages}
    test_ids = [passage_id(str(r.get("item_description") or ""))
                for r in test_rows]
    overlapping = set(test_ids) & train_ids
    return {
        "train_passage_count": len(passages),
        "test_record_count": len(test_rows),
        "overlap_test_records": sum(pid in overlapping for pid in test_ids),
        "overlap_unique_passages": len(overlapping),
        "exclude_train_passage_ids": sorted(overlapping),
    }


def validate_no_test_content(passages: list[dict], exclude: set[str]) -> None:
    if set(p["passage_id"] for p in passages) & exclude:
        raise ValueError("Strict corpus contains a test-overlapping passage")
    for p in passages:
        assert set(p) <= {"passage_id", "text", "dataset", "split", "sources",
                          "categories", "train_row_ids", "culture"}, (
            "Unexpected field in passage corpus; test/answer leakage risk"
        )
