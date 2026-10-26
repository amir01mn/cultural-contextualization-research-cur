"""Deterministic Phase 1 regression tests. No model or network needed."""
from __future__ import annotations

import json

import pytest

from src.evaluation.kazculture_phase1 import (
    choose_highest, evaluate_batch, question_text, summarize,
)
from src.loaders.kazculture_loader import convert_test_rows, load_test


def row(label="C"):
    return {
        "question": "Қазақ дәстүрінде не?", "a": "Бір",
        "b": "Екі", "c": "Үш", "d": "Төрт",
        "answer_label": label, "category_english": "Customs",
        "item_description": "SECRET PASSAGE",
        "answer": "SECRET GOLD ANSWER",
    }


class FixedScorer:
    def score(self, questions):
        # Prove scorer only receives sanitized question records, not raw rows.
        for q in questions:
            assert "SECRET" not in repr(q)
            assert "SECRET" not in question_text(q)
        return [{"A": -2, "B": -3, "C": -0.1, "D": -4} for _ in questions]


def test_sanitization_and_scoring():
    items = convert_test_rows([row()])
    assert not hasattr(items[0], "item_description")
    assert not hasattr(items[0], "answer")
    assert "SECRET" not in question_text(items[0])
    records = evaluate_batch(items, FixedScorer())
    assert records[0]["correct"]
    assert summarize(records)["accuracy"] == 1
    assert records[0]["prediction"] == "C"


def test_argmax_and_validation():
    assert choose_highest({"A": -1, "B": -1, "C": -2, "D": -3}) == "A"
    with pytest.raises(ValueError):
        choose_highest({"A": 1})
    with pytest.raises(ValueError):
        convert_test_rows([row("Z")])
    with pytest.raises(ValueError):
        convert_test_rows([{**row(), "a": ""}])


def test_local_fixture_expected_size(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text(json.dumps(row(), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected exactly"):
        load_test(local_file=str(p))
    items, meta = load_test(local_file=str(p), expected_size=1)
    assert len(items) == meta["full_test_count"] == 1
    assert meta["scored_subset_size"] == 1


def test_duplicates_have_unique_ids():
    items = convert_test_rows([row(), row()])
    assert len({q.question_id for q in items}) == 2
