"""Offline Phase 2-5 provenance, leakage, audit and paired-analysis tests."""
from __future__ import annotations

import csv
import json
import pytest

from src.loaders.kazculture_corpus import (
    corpus_from_train, deterministic_pilot, passage_id,
    test_overlap_report as passage_overlap_report, validate_no_test_content,
)
from src.construction.kazculture_kg import (
    extract_frame, pathways, audit_rows, write_audit_csv, validate_audit,
)
from src.retrieval.kazculture_index import make_units, render_evidence
from src.evaluation.kazculture_rag import paired_statistics, selected_evidence


def _rows():
    return [
        {"set": "train", "item_description": "Қазақтың домбырасы екі ішекті аспап.",
         "question": "Домбырада неше ішек бар?", "answer": "Екі",
         "source": "Book A", "category_english": "Music"},
        {"set": "train", "item_description": "  Қазақтың домбырасы екі ішекті аспап.  ",
         "question": "Бұл қандай аспап?", "answer": "Домбыра",
         "source": "Book A", "category_english": "Music"},
        {"set": "train", "item_description": "Наурыз көктемде тойланады.",
         "question": "Наурыз қашан?", "answer": "Көктемде",
         "source": "Book B", "category_english": "Tradition"},
    ]


def _frame(passages):
    passage = passages[0]
    quote = passage["text"][:19]
    raw = json.dumps({
        "triples": [{"subject": "домбыра", "predicate": "is", "object": "аспап",
                     "evidence": quote},
                    {"subject": "сырттан", "predicate": "imaginary",
                     "object": "value", "evidence": "NOT A PASSAGE QUOTE"}],
        "norm": "Домбыра қадірлі", "norm_type": "music",
        "values": ["tradition"], "unrelated": "no",
    }, ensure_ascii=False)
    return extract_frame(passage, raw, model="mock")


def test_dedupe_and_no_leakage():
    passages, qa, stats = corpus_from_train(_rows())
    assert len(passages) == 2
    assert len(qa) == 3
    assert stats["duplicate_passage_rows"] == 1
    assert all("answer" not in p and "question" not in p for p in passages)
    assert deterministic_pilot(passages, 1)
    overlap = passage_overlap_report(
        passages, [{"item_description": "Наурыз көктемде тойланады."},
                   {"item_description": "unrelated"}])
    assert overlap["overlap_test_records"] == 1
    exclude = set(overlap["exclude_train_passage_ids"])
    with pytest.raises(ValueError):
        validate_no_test_content(passages, exclude)
    kept = [p for p in passages if p["passage_id"] not in exclude]
    validate_no_test_content(kept, exclude)
    with pytest.raises(ValueError, match="Non-train"):
        corpus_from_train([{"set": "test", "item_description": "X"}])


def test_llm_quotes_are_required_and_values_provisional():
    passages, _, _ = corpus_from_train(_rows())
    frame = _frame(passages)
    assert frame["label_status"] == "candidate"
    assert len(frame["triples"]) == 1
    assert frame["triples"][0]["quote_verified"]
    assert frame["triples"][0]["generated"]
    assert frame["interpretation_status"] == "provisional"
    assert frame["values"] == ["tradition"]
    route = pathways(frame)
    assert len(route) == 1
    assert route[0]["passage_id"] == frame["passage_id"]
    assert route[0]["provisional"]


def test_manual_audit_must_be_completed(tmp_path):
    passages, _, _ = corpus_from_train(_rows())
    frame = _frame(passages)
    sample = audit_rows([frame], n=1)
    audit = tmp_path / "review.csv"
    write_audit_csv(audit, sample)
    with pytest.raises(ValueError, match="unanswered"):
        validate_audit(audit, min_rows=1)
    sample[0]["triple_correct_0_or_1"] = "1"
    sample[0]["pathway_correct_0_or_1"] = "1"
    write_audit_csv(audit, sample)
    assert validate_audit(audit, min_rows=1)["passed"]
    sample[0]["triple_correct_0_or_1"] = "0"
    write_audit_csv(audit, sample)
    with pytest.raises(ValueError, match="gate failed"):
        validate_audit(audit, min_rows=1)


def test_evidence_source_provenance_and_alignment():
    passages, _, _ = corpus_from_train(_rows())
    frame = _frame(passages)
    cb = {
        "frame_id": "cb33", "label_status": "ok",
        "culture_label": "Italian", "text": "People celebrate music",
        "values": ["tradition"], "norm_type_label": "music", "norm": "festival"
    }
    units = make_units(passages, [frame], [cb])
    assert any(u["system"] == "S1" and u["dataset"] == "KazCulture" for u in units)
    assert any(u["system"] == "S2" and "domбыра" not in u["text"] for u in units)
    assert any(u["system"] == "S4R" and u["dataset"] == "CultureBank"
               and u["indirect"] and "Italian" in u["text"] for u in units)
    assert len([u for u in units if u["system"] == "S3"]) == 1
    doc = render_evidence([u for u in units if u["dataset"] == "CultureBank"])
    assert "INDIRECT COMPARISON" in doc


def test_paired_statistics_stable():
    before = [
        {"question_id": "a", "correct": True},
        {"question_id": "b", "correct": False},
        {"question_id": "c", "correct": False},
        {"question_id": "d", "correct": True},
    ]
    after = [
        {"question_id": "a", "correct": True},
        {"question_id": "b", "correct": True},
        {"question_id": "c", "correct": False},
        {"question_id": "d", "correct": False},
    ]
    stats = paired_statistics(before, after, iterations=100)
    assert stats["difference_pp"] == 0
    assert stats["evidence_induced_errors_n"] == 1
    assert stats["evidence_rescued_n"] == 1
    with pytest.raises(ValueError):
        paired_statistics(before, after[:2])


def test_combined_requires_shared_values():
    class MockIndex:
        def search(self, query, *, system, combined, **kwargs):
            if not combined:
                return [{
                    "unit_id": "kaz1", "passage_id": "k1",
                    "dataset": "KazCulture", "indirect": False,
                    "values": ["tradition"], "source": ["Book"], "text": "Kazakh",
                    "similarity": 0.8
                }]
            return [
                {"unit_id": "cb1", "passage_id": "cb1",
                 "dataset": "CultureBank", "indirect": True,
                 "values": ["hedonism"], "source": ["CB"], "text": "Different",
                 "similarity": 0.95},
                {"unit_id": "cb2", "passage_id": "cb2",
                 "dataset": "CultureBank", "indirect": True,
                 "values": ["tradition"], "source": ["CB"], "text": "Similar",
                 "similarity": 0.6},
            ]
    from src.loaders.kazculture_loader import convert_test_rows
    q = convert_test_rows([{
        "question": "Question", "a": "one", "b": "two",
        "c": "three", "d": "four", "answer_label": "A",
    }])[0]
    txt, hits = selected_evidence(
        MockIndex(), q, "S4R_combined", top_k=4, min_similarity=0.0
    )
    assert [x["unit_id"] for x in hits] == ["kaz1", "cb2"]
    assert "INDIRECT COMPARISON" in txt
