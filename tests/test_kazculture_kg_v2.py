"""Offline regression tests for source-grounded KazCulture KG v2 pilot."""
from __future__ import annotations

import csv
import json

import pytest

from src.construction.kazculture_chunking import (
    prompt_tokens, split_passage,
)
from src.construction.kazculture_kg_v2 import (
    SCHEMA_VERSION, SYSTEM_V2, aggregate_v2, extraction_user_v2,
    make_v2_audit, parse_v2_chunk, save_v2_audit, summarize_v2_audit,
)


def _p(text="Садақтың кермесі малдың сіңірінен жасалады."):
    return {
        "passage_id": "kzp_v2_example",
        "text": text, "dataset": "issai/KazCulture",
        "split": "train", "sources": ["Cultural encyclopedia"],
        "categories": ["Craftsmanship"], "train_row_ids": [2],
        "culture": "Kazakh",
    }


def _raw(triples, pathways=None):
    return json.dumps({
        "triples": triples, "pathways": pathways if pathways is not None else [],
    }, ensure_ascii=False)


def test_factual_material_relation_has_no_invented_pathway():
    p = _p()
    raw = _raw([{
        "subject": "Садақтың кермесі", "predicate": "MADE_FROM",
        "object": "малдың сіңірі", "evidence": p["text"],
    }])
    parsed = parse_v2_chunk(p, raw, model="fake")
    assert parsed["parse_status"] == "parsed"
    assert len(parsed["triples"]) == 1
    triple = parsed["triples"][0]
    assert triple["knowledge_kind"] == "factual"
    assert triple["pathway"] is None
    assert triple["quote_verified"]
    assert triple["predicate"] == "MADE_FROM"
    assert triple["subject"] == "Садақтың кермесі"


def test_normative_pathways_need_quote_and_valid_values():
    p = _p("Тойда үлкендерге құрмет көрсету керек. Бұл дәстүр сақталады.")
    fact = {
        "subject": "Тойға қатысушылар", "predicate": "PRACTICED_BY",
        "object": "үлкендерге құрмет", "evidence": p["text"][:36],
    }
    good = {
        "triple_index": 0, "norm": "Үлкендерге құрмет көрсету керек",
        "norm_type": "respect for elders", "values": ["tradition"],
        "evidence": "Тойда үлкендерге құрмет көрсету керек.",
    }
    parsed = parse_v2_chunk(p, _raw([fact], [good]), model="fake")
    assert len(parsed["triples"]) == 1
    assert parsed["triples"][0]["knowledge_kind"] == "normative"
    assert parsed["triples"][0]["pathway"]["values"] == ["tradition"]

    wrong_quote = dict(good, evidence="not in original passage")
    bad = parse_v2_chunk(p, _raw([fact], [wrong_quote]), model="fake")
    assert bad["triples"][0]["knowledge_kind"] == "factual"
    assert bad["triples"][0]["pathway"] is None
    assert bad["rejections"]["unsupported_or_incomplete_pathway"] == 1

    wrong_value = dict(good, values=["Made up value"])
    invalid = parse_v2_chunk(p, _raw([fact], [wrong_value]), model="fake")
    assert invalid["triples"][0]["pathway"] is None


def test_invalid_quote_predicate_source_metadata_and_orphan_are_rejected():
    p = _p()
    raw = _raw([
        {
            "subject": "Cultural encyclopedia", "predicate": "CREATED_BY",
            "object": "X", "evidence": p["text"],
        },
        {
            "subject": "керме", "predicate": "CAUSES_MAGIC",
            "object": "сіңір", "evidence": p["text"],
        },
        {
            "subject": "керме", "predicate": "MADE_FROM",
            "object": "сіңір", "evidence": "Invented quote",
        },
        {
            "subject": "керме", "predicate": "MADE_FROM",
            "object": "сіңір", "evidence": p["text"],
        },
    ], [{
        "triple_index": 1, "norm": "bogus", "norm_type": "bogus",
        "values": ["tradition"], "evidence": p["text"],
    }])
    parsed = parse_v2_chunk(p, raw, model="fake")
    assert len(parsed["triples"]) == 1
    assert parsed["rejections"]["source_metadata_as_subject"] == 1
    assert parsed["rejections"]["invalid_predicate"] == 1
    assert parsed["rejections"]["unverified_source_quote"] == 1
    assert parsed["rejections"]["orphan_pathway"] == 1
    assert "Cultural encyclopedia" not in extraction_user_v2(p)
    assert "Craftsmanship" not in extraction_user_v2(p)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "\n".join(m["content"] for m in messages) + "\nassistant:"

    def encode(self, text, add_special_tokens=False):
        return range(len(text))


def test_long_passage_with_no_metadata_and_segment_quote_provenance():
    original = _p("Үлкендерге құрмет көрсету керек. " * 130)
    tok = FakeTokenizer()
    parts = split_passage(
        tok, original, max_model_len=3000, max_tokens=600,
        system=SYSTEM_V2, user_builder=extraction_user_v2,
    )
    assert len(parts) > 1
    assert parts[0]["_chunk_start"] == 0
    assert parts[-1]["_chunk_end"] == len(original["text"])
    assert all(
        prompt_tokens(tok, p, system=SYSTEM_V2,
                      user_builder=extraction_user_v2) <= 3000 - 600 - 64
        for p in parts
    )
    responses = [
        _raw([{
            "subject": f"жас {i}", "predicate": "REQUIRES",
            "object": "құрмет", "evidence": p["text"][:30],
        }])
        for i, p in enumerate(parts)
    ]
    merged = aggregate_v2(original, parts, responses, model="fake")
    assert merged["schema_version"] == SCHEMA_VERSION
    assert merged["chunk_count"] == len(parts)
    assert len(merged["triples"]) == len(parts)
    assert all(
        t["evidence"] in parts[t["source_chunk_index"]]["text"]
        for t in merged["triples"]
    )
    assert merged["pathways"] == []


def test_v2_audit_separates_pathway_nonapplicability(tmp_path):
    p1 = _p("Тойда үлкендерге құрмет көрсету керек.")
    p2 = _p("Бөрік қой терісінен жасалады.")
    p2["passage_id"] = "kzp_2"
    r1 = _raw([{
        "subject": "жастар", "predicate": "REQUIRES",
        "object": "құрмет", "evidence": p1["text"],
    }], [{
        "triple_index": 0, "norm": "үлкендерге құрмет",
        "norm_type": "respect for elders", "values": ["tradition"],
        "evidence": p1["text"],
    }])
    r2 = _raw([{
        "subject": "Бөрік", "predicate": "MADE_FROM",
        "object": "қой терісі", "evidence": p2["text"],
    }])
    frames = []
    for p, raw in [(p1, r1), (p2, r2)]:
        chunk = {**p, "_chunk_start": 0, "_chunk_end": len(p["text"])}
        frames.append(aggregate_v2(p, [chunk], [raw], model="fake"))
    rows = make_v2_audit(frames, n=2)
    file = tmp_path / "review.csv"
    save_v2_audit(file, rows)
    with pytest.raises(FileExistsError):
        save_v2_audit(file, rows)
    with pytest.raises(ValueError, match="human 0/1"):
        summarize_v2_audit(file, min_rows=2)
    for row in rows:
        row["triple_correct_0_or_1"] = "1"
        row["normative_eligible_0_or_1"] = row["model_proposed_pathway_0_or_1"]
        if row["model_proposed_pathway_0_or_1"] == 1:
            row["pathway_correct_0_or_1_if_proposed"] = "1"
    with file.open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = summarize_v2_audit(file, min_rows=2, min_proposed_paths=1)
    assert out["triple_precision"] == 1.0
    assert out["pathway_precision_on_proposed"] == 1.0
    assert out["human_normatively_eligible"] == 1
    assert out["eligible_pathway_coverage"] == 1.0
    assert out["pilot_gate"] == "pass"


def test_normative_eligible_but_model_missed_pathway():
    p = _p()
    parse = parse_v2_chunk(p, _raw([{
        "subject": "керме", "predicate": "MADE_FROM", "object": "сіңір",
        "evidence": p["text"],
    }]), model="fake")
    assert parse["triples"][0]["pathway"] is None
    # The independent reviewer can still mark normative_eligible=1,
    # revealing missed pathways rather than calling an absent pathway wrong.
