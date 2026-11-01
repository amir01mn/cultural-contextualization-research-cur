"""Offline regression tests for v2.1's conservative normative QA."""
from __future__ import annotations

import json
import pytest

from src.construction.kazculture_kg_v21 import (
    SCHEMA_VERSION, SYSTEM_V21, aggregate_v21,
)
from scripts.export_kazculture_pathway_audit import pathway_audit_rows


def sample(text):
    return {
        "passage_id": "kzp_v21_test", "text": text,
        "sources": ["Regional folklore book"],
        "categories": ["Traditions"], "dataset": "issai/KazCulture",
        "split": "train", "culture": "Kazakh",
        "train_row_ids": [1],
    }


def frame(text, triples, paths):
    p = sample(text)
    piece = {**p, "_chunk_start": 0, "_chunk_end": len(p["text"])}
    raw = json.dumps({"triples": triples, "pathways": paths}, ensure_ascii=False)
    return aggregate_v21(p, [piece], [raw], model="frozen-test")


def test_v21_no_forced_pathway_for_material_fact():
    text = "Садақтың кермесі малдың сіңірінен жасалады."
    f = frame(text, [{
        "subject": "Садақтың кермесі", "predicate": "MADE_FROM",
        "object": "малдың сіңірі", "evidence": text,
    }], [])
    assert f["schema_version"] == SCHEMA_VERSION
    assert len(f["triples"]) == 1
    assert f["pathways"] == []
    assert f["triples"][0]["knowledge_kind"] == "factual"


def test_explicit_custom_survives_and_unknown_rule_kind_is_rejected():
    text = "Оқшантайды белдікке тағып жүру дәстүрі болған."
    triples = [{
        "subject": "Оқшантай", "predicate": "PART_OF",
        "object": "белдік", "evidence": text,
    }]
    p = {
        "triple_index": 0, "norm": "Оқшантайды белдікке тағу дәстүрі",
        "norm_type": "dress custom", "values": ["tradition"],
        "evidence": text, "rule_kind": "explicit_custom",
    }
    good = frame(text, triples, [p])
    assert len(good["pathways"]) == 1
    assert good["pathways"][0]["rule_kind"] == "explicit_custom"
    assert len(pathway_audit_rows([good])) == 1
    bad = frame(text, triples, [dict(p, rule_kind="made_up")])
    assert not bad["pathways"]
    assert bad["v21_rejections"]["missing_or_invalid_rule_kind"] == 1


def test_v21_can_skip_unsupported_pathway_while_retaining_fact():
    text = "Аударыспақ мерекелерде өткізіледі."
    triple = {
        "subject": "Аударыспақ", "predicate": "USED_IN",
        "object": "мерекелер", "evidence": text,
    }
    path = {
        "triple_index": 0, "norm": "Бәрі міндетті түрде ойнауы керек",
        "norm_type": "mandatory participation", "values": ["tradition"],
        "evidence": "not in the passage", "rule_kind": "explicit_obligation",
    }
    f = frame(text, [triple], [path])
    assert len(f["triples"]) == 1
    assert f["pathways"] == []
    assert f["rejections"]["unsupported_or_incomplete_pathway"] == 1


def test_v21_limits_triplets_to_three_without_uncontrolled_extra_relationships():
    text = "Бөрік қой терісінен жасалады."
    triples = [{
        "subject": f"Бөрік{i}", "predicate": "MADE_FROM",
        "object": "қой терісі", "evidence": text,
    } for i in range(5)]
    f = frame(text, triples, [])
    assert len(f["triples"]) == 3
    assert f["v21_rejections"]["extra_triples_over_3"] == 2


def test_v21_prompt_specifies_semantic_guardrails():
    assert "BOWSTRING" in SYSTEM_V21
    assert "descriptive" in SYSTEM_V21
    assert "same cultural behavior" in SYSTEM_V21.lower()
    assert "publication names" in SYSTEM_V21


def test_complete_pathway_audit_does_not_autofill_human_scores():
    text = "Оқшантайды белдікке тағып жүру дәстүрі болған."
    f = frame(text, [{
        "subject": "Оқшантай", "predicate": "PART_OF",
        "object": "белдік", "evidence": text,
    }], [{
        "triple_index": 0, "norm": "Оқшантай тағу",
        "norm_type": "carrying custom", "values": ["tradition"],
        "evidence": text, "rule_kind": "explicit_custom",
    }])
    rows = pathway_audit_rows([f])
    assert len(rows) == 1
    assert rows[0]["rule_kind_provisional"] == "explicit_custom"
    assert rows[0]["pathway_correct_0_or_1"] == ""
    assert rows[0]["triple_correct_0_or_1"] == ""
    assert rows[0]["normatively_eligible_0_or_1"] == ""
