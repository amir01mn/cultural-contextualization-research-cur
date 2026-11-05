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


def test_exact_mcnemar_without_scipy():
    """Known two-sided exact binomial tails for the discordant pairs."""
    # Four baseline-wrong cases rescued by evidence, zero harmed:
    # 2 * P(Binomial(4, 1/2) <= 0) = 2/16 = 0.125.
    before = [{"question_id": str(i), "correct": False} for i in range(4)]
    after = [{"question_id": str(i), "correct": True} for i in range(4)]
    assert paired_statistics(before, after, iterations=100)["mcnemar_exact_p"] == 0.125

    # Three rescued and one harmed: 2*(1 + 4)/16 = 0.625.
    before = [{"question_id": str(i), "correct": i == 3} for i in range(4)]
    after = [{"question_id": str(i), "correct": i != 3} for i in range(4)]
    assert paired_statistics(before, after, iterations=100)["mcnemar_exact_p"] == 0.625

    # No discordant pairs means no evidence for a change.
    assert paired_statistics(before, before, iterations=100)["mcnemar_exact_p"] == 1.0


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


def test_index_shard_resume_and_source_filter(tmp_path, monkeypatch):
    """Fake embedding model verifies checkpoint/resume with zero network/GPU."""
    import sys
    import types
    import numpy as np
    from src.retrieval.kazculture_index import MultilingualIndex
    calls = {"n": 0}

    class FakeSentenceTransformer:
        def __init__(self, model_name, revision=None, device="cpu"):
            assert model_name == "BAAI/bge-m3"
            assert revision == "5617a9f61b028005a4858fdac845db406aefb181"

        def encode(self, texts, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("Simulated compute preemption")
            vectors = np.array([[len(s) + 1, 1] for s in texts], dtype=np.float32)
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    units = [{
        "unit_id": f"S1:{i}", "passage_id": f"kzp_{i}",
        "system": "S1", "dataset": "KazCulture", "culture": "Kazakh",
        "source": ["Train book"], "indirect": False,
        "values": [], "text": "Kazakh text " + str(i),
    } for i in range(5)]
    folder = tmp_path / "index"
    with pytest.raises(RuntimeError, match="preemption"):
        MultilingualIndex.create(folder, units, chunk_size=2, batch_size=2)
    assert len(list((folder / "embedding_shards").glob("*.npy"))) == 1
    assert not (folder / "metadata.json").exists()
    meta = MultilingualIndex.create(folder, units, chunk_size=2, batch_size=2)
    assert meta["n_units"] == 5
    assert meta["embedding_model_revision"] == "5617a9f61b028005a4858fdac845db406aefb181"
    assert (folder / "embeddings.npy").exists()
    ix = MultilingualIndex(folder)
    hits = ix.search("Kazakh text", system="S1", k=3,
                     exclude_passage_ids={"kzp_0"})
    assert len(hits) == 3
    assert all(h["passage_id"] != "kzp_0" for h in hits)
    with pytest.raises(FileExistsError):
        MultilingualIndex.create(folder, units, chunk_size=2)


def test_overlength_passage_is_chunked_losslessly_and_keeps_local_pathways():
    """Regression for Narval 7474841: vLLM rejected a >4096-token passage."""
    from src.construction.kazculture_chunking import (
        aggregate_chunks, prompt_tokens, split_passage,
    )
    from src.construction.kazculture_kg import pathways
    from src.retrieval.kazculture_index import make_units
    import json

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "\\n".join(m["content"] for m in messages) + "\\nAssistant:"

        def encode(self, text, add_special_tokens=False):
            # Character-level stand-in makes the limit deterministic and
            # exercises the exact templated prompt, including its headers.
            return range(len(text))

    text = ("Қазақтың дәстүрлі күйлері домбырада орындалады. " * 140)
    original = {
        "passage_id": "kzp_test_long", "text": text, "sources": ["Train book"],
        "categories": ["Music"], "dataset": "issai/KazCulture",
        "split": "train", "culture": "Kazakh", "train_row_ids": [1],
    }
    tok = FakeTokenizer()
    budget = 3000 - 240 - 64
    parts = split_passage(
        tok, original, max_model_len=3000, max_tokens=240,
    )
    assert len(parts) > 1
    assert parts[0]["_chunk_start"] == 0
    assert parts[-1]["_chunk_end"] == len(text)
    assert all(p["text"] == text[p["_chunk_start"]:p["_chunk_end"]]
               for p in parts)
    assert all(prompt_tokens(tok, p) <= budget for p in parts)
    assert all(next_part["_chunk_start"] <= prev["_chunk_end"]
               for prev, next_part in zip(parts, parts[1:]))

    choices = ["tradition", "benevolence (caring)"]
    replies = []
    for i, p in enumerate(parts):
        replies.append(json.dumps({
            "triples": [{
                "subject": f"dombyra_segment_{i}", "predicate": "part of",
                "object": "Kazakh music",
                "evidence": p["text"][:40],
            }],
            "norm": f"local norm {i}",
            "norm_type": f"type {i}",
            "values": [choices[i % 2]],
        }, ensure_ascii=False))
    merged = aggregate_chunks(original, parts, replies, model="test")
    assert merged["text"] == text
    assert merged["chunk_count"] == len(parts)
    assert merged["label_status"] == "candidate"
    assert len(merged["triples"]) == len(parts)
    assert len(merged["chunk_interpretations"]) == len(parts)
    for i, triple in enumerate(merged["triples"]):
        assert triple["evidence"] in parts[i]["text"]
        assert triple["source_chunk_index"] == i
        assert triple["chunk_values"] == [choices[i % 2]]
    routes = pathways(merged)
    assert len(routes) == len(parts)
    assert all(r["values"] == [choices[i % 2]] for i, r in enumerate(routes))
    units = make_units([original], [merged])
    s3 = [u for u in units if u["system"] == "S3"]
    assert len(s3) == len(parts)
    assert all(choices[i % 2] in unit["text"] for i, unit in enumerate(s3))


def test_overlength_quote_cannot_be_laundered_from_other_chunk():
    from src.construction.kazculture_chunking import aggregate_chunks
    import json
    orig = {
        "passage_id": "kzp_quotes", "text": "Алма жеміс. Домбыра аспап.",
        "sources": ["Book"], "categories": ["Culture"],
        "dataset": "issai/KazCulture", "culture": "Kazakh",
    }
    chunks = [
        {**orig, "text": "Алма жеміс.", "_chunk_start": 0, "_chunk_end": 11},
        {**orig, "text": " Домбыра аспап.", "_chunk_start": 11,
         "_chunk_end": len(orig["text"])},
    ]
    # The quote exists in the WHOLE passage but was never shown to model
    # in the first chunk: it must not become a grounded triple.
    responses = [
        json.dumps({
            "triples": [{"subject": "invented", "predicate": "related",
                         "object": "value", "evidence": "Домбыра"}],
            "norm": "fake", "norm_type": "fake", "values": ["tradition"],
        }),
        json.dumps({
            "triples": [{"subject": "домбыра", "predicate": "is",
                         "object": "аспап", "evidence": "Домбыра"}],
            "norm": "music", "norm_type": "performance", "values": ["tradition"],
        }, ensure_ascii=False),
    ]
    merged = aggregate_chunks(orig, chunks, responses, model="fake")
    assert len(merged["triples"]) == 1
    assert merged["triples"][0]["source_chunk_index"] == 1
