"""KG v3: leakage filter, frames, labels, norm types, routing, record pathways, S3."""

import numpy as np
import pytest

from src.construction import frame_builder as fb
from src.construction.record_pathways import build_record_pathways
from src.core.leakage import triple_uses_eval_text
from src.core.schemas import CulturalTriple, LayerLabel, MCQQuestion, MCQVariant
from src.retrieval.frame_retriever import (
    ROUTE_ABSTRACT,
    ROUTE_NONE,
    ROUTE_REFINE,
    FrameRetriever,
    build_frame_index,
    render_route,
)


def _row(iid, group, behavior, topic="Etiquette", context="at home", **kw):
    return {"instance_id": iid, "cultural group": group, "actor_behavior": behavior,
            "topic": topic, "context": context, "actor": "people", "agreement": "0.9",
            "eval_whole_desc": "SECRET EVAL DESCRIPTION", "eval_question": "SECRET Q", **kw}


def test_triple_eval_leak_detector():
    raw = {"actor_behavior": "dress casually", "eval_whole_desc":
           "reflects a relaxed and informal attitude toward clothing"}
    assert triple_uses_eval_text("people", "a relaxed informal attitude", raw)
    assert not triple_uses_eval_text("people", "dress casually", raw)


def test_frame_has_no_eval_text_and_prompt_is_clean():
    f = fb.frame_from_row(_row("cb_1", "Japanese", "remove shoes before entering"))
    assert f.culture == "japan" and f.culture_label == "Japanese"
    assert "remove shoes" in f.text
    prompt = fb.label_prompt(f)
    assert "SECRET" not in prompt and "SECRET" not in f.model_dump_json()
    assert fb.frame_from_row(_row("cb_2", "", "x")) is None


@pytest.mark.parametrize("text,ok", [
    ('{"norm":"n","norm_type":"Shoes Indoors","values":["Tradition","face"]}', True),
    ('```json\n{"norm":"n","norm_type":"t","values":["benevolence - caring"]}\n```', True),
    ('{"norm":"n","norm_type":"t","values":["kindness"]}', False),
    ('{"norm":"n","norm_type":"t","values":["benevolence"]}', False),  # ambiguous family
    ("no json here", False),
])
def test_parse_label(text, ok):
    out = fb.parse_label(text)
    assert (out is not None) == ok
    if ok:
        assert all(v in fb.VALUE_NAMES for v in out["values"])
        assert out["norm_type"] == out["norm_type"].lower()


def _labeled_frames():
    rows = [
        _row("a1", "Japanese", "remove shoes before entering homes"),
        _row("a2", "Japanese", "bow when greeting elders", topic="Greetings"),
        _row("b1", "Korean", "take off shoes at the entrance of houses"),
        _row("c1", "American", "keep shoes on indoors at parties"),
        _row("c2", "American", "tip waiters around twenty percent", topic="Dining"),
    ]
    frames = fb.frames_from_rows(rows)
    labels = {
        "a1": ("removing shoes indoors", ["tradition"]),
        "a2": ("greeting elders", ["tradition", "humility"]),
        "b1": ("taking shoes off at home", ["tradition"]),
        "c1": ("removing shoes indoors", ["self-direction (action)"]),
        "c2": ("tipping service staff", ["benevolence (caring)"]),
    }
    fb.apply_labels(frames, {k: {"label": {"norm": f"norm {k}", "norm_type": nt, "values": v}}
                             for k, (nt, v) in labels.items()})
    return frames


def _fake_embed(texts):
    # Shoe-related labels share a direction; everything else is orthogonal.
    vecs = []
    for t in texts:
        v = np.zeros(4, dtype=np.float32)
        v[0 if "shoe" in t else (1 if "greet" in t else 2 if "tip" in t else 3)] = 1.0
        vecs.append(v)
    return np.stack(vecs)


def test_norm_type_clustering_links_cultures():
    kg = fb.build_frame_kg(_labeled_frames(), _fake_embed)
    shoes = [n for n in kg["norm_types"] if "shoe" in n.label]
    assert len(shoes) == 1 and set(shoes[0].cultures) == {"japan", "korea", "united states"}
    assert shoes[0].label == "removing shoes indoors"  # most frequent label leads
    assert kg["profiles"]["japan"]["values"]["tradition"] > 0


def test_heldout_build_excludes_records():
    kg = fb.build_frame_kg(_labeled_frames(), _fake_embed, exclude_ids={"a1"})
    assert "a1" not in {f.frame_id for f in kg["frames"]}
    assert all("a1" not in n.frame_ids for n in kg["norm_types"])


@pytest.fixture()
def retriever(tmp_path, monkeypatch):
    from src.core import config

    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "tfidf")
    kg = fb.build_frame_kg(_labeled_frames(), _fake_embed)
    fb.save_frame_kg(kg, tmp_path / "full")
    build_frame_index(tmp_path / "full")
    return FrameRetriever(tmp_path / "full")


def test_route_refine_abstract_none(retriever):
    r = retriever.route("remove shoes before entering homes", "Japan", tau_ref=0.2, tau_abs=0.05)
    assert r["mode"] == ROUTE_REFINE and r["frames"][0][0].frame_id == "a1"
    # Korea's only frame is about shoes; a tipping question finds no Korean frame
    # but does match a norm type -> abstraction, with an analog from another culture.
    r = retriever.route("tipping service staff waiters", "Korea", tau_ref=0.5, tau_abs=0.05)
    assert r["mode"] == ROUTE_ABSTRACT and r["norm_types"]
    text = render_route(r, retriever.profiles, retriever.nt_by_id, "Korea")
    assert "No record matches" in text and "tipping" in text
    r = retriever.route("anything", "Fiji", tau_ref=0.2, tau_abs=0.05)
    assert r["mode"] == ROUTE_NONE
    r = retriever.route("tipping service staff", "Korea", tau_ref=0.5, tau_abs=0.05,
                        allow_abstraction=False)
    assert r["mode"] == ROUTE_NONE


def test_route_respects_mask(retriever):
    mask = retriever.mask_for_instance_ids({"a1"})
    r = retriever.route("remove shoes before entering homes", "Japan", tau_ref=0.2,
                        tau_abs=0.05, exclude_mask=mask)
    assert all(f.frame_id != "a1" for f, _ in r["frames"])


def _triple(tid, iid, s, sl, p, o, ol, raw):
    return CulturalTriple(triple_id=tid, subject=s, predicate=p, object=o, subject_layer=sl,
                          object_layer=ol, instance_id=iid, raw_fields=raw)


def test_record_pathways_stay_local_and_drop_eval_text():
    L1, L2, L3 = (LayerLabel.LAYER_1_SURFACE, LayerLabel.LAYER_2_SOCIAL,
                  LayerLabel.LAYER_3_DEEP_VALUE)
    raw1 = {"cultural group": "Japanese", "actor_behavior": "remove shoes indoors",
            "eval_whole_desc": "showing deep reverence for cleanliness rituals"}
    raw2 = {"cultural group": "Japanese", "actor_behavior": "bow to elders",
            "eval_whole_desc": "x"}
    ts = [
        _triple("t1", "r1", "shoe removal", L2, "expresses", "respect", L3, raw1),
        _triple("t2", "r1", "shoe removal", L2, "done as", "remove shoes indoors", L1, raw1),
        _triple("t3", "r1", "people", L1, "show", "deep reverence cleanliness rituals", L3, raw1),
        _triple("t4", "r2", "bowing", L2, "done as", "bow to elders", L1, raw2),
    ]
    pws, stats = build_record_pathways(ts)
    assert stats["triples_dropped"] == {"eval_text": 1}
    assert len(pws) == 2 and all(len(p.instance_ids) == 1 for p in pws)
    p1 = next(p for p in pws if p.instance_ids == ["r1"])
    assert [n.layer for n in p1.nodes] == [L3, L2, L1]
    assert p1.completeness.value == "FULL" and "remove shoes" in p1.source_text
    assert "reverence" not in p1.source_text


def test_s3_uses_raw_triples():
    from src.evaluation.mcq_systems import SYSTEM_S3, SYSTEM_S4_TRIPLE, RetrievalContext, build_prompt

    raw = {"cultural group": "Japanese"}
    L1, L3 = LayerLabel.LAYER_1_SURFACE, LayerLabel.LAYER_3_DEEP_VALUE
    raw_t = [_triple("r", "i", "japanese hosts", L1, "serve", "green tea rawversion", L1, raw)]
    ref_t = [_triple("f", "i", "japanese hosts", L1, "serve", "green tea refinedversion", L3, raw)]
    ctx = RetrievalContext(triples=raw_t, refined_triples=ref_t)
    q = MCQQuestion(question_id="q", group_id="q", variant=MCQVariant.EASY,
                    question="What do japanese hosts serve?",
                    cultural_group="Japan", options={"A": "tea", "B": "coffee"}, answer_key="A")
    _, up3, _, _ = build_prompt(SYSTEM_S3, q, ctx, top_k=1)
    _, upa, _, _ = build_prompt(SYSTEM_S4_TRIPLE, q, ctx, top_k=1)
    assert "rawversion" in up3 and "refinedversion" not in up3
    assert "refinedversion" in upa
