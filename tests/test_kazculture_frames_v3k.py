"""KazCulture frames v3k: embedding cache, labelling, build script, S4R adapter,
routing tuner and frame audit. Tiny synthetic data, TF-IDF, no model loads."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.construction import frame_builder as fb
from src.construction import kazculture_frames_v3k as kf
from src.core import config, embedding_utils as eu
from src.llm import prompts

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _tfidf(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "tfidf")
    eu.reset_embedding_state()
    yield
    eu.reset_embedding_state()


# ---------------------------------------------------------------------------
# 1. BGE-M3 embedding backend (fake encoder) and its cache
# ---------------------------------------------------------------------------

class FakeEncoder:
    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts, batch_size=16, normalize_embeddings=True, **kw):
        self.seen += list(texts)
        v = np.array([[len(t), sum(map(ord, t)) % 97 + 1, 1.0] for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_bge_backend_batches_normalizes_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "bge-m3")
    monkeypatch.setattr(config, "EMBEDDING_CACHE_DIR", tmp_path)
    enc = FakeEncoder()
    monkeypatch.setattr(eu, "_get_bge_model", lambda: enc)
    a = eu.embed_texts(["alpha", "beta", "alpha"])
    assert a.shape == (3, 3) and np.allclose(np.linalg.norm(a, axis=1), 1)
    assert enc.seen == ["alpha", "beta"]                     # dedup within a call
    assert eu.backend_name() == f"{eu.BGE_M3_MODEL}@{eu.BGE_M3_REVISION}"
    cache_dir = tmp_path / f"BAAI--bge-m3@{eu.BGE_M3_REVISION}"
    assert (cache_dir / "keys.txt").read_text().split() == [eu.text_key("alpha"),
                                                            eu.text_key("beta")]
    eu.reset_embedding_state()                               # new process
    enc2 = FakeEncoder()
    monkeypatch.setattr(eu, "_get_bge_model", lambda: enc2)
    b = eu.embed_texts(["beta", "gamma"])
    assert enc2.seen == ["gamma"] and np.allclose(b[0], a[1])
    # tf-idf stays selectable per call
    assert eu.embed_texts(["felt yurt", "yurt tea"], fit=True, backend="tfidf").shape[0] == 2


def test_embedding_cache_repairs_a_truncated_tail(tmp_path):
    c = eu.EmbeddingCache(tmp_path, "m/x", "r1")
    c.add(["k1", "k2"], np.eye(2, 3, dtype=np.float32))
    with c.vec_path.open("ab") as fh:                        # half-written vector
        fh.write(b"\x00" * 5)
    with c.keys_path.open("a") as fh:
        fh.write("k3")                                       # key without newline
    c2 = eu.EmbeddingCache(tmp_path, "m/x", "r1")
    assert set(c2.index) == {"k1", "k2"} and c2.vec_path.stat().st_size == 2 * 3 * 4
    c2.add(["k4"], np.ones((1, 3), dtype=np.float32))
    c3 = eu.EmbeddingCache(tmp_path, "m/x", "r1")
    assert c3.get(["k4"])[0].tolist() == [1, 1, 1]
    assert eu.EmbeddingCache(tmp_path, "m/x", "r2").index == {}   # revision is the key


def test_tfidf_stays_the_test_default():
    assert eu.backend_name() == "tfidf"


# ---------------------------------------------------------------------------
# 2. Prompt and vLLM options
# ---------------------------------------------------------------------------

def test_kazakh_label_prompt_keeps_v3_schema():
    f = kf.frame_from_passage({"passage_id": "kzp_1", "text": "Қонақты құрметтеу.",
                               "categories": ["Traditions"]})
    text = kf.render_label_prompt(f, f.text)
    assert "KazCulture" in text and "Kazakh" in text and "Қонақты" in text
    assert '{"norm": "...", "norm_type": "...", "values": ["..."]}' in text
    assert "tradition: maintaining cultural" in text
    assert kf.LABEL_MAX_TOKENS == fb.LABEL_MAX_TOKENS == 200
    none = '{"norm": "none", "norm_type": "none", "values": []}'
    assert fb.parse_label(none) is None and kf.is_explicit_none(none)
    assert not kf.is_explicit_none("garbage") and not kf.is_explicit_none(
        '{"norm": "Guests are honoured.", "norm_type": "x", "values": []}')
    ok = '{"norm": "Guests are honoured.", "norm_type": "Hosting guests", "values": ["tradition"]}'
    assert fb.parse_label(ok)["norm_type"] == "hosting guests"
    assert len(kf.prompt_sha256()) == 64 and kf.prompt_sha256() != kf.prompt_sha256(300)


def test_vllm_backend_passes_revision_and_template_kwargs():
    from src.llm.llm_backend import VLLMBackend

    b = VLLMBackend(model="Qwen/Qwen3-32B", revision="abc", tensor_parallel_size=2,
                    chat_template_kwargs={"enable_thinking": False})
    seen = {}

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            seen.update(kw)
            return "rendered"

    b._tokenizer = Tok()
    assert b._build_prompt("hi", "sys") == "rendered"
    assert seen["enable_thinking"] is False and b.revision == "abc"
    assert b.tensor_parallel_size == 2


class CharTok:
    """One token per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, msgs, tokenize=True, **kw):
        return self.encode("".join(m["content"] for m in msgs))


def test_truncation_is_prompt_only_and_fits_context():
    text = "а" * 50
    t, cut, n = kf.truncate_practice(text, 20, CharTok())
    assert cut and n == 50 and t.startswith("а" * 20) and t.endswith(kf.TRUNCATION_MARK)
    assert kf.truncate_practice(text, 100, CharTok()) == (text, False, 50)
    t2, cut2, _ = kf.truncate_practice(text, 10)            # char fallback
    assert cut2 and len(t2) == 20 + len(kf.TRUNCATION_MARK)
    budget = kf.practice_budget(CharTok(), max_model_len=6000, practice_max_tokens=10_000)
    overhead = len(kf.render_label_prompt(
        kf.frame_from_passage({"passage_id": "p", "text": "", "categories": ["x" * 40]}), "")
        + prompts.FRAME_LABEL_SYSTEM + prompts.FRAME_LABEL_RETRY_SUFFIX)
    assert budget + overhead + 200 <= 6000
    with pytest.raises(ValueError):
        kf.practice_budget(CharTok(), max_model_len=1000)


# ---------------------------------------------------------------------------
# 3. Labelling, checkpoint and merge
# ---------------------------------------------------------------------------

def _passages(n=12):
    cats = ["Hospitality", "Clothing", "Cuisine"]
    out = []
    for i in range(n):
        c = cats[i % 3]
        body = (f"Passage {i} about {c.lower()} customs: guests elders tea yurt felt "
                f"{c.lower()} meal hat {'descriptive' if i == 4 else 'respect'} word{i}")
        out.append({"passage_id": f"kzp_{i:04d}", "text": body, "dataset": "issai/KazCulture",
                    "split": "train", "sources": ["book"], "categories": [c],
                    "train_row_ids": [i], "culture": "Kazakh"})
    return out


def _responder(prompt, system):
    if "Your previous reply was not valid" in prompt:
        return '{"norm": "Elders are served first.", "norm_type": "serving elders", ' \
               '"values": ["tradition"]}'
    if "word7" in prompt:
        return "not json"
    if "descriptive" in prompt.split("Fill in three fields")[0]:
        return '{"norm": "none", "norm_type": "none", "values": []}'
    if "word9" in prompt:
        return "still not json"
    return '{"norm": "Guests are honoured.", "norm_type": "hosting guests", ' \
           '"values": ["benevolence (caring)"]}'


def test_label_passages_statuses_retry_and_resume(tmp_path, fake_backend_factory):
    ps = _passages(10)
    ck = tmp_path / "labels" / "s.jsonl"
    be = fake_backend_factory(_responder)
    rows = kf.label_passages(be, ps, ck, model="m", revision="r", batch_size=4)
    st = {pid: r["status"] for pid, r in rows.items()}
    assert st["kzp_0004"] == "none"              # explicit none, no retry
    assert st["kzp_0007"] == "ok"                # fixed by the retry
    assert all(v == "ok" for k, v in st.items() if k != "kzp_0004")
    retried = [p for prompts_, _ in be.calls for p in prompts_ if "previous reply" in p]
    assert len(retried) == 2 and not any("word4" in p for p in retried)   # 7 and 9
    # truncated last line: dropped and repaired, then relabelled on resume
    lines = ck.read_text().splitlines()
    ck.write_text("\n".join(lines[:-1]) + "\n" + lines[-1][:15])
    be2 = fake_backend_factory(_responder)
    rows2 = kf.label_passages(be2, ps, ck, model="m", revision="r")
    assert len(be2.calls[0][0]) == 1 and len(rows2) == 10
    assert all(json.loads(x) for x in ck.read_text().splitlines())
    # a different revision never reuses rows
    be3 = fake_backend_factory(_responder)
    kf.label_passages(be3, ps, ck, model="m", revision="r2")
    assert len(be3.calls[0][0]) == 10


def test_checkpoint_refuses_corruption_in_the_middle(tmp_path):
    ck = tmp_path / "c.jsonl"
    ck.write_text('{"passage_id": "a"}\nBROKEN\n{"passage_id": "b"}\n')
    with pytest.raises(ValueError):
        kf.read_checkpoint(ck, model="m", revision="r", prompt_hash="h", max_tokens=1)


def test_shards_partition_and_merge_checks(tmp_path, fake_backend_factory):
    ps = _passages(9)
    parts = [kf.shard(ps, i, 3) for i in range(3)]
    assert sorted(p["passage_id"] for part in parts for p in part) == sorted(
        p["passage_id"] for p in ps)
    paths = []
    for i, part in enumerate(parts[:2]):
        path = kf.shard_path(tmp_path, i, 3)
        kf.label_passages(fake_backend_factory(_responder), part, path, model="m", revision="r")
        paths.append(path)
    with pytest.raises(ValueError, match="no label row"):
        kf.merge_label_shards(paths, ps, model="m", revision="r")
    path = kf.shard_path(tmp_path, 2, 3)
    kf.label_passages(fake_backend_factory(_responder), parts[2], path, model="m", revision="r")
    merged, stats = kf.merge_label_shards(paths + [path], ps, model="m", revision="r")
    assert [r["passage_id"] for r in merged] == sorted(p["passage_id"] for p in ps)
    assert stats["n"] == 9
    with pytest.raises(ValueError, match="not in the corpus"):
        kf.merge_label_shards(paths + [path], ps[:5], model="m", revision="r")


# ---------------------------------------------------------------------------
# 4/5. Build script end to end + the S4R adapter
# ---------------------------------------------------------------------------

def _cb_frames():
    rows = []
    specs = [("japanese", "Japanese", "hospitality customs", ["benevolence (caring)", "face"]),
             ("korean", "Korean", "hospitality customs", ["tradition"]),
             ("french", "French", "clothing customs", ["stimulation"])]
    for culture, label, nt, vals in specs:
        for j in range(6):
            rows.append({"frame_id": f"cb_{culture}_{j}", "culture": culture,
                         "culture_label": label, "topic": "Etiquette",
                         "actor_behavior": f"{label} hosts pour tea for guests {j}",
                         "text": f"{label} hosts pour tea for guests and elders {j}.",
                         "norm": f"{label} hosts honour guests {j}.", "norm_type_label": nt,
                         "norm_type_id": "nt_old", "values": vals, "label_status": "ok"})
    rows.append({"frame_id": "cb_kz_0", "culture": "kazakhstani",
                 "culture_label": "Kazakhstani", "topic": "x", "text": "dropped",
                 "norm": "n", "norm_type_label": "hospitality customs", "values": ["tradition"],
                 "label_status": "ok"})
    return rows


@pytest.fixture
def built(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    ps = _passages(12)
    with (corpus / "passages.jsonl").open("w") as fh:
        for p in ps:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    overlap = {"exclude_train_passage_ids": ["kzp_0001", "kzp_0002"]}
    (corpus / "test_overlap.json").write_text(json.dumps(overlap))
    cb = tmp_path / "cb_full"
    cb.mkdir()
    (cb / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _cb_frames()))
    out = tmp_path / "frames_v3k"
    mod = _script("build_kazculture_frames")
    common = ["--passages", str(corpus / "passages.jsonl"), "--overlap_file",
              str(corpus / "test_overlap.json"), "--culturebank_kg", str(cb), "--out", str(out),
              "--fake_labeler", "--embedding_backend", "tfidf"]
    for i in range(2):
        assert mod.main(["--stages", "label", "--shard", f"{i}/2", *common]) == 0
    assert mod.main(["--stages", "merge,kg", "--n_shards", "2", *common]) == 0
    with pytest.raises(FileExistsError):
        mod.main(["--stages", "kg", *common])
    config.EMBEDDING_BACKEND = "tfidf"
    return {"out": out, "passages": ps, "corpus": corpus, "overlap": overlap}


def test_build_writes_versioned_outputs(built):
    out = built["out"]
    man = json.loads((out / "manifest.json").read_text())
    assert man["label_model"] == "fake-labeler" and len(man["label_prompt_sha256"]) == 64
    assert man["n_passages"] == 12 and man["strict_excluded_passages"] == 2
    assert man["dropped_kazakh_culturebank_frames"] == {"Kazakhstani": 1}
    assert man["label_status"] == {"ok": 11, "none": 1}
    kaz = [json.loads(x) for x in (out / "frames.jsonl").read_text().splitlines()]
    assert len(kaz) == 12 and all(f["culture"] == "kazakh" for f in kaz)
    assert {f["text"] for f in kaz} == {p["text"] for p in built["passages"]}   # full L1
    strict = [json.loads(x)["frame_id"] for x in (out / "strict" / "frames.jsonl").open()]
    assert "kzp_0001" not in strict and "kzp_0003" in strict and "cb_kz_0" not in strict
    nts = [json.loads(x) for x in (out / "full" / "norm_types.jsonl").open()]
    shared = [n for n in nts if n["label"] == "hospitality customs"][0]
    assert {"kazakh", "japanese", "korean"} <= set(shared["cultures"])
    prof = json.loads((out / "full" / "culture_profiles.json").read_text())
    assert prof["kazakh"]["n_frames"] == 12 and "japanese" in prof
    assert (out / "full" / "frame_embeddings.npy").exists()


def _adapter(out, setting, **kw):
    from src.retrieval.kazculture_frame_retriever import KazCultureFrameRetriever
    return KazCultureFrameRetriever(out, settings=(setting,), **kw)


def test_refine_blocks_reproduce_render_route_and_budget(built):
    from src.retrieval.frame_retriever import render_route
    from src.retrieval.kazculture_frame_retriever import fit_blocks, render_s4r_evidence

    kr = _adapter(built["out"], "full", tau_ref=0.05, tau_abs=0.05)
    q = "hospitality customs guests elders tea respect"
    blocks = kr.retrieve_s4r(q, {"A": "x"}, "full", 3, 100_000, "full")
    assert blocks and all(b["route"] == "refine" and b["layer"] == "frame" for b in blocks)
    assert set(blocks[0]) == {"text", "source_id", "layer", "route", "indirect", "similarity"}
    assert not any(b["indirect"] for b in blocks) and len(blocks) <= 3
    fr = kr.retrievers["full"]
    r = kr.route(q, setting="full", top_k=3)
    assert render_s4r_evidence(blocks) == render_route(r, fr.profiles, fr.nt_by_id, "Kazakh")
    assert blocks[0]["text"].startswith("Specific knowledge about Kazakh")
    # budget: same semantics as render_evidence_with_hits (prefix, separators counted)
    cap = len(render_s4r_evidence(blocks[:2]))
    small = kr.retrieve_s4r(q, None, "full", 3, cap, "full")
    assert small == blocks[:2] and len(render_s4r_evidence(small)) <= cap
    assert kr.retrieve_s4r(q, None, "full", 3, cap - 1, "full") == blocks[:1]
    assert fit_blocks(blocks, 5) == []
    # leave-source-out
    top = blocks[0]["source_id"]
    assert top not in {b["source_id"] for b in kr.retrieve_s4r(
        q, None, "full", 3, 100_000, "full", exclude_passage_ids={top})}


def test_abstract_route_analogs_are_indirect_and_noabs_disables(built):
    from src.retrieval.frame_retriever import render_route
    from src.retrieval.kazculture_frame_retriever import render_s4r_evidence

    kr = _adapter(built["out"], "full", tau_ref=1.01, tau_abs=-1.0)
    q = "hospitality customs guests"
    blocks = kr.retrieve_s4r(q, None, "full", 8, 100_000, "full")
    layers = [b["layer"] for b in blocks]
    assert blocks[0]["route"] == "abstract" and "norm_type" in layers
    assert "culture_profile" in layers and "norm" in layers and "analog" in layers
    analogs = [b for b in blocks if b["layer"] == "analog"]
    assert all(b["indirect"] and b["source_id"].startswith("cb_") for b in analogs)
    assert all(not b["indirect"] for b in blocks if b["layer"] == "norm")
    fr = kr.retrievers["full"]
    r = kr.route(q, setting="full")
    assert render_s4r_evidence(blocks) == render_route(r, fr.profiles, fr.nt_by_id, "Kazakh")
    assert kr.retrieve_s4r(q, None, "full", 8, 100_000, "noabs") == []
    with pytest.raises(ValueError):
        kr.retrieve_s4r(q, None, "full", 8, 100, "other")


def test_strict_index_excludes_test_overlap(built):
    kr = _adapter(built["out"], "strict", tau_ref=-1.0, tau_abs=-1.0)
    ids = {b["source_id"] for b in kr.retrieve_s4r("customs guests", None, "strict",
                                                     50, 10**6, "full")}
    assert ids and not ids & set(built["overlap"]["exclude_train_passage_ids"])
    assert kr.labels_for_passage("kzp_0001", setting="strict") is None   # not in strict
    with pytest.raises(ValueError):
        kr.retrieve_s4r("x", None, "full", 3, 100, "full")      # setting not loaded


def test_labels_for_passage_and_shuffled_control(built):
    kr = _adapter(built["out"], "full", tau_ref=0.05, tau_abs=0.05)
    lab = kr.labels_for_passage("kzp_0000", setting="full")
    assert set(lab) == {"norm", "norm_type", "values"} and lab["values"]
    assert lab["norm_type"] == "hospitality customs"
    assert kr.labels_for_passage("kzp_0004", setting="full") is None     # "none" label
    donors = kr.donors("full")["frame"]
    kaz = kr.kazakh_frames("full")
    assert all(d != k and kaz[d].topic != kaz[k].topic for k, d in donors.items())
    assert donors == _adapter(built["out"], "full", tau_ref=0, tau_abs=0).donors("full")["frame"]
    shuf = kr.labels_for_passage("kzp_0000", setting="full", shuffled=True)
    assert shuf == kr.labels_for_passage(donors["kzp_0000"], setting="full") != lab
    q = "hospitality customs guests elders tea respect"
    a = kr.retrieve_s4r(q, None, "full", 3, 10**6, "full")
    b = kr.retrieve_s4r(q, None, "full", 3, 10**6, "full", shuffled=True)
    assert [x["source_id"] for x in a] == [x["source_id"] for x in b]   # same practices
    assert [x["text"] for x in a] != [x["text"] for x in b]
    kr2 = _adapter(built["out"], "full", tau_ref=1.01, tau_abs=-1.0)
    a2 = kr2.retrieve_s4r("hospitality customs guests", None, "full", 8, 10**6, "full")
    b2 = kr2.retrieve_s4r("hospitality customs guests", None, "full", 8, 10**6, "full",
                          shuffled=True)
    an_a = {x["source_id"] for x in a2 if x["layer"] == "analog"}
    an_b = {x["source_id"] for x in b2 if x["layer"] == "analog"}
    assert an_a and an_b and not an_a & an_b


def test_thresholds_required(built):
    kr = _adapter(built["out"], "full")
    with pytest.raises(ValueError, match="thresholds unset"):
        kr.retrieve_s4r("x", None, "full", 3, 100, "full")
    (built["out"] / "routing_thresholds.json").write_text('{"tau_ref": 0.4, "tau_abs": 0.3}')
    kr = _adapter(built["out"], "full")
    assert (kr.tau_ref, kr.tau_abs) == (0.4, 0.3)


# ---------------------------------------------------------------------------
# 6. Routing tuner (stub scorer)
# ---------------------------------------------------------------------------

def test_tune_routing_train_only_disjoint_sample(built, tmp_path):
    mod = _script("tune_kazculture_routing")
    rows = [{"train_row_id": i, "question": f"customs guests tea question {i}",
             "answer": f"ans {i}", "passage_id": f"kzp_{i % 12:04d}", "category": "c"}
            for i in range(420)]
    qa = tmp_path / "train_dev_qa.jsonl"
    qa.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sample = mod.sample_questions(rows, n=10, seed=0)
    assert all(r["train_row_id"] >= 400 for r in sample)
    assert sample == mod.sample_questions(rows, n=10, seed=0)
    calls = []

    class Scorer:
        tokenizer = None

        def score_gold_loglik(self, prompts_, golds):
            calls.append(len(prompts_))
            assert len(prompts_) == len(golds)
            return [{"mean_loglik": -len(p) / 1000} for p in prompts_]   # short wins

    class Ev:
        def __init__(self, items):
            self.items = items

        def text(self):
            return "\n\n".join(f"[{i}] {x['text']}" for i, x in enumerate(self.items, 1))

    rendering = (lambda items, max_chars: Ev(items),
                 lambda tok, question, evidence: f"{evidence}\nQuestion:\n{question}\nAnswer:")
    args = ["--frames_dir", str(built["out"]), "--train_qa", str(qa), "--n", "10",
            "--embedding_backend", "tfidf"]
    assert mod.main(args + ["--dry_run"]) == 0
    assert not (built["out"] / "routing_thresholds.json").exists()
    assert mod.main(args, scorer=Scorer(), rendering=rendering) == 0
    ids = json.loads((built["out"] / "routing_tuning_train_row_ids.json").read_text())
    assert ids == [r["train_row_id"] for r in sample]
    res = json.loads((built["out"] / "routing_thresholds.json").read_text())
    assert res["source"] == "train_only" and len(res["grid"]) == 12
    assert res["tau_ref"] in mod.TAU_REF_GRID and res["tau_abs"] in mod.TAU_ABS_GRID
    best = max(g["mean_gold_loglik"] for g in res["grid"])
    assert res["best_mean_gold_loglik"] == best and calls == [res["n_unique_prompts"]]
    with pytest.raises(FileExistsError):
        mod.main(args, scorer=Scorer(), rendering=rendering)
    with pytest.raises(ValueError, match="train"):
        bad = tmp_path / "test_qa.jsonl"
        bad.write_text(qa.read_text())
        mod.main(["--frames_dir", str(built["out"]), "--train_qa", str(bad), "--n", "10",
                  "--out", str(tmp_path / "t.json"), "--embedding_backend", "tfidf"],
                 scorer=Scorer(), rendering=rendering)


# ---------------------------------------------------------------------------
# 7. Frame audit
# ---------------------------------------------------------------------------

def test_frame_audit_is_blinded_and_refuses_blanks(built, tmp_path):
    mod = _script("build_kazculture_frame_audit")
    out = tmp_path / "audit"
    assert mod.main(["build", "--frames_dir", str(built["out"]), "--n", "10",
                     "--out_dir", str(out)]) == 0
    with pytest.raises(FileExistsError):
        mod.main(["build", "--frames_dir", str(built["out"]), "--n", "10", "--out_dir", str(out)])
    with (out / "frame_review_10.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == mod.COLUMNS and len(rows) == 10
    assert all(r[h] == "" for r in rows for h in mod.HUMAN)
    assert not any("kzp_" in v for r in rows for v in r.values())
    key = json.loads((out / "frame_review_10_key.json").read_text())
    with pytest.raises(ValueError, match="blanks are refused|must be 0 or 1"):
        mod.score(rows, key)
    for r in rows:
        ok = key["keys"][r["review_id"]]["label_status"] == "ok"
        r.update({"norm_warranted_0_or_1": "1", "norm_correct_0_or_1": "1" if ok else "NA",
                  "values_correct_0_or_1": "0" if ok else "NA"})
    res = mod.score(rows, key)
    assert res["n_reviewed"] == 10 and res["proposed_values_correct"]["rate"] in (0.0, None)
    rows[0]["values_correct_0_or_1"] = ""
    with pytest.raises(ValueError):
        mod.score(rows, key)


def test_module_level_api_for_the_eval_harness(built):
    import inspect

    import src.retrieval.kazculture_frame_retriever as m

    assert "exclude_passage_ids" in inspect.signature(m.retrieve_s4r).parameters
    m.configure(frames_dir=built["out"], settings=("full",), tau_ref=0.05, tau_abs=0.05)
    q = "hospitality customs guests elders tea respect"
    routed = m.retrieve_s4r(q, None, setting="full", top_k=3, budget_chars=10**6,
                            route_mode="routed")
    assert routed == m.retrieve_s4r(q, None, "full", 3, 10**6, "full")
    shuf = m.retrieve_s4r(q, None, setting="full", top_k=3, budget_chars=10**6,
                          route_mode="shuffled")
    assert [b["source_id"] for b in shuf] == [b["source_id"] for b in routed]
    assert m.labels_for_passage("kzp_0003")["values"]
    assert m.labels_for_passage("kzp_missing") is None
    m.configure()          # reset to defaults for other tests
