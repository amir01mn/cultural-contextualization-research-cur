"""Offline tests for the final KazCulture harness (no model or index loads)."""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from src.evaluation import kazculture_final as kf
from src.retrieval.kazculture_index import (
    oversize_break_affected, render_evidence_with_hits, render_s1_blocks,
)

# ----------------------------------------------------------------- fakes

class FakeTokenizer:
    """Chat template stand-in; one 'token' per character."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        self.calls.append(kw)
        return "<user>" + messages[0]["content"] + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class FakeScorer:
    """Picks the option whose text appears in the evidence, else A."""

    def __init__(self, family="qwen3", limit=None):
        self.tokenizer = FakeTokenizer()
        self.family = family
        self.limit = limit
        self.rendered = []

    def render(self, q, evidence):
        from src.evaluation.kazculture_rag import evidence_prompt
        kw = {} if self.family == "olmo2" else None
        return evidence_prompt(self.tokenizer, q, evidence, template_kwargs=kw)

    def count_tokens(self, rendered):
        return len(rendered)

    def score_rendered(self, rendered):
        self.rendered.append(rendered)
        if self.limit is not None:
            assert len(rendered) <= self.limit
        ev = rendered.split("Question:")[0]
        out = {k: -5.0 for k in "ABCD"}
        for k in "ABCD":
            if f"OPT{k}" in ev:
                out[k] = -0.1
                break
        else:
            out["A"] = -1.0
        return out

    def score_gold_loglik(self, prompts, golds):
        return [{"mean_loglik": -1.0 + 0.1 * p.count("[") , "sum_loglik": -1.0,
                 "n_gold_tokens": 1, "n_prompt_tokens": len(p)} for p in prompts]


class FakeIndex:
    def __init__(self, units):
        self.units = units
        self.searches = []

    def search(self, query, *, system, k=5, min_similarity=0.0, exclude_passage_ids=None,
               combined=False):
        self.searches.append((query, k, frozenset(exclude_passage_ids or ())))
        exclude = exclude_passage_ids or set()
        scored = []
        for u in self.units:
            if u["system"] != system or u["passage_id"] in exclude:
                continue
            words = set(query.split())
            sim = round(len(words & set(u["text"].split())) / (len(words) or 1), 6)
            if sim >= min_similarity:
                scored.append({**u, "similarity": sim})
        scored.sort(key=lambda h: (-h["similarity"], h["unit_id"]))
        return scored[:k]


def unit(i, text):
    return {"unit_id": f"S1:kzp_tr{i}", "passage_id": f"kzp_tr{i}", "system": "S1",
            "dataset": "KazCulture", "source": ["Book"], "culture": "Kazakh",
            "category": ["Clothing"], "values": [], "indirect": False, "text": text}


def make_test_file(tmp_path, n=6, desc=lambda i: f"test passage {i}"):
    rows = []
    for i in range(n):
        gold = "ABCD"[i % 4]
        rows.append({"set": "test", "question": f"q{i} shapan kiiz",
                     "a": "x", "b": "y", "c": "z", "d": "w", "answer_label": gold,
                     "category_english": "Clothing" if i % 2 else "Cuisine",
                     "item_description": desc(i)})
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path, rows


def make_index_dir(tmp_path, name, strict):
    d = tmp_path / name
    d.mkdir()
    (d / "build_manifest.json").write_text(json.dumps({"strict_passage_disjoint": strict}))
    (d / "metadata.json").write_text(json.dumps({"units_sha256": name}))
    return d


def install_fake_adapter(monkeypatch, *, leak_pid=None):
    mod = types.ModuleType("fake_frame_adapter")
    mod.calls = []

    def retrieve_s4r(question_text, options, setting, top_k, budget_chars, route_mode,
                     exclude_passage_ids=None):
        mod.calls.append((question_text, setting, route_mode, exclude_passage_ids))
        i = int(question_text.split()[0][1:]) if question_text.startswith("q") else 0
        gold = "ABCD"[i % 4]
        items = [{"text": f"Norm about OPT{gold} {'x' * 300}", "source_id": "kzp_tr1",
                  "layer": "L2", "route": "refine", "indirect": False, "similarity": 0.8},
                 {"text": "Value: tradition " + "y" * 200, "source_id": "cb_7",
                  "layer": "L3", "route": "abstract", "indirect": True, "similarity": 0.6}]
        if route_mode == "shuffled":
            items[0]["text"] = "Unrelated norm " + "z" * 300
        if leak_pid:
            items.append({"text": "leak", "source_id": leak_pid, "layer": "L1",
                          "route": "refine", "indirect": False, "similarity": 0.5})
        if route_mode == "noabs":
            items = items[:1]
        if exclude_passage_ids:
            items = [x for x in items if x["source_id"] not in exclude_passage_ids]
        return items

    def labels_for_passage(pid):
        if pid == "kzp_tr0":
            raise KeyError(pid)
        return {"norm": "guests are honoured", "norm_type": "hospitality",
                "values": ["benevolence"]}

    mod.retrieve_s4r = retrieve_s4r
    mod.labels_for_passage = labels_for_passage
    monkeypatch.setitem(sys.modules, "fake_frame_adapter", mod)
    return mod


# ----------------------------------------------------------------- checkpoints

def test_checkpoint_drops_truncated_last_line_and_repairs(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b"')
    assert kf.read_jsonl_checkpoint(p) == [{"a": 1}, {"a": 2}]
    assert p.read_text() == '{"a": 1}\n{"a": 2}\n'
    with p.open("a") as fh:
        kf.append_jsonl(fh, {"a": 4})
    assert [r["a"] for r in kf.read_jsonl_checkpoint(p)] == [1, 2, 4]


def test_checkpoint_restores_missing_newline_and_rejects_mid_corruption(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}')
    assert len(kf.read_jsonl_checkpoint(p)) == 2
    assert p.read_text().endswith("\n")
    p.write_text('{"a": 1}\n{"a": \n{"a": 3}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        kf.read_jsonl_checkpoint(p)


# ----------------------------------------------------------------- rendering

def test_default_render_keeps_frozen_break_and_skip_is_opt_in():
    hits = [{**unit(0, "a" * 100), "similarity": 1}, {**unit(1, "b" * 900), "similarity": .9},
            {**unit(2, "c" * 50), "similarity": .8}]
    text, used = render_evidence_with_hits(hits, max_chars=400)
    assert [u["passage_id"] for u in used] == ["kzp_tr0"]
    text2, used2 = render_evidence_with_hits(hits, max_chars=400, skip_oversize=True)
    assert [u["passage_id"] for u in used2] == ["kzp_tr0", "kzp_tr2"]
    assert "[2] KAZAKH TRAINING SOURCE" in text2 and "[3]" not in text2
    assert oversize_break_affected(hits, max_chars=400)
    assert not oversize_break_affected(hits[:1], max_chars=400)
    blocks, _ = render_s1_blocks(hits, max_chars=400)
    assert "\n\n".join(blocks) == text


def test_s1_evidence_matches_frozen_renderer_exactly():
    hits = [{**unit(i, f"text {i} " * (40 * i + 5)), "similarity": 1 - i / 10} for i in range(10)]
    ev = kf.s1_evidence(hits, top_k=8, max_chars=5000)
    frozen_text, frozen_used = render_evidence_with_hits(hits[:8])
    assert ev.text() == frozen_text
    assert [m["unit_id"] for m in ev.meta] == [u["unit_id"] for u in frozen_used]


def test_evidence_prompt_template_unchanged_from_frozen_v2():
    from src.evaluation.kazculture_rag import evidence_prompt
    from src.evaluation.kazculture_hf_scorer_v2 import render_prompt, user_question
    from src.loaders.kazculture_loader import KazCultureQuestion
    q = KazCultureQuestion("id", "Сұрақ?", {k: k.lower() for k in "ABCD"}, "A", "C")
    tok = FakeTokenizer()
    frozen = ("<user>Retrieved cultural evidence (may be incomplete; answer only from "
              "supported facts):\nEV\n\n" + user_question(q) + "<assistant>Answer:")
    assert evidence_prompt(tok, q, "EV") == frozen
    assert evidence_prompt(tok, q, "  ") == render_prompt(tok, q)
    assert tok.calls[0] == {"enable_thinking": False}
    evidence_prompt(tok, q, "EV", template_kwargs={})
    assert tok.calls[-1] == {}


def test_ctrl_len_append_is_s1_plus_exact_extra_budget():
    hits = [{**unit(i, f"p{i} " + "w" * (100 + 10 * i)), "similarity": 1 - i / 50}
            for i in range(20)]
    s1 = kf.s1_evidence(hits, top_k=3, max_chars=5000)
    same = kf.ctrl_len_evidence(hits, s1, extra_chars=0, top_k=3, base_budget=5000)
    assert same.text() == s1.text()
    ev = kf.ctrl_len_evidence(hits, s1, extra_chars=500, top_k=3, base_budget=5000)
    assert ev.text().startswith(s1.text() + "\n\n[4] KAZAKH")
    assert 0 < ev.info["ctrl_len_added_chars"] <= 500
    assert len(ev.text()) <= len(s1.text()) + 500
    assert [m["unit_id"] for m in ev.meta[:3]] == [m["unit_id"] for m in s1.meta]
    assert len({m["unit_id"] for m in ev.meta}) == len(ev.meta)
    budget = kf.ctrl_len_evidence(hits, s1, extra_chars=500, top_k=3, base_budget=5000,
                                  mode="budget")
    assert len(budget.text()) <= 5500


def test_format_labels_and_frame_blocks_number_continuously():
    assert kf.format_labels(None) is None
    lab = kf.format_labels({"norm": "n", "norm_type": "t", "values": ["v1", "v2"]})
    assert lab.startswith("Provisional labels") and "values: v1, v2" in lab
    ev = kf.frame_evidence([{"text": "a", "source_id": "s1", "layer": "L2", "route": "refine"},
                            {"text": "b", "source_id": "cb", "indirect": True}], max_chars=1000,
                           render="numbered")
    assert ev.text().startswith("[1] KAZAKH TRAINING SOURCE | layer=L2 | route=refine")
    assert "[2] INDIRECT COMPARISON" in ev.text()


def test_frame_evidence_adapter_layout_is_verbatim_and_budgeted():
    items = [{"text": "Specific knowledge about Kazakh:\n[1] Kazakh\n  norm: n1",
              "source_id": "kzp_a", "layer": "frame", "route": "refine"},
             {"text": "[2] Kazakh\n  norm: n2", "source_id": "kzp_b", "layer": "frame",
              "route": "refine"},
             {"text": "x" * 500, "source_id": None, "layer": "frame", "route": "refine"}]
    ev = kf.frame_evidence(items, max_chars=200)
    assert ev.text() == items[0]["text"] + "\n" + items[1]["text"]
    assert ev.info["frame_items_dropped"] == 1 and ev.meta[1]["route"] == "refine"
    assert kf.frame_meta(items[2])["id"] == "-"


def test_real_frame_adapter_contract_if_present():
    mod = pytest.importorskip("src.retrieval.kazculture_frame_retriever")
    import inspect
    params = inspect.signature(mod.retrieve_s4r).parameters
    for name in ("question_text", "options", "setting", "top_k", "budget_chars", "route_mode",
                 "exclude_passage_ids"):
        assert name in params
    assert set(kf.DEFAULT_ROUTE_MODES.values()) <= set(mod.ROUTE_MODES)
    assert callable(mod.labels_for_passage) and callable(mod.configure)
    assert kf.FrameAdapter(mod).exclude_kw == "exclude_passage_ids"


def test_fit_blocks_largest_prefix():
    assert kf.fit_blocks(5, lambda k: k <= 3) == 3
    assert kf.fit_blocks(2, lambda k: False) == 0


# ----------------------------------------------------------------- statistics

def test_mcnemar_and_bootstrap_and_holm():
    m = kf.mcnemar_exact([0] * 10, [1] * 10)
    assert m["rescued"] == 10 and abs(m["p"] - 2 / 1024) < 1e-12
    assert kf.mcnemar_exact([1, 0], [1, 0])["p"] == 1.0
    lo, hi = kf.paired_bootstrap_ci([1] * 50 + [0] * 50, n_boot=2000, seed=0)
    assert 0.35 < lo < 0.5 < hi < 0.65
    assert kf.paired_bootstrap_ci([1, 0, 1], n_boot=100, seed=0) == \
        kf.paired_bootstrap_ci([1, 0, 1], n_boot=100, seed=0)
    from src.evaluation.intrinsic_eval import holm
    assert holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}


def test_compare_paired_requires_matching_ids_and_reports_categories():
    b = [{"question_id": str(i), "correct": i < 5, "category": "X"} for i in range(10)]
    s = [{"question_id": str(i), "correct": i < 8, "category": "X"} for i in range(10)]
    res = kf.compare_paired(b, s, n_boot=200)
    assert res["delta_pp"] == pytest.approx(30.0) and res["rescued"] == 3
    assert res["category_delta"]["X"]["net"] == 3
    with pytest.raises(ValueError):
        kf.compare_paired(b, s[:-1])


# ----------------------------------------------------------------- scorer

def test_score_gold_loglik_scores_only_gold_tokens():
    torch = pytest.importorskip("torch")
    from src.evaluation.kazculture_hf_scorer_v2 import (
        HFQwenLetterScorerV2, gold_continuation_ids,
    )
    tok = FakeTokenizer()
    ids, n = gold_continuation_ids(tok, "PQ", "ab")
    assert ids == [ord(c) for c in "PQ ab"] and n == 2

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids):
            # Uniform over 200 tokens except next-token 'b' gets a big logit.
            logits = torch.zeros(1, input_ids.shape[1], 200)
            logits[..., ord("b")] = 10.0
            return types.SimpleNamespace(logits=logits)

    s = HFQwenLetterScorerV2.__new__(HFQwenLetterScorerV2)
    s.torch, s.tokenizer, s.model, s.max_context_tokens = torch, tok, Model(), 100
    res = s.score_gold_loglik(["PQ"], ["ab"])[0]
    assert res["n_gold_tokens"] == 3 and res["n_prompt_tokens"] == 2
    lp_b = 10 - torch.logsumexp(torch.tensor([10.0] + [0.0] * 199), 0).item()
    lp_other = -torch.logsumexp(torch.tensor([10.0] + [0.0] * 199), 0).item()
    assert res["mean_loglik"] == pytest.approx((2 * lp_other + lp_b) / 3, abs=1e-4)
    with pytest.raises(ValueError):
        s.score_gold_loglik(["PQ"], [" "])


def test_olmo_scorer_has_no_thinking_kwarg_and_pinned_revision():
    from src.evaluation import kazculture_hf_scorer_olmo as olmo
    assert olmo.HFOlmoLetterScorer.template_kwargs == {}
    assert olmo.OLMO_REVISION == "470b1fba1ae01581f270116362ee4aa1b97f4c84"
    assert olmo.OLMO_MAX_CONTEXT_TOKENS == 4096


# ----------------------------------------------------------------- evaluator end to end

@pytest.fixture
def harness(tmp_path, monkeypatch):
    from scripts import eval_kazculture_final as ev
    test_file, rows = make_test_file(tmp_path)
    train = [unit(i, f"shapan kiiz train {i} OPT{'ABCD'[i % 4]} " + "t" * (60 * i))
             for i in range(12)]
    indexes = {"s_strict": FakeIndex(train), "s_full": FakeIndex(train)}
    monkeypatch.setattr(ev, "make_index", lambda path: indexes[path.name])
    scorers = []
    monkeypatch.setattr(ev, "make_scorer", lambda a: scorers.append(
        FakeScorer(a.model_family, a.max_context_tokens if a.model_family == "olmo2" else None))
        or scorers[-1])
    adapter = install_fake_adapter(monkeypatch)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"source": "train_dev_only", "top_k": 8, "min_similarity": 0.0,
                               "evidence_budget_chars": 1500}))
    base = ["--test_file", str(test_file), "--s1_config", str(cfg),
            "--index_strict", str(make_index_dir(tmp_path, "s_strict", True)),
            "--index_full", str(make_index_dir(tmp_path, "s_full", False)),
            "--frame_adapter", "fake_frame_adapter", "--expected_size", str(len(rows)),
            "--out", str(tmp_path / "out")]
    return types.SimpleNamespace(ev=ev, base=base, out=tmp_path / "out", scorers=scorers,
                                 indexes=indexes, adapter=adapter, rows=rows, tmp=tmp_path,
                                 train=train)


def test_final_evaluator_all_conditions_one_model_load(harness):
    h = harness
    rc = h.ev.main(h.base + ["--conditions", ",".join(kf.CONDITIONS), "--settings", "strict,full"])
    assert rc == 0
    assert len(h.scorers) == 1  # one model load for every condition and setting
    for sett, cond in [("closed_book", "S0")] + [(s, c) for s in ("strict", "full")
                                                for c in kf.CONDITIONS if c != "S0"]:
        rows = kf.read_jsonl_checkpoint(h.out / sett / f"{cond}.jsonl")
        assert len(rows) == len(h.rows), (sett, cond)
        r = rows[0]
        for key in ("condition", "setting", "retrieved_ids", "rendered_chars", "prediction",
                    "gold", "correct"):
            assert key in r
        assert r["condition"] == cond and r["setting"] == sett
    # Strict S1 used the frozen exclusion (all test passages) and frozen rendering.
    strict_s1 = kf.read_jsonl_checkpoint(h.out / "strict" / "S1.jsonl")
    ix = h.indexes["s_strict"]
    from src.loaders.kazculture_corpus import passage_id
    test_pids = {passage_id(r["item_description"]) for r in h.rows}
    assert all(ex == test_pids for _, _, ex in ix.searches)
    q0 = h.rows[0]["question"]
    frozen_text, frozen_used = render_evidence_with_hits(
        ix.search(q0, system="S1", k=8, min_similarity=0.0, exclude_passage_ids=test_pids),
        max_chars=1500)
    assert strict_s1[0]["evidence_sha256"] == hashlib.sha256(frozen_text.encode()).hexdigest()
    assert strict_s1[0]["retrieved_unit_ids"] == [u["unit_id"] for u in frozen_used]
    # S1+L attaches labels except for the passage without labels.
    s1l = kf.read_jsonl_checkpoint(h.out / "strict" / "S1+L.jsonl")
    assert all(r["labels_attached"] >= 1 for r in s1l)
    # CTRL-LEN extends S1 by exactly the S4R excess at most.
    cl = kf.read_jsonl_checkpoint(h.out / "strict" / "CTRL-LEN.jsonl")
    for r, s in zip(cl, strict_s1):
        assert r["ctrl_len_s1_chars"] == s["rendered_chars"]
        assert r["ctrl_len_target_extra_chars"] == max(0, r["ctrl_len_s4r_chars"] - s["rendered_chars"])
        assert r["rendered_chars"] - s["rendered_chars"] == r["ctrl_len_added_chars"]
        assert r["ctrl_len_added_chars"] <= r["ctrl_len_target_extra_chars"]
        assert r["retrieved_unit_ids"][:len(s["retrieved_unit_ids"])] == s["retrieved_unit_ids"]
    # Route modes reach the adapter; S4R items are cached and reused by CTRL-LEN.
    modes = {m for _, _, m, _ in h.adapter.calls}
    assert modes == {"routed", "noabs", "shuffled"}
    s4r_calls = [c for c in h.adapter.calls if c[1] == "strict" and c[2] == "routed"]
    assert len(s4r_calls) == len(h.rows)
    s4r = kf.read_jsonl_checkpoint(h.out / "strict" / "S4R.jsonl")
    assert s4r[0]["routes"] == ["refine", "abstract"] and s4r[0]["indirect"] == [False, True]
    summary = json.loads((h.out / "strict" / "S1_summary.json").read_text())
    assert summary["n"] == len(h.rows) and "oversize_break_affected_n" in summary
    runs = kf.read_jsonl_checkpoint(h.out / "runs.jsonl")
    assert len(runs) == 1 and runs[0]["conditions"] == list(kf.CONDITIONS)


def test_final_evaluator_resume_after_truncation_and_reproduction(harness):
    h = harness
    assert h.ev.main(h.base + ["--conditions", "S0,S1"]) == 0
    s1 = h.out / "strict" / "S1.jsonl"
    lines = s1.read_text().splitlines(keepends=True)
    ref = h.tmp / "ref_s1.jsonl"
    ref.write_text("".join(lines))
    ref0 = h.tmp / "ref_s0.jsonl"
    ref0.write_text((h.out / "closed_book" / "S0.jsonl").read_text())
    s1.write_text("".join(lines[:3]) + lines[3][:25])  # killed mid-write
    args = h.base + ["--conditions", "S0,S1,S4R", "--reference_s0", str(ref0),
                     "--reference_s1_strict", str(ref), "--require_reproduction"]
    assert h.ev.main(args) == 0
    rep = json.loads((h.out / "reproduction_strict_S1.json").read_text())
    assert rep["exact"] and rep["n"] == len(h.rows)
    assert len(kf.read_jsonl_checkpoint(s1)) == len(h.rows)
    assert (h.out / "strict" / "S4R.jsonl").exists()


def test_reproduction_gate_blocks_structured_conditions(harness):
    h = harness
    assert h.ev.main(h.base + ["--conditions", "S0,S1"]) == 0
    bad = h.tmp / "bad.jsonl"
    rows = kf.read_jsonl_checkpoint(h.out / "strict" / "S1.jsonl")
    rows[0]["prediction"] = "D" if rows[0]["prediction"] != "D" else "C"
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows))
    import shutil
    shutil.rmtree(h.out)
    rc = h.ev.main(h.base + ["--conditions", "S1,S4R", "--reference_s1_strict", str(bad),
                             "--require_reproduction"])
    assert rc == 3
    assert not (h.out / "strict" / "S4R.jsonl").exists()
    with pytest.raises(SystemExit):
        h.ev.main(h.base + ["--conditions", "S4R", "--require_reproduction"])


def test_settings_change_and_strict_leak_are_refused(harness, monkeypatch):
    h = harness
    assert h.ev.main(h.base + ["--conditions", "S1"]) == 0
    with pytest.raises(ValueError, match="Settings changed"):
        h.ev.main(h.base + ["--conditions", "S1", "--skip_oversize"])
    from src.loaders.kazculture_corpus import passage_id
    install_fake_adapter(monkeypatch, leak_pid=passage_id(h.rows[2]["item_description"]))
    with pytest.raises(ValueError, match="Strict leakage"):
        h.ev.main(h.base[:-1] + [str(h.tmp / "out2"), "--conditions", "S4R"])


def test_time_guard_stops_resumably(harness):
    h = harness
    assert h.ev.main(h.base + ["--conditions", "S1", "--max_minutes", "0"]) == 0
    assert kf.read_jsonl_checkpoint(h.out / "strict" / "S1.jsonl") == []


def test_retrieval_only_reports_oversize_without_model(harness):
    h = harness
    assert h.ev.main(h.base + ["--conditions", "S1,CTRL-LEN", "--retrieval_only"]) == 0
    assert h.scorers == []
    rep = json.loads((h.out / "retrieval_report_strict.json").read_text())
    assert rep["n_questions"] == len(h.rows) and rep["oversize_break_affected_n"] is not None
    assert (h.out / "retrieval" / "strict_S4R.jsonl").exists()


def test_olmo_truncates_evidence_by_blocks(harness):
    h = harness
    with pytest.raises(ValueError, match="OLMo"):
        h.ev.main(h.base + ["--conditions", "S1+L", "--model_family", "olmo2"])
    out = h.tmp / "olmo"
    args = h.base[:-1] + [str(out), "--conditions", "S0,S1,S4R", "--model_family", "olmo2",
                          "--max_context_tokens", "700"]
    assert h.ev.main(args) == 0
    rows = kf.read_jsonl_checkpoint(out / "strict" / "S1.jsonl")
    assert any(r["truncated"] for r in rows)
    assert all(r["prompt_tokens"] <= 700 for r in rows)
    summ = json.loads((out / "strict" / "S1_summary.json").read_text())
    assert summ["truncated_n"] == sum(r["truncated"] for r in rows)
    st = json.loads((out / "settings.json").read_text())
    assert st["model"] == "allenai/OLMo-2-1124-7B-Instruct" and st["truncate_to_fit"]


# ----------------------------------------------------------------- gate

def test_traindev_gate_leave_own_source_out_and_decision(tmp_path, monkeypatch):
    from scripts import eval_kazculture_traindev_gate as gate
    train = [unit(i, f"shapan kiiz train {i}") for i in range(30)]
    ix = FakeIndex(train)
    monkeypatch.setattr(gate, "make_index", lambda p: ix)
    monkeypatch.setattr(gate, "make_scorer", lambda a: FakeScorer())
    adapter = install_fake_adapter(monkeypatch)
    dev = tmp_path / "dev.jsonl"
    dev.write_text("".join(json.dumps({"train_row_id": i, "question": f"q{i} shapan kiiz",
                                       "answer": "an answer", "passage_id": f"kzp_tr{i % 30}"})
                           + "\n" for i in range(60)))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"source": "train_dev_only", "top_k": 8, "min_similarity": 0.0,
                               "evidence_budget_chars": 5000}))
    out = tmp_path / "gate"
    args = ["--dev", str(dev), "--index", str(tmp_path), "--s1_config", str(cfg),
            "--out", str(out), "--n", "20", "--exclude_first_n", "10",
            "--frame_adapter", "fake_frame_adapter", "--n_boot", "200"]
    assert gate.main(args) == 0
    dec = json.loads((out / "gate_decision.json").read_text())
    assert set(dec["condition_means"]) == {"S1", "S4R", "S4R-noabs", "S1+L"}
    assert dec["gate"]["decision"] in ("proceed_s4r_primary",
                                       "not_met_run_test_once_report_negative_no_iteration")
    assert dec["gate"]["passed"] == (dec["paired"]["S4R_minus_S1"]["mean_delta"] > 0)
    s1 = kf.read_jsonl_checkpoint(out / "S1.jsonl")
    assert len(s1) == 20 and all(r["train_row_id"] >= 10 for r in s1)
    assert all(r["passage_id"] not in r["retrieved_ids"] for r in s1)
    assert all(c[3] == {f"kzp_tr{int(c[0].split()[0][1:]) % 30}"} for c in adapter.calls)
    # Same seed -> same sample; resume is a no-op.
    assert gate.main(args) == 0


def test_gate_decision_rule():
    rows = lambda vals: [{"train_row_id": i, "mean_loglik": v} for i, v in enumerate(vals)]
    d = kf.gate_decision({"S1": rows([-2, -2]), "S4R": rows([-1.5, -2.1])}, n_boot=100)
    assert d["gate"]["passed"] and d["paired"]["S4R_minus_S1"]["wins"] == 1
    d = kf.gate_decision({"S1": rows([-2, -2]), "S4R": rows([-2.5, -2])}, n_boot=100)
    assert not d["gate"]["passed"]


# ----------------------------------------------------------------- analysis

def _write_run(folder, setting, cond, correct, cats):
    d = folder / setting
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{cond}.jsonl").open("w") as fh:
        for i, (c, cat) in enumerate(zip(correct, cats)):
            fh.write(json.dumps({"question_id": f"q{i}", "correct": bool(c), "category": cat,
                                 "condition": cond, "setting": setting}) + "\n")


def test_analysis_checklist(tmp_path):
    from scripts import analyze_kazculture_final as an
    n = 400
    cats = [f"C{(i // 2) % 8}" for i in range(n)]
    ev = tmp_path / "ev"
    ev.mkdir()
    test_file = tmp_path / "t.jsonl"
    test_file.write_text("x")
    (ev / "settings.json").write_text(json.dumps({"n_questions": n,
                                                 "test_file_sha256": kf.file_sha256(test_file)}))
    base = [i % 2 == 0 for i in range(n)]
    better = [b or i % 5 == 1 for i, b in enumerate(base)]
    _write_run(ev, "closed_book", "S0", [i % 3 == 0 for i in range(n)], cats)
    for cond in ("S1", "CTRL-LEN", "CTRL-SHUF", "S1+L", "S4R-noabs"):
        _write_run(ev, "strict", cond, base, cats)
    _write_run(ev, "strict", "S4R", better, cats)
    _write_run(ev, "full", "S1", base[:-1], cats)  # incomplete -> pending, never analyzed
    olmo = tmp_path / "olmo"
    _write_run(olmo, "strict", "S1", base, cats)
    _write_run(olmo, "strict", "S4R", better, cats)
    _write_run(olmo, "closed_book", "S0", base, cats)
    nd = tmp_path / "nd.json"
    nd.write_text(json.dumps({"flagged_question_ids": ["q1", "q2"], "criteria": {}}))
    out = tmp_path / "an"
    assert an.main(["--eval_dir", str(ev), "--olmo_dir", str(olmo), "--test_file",
                    str(test_file), "--out", str(out), "--n_boot", "500",
                    "--near_dedup_list", str(nd)]) == 0
    rep = json.loads((out / "final_analysis.json").read_text())
    assert rep["primary"]["delta_pp"] == pytest.approx(10.0)
    assert rep["controls"]["strict/S4R vs strict/CTRL-LEN"]["holm_family_size"] == 2
    assert rep["secondary"]["full/S4R vs full/S1"]["status"] == "pending"
    assert rep["secondary"]["strict/S1+L vs strict/S1"]["holm_family_size"] == 2
    ck = rep["checklist"]
    assert ck["1_primary_significant_and_ge_2pp"]["status"] == "met"
    assert ck["2_beats_ctrl_len_and_ctrl_shuf"]["status"] == "met"
    assert ck["3_test_touched_once_tuning_on_train"]["status"].startswith("pending")
    assert ck["3_test_touched_once_tuning_on_train"]["test_file_hash_matches_eval"]
    assert ck["4_frame_labels_human_validated"]["status"] == "pending"
    assert ck["5_not_concentrated_and_replicates"]["status"] == "met"
    assert ck["5_not_concentrated_and_replicates"]["categories_n_ge_10"] == 8
    assert rep["near_dedup_subset"]["kept_questions"] == n - 2
    assert (out / "final_analysis.md").read_text().startswith("# KazCulture final analysis")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"passed": False}))
    assert an.frame_audit_status(audit)["status"] == "not met"


# ----------------------------------------------------------------- near-dup audit

def test_char_ngram_containment():
    from scripts.audit_kazculture_near_duplicates import char_ngrams, max_containment
    assert char_ngrams("Abc  DEFg", 5) == {"abc d", "bc de", "c def", " defg"}
    test = ["ақ киіз үй дәстүрі", "мүлде басқа мәтін"]
    train = ["бұл ақ киіз үй дәстүрі туралы ұзын мәтін", "домбыра"]
    best, arg = max_containment(test, train, 5)
    assert best[0] == pytest.approx(1.0) and arg[0] == 0
    assert best[1] < 0.5


def test_near_duplicate_audit_end_to_end(tmp_path, monkeypatch):
    import numpy as np
    from scripts import audit_kazculture_near_duplicates as audit
    test_file, rows = make_test_file(
        tmp_path, n=4, desc=lambda i: "shared superset text" if i == 0 else f"unique{i}" * 3)
    train = [unit(0, "this is the shared superset text plus more"), unit(1, "unrelated")]

    class Ix(FakeIndex):
        emb = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        def _load_model(self):
            return types.SimpleNamespace(encode=lambda texts, **k: np.asarray(
                [[0.96, 0.28] if "shared" in t else [0.5, 0.5] for t in texts]))

    monkeypatch.setattr(audit, "make_index", lambda p: Ix(train))
    out = tmp_path / "nd"
    assert audit.main(["--test_file", str(test_file), "--index_strict", str(tmp_path),
                       "--out", str(out), "--expected_size", "4"]) == 0
    rep = json.loads((out / "near_duplicate_audit.json").read_text())
    assert rep["counts"]["cosine_ge_0.95"]["passages"] == 1
    assert rep["counts"]["containment_ge_0.8"]["passages"] == 1
    lst = json.loads((out / "near_dedup_list.json").read_text())
    assert len(lst["flagged_question_ids"]) == 1
