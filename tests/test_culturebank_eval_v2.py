"""Eval v2 / Evaluation 1 (CultureBank grounded): gold mapping, masking,
leakage guard, parsing, and the stage functions on tiny synthetic data."""

import csv
import json

import pytest

from src.core import config
from src.core.schemas import CulturalPathway, CulturalTriple, PathwayNodeRef
from src.core.utils import stable_id
from src.evaluation import culturebank_grounded as cg
from src.evaluation.mcq_systems import SYSTEM_S1, SYSTEM_S2, RetrievalContext, run_system
from src.loaders.culturebank_eval_loader import instance_id_for_row, rows_with_instance_ids
from src.retrieval.pathway_retriever import PathwayRetriever


# ---------------------------------------------------------------- fixtures

def _item(i, culture, beh, topic="Food"):
    iid = instance_id_for_row(i)
    return {
        "qid": f"cbq_{iid}", "instance_id": iid, "row_index": i, "split": "test",
        "cultural_group": culture, "culture_key": culture.lower(), "topic": topic,
        "context": "at dinner", "eval_question": f"How do people {beh} at dinner in {culture}?",
        "eval_whole_desc": f"In {culture}, people {beh} during dinner as a sign of respect.",
        "gold": beh, "distractors": ["wear hats indoors", "sing loudly", "arrive very late"],
        "heldout": True, "open_ended": True,
    }


@pytest.fixture
def tiny():
    specs = [("Japanese", "remove shoes before entering"), ("Indian", "eat with the right hand"),
             ("Brazilian", "greet with a cheek kiss"), ("Chinese", "offer tea to guests")]
    items = [_item(i, c, b) for i, (c, b) in enumerate(specs)]
    passages = [{"passage_id": it["instance_id"], "instance_id": it["instance_id"],
                 "cultural_group": it["cultural_group"],
                 "text": f"{it['cultural_group']}. at dinner. people {it['gold']}."} for it in items]
    triples = [CulturalTriple(triple_id=f"t_{k}", subject="people", predicate=it["gold"].split()[0],
                              object=it["gold"], instance_id=it["instance_id"],
                              raw_fields={"cultural group": it["cultural_group"]})
               for k, it in enumerate(items)]
    return items, passages, triples


class _Backend:
    name = model = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, prompts, system_prompt=None, *, temperature=None, max_tokens=None):
        from src.llm import prompts as P

        self.calls.append((system_prompt, max_tokens, len(prompts)))
        if system_prompt == P.CB_PARAPHRASE_SYSTEM:
            return ["1: one\n2: \n3: three\n4: four" for _ in prompts]
        if system_prompt == P.CB_OPEN_SYSTEM:
            return ["short answer" for _ in prompts]
        return ["ANSWER: A" for _ in prompts]


# ---------------------------------------------------------------- gold mapping

def test_instance_id_matches_loader_stable_id():
    assert instance_id_for_row(0) == stable_id("culturebank", 0, prefix="cb_")
    assert instance_id_for_row(17).startswith("cb_")


def test_rows_with_instance_ids_counts_across_splits(tmp_path):
    for split, n in (("tiktok", 2), ("reddit", 3)):
        d = tmp_path / split
        d.mkdir()
        with (d / f"culturebank_{split}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cultural group", "actor_behavior", "agreement",
                                              "eval_question"])
            w.writeheader()
            for k in range(n):
                w.writerow({"cultural group": f"G{split}{k}", "actor_behavior": "b\nx",
                            "agreement": "0.7", "eval_question": "q?"})
    rows = rows_with_instance_ids([tmp_path / "tiktok" / "culturebank_tiktok.csv",
                                   tmp_path / "reddit" / "culturebank_reddit.csv"])
    assert [r["row_index"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[2]["instance_id"] == instance_id_for_row(2) and rows[2]["split"] == "reddit"
    assert rows[0]["actor_behavior"] == "b x" and rows[0]["agreement_f"] == 0.7


# ---------------------------------------------------------------- masking

def test_mask_excludes_passages_and_triples_and_unset_is_unchanged(tiny):
    items, passages, triples = tiny
    ctx = RetrievalContext(passages=passages, triples=triples)
    q = cg.build_mcq(items[0], 42)
    base_ids = cg.compute_evidence("S1", [q], ctx, 8)[0][1]
    assert items[0]["instance_id"] in base_ids
    with cg.masked(ctx, [items[0]["instance_id"]]):
        ids = cg.compute_evidence("S1", [q], ctx, 8)[0][1]
        tids = cg.compute_evidence("S2", [q], ctx, 8)[0][1]
    assert items[0]["instance_id"] not in ids
    assert "t_0" not in tids
    assert ctx.masked_instance_ids is None
    assert cg.compute_evidence("S1", [q], ctx, 8)[0][1] == base_ids


def test_pathway_masked_if_any_triple_masked(tmp_path):
    import numpy as np

    pws = [CulturalPathway(pathway_id=f"pw_{k}", culture="Japanese", triple_ids=tids,
                           nodes=[PathwayNodeRef(entity_id="e", label=f"shoe custom {k}")])
           for k, tids in enumerate([["t_a", "t_b"], ["t_c"]])]
    (tmp_path / "pathways.jsonl").write_text("\n".join(p.model_dump_json() for p in pws))
    np.save(tmp_path / "embeddings.npy", np.zeros((2, 3)))
    pr = PathwayRetriever(tmp_path)
    mask = pr.mask_for_instance_ids({"cb_x"}, {"t_a": "cb_y", "t_b": "cb_x", "t_c": "cb_z"})
    assert mask.tolist() == [True, False]


# ---------------------------------------------------------------- leakage + parsing

def test_leak_counter(tiny):
    items = tiny[0][:2]
    blocks = ["[1] " + items[0]["eval_whole_desc"].upper(), "[1] unrelated evidence"]
    assert cg.leak_count(blocks, items) == 1
    assert cg.leak_count(["", ""], items) == 0


def test_parse_paraphrases_and_fallback():
    assert cg.parse_paraphrases("1: a\n2) b\n**3:** c\nnoise", 4) == ["a", "b", "c", ""]
    it = _item(0, "Japanese", "remove shoes")
    opts = cg.item_options(it, {it["qid"]: ["x", "", "y", "z"]})
    assert opts == ["x", "wear hats indoors", "y", "z"]


def test_build_mcq_seeds_permute_and_keep_gold():
    it = _item(0, "Japanese", "remove shoes")
    keys = set()
    for seed in (42, 43, 44, 45, 46):
        q = cg.build_mcq(it, seed)
        assert q.options[q.answer_key] == "remove shoes"
        assert "Japanese" in q.question and it["eval_question"] in q.question
        keys.add(q.answer_key)
    assert len(keys) > 1
    assert cg.build_mcq(it, 42).options == cg.build_mcq(it, 42).options


def test_gated_config_restores():
    old = (config.RETRIEVAL_STRICT_CULTURE, config.RETRIEVAL_MIN_SCORE)
    with cg.gated_config("S4g", 0.35):
        assert config.RETRIEVAL_STRICT_CULTURE is True and config.RETRIEVAL_MIN_SCORE == 0.35
    assert (config.RETRIEVAL_STRICT_CULTURE, config.RETRIEVAL_MIN_SCORE) == old


def test_run_system_precomputed_evidence_matches_default(tiny, fake_backend_factory):
    items, passages, triples = tiny
    ctx = RetrievalContext(passages=passages, triples=triples)
    qs = [cg.build_mcq(it, 42) for it in items]
    b1 = fake_backend_factory(lambda p, s: "ANSWER: B")
    b2 = fake_backend_factory(lambda p, s: "ANSWER: B")
    a = run_system(SYSTEM_S2, qs, b1, ctx, top_k=3)
    ev = cg.compute_evidence("S2", qs, ctx, 3)
    b = run_system(SYSTEM_S2, qs, b2, ctx, top_k=3, evidence=ev)
    assert b1.calls == b2.calls
    assert [x.retrieved_ids for x in a] == [x.retrieved_ids for x in b]


# ---------------------------------------------------------------- stages end-to-end

def test_stages_end_to_end(tmp_path, tiny):
    items, passages, triples = tiny
    ctx = RetrievalContext(passages=passages, triples=triples)
    data = {"dev": [dict(it, split="dev") for it in items[:2]], "test": items,
            "masks": {"near": [items[0]["instance_id"]], "far": [items[0]["instance_id"]]}}
    be = _Backend()
    cache, leaks = {}, {}
    para = cg.run_paraphrase(be, ctx, data, tmp_path, cache_path=tmp_path / "p.jsonl")
    assert para["options_fallback"] == len(items) + 2
    paraphrases = cg.load_paraphrases(tmp_path / "p.jsonl")
    common = dict(top_k=3, cache=cache, leak_log=leaks, paraphrases=paraphrases)
    og = cg.run_open_gen(be, ctx, data, tmp_path, systems=["S0", "S1", "S4g"], **common)
    assert (tmp_path / "open_answers.DONE").exists()
    rows = [json.loads(line) for line in (tmp_path / "open_answers.jsonl").open()]
    assert set(rows[0]) == {"qid", "system", "cultural_group", "question", "answer", "gold_desc"}
    assert og["n_rows"] == 3 * len(items)
    mcq = cg.run_mcq(be, ctx, data, tmp_path, systems=["S0", "S1", "S1g"], **common)
    assert set(mcq["tau"]) == {"S1g"} and mcq["tau"]["S1g"] in cg.TAU_GRID
    held = cg.run_heldout(be, ctx, data, tmp_path, systems=["S0", "S1", "S1g"], **common)
    assert held["mask_violations"] == {}
    rm = cg.run_retrieval_metrics(be, ctx, data, tmp_path, top_k=3, cache=cache,
                                  leak_log=leaks, systems=["S1", "S2"])
    assert rm["test"]["S1"]["recall@8"] == 1.0 and rm["test"]["S1"]["mrr"] == 1.0
    assert sum(leaks.values()) == 0
    assert all(mt is not None for sp, mt, _n in be.calls)  # explicit max_tokens always
    summ = cg.write_summary(tmp_path, run_meta={"model": "fake"}, n_boot=50)
    assert "S1__vs__S0" in summ["comparisons"]
    assert (tmp_path / "summary.md").exists()


def test_holm_monotone():
    adj = cg.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["c"] == 0.06 and adj["b"] == 0.06
