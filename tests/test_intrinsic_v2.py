"""Evaluation 3 (intrinsic + judge) and the CulturalBench coverage strata."""
import json
import sys
from pathlib import Path

import pytest

from src.evaluation import intrinsic_eval as ie

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))


class FakeBackend:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, prompts, system_prompt=None, *, temperature=None, max_tokens=None):
        self.calls.append({"n": len(prompts), "max_tokens": max_tokens,
                           "temperature": temperature})
        return [self.reply(p) for p in prompts]


# ---------------------------------------------------------------- parsers

@pytest.mark.parametrize("text,expected", [
    ("Yes", True), ("no.", False), ("Answer: YES", True), ("**No**, because", False),
    ("The fact is not stated. no", False), ("true", True), ("", None), ("maybe", None),
])
def test_parse_yes_no(text, expected):
    assert ie.parse_yes_no(text) is expected


def test_parse_json_list_variants():
    assert ie.parse_json_list('["A.", "B."]') == ["A.", "B."]
    assert ie.parse_json_list('Here:\n```json\n["A.", "B.",]\n```') == ["A.", "B."]
    assert ie.parse_json_list('["A "quoted" x", "B."') is not None  # damaged, no crash
    assert ie.parse_json_list("1. First fact.\n2. Second fact.") == ["First fact.", "Second fact."]
    assert ie.parse_json_list("nothing here") is None


def test_parse_json_flags_json_and_regex_fallback():
    keys = ie.TRIPLE_AUDIT_KEYS
    out = ie.parse_json_flags('```json\n{"faithful": "yes", "layers_correct": false, '
                              '"culture_specific": "No", "stereotyping": "no"}```', keys)
    assert out == {"faithful": True, "layers_correct": False,
                   "culture_specific": False, "stereotyping": False}
    broken = 'faithful: yes, "layers_correct": "no", culture_specific = yes'
    out = ie.parse_json_flags(broken, keys)
    assert out["faithful"] is True and out["layers_correct"] is False
    assert out["culture_specific"] is True and out["stereotyping"] is None


def test_normalize_facts_dedupes_and_caps():
    facts = ["A.", "a.", "B.", "C.", "D.", "E.", "F.", "G."]
    assert ie.normalize_facts(facts) == ["A.", "B.", "C.", "D.", "E.", "F."]


# ---------------------------------------------------------------- sampling

def test_stratified_sample_equal_allocation_and_seed():
    items = [{"c": "us", "i": i} for i in range(100)] + \
            [{"c": "jp", "i": i} for i in range(3)] + [{"c": "", "i": 0}]
    s = ie.stratified_sample(items, lambda x: x["c"], 10, seed=42)
    assert len(s) == 10
    assert sum(x["c"] == "jp" for x in s) == 3  # small stratum fully used
    assert all(x["c"] for x in s)               # empty key never sampled
    assert s == ie.stratified_sample(items, lambda x: x["c"], 10, seed=42)


# ---------------------------------------------------------------- retention

def test_aggregate_retention_rates_and_unparsed():
    rows = [
        {"instance_id": "r1", "culture": "japan", "fact_idx": 0, "source": "raw_triples", "inferable": True},
        {"instance_id": "r1", "culture": "japan", "fact_idx": 1, "source": "raw_triples", "inferable": False},
        {"instance_id": "r2", "culture": "peru", "fact_idx": 0, "source": "raw_triples", "inferable": True},
        {"instance_id": "r2", "culture": "peru", "fact_idx": 1, "source": "raw_triples", "inferable": None},
        {"instance_id": "r1", "culture": "japan", "fact_idx": 0, "source": "pathways", "inferable": False},
    ]
    agg = ie.aggregate_retention(rows)
    raw = agg["overall"]["raw_triples"]
    assert raw["rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert raw["n_unparsed"] == 1
    assert raw["record_macro"] == pytest.approx((0.5 + 1.0) / 2)
    assert agg["overall"]["pathways"]["rate"] == 0.0
    assert agg["per_culture"]["japan"]["raw_triples"]["rate"] == 0.5


def _record(pathways=True):
    return {"instance_id": "r1", "culture": "japan", "cultural_group": "Japanese",
            "eval_whole_desc": "In Japan people bow. Bowing shows respect.",
            "raw_triples": [{"subject": "people", "predicate": "bow", "object": "greeting"}],
            "refined_triples": [],
            "pathways": [{"culture": "japan", "predicates": ["show", "do"],
                          "nodes": [{"label": "respect", "layer": "LAYER_3_DEEP_VALUE"},
                                    {"label": "greeting norm", "layer": "LAYER_2_SOCIAL"},
                                    {"label": "bowing", "layer": "LAYER_1_SURFACE"}]}]
            if pathways else []}


def test_run_facts_and_retention_end_to_end(tmp_path):
    backend = FakeBackend(lambda p: '["Japanese bow.", "Bowing shows respect.", "x"]'
                          if "atomic facts" in p else ("yes" if "bowing" in p else "no"))
    fs = ie.run_facts(backend, [_record()], tmp_path)
    assert fs["n_facts_total"] == 3 and fs["n_parse_failed"] == 0
    facts = {json.loads(l)["instance_id"]: json.loads(l)["facts"]
             for l in (tmp_path / "facts.jsonl").read_text().splitlines()}
    rs = ie.run_retention(backend, [_record()], facts, tmp_path)
    # refined is empty -> no prompts, counted as not retained
    assert rs["n_empty_evidence_facts"] == {"refined_triples": 3}
    assert rs["overall"]["refined_triples"]["rate"] == 0.0
    assert rs["overall"]["pathways"]["rate"] == 1.0   # pathway text contains "bowing"
    assert rs["n_judge_prompts"] == 6
    assert all(c["max_tokens"] and c["temperature"] == 0.0 for c in backend.calls)


def test_format_pathway_orders_layers():
    s = ie.format_pathway(_record()["pathways"][0])
    assert s.startswith("[L3] respect --show--> [L2] greeting norm --do--> [L1] bowing")


def test_triple_and_pathway_audit(tmp_path):
    backend = FakeBackend(lambda p: '{"faithful": "yes", "layers_correct": "no", '
                          '"culture_specific": "yes", "stereotyping": "no", '
                          '"coherent": "yes", "culture_consistent": "no", "correct": "garbled"}')
    t = {"triple_id": "t1", "culture": "japan", "subject": "a", "predicate": "b", "object": "c",
         "subject_layer": "LAYER_1_SURFACE", "object_layer": "LAYER_2_SOCIAL",
         "source_text": "x", "generated": True}
    s = ie.run_triple_audit(backend, [t], tmp_path)
    assert s["overall"]["faithful"]["rate"] == 1.0
    assert s["overall"]["layers_correct"]["rate"] == 0.0
    assert s["by_generated"]["True"]["culture_specific"]["rate"] == 1.0
    pw = dict(_record()["pathways"][0], pathway_id="p1", sources=["src"])
    s = ie.run_pathway_audit(backend, [pw], tmp_path)
    assert s["overall"]["coherent"]["rate"] == 1.0
    assert s["overall"]["correct"]["n_unparsed"] == 1


# ---------------------------------------------------------------- open judge

def test_holm_and_resolve_system():
    adj = ie.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == {"a": 0.03, "c": 0.06, "b": 0.06}
    names = ["S0_no_kg", "S1_raw_text_rag", "S4_abstraction_refinement_kg", "S4g"]
    assert ie.resolve_system("S4", names) == "S4_abstraction_refinement_kg"
    assert ie.resolve_system("S4g", names) == "S4g"
    assert ie.resolve_system("S3", names) is None


def test_run_open_judge_pairs_and_writes(tmp_path):
    answers = []
    for q in range(6):
        for sys_name in ("S0_no_kg", "S1_raw_text_rag", "S4_abstraction_refinement_kg", "S4g"):
            answers.append({"qid": q, "system": sys_name, "cultural_group": "Japan",
                            "question": "?", "answer": f"{sys_name} good" if q < 4 else "bad",
                            "gold_desc": "desc"})
    backend = FakeBackend(lambda p: "Yes" if "S4" in p and "good" in p else "No")
    s = ie.run_open_judge(backend, answers, tmp_path, n_boot=100)
    assert s["per_system"]["S4g"]["mean_entailment"] == pytest.approx(4 / 6, abs=1e-4)
    assert s["per_system"]["S0_no_kg"]["mean_entailment"] == 0.0
    cmp = s["comparisons"]["S4 vs S0"]
    assert cmp["b_a_only"] == 4 and cmp["n_paired"] == 6 and "p_holm" in cmp
    assert "S3 vs S0" not in s["comparisons"]
    assert (tmp_path / "open_judge_summary.md").exists()
    limited = ie.run_open_judge(backend, answers, tmp_path, limit=2, n_boot=50)
    assert limited["n_answers"] == 8  # 2 qids x 4 systems


# ---------------------------------------------------------------- strata

def test_strata_assignment_boundaries():
    import culturalbench_strata_v2 as cs

    assert [cs.stratum_for(n) for n in (0, 19, 20, 99, 100)] == \
        ["<20", "<20", "20-99", "20-99", ">=100"]
    counts = cs.descriptor_counts(["Japanese"] * 25 + ["Americans"] * 100 + ["Peruvian"])
    strata = cs.assign_strata([{"cultural_group": "Japan"}, {"cultural_group": "United States"},
                               {"cultural_group": "Peru"}, {"cultural_group": "Atlantis"}], counts)
    assert strata["Japan"]["stratum"] == "20-99"
    assert strata["United States"]["stratum"] == ">=100"
    assert strata["Peru"]["n_descriptors"] == 1
    assert strata["Atlantis"]["stratum"] == "<20"


def test_strata_hard_uses_group_metric():
    import culturalbench_strata_v2 as cs

    def item(qid, gid, variant, correct):
        return {"question_id": qid, "group_id": gid, "variant": variant,
                "cultural_group": "Japan",
                "answers": {"S0_no_kg": {"correct": correct}, "S4": {"correct": True}}}

    results = [item("e0", "e0", "easy", True),
               item("h0_1", "h0", "hard", True), item("h0_2", "h0", "hard", False),
               item("h1_1", "h1", "hard", True)]
    units, corr = cs.units_from_results(results)
    assert len(units) == 3
    table = cs.stratified_table(units, corr, {"Japan": {"stratum": "20-99"}}, n_boot=50)
    assert table["hard|20-99"]["systems"]["S0_no_kg"]["accuracy"] == 0.5
    assert table["hard|20-99"]["systems"]["S4"]["accuracy"] == 1.0
    assert table["easy|all"]["n_units"] == 1
