"""Eval v2 / Evaluation 2 (NormAd): parser, metrics, prompts, stage (fake backend)."""

import json
import sys
from pathlib import Path

import pytest

from src.core import config
from src.evaluation import normad_eval as ne
from src.evaluation.mcq_systems import RetrievalContext
from src.loaders.normad_loader import country_to_culture, normalize_label, normalize_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _story(i, gold="yes", country="egypt", covered=True, dbin=">=100"):
    return {"id": f"normad_{i}", "country": country, "country_display": country.title(),
            "culture": country, "value": "Respect for elders.",
            "rule_of_thumb": "Greet everyone before sitting down.",
            "story": f"Sarah greeted everyone ({i}). Is what Sarah did socially acceptable?",
            "gold": gold, "covered": covered, "descriptor_bin": dbin}


class FakeBackend:
    def __init__(self, answer_fn):
        self.answer_fn = answer_fn
        self.calls = []

    def generate(self, prompts, system_prompt=None, *, temperature=None, max_tokens=None):
        self.calls.append({"n": len(prompts), "max_tokens": max_tokens, "system": system_prompt})
        return [self.answer_fn(p) for p in prompts]


# --- loader ---------------------------------------------------------------

def test_label_and_country_normalization():
    assert normalize_label("neutral") == "neither"
    assert normalize_label("Yes") == "yes"
    assert normalize_label("maybe") is None
    assert country_to_culture("united_states_of_america") == "united states"
    assert country_to_culture("türkiye") == "turkey"
    assert country_to_culture("south_korea") == country_to_culture("korea")
    row = normalize_row({"ID": 3, "Country": "egypt", "Story": "s", "Gold Label": "neutral",
                         "Rule-of-Thumb": "r", "Value": "v"}, 0)
    assert row["id"] == "normad_3" and row["gold"] == "neither" and row["culture"] == "egypt"


# --- parser ---------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Reasoning.\nANSWER: Yes", "yes"),
    ("ANSWER: <No>", "no"),
    ("ANSWER: **Neither**", "neither"),
    ("answer - neutral", "neither"),
    ("First ANSWER: Yes\nOn reflection...\nANSWER: No", "no"),
    ("No, this is rude.", "no"),
    ("Yes", "yes"),
    ("It depends.\nThe answer is neither.", "neither"),
    ("Could be yes or no.", None),
    ("", None),
    (None, None),
])
def test_parse_normad_answer(text, expected):
    assert ne.parse_normad_answer(text) == expected


# --- metrics --------------------------------------------------------------

def test_macro_f1_and_accuracy():
    golds = ["yes", "no", "neither", "yes"]
    assert ne.macro_f1(golds, golds) == 1.0
    preds = ["yes", "yes", None, "yes"]
    # yes: tp=2 fp=1 fn=0 -> 0.8; no: 0; neither: 0
    assert ne.macro_f1(golds, preds) == pytest.approx(0.8 / 3)
    assert ne.accuracy([1, 0, 1, 1]) == 0.75
    assert ne.accuracy([]) == 0.0


def test_gap_closed():
    assert ne.gap_closed(0.6, 0.5, 0.7) == pytest.approx(0.5)
    assert ne.gap_closed(0.7, 0.5, 0.7) == pytest.approx(1.0)
    assert ne.gap_closed(0.4, 0.5, 0.7) == pytest.approx(-0.5)
    assert ne.gap_closed(0.6, 0.5, 0.5) is None


def test_holm():
    adj = ne.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)  # monotone


def test_paired_bootstrap_delta():
    r = ne.paired_bootstrap_delta([1] * 50 + [0] * 50, [0] * 100, n_boot=500)
    assert r["delta"] == 0.5 and r["ci_low"] > 0 and r["p_boot"] < 0.01


def test_primary_family():
    pairs = ne.primary_comparisons(ne.ALL_CONDITIONS)
    assert ("S4_abstraction_refinement_kg", "S1_raw_text_rag") in pairs
    assert ("S4g", "S4_abstraction_refinement_kg") in pairs
    assert ("S1g", ne.C1) in pairs
    assert all(a not in (ne.C2, ne.C3) for a, _ in pairs)


# --- prompts --------------------------------------------------------------

def test_condition_prompts_differ_only_in_context():
    s = _story(0)
    c1 = ne.build_user_prompt(ne.C1, s)
    c2 = ne.build_user_prompt(ne.C2, s)
    c3 = ne.build_user_prompt(ne.C3, s)
    ev = ne.build_user_prompt("S1_raw_text_rag", s, "Header:\n[1] passage\n\n")
    for p in (c1, c2, c3, ev):
        assert "Country: Egypt" in p and s["story"] in p and "ANSWER: <Yes|No|Neither>" in p
    assert "Value: Respect for elders." in c2 and "Rule-of-Thumb" not in c2
    assert "Rule-of-Thumb: Greet everyone" in c3 and "Value:" not in c3
    assert "Value:" not in c1 and "Rule-of-Thumb" not in c1
    assert "[1] passage" in ev and "Rule-of-Thumb" not in ev


def test_story_question_query_is_story_plus_country():
    q = ne.story_question(_story(1))
    assert q.question.startswith("Sarah greeted") and q.cultural_group == "egypt"


# --- stage ----------------------------------------------------------------

def test_run_normad_end_to_end(tmp_path):
    ctx = RetrievalContext(passages=[
        {"passage_id": "p1", "cultural_group": "Egyptian", "text": "Egyptians greet everyone present."},
        {"passage_id": "p2", "cultural_group": "Japanese", "text": "Japanese people bow when greeting."},
    ])

    def answer(prompt):
        # C3 always right (gold yes), everything else says "No".
        return "ANSWER: Yes" if "Rule-of-Thumb:" in prompt else "Reasoning\nANSWER: No"

    backend = FakeBackend(answer)
    test = [_story(i, gold="yes", covered=i % 2 == 0) for i in range(6)]
    dev = [_story(100 + i) for i in range(3)]
    old = (config.RETRIEVAL_STRICT_CULTURE, config.RETRIEVAL_MIN_SCORE)
    conds = [ne.C1, ne.C2, ne.C3, "S1_raw_text_rag", "S1g"]
    summary = ne.run_normad(backend, ctx, test, tmp_path, dev_stories=dev, conditions=conds)
    assert (config.RETRIEVAL_STRICT_CULTURE, config.RETRIEVAL_MIN_SCORE) == old
    allm = summary["subsets"]["all"]["conditions"]
    assert allm[ne.C3]["accuracy"] == 1.0 and allm[ne.C1]["accuracy"] == 0.0
    assert allm[ne.C3]["gap_closed"] == 1.0 and allm[ne.C2]["gap_closed"] == 0.0
    assert summary["subsets"]["covered"]["conditions"][ne.C1]["n"] == 3
    assert summary["taus"]["S1g"] in ne.TAU_GRID
    # 4 taus on dev for S1g + one batch per test condition; explicit max_tokens always
    assert len(backend.calls) == 4 + len(conds)
    assert all(c["max_tokens"] == ne.MAX_TOKENS for c in backend.calls)
    rows = [json.loads(l) for l in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(rows) == len(test) * len(conds)
    s1 = [r for r in rows if r["condition"] == "S1_raw_text_rag"]
    assert all({"retrieved_ids", "evidence", "raw_output"} <= set(r) for r in s1)
    assert (tmp_path / "summary.md").read_text().startswith("# NormAd")
    assert "S1g vs C1_country" in summary["subsets"]["all"]["tests_primary_holm"]


def test_gated_needs_tau_or_dev(tmp_path):
    with pytest.raises(ValueError):
        ne.run_normad(FakeBackend(lambda p: "ANSWER: Yes"), None, [_story(0)], tmp_path,
                      conditions=["S4g"])


def test_dev_split_is_stratified_and_disjoint():
    from prep_normad_v2 import stratified_dev_split

    stories = [_story(i, country=c) for i, c in enumerate(["a"] * 60 + ["b"] * 30 + ["c"] * 10)]
    dev, test = stratified_dev_split(stories, 10, 42)
    assert len(dev) == 10 and len(test) == 90
    assert not ({s["id"] for s in dev} & {s["id"] for s in test})
    counts = {c: sum(s["country"] == c for s in dev) for c in "abc"}
    assert counts == {"a": 6, "b": 3, "c": 1}
    dev2, _ = stratified_dev_split(stories, 10, 42)
    assert [s["id"] for s in dev] == [s["id"] for s in dev2]
