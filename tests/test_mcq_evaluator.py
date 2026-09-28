from src.evaluation import mcq_evaluator
from src.evaluation.mcq_evaluator import compute_units, mcnemar, paired_bootstrap, summarize
from src.core.schemas import MCQQuestion, MCQSystemAnswer, MCQVariant


def _easy(qid, key="B"):
    return MCQQuestion(question_id=qid, variant=MCQVariant.EASY, group_id=qid,
                       question="q", cultural_group="X",
                       options={"A": "a", "B": "b"}, answer_key=key)


def _hard(qid, gid, gold):
    return MCQQuestion(question_id=qid, variant=MCQVariant.HARD, group_id=gid,
                       question="q", cultural_group="Y", option_text="o",
                       answer_bool=gold)


def _ans(system, correct):
    return MCQSystemAnswer(system=system, correct=correct)


def test_compute_units_easy_and_hard_grouping():
    questions = [
        _easy("e1"), _easy("e2"),
        _hard("h1_0", "h1", True), _hard("h1_1", "h1", False),
    ]
    answers = {
        "S0": [_ans("S0", True), _ans("S0", False),
               _ans("S0", True), _ans("S0", True)],  # hard group: not all correct
    }
    units, correctness = compute_units(questions, answers)
    # 2 easy units + 1 hard group unit = 3.
    assert len(units) == 3
    variants = [u["variant"] for u in units]
    assert variants.count("easy") == 2 and variants.count("hard") == 1
    # Hard group correct iff all items correct -> here item2 correct=True but
    # gold differs; the answer.correct flags drive it: [True, True] -> all True.
    hard_idx = variants.index("hard")
    assert correctness["S0"][hard_idx] == 1


def test_hard_group_requires_all_items_correct():
    questions = [_hard("h1_0", "h1", True), _hard("h1_1", "h1", False)]
    answers = {"S0": [_ans("S0", True), _ans("S0", False)]}  # one item wrong
    units, correctness = compute_units(questions, answers)
    assert len(units) == 1
    assert correctness["S0"][0] == 0


def test_paired_bootstrap_ci_bounds():
    correctness = {"S0": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]}
    out = paired_bootstrap(correctness, n_boot=500, seed=1)
    s0 = out["S0"]
    assert 0.0 <= s0["ci_low"] <= s0["accuracy"] <= s0["ci_high"] <= 1.0


def test_mcnemar_detects_difference():
    # A correct everywhere, B wrong everywhere -> highly discordant.
    a = [1] * 20
    b = [0] * 20
    res = mcnemar(a, b)
    assert res["n_discordant"] == 20
    assert res["delta_accuracy"] == 1.0
    assert res["p_value"] < 0.05


def test_mcnemar_no_discordant():
    res = mcnemar([1, 0, 1], [1, 0, 1])
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0


def test_summarize_structure():
    questions = [_easy("e1"), _easy("e2"), _hard("h1_0", "h1", True)]
    answers = {
        "S0_no_kg": [_ans("S0_no_kg", False), _ans("S0_no_kg", False), _ans("S0_no_kg", True)],
        "S3_layered_kg": [_ans("S3_layered_kg", True), _ans("S3_layered_kg", True),
                          _ans("S3_layered_kg", True)],
    }
    summary, units, correctness = summarize(questions, answers, n_boot=200, seed=3)
    assert "accuracy" in summary and "combined" in summary["accuracy"]
    assert summary["num_units_total"] == 3
    assert "mcnemar_combined" in summary
    assert "per_culture_accuracy" in summary
    # S3 should outscore S0 here.
    acc = summary["accuracy"]["combined"]
    assert acc["S3_layered_kg"]["accuracy"] >= acc["S0_no_kg"]["accuracy"]
