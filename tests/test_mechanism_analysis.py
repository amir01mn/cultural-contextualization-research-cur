from src.mechanism_analysis import summarize_mechanisms
from src.mcq_systems import SYSTEM_S2, SYSTEM_S4
from src.schemas import MCQQuestion, MCQSystemAnswer, MCQVariant


def _easy(qid, culture="Japan"):
    return MCQQuestion(
        question_id=qid, variant=MCQVariant.EASY, group_id=qid,
        question="q", cultural_group=culture,
        options={"A": "a", "B": "b"}, answer_key="B",
    )


def _ans(system, correct, *, completeness="FULL", shared=False):
    return MCQSystemAnswer(
        system=system,
        correct=correct,
        completeness=completeness,
        completeness_score=1.0 if completeness == "FULL" else 0.33,
        shared_node_available=shared,
    )


def test_mechanism_summary_by_completeness():
    questions = [_easy("e1"), _easy("e2")]
    answers = {
        SYSTEM_S4: [
            _ans(SYSTEM_S4, True, completeness="FULL"),
            _ans(SYSTEM_S4, False, completeness="NONE"),
        ],
        SYSTEM_S2: [
            _ans(SYSTEM_S2, True, completeness="SOCIAL"),
            _ans(SYSTEM_S2, True, completeness="SURFACE"),
        ],
    }
    mech = summarize_mechanisms(questions, answers)
    s4_full = mech["by_completeness"][SYSTEM_S4]["FULL"]
    assert s4_full["n"] == 1
    assert s4_full["accuracy"] == 1.0
    s2_social = mech["by_completeness"][SYSTEM_S2]["SOCIAL"]
    assert s2_social["n"] == 1
