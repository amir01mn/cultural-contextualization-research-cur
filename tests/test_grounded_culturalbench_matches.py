import importlib.util
from pathlib import Path

from src.schemas import MCQQuestion

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "13_discover_grounded_culturalbench_matches.py"
SPEC = importlib.util.spec_from_file_location("grounded_match_script", MODULE_PATH)
gm = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gm)


def test_easy_gold_target_preserves_question_qualifier():
    q = MCQQuestion(
        question_id="easy_x", group_id="easy_x", variant="easy",
        question="What practice is considered beneficial for health?",
        cultural_group="Spain",
        options={"A": "Taking a nap", "B": "Running"},
        answer_key="A",
    )
    target = gm.gold_target(q)
    assert "beneficial for health" in target
    assert "Taking a nap" in target
    assert "including all qualifiers" in target


def test_hard_false_target_requires_direct_refutation():
    q = MCQQuestion(
        question_id="hard_x", group_id="hard_x", variant="hard",
        question="What is common?", cultural_group="South Korea",
        option_text="Everyone splits the bill evenly.", answer_bool=False,
    )
    target = gm.gold_target(q)
    assert "Gold judgment: FALSE" in target
    assert "directly contradict/refute" in target
    assert "absence is insufficient" in target


def test_validate_verdict_accepts_only_exact_source_quotes():
    source = "Japanese. in homes. remove shoes as a sign of respect and cleanliness."
    verdict = {
        "label": "MATCH",
        "element_checks": [
            {
                "element": "Shoes are removed in homes for respect and cleanliness.",
                "supported": True,
                "quote": "remove shoes as a sign of respect and cleanliness",
            }
        ],
        "rationale": "direct",
        "confidence": 0.95,
    }
    out = gm.validate_verdict(verdict, source)
    assert out["accepted"] is True
    assert out["validation_problems"] == []


def test_validate_verdict_rejects_missing_element_even_with_related_quote():
    source = "Spanish. daily life. take regular naps as a cultural norm."
    verdict = {
        "label": "MATCH",
        "element_checks": [
            {
                "element": "Taking naps is common.",
                "supported": True,
                "quote": "take regular naps as a cultural norm",
            },
            {
                "element": "Naps are considered beneficial for health.",
                "supported": False,
                "quote": "",
            },
        ],
        "rationale": "health qualifier missing",
        "confidence": 0.9,
    }
    out = gm.validate_verdict(verdict, source)
    assert out["accepted"] is False
    assert "element 2 unsupported" in out["validation_problems"]


def test_validate_verdict_rejects_hallucinated_quote():
    source = "Dutch. engage in going Dutch."
    verdict = {
        "label": "MATCH",
        "element_checks": [
            {
                "element": "Each person pays for their own meal.",
                "supported": True,
                "quote": "each person pays for their own meal",
            }
        ],
        "rationale": "definition supplied from outside knowledge",
        "confidence": 0.9,
    }
    out = gm.validate_verdict(verdict, source)
    assert out["accepted"] is False
    assert "quote not found" in out["validation_problems"][0]
