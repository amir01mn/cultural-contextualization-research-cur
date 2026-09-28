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


def test_stage2_prompt_reviews_prior_elements_without_grammatical_over_split():
    q = MCQQuestion(
        question_id="easy_japan",
        group_id="easy_japan",
        variant="easy",
        question="In Japanese culture, why is it customary to remove shoes before entering a home?",
        cultural_group="Japan",
        options={"A": "Show respect and cleanliness", "B": "Other"},
        answer_key="A",
    )
    rec = {
        "instance_id": "jp1",
        "cultural_group": "Japanese",
        "topic": "Social Norms and Etiquette",
        "context": "in homes",
        "actor_behavior": "remove shoes before entering the home as a sign of respect and cleanliness",
        "recipient_behavior": None,
        "source_text": "Japanese. in homes. remove shoes before entering the home as a sign of respect and cleanliness.",
    }
    prior = {
        "label": "MATCH",
        "element_checks": [
            {
                "element": "customary to remove shoes before entering a home",
                "supported": True,
                "quote": "remove shoes before entering the home",
            },
            {
                "element": "to show respect and cleanliness",
                "supported": True,
                "quote": "as a sign of respect and cleanliness",
            },
        ],
        "rationale": "direct",
    }

    prompt = gm.pair_prompt(q, rec, stage2=True, prior_verdict=prior)

    assert "PREVIOUS VERIFIER ELEMENT CHECKS" in prompt
    assert "customary to remove shoes before entering a home" in prompt
    assert "as a sign of respect and cleanliness" in prompt


def test_stage2_system_allows_cultural_record_to_support_common_wording():
    assert "may support wording such as \"common\", \"customary\", or" in gm.STAGE2_SYSTEM
    assert "Do NOT create standalone elements for grammatical/question words" in gm.STAGE2_SYSTEM


def test_real_case_regression_fixture_pairing_and_expectation():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent / "fixtures"
    questions = gm.load_questions(root / "grounded_real_cases_questions.jsonl")
    records = list(gm.read_jsonl(root / "grounded_real_cases_records.jsonl"))
    expected = json.loads(
        (root / "grounded_real_cases_expected.json").read_text(encoding="utf-8")
    )

    counts = {
        q.question_id: len(gm.culture_records_for_question(q, records))
        for q in questions
    }

    assert counts == {
        "easy_31": 2,
        "easy_214": 1,
        "easy_220": 1,
        "easy_234": 1,
        "hard_5_0": 4,
    }
    assert expected["expected_supported_units"] == ["easy_214"]
    assert expected["expected_pairs"] == [
        {"question_id": "easy_214", "instance_id": "cb_cfa83c007b0d"}
    ]


def test_run_stage_retries_unparseable_output(tmp_path):
    q = MCQQuestion(
        question_id="retry_q",
        group_id="retry_q",
        variant="easy",
        question="Why remove shoes before entering a Japanese home?",
        cultural_group="Japan",
        options={"A": "Respect and cleanliness", "B": "Other"},
        answer_key="A",
    )
    rec = {
        "instance_id": "retry_rec",
        "cultural_group": "Japanese",
        "topic": "Social Norms and Etiquette",
        "context": "in homes",
        "actor_behavior": "remove shoes as a sign of respect and cleanliness",
        "recipient_behavior": None,
        "source_text": "Japanese. in homes. remove shoes as a sign of respect and cleanliness.",
    }

    class FakeBackend:
        def generate(self, prompts, system_prompt=None, **kwargs):
            return ["analysis only, no json"]

        def generate_one(self, prompt, system_prompt=None, **kwargs):
            return (
                '{"label":"MATCH","element_checks":['
                '{"element":"Japanese culture","supported":true,"quote":"Japanese."},'
                '{"element":"remove shoes","supported":true,"quote":"remove shoes"},'
                '{"element":"respect and cleanliness","supported":true,'
                '"quote":"respect and cleanliness"}],'
                '"rationale":"direct","confidence":1.0}'
            )

    state_path = tmp_path / "state.jsonl"
    state = gm.run_stage(
        stage=1,
        pairs=[(q, rec)],
        backend=FakeBackend(),
        state_path=state_path,
        batch_size=1,
    )

    row = next(iter(state.values()))
    assert row["accepted"] is True
    assert row["retry_used"] is True
    assert row["parse_error"] is None
    assert row["initial_raw_output"] == "analysis only, no json"


def test_parse_stage2_review_accepts_compact_audit():
    review = gm.parse_stage2_review(
        '{"decision":"ACCEPT","unsupported_prior_element_indices":[],'
        '"missing_material_elements":[],"rationale":"all good","confidence":1.0}'
    )
    assert review["decision"] == "ACCEPT"
    assert review["unsupported_prior_element_indices"] == []
    assert review["missing_material_elements"] == []


def test_validate_stage2_review_accepts_existing_stage1_elements():
    prior = {
        "accepted": True,
        "element_checks": [
            {
                "element": "customary shoe removal",
                "supported": True,
                "quote": "remove shoes and wear slippers",
            },
            {
                "element": "respect and cleanliness",
                "supported": True,
                "quote": "as a sign of respect and cleanliness",
            },
        ],
    }
    review = {
        "decision": "ACCEPT",
        "unsupported_prior_element_indices": [],
        "missing_material_elements": [],
        "rationale": "all material facts covered",
        "confidence": 1.0,
    }
    out = gm.validate_stage2_review(review, prior)

    assert out["accepted"] is True
    assert out["label"] == "MATCH"
    assert out["element_checks"] == prior["element_checks"]
    assert out["validation_problems"] == []


def test_validate_stage2_review_rejects_missing_material_qualifier():
    prior = {
        "accepted": True,
        "element_checks": [
            {
                "element": "Spanish people take afternoon naps",
                "supported": True,
                "quote": "take regular naps",
            }
        ],
    }
    review = {
        "decision": "REJECT",
        "unsupported_prior_element_indices": [],
        "missing_material_elements": ["beneficial for health"],
        "rationale": "health qualifier missing",
        "confidence": 0.95,
    }
    out = gm.validate_stage2_review(review, prior)

    assert out["accepted"] is False
    assert out["label"] == "NO_MATCH"
    assert "missing material elements: beneficial for health" in out["validation_problems"]


def test_stage2_schema_does_not_request_new_element_checks():
    assert "unsupported_prior_element_indices" in gm.STAGE2_SYSTEM
    assert "missing_material_elements" in gm.STAGE2_SYSTEM
    assert '"element_checks"' not in gm.STAGE2_SYSTEM.split("Return exactly:", 1)[1]


def test_validate_stage2_review_ignores_question_grammar_and_norm_words():
    prior = {
        "accepted": True,
        "element_checks": [
            {
                "element": "customary shoe removal",
                "supported": True,
                "quote": "remove shoes and wear slippers",
            }
        ],
    }
    review = {
        "decision": "ACCEPT",
        "unsupported_prior_element_indices": [],
        "missing_material_elements": ["why", "customary", "common practice"],
        "rationale": "no material omission",
        "confidence": 1.0,
    }
    out = gm.validate_stage2_review(review, prior)
    assert out["accepted"] is True
    assert out["stage2_review"]["missing_material_elements"] == []


def test_run_stage_stage2_uses_prior_element_audit(tmp_path):
    q = MCQQuestion(
        question_id="stage2_q",
        group_id="stage2_q",
        variant="easy",
        question="Why remove shoes before entering a Japanese home?",
        cultural_group="Japan",
        options={"A": "Respect and cleanliness", "B": "Other"},
        answer_key="A",
    )
    rec = {
        "instance_id": "stage2_rec",
        "cultural_group": "Japanese",
        "topic": "Social Norms and Etiquette",
        "context": "in homes",
        "actor_behavior": "remove shoes as a sign of respect and cleanliness",
        "recipient_behavior": None,
        "source_text": "Japanese. in homes. remove shoes as a sign of respect and cleanliness.",
    }
    pair_id = gm.stable_id(q.question_id, rec["instance_id"], prefix="gm_")
    prior = {
        pair_id: {
            "pair_id": pair_id,
            "accepted": True,
            "label": "MATCH",
            "element_checks": [
                {
                    "element": "remove shoes",
                    "supported": True,
                    "quote": "remove shoes",
                },
                {
                    "element": "respect and cleanliness",
                    "supported": True,
                    "quote": "respect and cleanliness",
                },
            ],
            "rationale": "direct",
        }
    }

    class FakeBackend:
        def generate(self, prompts, system_prompt=None, **kwargs):
            return [
                '{"decision":"ACCEPT",'
                '"unsupported_prior_element_indices":[],'
                '"missing_material_elements":[],'
                '"rationale":"all covered","confidence":1.0}'
            ]

        def generate_one(self, prompt, system_prompt=None, **kwargs):
            raise AssertionError("retry should not be needed")

    state = gm.run_stage(
        stage=2,
        pairs=[(q, rec)],
        backend=FakeBackend(),
        state_path=tmp_path / "stage2.jsonl",
        batch_size=1,
        prior_state=prior,
    )
    row = state[pair_id]
    assert row["accepted"] is True
    assert row["label"] == "MATCH"
    assert row["element_checks"] == prior[pair_id]["element_checks"]
    assert row["stage2_review"]["decision"] == "ACCEPT"
