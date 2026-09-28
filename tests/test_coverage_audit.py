import importlib.util
from pathlib import Path

from src.retriever import TextRetriever
from src.schemas import MCQQuestion

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "10_audit_culturalbench_coverage.py"
SPEC = importlib.util.spec_from_file_location("coverage_audit_script", MODULE_PATH)
coverage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(coverage)


def test_parse_audit_output_accepts_json_and_code_fence():
    plain = coverage.parse_audit_output(
        '{"label":"COVERED","supporting_instance_ids":["r1"],'
        '"rationale":"Direct support.","confidence":0.9}'
    )
    fenced = coverage.parse_audit_output(
        '```json\\n{"label":"ABSENT","supporting_instance_ids":[],'
        '"rationale":"No support.","confidence":0.8}\\n```'
    )
    assert plain["label"] == "COVERED"
    assert plain["supporting_instance_ids"] == ["r1"]
    assert plain["confidence"] == 0.9
    assert fenced["label"] == "ABSENT"
    assert fenced["supporting_instance_ids"] == []


def test_select_candidates_includes_all_target_culture_alias_records():
    q = MCQQuestion(
        question_id="q1", group_id="q1", variant="easy",
        question="What is common in Spanish culture?", cultural_group="Spain",
        options={"A": "Taking a nap", "B": "Other"}, answer_key="A",
    )
    passages = [
        {"instance_id": "spain_1", "cultural_group": "Spanish", "topic": "Health and Hygiene", "text": "Spanish people describe an afternoon practice."},
        {"instance_id": "spain_2", "cultural_group": "Spaniards", "topic": "Food and Dining", "text": "Spaniards describe dining practices."},
        {"instance_id": "mexico_1", "cultural_group": "Mexican", "topic": "Food and Dining", "text": "A Mexican food record."},
    ]
    retriever = TextRetriever(passages)
    candidates, culture_count, truncated = coverage.select_candidates(
        q, passages, retriever, max_culture_records=80, semantic_fallback_k=1
    )
    ids = {row["instance_id"] for row in candidates}
    assert culture_count == 2
    assert truncated is False
    assert {"spain_1", "spain_2"}.issubset(ids)


def test_gold_description_hard_false_is_explicit():
    q = MCQQuestion(
        question_id="hard_1_0", group_id="hard_1", variant="hard",
        question="What is expected?", cultural_group="Spain",
        option_text="Everyone splits the bill.", answer_bool=False,
    )
    desc = coverage.gold_description(q)
    assert "Everyone splits the bill." in desc
    assert "FALSE" in desc


def test_record_to_passage_matches_corpus_view_fields():
    rec = {
        "dataset_id": "culturebank",
        "instance_id": "r1",
        "cultural_group": "Spanish",
        "context": "at home",
        "actor_behavior": "take an afternoon rest",
        "topic": "Health and Hygiene",
        "source_text": "Spanish. at home. take an afternoon rest.",
    }
    passage = coverage.record_to_passage(rec)
    assert passage["instance_id"] == "r1"
    assert passage["cultural_group"] == "Spanish"
    assert passage["topic"] == "Health and Hygiene"
    assert "afternoon rest" in passage["text"]
