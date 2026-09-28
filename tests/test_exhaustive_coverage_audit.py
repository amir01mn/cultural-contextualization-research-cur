import importlib.util
from pathlib import Path

from src.schemas import MCQQuestion

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "12_audit_culturalbench_coverage_exhaustive.py"
SPEC = importlib.util.spec_from_file_location("coverage_exhaustive_script", MODULE_PATH)
coverage = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(coverage)


def _question():
    return MCQQuestion(
        question_id="q1",
        group_id="q1",
        variant="easy",
        question="What is the practice?",
        cultural_group="Spain",
        options={"A": "Afternoon nap", "B": "Other"},
        answer_key="A",
    )


def test_chunks_cover_every_record_once():
    rows = [{"instance_id": f"r{i}"} for i in range(10)]
    out = coverage.chunks(rows, 4)
    assert [len(x) for x in out] == [4, 4, 2]
    assert [r["instance_id"] for chunk in out for r in chunk] == [f"r{i}" for i in range(10)]


def test_aggregate_prefers_covered_over_partial_and_absent():
    q = _question()
    docs = [{"instance_id": "r1"}, {"instance_id": "r2"}]
    chunk_results = [
        {"label": "ABSENT", "supporting_instance_ids": [], "rationale": "none", "confidence": 0.9},
        {"label": "COVERED", "supporting_instance_ids": ["r2"], "rationale": "direct", "confidence": 0.8},
        {"label": "PARTIAL", "supporting_instance_ids": ["r1"], "rationale": "partial", "confidence": 0.95},
    ]
    row = coverage.aggregate_question(q, docs, chunk_results)
    assert row["label"] == "COVERED"
    assert row["supporting_instance_ids"] == ["r2"]
    assert row["audited_all_target_culture_records"] is True


def test_validate_partial_without_cited_target_record_becomes_absent():
    q = _question()
    candidates = [
        {"instance_id": "r1", "cultural_group": "Spanish", "topic": "Health", "text": "A related record."}
    ]
    raw = '{"label":"PARTIAL","supporting_instance_ids":[],"rationale":"related","confidence":0.9}'
    result = coverage.validate_chunk_audit(q, candidates, raw)
    assert result["label"] == "ABSENT"
    assert result["supporting_instance_ids"] == []


def test_validate_covered_rejects_cross_culture_support():
    q = _question()
    candidates = [
        {"instance_id": "r1", "cultural_group": "Mexican", "topic": "Health", "text": "Afternoon nap."}
    ]
    raw = '{"label":"COVERED","supporting_instance_ids":["r1"],"rationale":"direct","confidence":0.9}'
    result = coverage.validate_chunk_audit(q, candidates, raw)
    assert result["label"] == "ABSENT"
    assert result["cross_culture_support_instance_ids"] == ["r1"]
