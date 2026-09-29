import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "15_build_grounded_easy_retrieval_set.py"
)
SPEC = importlib.util.spec_from_file_location("grounded_easy_builder", MODULE_PATH)
gb = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gb)


def _rec(iid, culture, topic):
    return {
        "instance_id": iid,
        "cultural_group": culture,
        "topic": topic,
        "source_text": iid,
    }


def test_fact_group_overrides_collapse_known_duplicate_facts():
    assert gb.fact_group_id("easy_1126") == "fact_japan_home_shoe_removal"
    assert gb.fact_group_id("easy_214") == "fact_japan_home_shoe_removal"
    assert gb.fact_group_id("easy_140") == "fact_dutch_direct_communication"
    assert gb.fact_group_id("easy_917") == "fact_dutch_direct_communication"
    assert gb.fact_group_id("easy_other") == "fact_easy_other"


def test_controlled_corpus_always_contains_support_and_is_deterministic():
    grounded = [
        {
            "question_id": "easy_1126",
            "cultural_group": "Japan",
            "support_instance_ids": ["support_jp"],
        },
        {
            "question_id": "easy_140",
            "cultural_group": "Netherlands",
            "support_instance_ids": ["support_nl"],
        },
    ]
    records = [
        _rec("support_jp", "Japan", "Etiquette"),
        _rec("jp_same_topic", "Japanese", "Etiquette"),
        _rec("jp_other", "Japan", "Food"),
        _rec("support_nl", "Dutch", "Communication"),
        _rec("nl_same_topic", "Netherlands", "Communication"),
        _rec("nl_other", "Dutch", "Food"),
        _rec("global_1", "Spain", "Food"),
        _rec("global_2", "China", "Food"),
    ]

    first, manifest_first = gb.select_controlled_corpus(
        grounded,
        records,
        corpus_size=8,
        seed=42,
        same_topic_per_question=1,
        same_culture_per_question=1,
    )
    second, manifest_second = gb.select_controlled_corpus(
        grounded,
        records,
        corpus_size=8,
        seed=42,
        same_topic_per_question=1,
        same_culture_per_question=1,
    )

    first_ids = [r["instance_id"] for r in first]
    assert first_ids == [r["instance_id"] for r in second]
    assert {"support_jp", "support_nl"}.issubset(first_ids)
    assert manifest_first == manifest_second
    roles = {row["instance_id"]: row["selection_roles"] for row in manifest_first}
    assert roles["support_jp"] == ["verified_support"]
    assert roles["support_nl"] == ["verified_support"]
