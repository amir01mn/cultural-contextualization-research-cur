from src import culturalbench_loader as cbl
from src.schemas import MCQVariant


def test_fallback_dataset_has_easy_and_hard():
    qs = cbl._fallback_dataset()
    easy = [q for q in qs if q.variant == MCQVariant.EASY]
    hard = [q for q in qs if q.variant == MCQVariant.HARD]
    assert len(easy) >= 2
    assert len(hard) >= 2
    # Hard items share a group_id.
    assert len({q.group_id for q in hard}) >= 1


def test_load_fallback_when_offline(monkeypatch):
    # Force the HF load path to fail so the loader uses the fallback sample.
    def boom(*args, **kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(cbl, "_load_hf_split", boom)
    qs = cbl.load_culturalbench(variants=("easy", "hard"), limit=10)
    assert len(qs) > 0
    assert any(q.variant == MCQVariant.EASY for q in qs)
    assert any(q.variant == MCQVariant.HARD for q in qs)


def test_variant_filtering(monkeypatch):
    monkeypatch.setattr(cbl, "_load_hf_split",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    easy_only = cbl.load_culturalbench(variants=("easy",), limit=10)
    assert all(q.variant == MCQVariant.EASY for q in easy_only)


def test_extract_easy_options_prompt_option_convention():
    raw = {
        "prompt_question": "Q?",
        "prompt_option_a": "first",
        "prompt_option_b": "second",
        "prompt_option_c": "third",
        "prompt_option_d": "fourth",
        "answer": "b",
        "country": "Testland",
    }
    qs = cbl._normalize_easy([raw], limit=5)
    assert len(qs) == 1
    q = qs[0]
    assert q.options["A"] == "first"
    assert q.answer_key == "B"
    assert q.cultural_group == "Testland"


def test_normalize_answer_letter_from_text():
    options = {"A": "Keep them on", "B": "Remove them"}
    assert cbl._normalize_answer_letter("Remove them", options) == "B"
    assert cbl._normalize_answer_letter("A", options) == "A"
    assert cbl._normalize_answer_letter("2", options) == "B"


def test_normalize_bool():
    assert cbl._normalize_bool("yes") is True
    assert cbl._normalize_bool("False") is False
    assert cbl._normalize_bool(True) is True
    assert cbl._normalize_bool("maybe") is None


def test_normalize_hard_groups_items():
    rows = [
        {"data_idx": 7, "prompt_question": "Q?", "prompt_option": "opt1",
         "answer": "true", "country": "X"},
        {"data_idx": 7, "prompt_question": "Q?", "prompt_option": "opt2",
         "answer": "false", "country": "X"},
    ]
    qs = cbl._normalize_hard(rows, limit_groups=5)
    assert len(qs) == 2
    assert all(q.group_id == "hard_7" for q in qs)
    assert qs[0].answer_bool is True and qs[1].answer_bool is False
