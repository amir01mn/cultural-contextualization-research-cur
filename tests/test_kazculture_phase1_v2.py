"""V2 assistant-prefix scoring invariants; no model or GPU needed."""
import pytest

from src.loaders.kazculture_loader import convert_test_rows
from src.evaluation.kazculture_phase1 import prompt_hash
from src.evaluation.kazculture_hf_scorer_v2 import (
    PROTOCOL_VERSION, candidate_token_ids, prompt_sha256,
    render_prompt, user_question,
)


def _question():
    return convert_test_rows([{
        "question": "Қай зат?", "a": "Бір", "b": "Екі",
        "c": "Үш", "d": "Төрт", "answer_label": "B",
        "item_description": "DO NOT SHOW", "answer": "SECRET",
    }])[0]


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        assert "SECRET" not in repr(messages)
        assert "DO NOT SHOW" not in repr(messages)
        assert not messages[0]["content"].endswith("Answer:")
        return "USER:" + messages[0]["content"] + "|ASSISTANT-NONTHINK:"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        for i, label in enumerate(("A", "B", "C", "D"), 10):
            if text.endswith("Answer: " + label):
                return [777, 800 + i]
        if text.endswith("Answer:"):
            return [777]
        raise AssertionError("Unexpected tokenization")


def test_candidate_ids_use_contextual_leading_space_and_nonthinking():
    q = _question()
    assert user_question(q).endswith("D. Төрт")
    rendered = render_prompt(FakeTokenizer(), q)
    assert rendered.endswith("|ASSISTANT-NONTHINK:Answer:")
    assert candidate_token_ids(FakeTokenizer(), rendered) == {
        "A": 810, "B": 811, "C": 812, "D": 813,
    }
    assert prompt_sha256(q) != prompt_hash(q)
    assert "v2" in PROTOCOL_VERSION


def test_mismatched_token_boundaries_fail_loudly():
    class BrokenTokenizer(FakeTokenizer):
        def encode(self, text, add_special_tokens=False):
            values = super().encode(text, add_special_tokens=add_special_tokens)
            if text.endswith("Answer: B"):
                return [123] + values
            return values
    with pytest.raises(ValueError, match="cannot use one-token"):
        candidate_token_ids(BrokenTokenizer(), render_prompt(FakeTokenizer(), _question()))
