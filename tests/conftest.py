import os
import sys
from pathlib import Path

# Ensure the project root is importable as `src` during tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep the suite offline and deterministic. Set before importing `src` so
# config picks them up: otherwise retrieval loads sentence-transformers from
# shared storage (minutes per run) and may reach for the network.
# setdefault, so an explicit override from the environment still wins.
os.environ.setdefault("EMBEDDING_BACKEND", "tfidf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pytest

from src.core.schemas import CulturalTriple, LayerLabel, MCQQuestion, MCQVariant


class FakeBackend:
    """A deterministic backend for tests: maps each prompt to a response.

    ``responder`` is a callable (user_prompt, system_prompt) -> str.
    """

    name = "fake"
    model = "fake-model"

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def generate(self, prompts, system_prompt=None, **kwargs):
        self.calls.append((list(prompts), system_prompt))
        return [self.responder(p, system_prompt) for p in prompts]

    def generate_one(self, prompt, system_prompt=None, **kwargs):
        return self.responder(prompt, system_prompt)

    def status(self):
        return {"available": True, "model_present": True, "message": "fake backend"}


@pytest.fixture
def fake_backend_factory():
    def _make(responder):
        return FakeBackend(responder)
    return _make


@pytest.fixture
def mcq_easy_question() -> MCQQuestion:
    return MCQQuestion(
        question_id="easy_1", variant=MCQVariant.EASY, group_id="easy_1",
        question="In Japan, what do you do with shoes when entering a home?",
        cultural_group="Japan",
        options={"A": "Keep them on", "B": "Remove them at the entrance",
                 "C": "Put them on the table", "D": "Give them away"},
        answer_key="B",
    )


@pytest.fixture
def mcq_hard_group() -> list[MCQQuestion]:
    specs = [
        ("Greeting an elder in India?", "Touch the elder's feet", True),
        ("Greeting an elder in India?", "Ignore the elder", False),
    ]
    return [
        MCQQuestion(
            question_id=f"hard_1_{i}", variant=MCQVariant.HARD, group_id="hard_1",
            question=q, cultural_group="India", option_text=opt, answer_bool=gold,
        )
        for i, (q, opt, gold) in enumerate(specs)
    ]


@pytest.fixture
def sample_record() -> dict:
    return {
        "dataset_id": "culturebank",
        "instance_id": "test_001",
        "cultural_group": "Japanese",
        "context": "Visiting someone's home",
        "goal": "show respect",
        "actor": "guest",
        "recipient": "host",
        "relation": "guest-host",
        "actor_behavior": "removes shoes at the entrance",
        "recipient_behavior": "offers slippers",
        "topic": "home etiquette",
        "agreement": "high",
        "source_text": "A guest removes shoes at the genkan; the host offers slippers.",
        "raw": {"cultural group": "Japanese", "topic": "home etiquette"},
    }


@pytest.fixture
def sample_triples() -> list[CulturalTriple]:
    return [
        CulturalTriple(
            triple_id="t_1",
            subject="guest",
            predicate="removes",
            object="shoes at the entrance",
            subject_layer=LayerLabel.LAYER_2_SOCIAL,
            predicate_layer=LayerLabel.LAYER_1_SURFACE,
            object_layer=LayerLabel.LAYER_1_SURFACE,
            confidence=0.8,
            source_text="A guest removes shoes at the genkan.",
            instance_id="test_001",
            generated=False,
            evidence="actor_behavior",
        ),
        CulturalTriple(
            triple_id="t_2",
            subject="removing shoes",
            predicate="expresses",
            object="respect",
            subject_layer=LayerLabel.LAYER_1_SURFACE,
            predicate_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            object_layer=LayerLabel.LAYER_3_DEEP_VALUE,
            confidence=0.4,
            source_text="A guest removes shoes at the genkan.",
            instance_id="test_001",
            generated=True,
            evidence="inferred",
        ),
    ]
