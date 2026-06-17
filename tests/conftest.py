import sys
from pathlib import Path

# Ensure the project root is importable as `src` during tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.schemas import CulturalTriple, LayerLabel


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
