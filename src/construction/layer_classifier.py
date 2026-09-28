"""Layer validation and lightweight fallback classification.

NOTE: This module is NOT the main extraction method. The local LLM assigns
layers during extraction. This module only:
  - validates that LLM-provided layer labels are well-formed,
  - converts invalid labels to UNKNOWN,
  - provides a keyword-based guess as a *fallback* when a label is missing
    or UNKNOWN, so the graph still has reasonable structure.
"""

from __future__ import annotations

from ..core.schemas import LayerLabel
from ..core.utils import normalize_text

VALID_LAYERS = {l.value for l in LayerLabel}

# Keyword cues per layer used only for fallback guessing.
_L1_CUES = {
    "shoe", "shoes", "food", "meal", "eat", "coffee", "tea", "bow", "kiss",
    "gesture", "clothing", "dress", "ritual", "ceremony", "holiday", "gift",
    "card", "hand", "nose", "feet", "slipper", "home", "house", "greeting",
    "greet", "serve", "drink", "cup", "touch", "wear", "dance", "song",
}
_L2_CUES = {
    "host", "guest", "elder", "role", "expectation", "expected", "norm",
    "polite", "politeness", "hierarchy", "relation", "relationship", "respectful",
    "obligation", "taboo", "should", "must", "communication", "family", "senior",
    "junior", "service", "etiquette", "rule", "duty", "status",
}
_L3_CUES = {
    "respect", "harmony", "hospitality", "modesty", "honor", "face", "dignity",
    "value", "belief", "moral", "identity", "spiritual", "spirituality",
    "collectivism", "individualism", "purity", "unity", "care", "trust",
    "friendship", "blessing", "sacred", "humility", "generosity",
}


def is_valid_layer(label: str) -> bool:
    return normalize_text(label).upper() in VALID_LAYERS


def validate_layer(label: object) -> LayerLabel:
    """Return a valid LayerLabel; invalid input collapses to UNKNOWN."""
    if isinstance(label, LayerLabel):
        return label
    if label is None:
        return LayerLabel.UNKNOWN
    norm = normalize_text(label).upper()
    if norm in VALID_LAYERS:
        return LayerLabel(norm)
    return LayerLabel.coerce(label)


def guess_layer(text: str) -> LayerLabel:
    """Keyword-based fallback layer guess for a single element."""
    tokens = set(normalize_text(text).lower().replace("-", " ").split())
    if not tokens:
        return LayerLabel.UNKNOWN
    scores = {
        LayerLabel.LAYER_1_SURFACE: len(tokens & _L1_CUES),
        LayerLabel.LAYER_2_SOCIAL: len(tokens & _L2_CUES),
        LayerLabel.LAYER_3_DEEP_VALUE: len(tokens & _L3_CUES),
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return LayerLabel.UNKNOWN
    return best


def validate_or_guess(label: object, text: str) -> LayerLabel:
    """Validate an LLM label; if UNKNOWN/invalid, attempt a keyword guess."""
    validated = validate_layer(label)
    if validated != LayerLabel.UNKNOWN:
        return validated
    return guess_layer(text)


def validate_triple_layers(triple: dict) -> dict:
    """Sanity-check the three layer fields on a triple dict in place."""
    triple["subject_layer"] = validate_or_guess(
        triple.get("subject_layer"), triple.get("subject", "")
    ).value
    triple["predicate_layer"] = validate_or_guess(
        triple.get("predicate_layer"), triple.get("predicate", "")
    ).value
    triple["object_layer"] = validate_or_guess(
        triple.get("object_layer"), triple.get("object", "")
    ).value
    return triple
