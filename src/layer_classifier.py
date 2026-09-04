"""Layer validation and lightweight fallback classification.

NOTE: This module is NOT the main extraction method. The local LLM assigns
layers during extraction. This module only:
  - validates that LLM-provided layer labels are well-formed,
  - converts invalid labels to UNKNOWN,
  - provides a keyword-based guess as a *fallback* when a label is missing
    or UNKNOWN,
  - applies narrow, high-confidence repairs for common extraction mistakes
    found in the controlled food/hospitality subset.
"""

from __future__ import annotations

from .schemas import LayerLabel
from .utils import normalize_text

VALID_LAYERS = {l.value for l in LayerLabel}

# Keyword cues per layer used for fallback guessing and conservative repairs.
_L1_CUES = {
    "shoe", "shoes", "food", "meal", "meals", "eat", "eating", "coffee",
    "tea", "bow", "kiss", "gesture", "clothing", "dress", "ritual",
    "ceremony", "holiday", "gift", "gifts", "card", "hand", "nose",
    "feet", "slipper", "home", "house", "greeting", "greet", "serve",
    "serving", "drink", "drinks", "cup", "touch", "wear", "dance",
    "song", "water", "ice", "snack", "snacks", "plate", "plates", "fork",
    "utensil", "utensils", "buffet", "refill", "refills", "tip", "tipping",
    "bill", "pay", "paying", "restaurant", "restaurants", "bar", "bars",
    "cooking", "cook", "offering", "offer", "providing", "provide",
    "consuming", "consume", "bring", "bringing", "celebrate", "celebration",
    "celebrations", "birthday", "candy", "to-go", "box", "boxes",
}
_L2_CUES = {
    "host", "hosts", "guest", "guests", "visitor", "visitors", "elder",
    "role", "roles", "actor", "recipient", "author", "authors",
    "commenter", "commenters", "expectation", "expectations", "expected",
    "norm", "norms", "polite", "politeness", "hierarchy", "relation",
    "relationship", "respectful", "obligation", "taboo", "should", "must",
    "communication", "family", "families", "friend", "friends", "parent",
    "parents", "child", "children", "senior", "junior", "service", "server",
    "servers", "customer", "customers", "staff", "restaurant staff",
    "etiquette", "rule", "rules", "duty", "status", "birthday person",
    "people", "individual", "individuals", "couple", "couples",
}
_L3_CUES = {
    "respect", "harmony", "hospitality", "modesty", "honor", "honour",
    "face", "dignity", "value", "values", "belief", "beliefs", "moral",
    "identity", "spiritual", "spirituality", "collectivism", "individualism",
    "purity", "unity", "care", "trust", "friendship", "blessing", "sacred",
    "humility", "generosity", "gratitude", "grateful", "appreciation",
    "appreciate", "reciprocity", "reciprocal", "reciprocate", "welcoming",
    "belonging", "cleanliness", "fairness", "comfort", "community connection",
}

# Phrase-level cues help with short labels where tokenization is not enough.
_L2_PHRASES = {
    "birthday person",
    "restaurant staff",
    "hosts and guests",
    "guests and hosts",
    "visitors and the author",
    "visitors and the author of the comment",
    "author of the comment",
    "customers and servers",
    "servers and customers",
    "friends and family",
    "family and friends",
    "parents and children",
    "people and couples",
}
_L3_PHRASES = {
    "host-guest reciprocity",
    "respect for guests",
    "hospitality and appreciation",
    "family connection",
    "community connection",
}
_L1_PHRASES = {
    "offering, providing, and consuming drinks and food",
    "offering drinks",
    "providing drinks",
    "consuming drinks",
    "bringing gifts",
    "gifts or their own drinks",
    "meals, gifts",
    "serve food",
    "birthday celebrations",
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


def _tokens(text: str) -> set[str]:
    return set(normalize_text(text).lower().replace("-", " ").split())


def _contains_phrase(text_l: str, phrases: set[str]) -> bool:
    return any(p in text_l for p in phrases)


def guess_layer(text: str) -> LayerLabel:
    """Keyword-based fallback layer guess for a single element."""
    text_l = normalize_text(text).lower()
    tokens = _tokens(text)
    if not tokens and not text_l:
        return LayerLabel.UNKNOWN

    # Deep values take precedence when a label explicitly names a value, e.g.
    # "hospitality and appreciation" or "host-guest reciprocity".
    if _contains_phrase(text_l, _L3_PHRASES):
        return LayerLabel.LAYER_3_DEEP_VALUE

    scores = {
        LayerLabel.LAYER_1_SURFACE: len(tokens & _L1_CUES),
        LayerLabel.LAYER_2_SOCIAL: len(tokens & _L2_CUES),
        LayerLabel.LAYER_3_DEEP_VALUE: len(tokens & _L3_CUES),
    }

    # Prefer L3 when explicit value words are present, even if the phrase also
    # contains social-role words such as "guest".
    if scores[LayerLabel.LAYER_3_DEEP_VALUE] > 0:
        return LayerLabel.LAYER_3_DEEP_VALUE

    # Prefer L2 for actor/role phrases such as "hosts and guests".
    if _contains_phrase(text_l, _L2_PHRASES):
        return LayerLabel.LAYER_2_SOCIAL

    # Prefer L1 for concrete practices, objects, and settings.
    if _contains_phrase(text_l, _L1_PHRASES):
        return LayerLabel.LAYER_1_SURFACE

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return LayerLabel.UNKNOWN
    return best


def repair_layer(label: object, text: str) -> LayerLabel:
    """Validate a layer and repair high-confidence LLM layer mistakes.

    This is intentionally conservative. It fixes cases like:
      - "hosts and guests" marked as L1 -> L2
      - "visitors and the author of the comment" marked as L1 -> L2
      - "gifts" marked as L2 -> L1
      - "hospitality and appreciation" marked as L1/L2 -> L3
    """
    validated = validate_layer(label)
    guessed = guess_layer(text)
    if guessed == LayerLabel.UNKNOWN:
        return validated

    text_l = normalize_text(text).lower()
    tokens = _tokens(text)

    # Strong value labels should be L3 regardless of the model's label.
    if guessed == LayerLabel.LAYER_3_DEEP_VALUE:
        return guessed

    # Strong actor/role labels should be L2 unless they explicitly name a value.
    if guessed == LayerLabel.LAYER_2_SOCIAL:
        if _contains_phrase(text_l, _L2_PHRASES) or tokens & _L2_CUES:
            return guessed

    # Strong concrete object/action labels should be L1 unless they explicitly
    # name a value or social role.
    if guessed == LayerLabel.LAYER_1_SURFACE:
        if _contains_phrase(text_l, _L1_PHRASES) or tokens & _L1_CUES:
            return guessed

    if validated != LayerLabel.UNKNOWN:
        return validated
    return guessed


def validate_or_guess(label: object, text: str) -> LayerLabel:
    """Validate an LLM label, then repair or guess when appropriate."""
    return repair_layer(label, text)


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
