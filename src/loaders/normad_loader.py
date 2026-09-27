"""Load and normalize NormAd (Rao et al. 2024; ``akhilayerukola/NormAd``).

NormAd has one ``train`` split of ~2.6k stories. Each row carries a Country, the
Value and Rule-of-Thumb (two context levels that mirror the KG's L3 and L2) and a
Gold Label in {yes, no, neutral}. ``neutral`` is renamed ``neither`` to match the
answer vocabulary of NormAd's official prompt (Yes / No / Neither).

Used by eval v2 (docs/EVAL_V2_PLAN.md, Evaluation 2). NormAd's source is the
Cultural Atlas, which must never be added to the KG (leakage).
"""

from __future__ import annotations

import unicodedata
from typing import Optional

from ..core.culture_normalizer import canonical_culture
from ..core.utils import get_logger

LOGGER = get_logger("normad_loader")

NORMAD_HF_ID = "akhilayerukola/NormAd"
LABELS = ("yes", "no", "neither")

_LABEL_MAP = {
    "yes": "yes", "no": "no", "neither": "neither", "neutral": "neither",
}

# NormAd country slugs that the shared normalizer does not resolve on its own.
_COUNTRY_ALIASES = {
    "palestinian territories": "palestine",
    "timor leste": "east timor",
}


def normalize_label(value: Optional[str]) -> Optional[str]:
    """Map a NormAd gold label to yes/no/neither (None if unrecognized)."""
    if value is None:
        return None
    return _LABEL_MAP.get(str(value).strip().lower())


def _deaccent(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))


def country_display(slug: str) -> str:
    """'united_states_of_america' -> 'United States Of America'."""
    text = str(slug or "").replace("_", " ").strip()
    return " ".join(w[:1].upper() + w[1:] for w in text.split())


def country_to_culture(slug: str) -> str:
    """Map a NormAd Country slug to the canonical culture key of the KG."""
    text = _deaccent(str(slug or "").replace("_", " ").replace("-", " ")).strip().lower()
    text = _COUNTRY_ALIASES.get(text, text)
    return canonical_culture(text)


def normalize_row(raw: dict, index: int) -> Optional[dict]:
    """Normalize one NormAd row; returns None when story or label is missing."""
    story = str(raw.get("Story") or "").strip()
    gold = normalize_label(raw.get("Gold Label"))
    if not story or gold is None:
        return None
    country = str(raw.get("Country") or "").strip()
    rid = raw.get("ID", index)
    return {
        "id": f"normad_{rid}",
        "country": country,
        "country_display": country_display(country),
        "culture": country_to_culture(country),
        "axis": raw.get("Axis"),
        "subaxis": raw.get("Subaxis"),
        "value": str(raw.get("Value") or "").strip(),
        "rule_of_thumb": str(raw.get("Rule-of-Thumb") or "").strip(),
        "story": story,
        "explanation": str(raw.get("Explanation") or "").strip(),
        "gold": gold,
    }


def load_normad(limit: Optional[int] = None) -> list[dict]:
    """Load NormAd from the HF cache (honours HF_DATASETS_OFFLINE)."""
    from datasets import load_dataset

    ds = load_dataset(NORMAD_HF_ID, split="train")
    rows = []
    dropped = 0
    for i, raw in enumerate(ds):
        row = normalize_row(dict(raw), i)
        if row is None:
            dropped += 1
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    LOGGER.info("Loaded %d NormAd stories (%d dropped).", len(rows), dropped)
    return rows
