"""Load and normalize CultureBank records.

Attempts to download CultureBank from Hugging Face ``datasets``. If that fails
(no network, dataset id changed, datasets not installed), a small built-in
fallback dataset (5-10 examples) is used so the pipeline always runs.

Every record is normalized to a fixed schema; missing fields become None.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from ..core import config
from ..core.schemas import CulturalRecord
from ..core.utils import coalesce, get_logger, normalize_text, stable_id, write_jsonl

LOGGER = get_logger("data_loader")


# Candidate source-field names in CultureBank-style datasets, mapped to our
# normalized field. We try each candidate in order.
FIELD_CANDIDATES = {
    "cultural_group": ["cultural group", "cultural_group", "culture", "country", "region"],
    "context": ["context", "scenario", "situation", "setting"],
    "goal": ["goal", "intent", "purpose"],
    "actor": ["actor", "agent", "subject", "person"],
    "recipient": ["recipient", "target", "other", "object_person"],
    "relation": ["relation", "relationship"],
    "actor_behavior": ["actor_behavior", "actor behavior", "behavior", "action"],
    "recipient_behavior": ["recipient_behavior", "recipient behavior", "response"],
    "topic": ["topic", "cultural topic", "category", "domain"],
    "agreement": ["agreement", "agreement_percentage", "norm", "vote"],
    "source_text": ["source_text", "desc", "description", "text", "summary", "content"],
}


def _pick(raw: dict, field: str) -> Optional[str]:
    for candidate in FIELD_CANDIDATES.get(field, [field]):
        if candidate in raw and raw[candidate] not in (None, "", []):
            return normalize_text(raw[candidate]) or None
    return None


def _build_source_text(raw: dict, normalized: dict) -> str:
    """Synthesize a source_text if the dataset has no obvious one."""
    explicit = _pick(raw, "source_text")
    if explicit:
        return explicit
    pieces = []
    for key in ("cultural_group", "context", "actor_behavior",
                "recipient_behavior", "topic"):
        val = normalized.get(key)
        if val:
            pieces.append(str(val))
    return normalize_text(". ".join(pieces))


def normalize_record(raw: dict, index: int, dataset_id: str = "culturebank") -> dict:
    """Map an arbitrary raw record to the fixed normalized schema."""
    raw = {str(k): v for k, v in dict(raw).items()}
    instance_id = coalesce(
        raw.get("instance_id"),
        raw.get("id"),
        raw.get("idx"),
    )
    normalized: dict[str, Any] = {
        "dataset_id": dataset_id,
        "instance_id": str(instance_id) if instance_id is not None
        else stable_id(dataset_id, index, prefix="cb_"),
        "cultural_group": _pick(raw, "cultural_group"),
        "context": _pick(raw, "context"),
        "goal": _pick(raw, "goal"),
        "actor": _pick(raw, "actor"),
        "recipient": _pick(raw, "recipient"),
        "relation": _pick(raw, "relation"),
        "actor_behavior": _pick(raw, "actor_behavior"),
        "recipient_behavior": _pick(raw, "recipient_behavior"),
        "topic": _pick(raw, "topic"),
        "agreement": _pick(raw, "agreement"),
        "raw": raw,
    }
    normalized["source_text"] = _build_source_text(raw, normalized) or None
    # Validate via Pydantic, then re-dump to ensure schema correctness.
    return CulturalRecord(**normalized).model_dump()


def _fallback_dataset() -> list[dict]:
    """A small, respectful set of CultureBank-style examples for offline runs."""
    examples = [
        {
            "instance_id": "fallback_001",
            "cultural group": "Japanese",
            "context": "Visiting someone's home",
            "actor": "guest",
            "recipient": "host",
            "relation": "guest-host",
            "actor_behavior": "removes shoes at the entrance before stepping inside",
            "recipient_behavior": "offers indoor slippers to the guest",
            "topic": "home etiquette",
            "agreement": "high",
            "desc": "In this context, a guest takes off their shoes at the genkan "
                    "before entering the home, and the host provides slippers.",
        },
        {
            "instance_id": "fallback_002",
            "cultural group": "Indian",
            "context": "Greeting an elder",
            "actor": "younger person",
            "recipient": "elder",
            "relation": "junior-senior",
            "actor_behavior": "touches the elder's feet as a sign of respect",
            "recipient_behavior": "places a hand on the head to bless",
            "topic": "greetings and respect",
            "agreement": "high",
            "desc": "A younger person greets an elder by touching their feet, and "
                    "the elder responds with a blessing.",
        },
        {
            "instance_id": "fallback_003",
            "cultural group": "Ethiopian",
            "context": "Sharing a meal",
            "actor": "host",
            "recipient": "guest",
            "relation": "host-guest",
            "actor_behavior": "feeds the guest a bite by hand (gursha)",
            "recipient_behavior": "accepts the gursha graciously",
            "topic": "hospitality and food",
            "agreement": "medium",
            "desc": "During a shared injera meal, the host offers gursha, hand-feeding "
                    "a guest as a gesture of friendship and care.",
        },
        {
            "instance_id": "fallback_004",
            "cultural group": "Brazilian",
            "context": "Arriving at a social gathering",
            "actor": "guest",
            "recipient": "other guests",
            "relation": "peers",
            "actor_behavior": "greets others with a kiss on the cheek",
            "recipient_behavior": "reciprocates the cheek greeting",
            "topic": "social greetings",
            "agreement": "medium",
            "desc": "At a casual gathering, people greet each other with cheek kisses; "
                    "the number varies by region.",
        },
        {
            "instance_id": "fallback_005",
            "cultural group": "Saudi Arabian",
            "context": "Being served coffee by a host",
            "actor": "guest",
            "recipient": "host",
            "relation": "guest-host",
            "actor_behavior": "shakes the cup gently to signal they have had enough",
            "recipient_behavior": "keeps refilling until the guest signals to stop",
            "topic": "hospitality",
            "agreement": "high",
            "desc": "When served Arabic coffee (gahwa), a guest gently wiggles the cup "
                    "to indicate they are finished; otherwise the host refills it.",
        },
        {
            "instance_id": "fallback_006",
            "cultural group": "Chinese",
            "context": "Receiving a business card",
            "actor": "recipient",
            "recipient": "giver",
            "relation": "business contacts",
            "actor_behavior": "accepts the card with both hands and examines it",
            "recipient_behavior": "presents the card with both hands",
            "topic": "business etiquette",
            "agreement": "high",
            "desc": "A business card is given and received with both hands and a moment "
                    "of attention, signaling respect for the other person.",
        },
        {
            "instance_id": "fallback_007",
            "cultural group": "Maori (New Zealand)",
            "context": "Formal greeting",
            "actor": "two people",
            "recipient": "each other",
            "relation": "peers",
            "actor_behavior": "press noses and foreheads together (hongi)",
            "recipient_behavior": "shares the breath of life in return",
            "topic": "greetings",
            "agreement": "high",
            "desc": "The hongi is a traditional greeting where two people press noses "
                    "and foreheads, symbolizing the sharing of breath and life.",
        },
    ]
    return examples


def load_culturebank(dataset: str = "culturebank",
                     split: str = "train",
                     limit: int = 100) -> list[dict]:
    """Load and normalize up to ``limit`` CultureBank records.

    Falls back to a built-in dataset if Hugging Face loading fails.
    """
    raw_records: list[dict] = []
    used_fallback = False

    try:
        from datasets import get_dataset_split_names, load_dataset

        LOGGER.info("Attempting to load '%s' from Hugging Face (%s)...",
                    dataset, config.CULTUREBANK_HF_ID)

        # Resolve which split(s) to read. CultureBank has no 'train' split; its
        # real splits are e.g. ['tiktok', 'reddit']. If the requested split is
        # not available, read from all available splits (in order) up to limit.
        try:
            available = get_dataset_split_names(config.CULTUREBANK_HF_ID)
        except Exception:  # noqa: BLE001
            available = []

        if split in available:
            splits_to_read = [split]
        elif available:
            LOGGER.warning(
                "Split '%s' not found for %s. Available splits: %s. "
                "Reading from available splits instead.",
                split, config.CULTUREBANK_HF_ID, available,
            )
            splits_to_read = list(available)
        else:
            # Could not introspect; try the requested split and let it raise.
            splits_to_read = [split]

        for sp in splits_to_read:
            if len(raw_records) >= limit:
                break
            ds = load_dataset(config.CULTUREBANK_HF_ID, split=sp)
            remaining = limit - len(raw_records)
            n = min(remaining, len(ds))
            for i in range(n):
                rec = dict(ds[i])
                rec.setdefault("_split", sp)
                raw_records.append(rec)
            LOGGER.info("Loaded %d records from split '%s'.", n, sp)

        if not raw_records:
            raise RuntimeError("No records returned from any split.")
        LOGGER.info("Loaded %d records total from Hugging Face.", len(raw_records))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Could not load CultureBank from Hugging Face: %s", exc)
        LOGGER.warning("Using built-in fallback dataset so the pipeline still runs.")
        raw_records = _fallback_dataset()
        used_fallback = True

    if not raw_records:
        raw_records = _fallback_dataset()
        used_fallback = True

    normalized = [
        normalize_record(rec, idx, dataset_id=dataset)
        for idx, rec in enumerate(raw_records[:limit])
    ]
    if used_fallback:
        LOGGER.info("Normalized %d fallback records.", len(normalized))
    return normalized


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Load and normalize CultureBank.")
    parser.add_argument("--dataset", default=config.DATASET_NAME)
    parser.add_argument("--split", default=config.DEFAULT_SPLIT)
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    parser.add_argument(
        "--output",
        default=str(config.PROCESSED_DIR / "culturebank_sample_100.jsonl"),
    )
    args = parser.parse_args(argv)

    records = load_culturebank(args.dataset, args.split, args.limit)
    count = write_jsonl(args.output, records)
    LOGGER.info("Wrote %d normalized records to %s", count, args.output)


if __name__ == "__main__":
    main()
