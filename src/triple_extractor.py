"""Cultural triple extraction.

The MAIN method is LLM-based (local model via Ollama). Rule/schema-based
extraction exists ONLY as a fallback to keep the pipeline running when the local
LLM is unavailable (and only when fallback is explicitly allowed).

Modes:
    - "llm_based"      : LLM only (default)
    - "schema_fallback": deterministic field-based triples only
    - "both"           : LLM triples plus schema triples (deduped)
"""

from __future__ import annotations

import json
from typing import Optional

from . import config, local_llm, prompts
from .layer_classifier import validate_triple_layers
from .schemas import CulturalRecord, CulturalTriple, LayerLabel
from .utils import get_logger, normalize_text, stable_id, truncate

LOGGER = get_logger("triple_extractor")


def _record_to_dict(record) -> dict:
    if isinstance(record, CulturalRecord):
        return record.model_dump()
    return dict(record)


def _build_triple(rec: dict, fields: dict, method: str) -> CulturalTriple:
    """Construct a CulturalTriple with provenance from a raw field dict."""
    subject = normalize_text(fields.get("subject"))
    predicate = normalize_text(fields.get("predicate"))
    obj = normalize_text(fields.get("object"))
    triple_id = stable_id(
        rec.get("instance_id", ""), subject, predicate, obj, method,
        prefix="t_",
    )
    return CulturalTriple(
        triple_id=triple_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        subject_layer=LayerLabel.coerce(fields.get("subject_layer")),
        predicate_layer=LayerLabel.coerce(fields.get("predicate_layer")),
        object_layer=LayerLabel.coerce(fields.get("object_layer")),
        confidence=fields.get("confidence", 0.5),
        source_text=rec.get("source_text") or "",
        dataset_id=rec.get("dataset_id", "culturebank"),
        instance_id=rec.get("instance_id", ""),
        raw_fields=rec.get("raw", {}) or {},
        extraction_method=method,
        generated=bool(fields.get("generated", False)),
        evidence=fields.get("evidence"),
        notes=fields.get("notes"),
    )


# --- LLM-based extraction (MAIN) ------------------------------------------

def _format_user_prompt(rec: dict) -> str:
    raw_str = truncate(json.dumps(rec.get("raw", {}), ensure_ascii=False), 1500)

    def f(key: str) -> str:
        return normalize_text(rec.get(key)) or "null"

    return prompts.TRIPLE_EXTRACTION_USER.format(
        dataset_id=f("dataset_id"),
        instance_id=f("instance_id"),
        cultural_group=f("cultural_group"),
        context=f("context"),
        goal=f("goal"),
        actor=f("actor"),
        recipient=f("recipient"),
        relation=f("relation"),
        actor_behavior=f("actor_behavior"),
        recipient_behavior=f("recipient_behavior"),
        topic=f("topic"),
        agreement=f("agreement"),
        source_text=f("source_text"),
        raw=raw_str,
    )


def extract_triples_llm(record,
                        *,
                        model: Optional[str] = None) -> list[CulturalTriple]:
    """Extract triples using the local LLM. Returns [] on parse failure.

    Raises local_llm.LLMUnavailableError if the server is unreachable.
    """
    rec = _record_to_dict(record)
    user_prompt = _format_user_prompt(rec)
    parsed = local_llm.generate_json(
        user_prompt,
        system_prompt=prompts.TRIPLE_EXTRACTION_SYSTEM,
        model=model,
    )
    if not parsed or not isinstance(parsed, dict):
        LOGGER.warning("LLM returned no valid JSON for record %s",
                       rec.get("instance_id"))
        return []

    raw_triples = parsed.get("triples", [])
    if not isinstance(raw_triples, list):
        return []

    triples: list[CulturalTriple] = []
    for ft in raw_triples:
        if not isinstance(ft, dict):
            continue
        if not normalize_text(ft.get("subject")) or not normalize_text(ft.get("object")):
            continue
        ft = validate_triple_layers(ft)
        triples.append(_build_triple(rec, ft, method="llm_based"))
    return triples


# --- Schema fallback (NOT the main method) --------------------------------

def extract_triples_schema(record) -> list[CulturalTriple]:
    """Deterministic field-based triples. Fallback only.

    Builds a handful of grounded triples directly from normalized record fields.
    No external model is used. Layers are guessed by the keyword classifier.
    """
    rec = _record_to_dict(record)
    cg = normalize_text(rec.get("cultural_group"))
    actor = normalize_text(rec.get("actor")) or "actor"
    recipient = normalize_text(rec.get("recipient")) or "recipient"
    candidates: list[dict] = []

    if cg and rec.get("actor_behavior"):
        candidates.append({
            "subject": cg,
            "predicate": "practices",
            "object": normalize_text(rec["actor_behavior"]),
            "subject_layer": "LAYER_2_SOCIAL",
            "predicate_layer": "LAYER_2_SOCIAL",
            "object_layer": "LAYER_1_SURFACE",
            "confidence": 0.6,
            "generated": False,
            "evidence": "actor_behavior",
        })
    if rec.get("actor_behavior"):
        candidates.append({
            "subject": actor,
            "predicate": "performs",
            "object": normalize_text(rec["actor_behavior"]),
            "subject_layer": "LAYER_2_SOCIAL",
            "predicate_layer": "LAYER_1_SURFACE",
            "object_layer": "LAYER_1_SURFACE",
            "confidence": 0.6,
            "generated": False,
            "evidence": "actor_behavior",
        })
    if rec.get("recipient_behavior"):
        candidates.append({
            "subject": recipient,
            "predicate": "responds with",
            "object": normalize_text(rec["recipient_behavior"]),
            "subject_layer": "LAYER_2_SOCIAL",
            "predicate_layer": "LAYER_1_SURFACE",
            "object_layer": "LAYER_1_SURFACE",
            "confidence": 0.6,
            "generated": False,
            "evidence": "recipient_behavior",
        })
    if rec.get("relation"):
        candidates.append({
            "subject": actor,
            "predicate": "has relation",
            "object": normalize_text(rec["relation"]) + " with " + recipient,
            "subject_layer": "LAYER_2_SOCIAL",
            "predicate_layer": "LAYER_2_SOCIAL",
            "object_layer": "LAYER_2_SOCIAL",
            "confidence": 0.55,
            "generated": False,
            "evidence": "relation",
        })
    if cg and rec.get("topic"):
        candidates.append({
            "subject": cg,
            "predicate": "has cultural topic",
            "object": normalize_text(rec["topic"]),
            "subject_layer": "LAYER_2_SOCIAL",
            "predicate_layer": "LAYER_2_SOCIAL",
            "object_layer": "LAYER_3_DEEP_VALUE",
            "confidence": 0.5,
            "generated": False,
            "evidence": "topic",
        })
    if rec.get("context") and rec.get("actor_behavior"):
        candidates.append({
            "subject": normalize_text(rec["actor_behavior"]),
            "predicate": "occurs in context",
            "object": normalize_text(rec["context"]),
            "subject_layer": "LAYER_1_SURFACE",
            "predicate_layer": "LAYER_2_SOCIAL",
            "object_layer": "LAYER_1_SURFACE",
            "confidence": 0.55,
            "generated": False,
            "evidence": "context",
        })

    triples = []
    for c in candidates:
        c = validate_triple_layers(c)
        triples.append(_build_triple(rec, c, method="schema_fallback"))
    return triples


# --- Dispatcher ------------------------------------------------------------

def _dedupe(triples: list[CulturalTriple]) -> list[CulturalTriple]:
    seen: dict[str, CulturalTriple] = {}
    for t in triples:
        key = stable_id(
            t.instance_id,
            t.subject.lower(),
            t.predicate.lower(),
            t.object.lower(),
        )
        if key not in seen:
            seen[key] = t
    return list(seen.values())


def extract_triples(record,
                    mode: str = "llm_based",
                    *,
                    allow_fallback: bool = False,
                    model: Optional[str] = None) -> list[CulturalTriple]:
    """Extract triples for a single record according to ``mode``.

    - llm_based: LLM only. On LLM unavailability, falls back to schema only
      when allow_fallback is True; otherwise re-raises LLMUnavailableError.
    - schema_fallback: schema only.
    - both: LLM + schema, deduped.
    """
    if mode == "schema_fallback":
        return extract_triples_schema(record)

    if mode in ("llm_based", "both"):
        try:
            llm_triples = extract_triples_llm(record, model=model)
        except local_llm.LLMUnavailableError:
            if not allow_fallback:
                raise
            LOGGER.warning("LLM unavailable; using schema fallback for %s",
                           _record_to_dict(record).get("instance_id"))
            return extract_triples_schema(record)

        if mode == "both":
            return _dedupe(llm_triples + extract_triples_schema(record))

        # llm_based: if the LLM produced nothing parseable, optionally fall back.
        if not llm_triples and allow_fallback:
            LOGGER.warning("LLM produced no triples; using schema fallback for %s",
                           _record_to_dict(record).get("instance_id"))
            return extract_triples_schema(record)
        return llm_triples

    raise ValueError(f"Unknown extraction mode: {mode}")


def extract_triples_for_records(records: list[dict],
                                mode: str = "llm_based",
                                *,
                                allow_fallback: bool = False,
                                model: Optional[str] = None,
                                progress: bool = True) -> list[CulturalTriple]:
    """Extract triples across many records."""
    iterator = records
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(records, desc=f"Extracting triples ({mode})")
        except Exception:  # noqa: BLE001
            iterator = records

    all_triples: list[CulturalTriple] = []
    for rec in iterator:
        all_triples.extend(
            extract_triples(rec, mode, allow_fallback=allow_fallback, model=model)
        )
    return all_triples
