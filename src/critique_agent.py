"""Cultural KG critique agent.

MAIN method: local LLM critique via Ollama. It inspects extracted triples for a
record and reports issues + cautious suggested bridge triples (generated=true,
low confidence) + triples to flag.

FALLBACK: a rule-based critique used only when the LLM is unavailable (and only
when fallback is allowed). It performs the same checks heuristically.

This module also builds the refined triple set by combining original valid
triples with critique suggestions, keeping (but annotating) flagged triples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import config, local_llm, prompts
from .layer_classifier import validate_triple_layers
from .schemas import (
    CritiqueIssue,
    CritiqueResult,
    CulturalTriple,
    FlaggedTriple,
    LayerLabel,
)
from .utils import get_logger, normalize_text, stable_id, truncate

LOGGER = get_logger("critique_agent")

_LAYER_LABEL_TEXT = {
    "LAYER_1_SURFACE",
    "LAYER_2_SOCIAL",
    "LAYER_3_DEEP_VALUE",
    "UNKNOWN",
}

_TOPIC_METADATA_PREDICATES = {
    "has cultural topic",
    "has topic",
    "topic",
    "is topic",
    "belongs to topic",
    "has category",
    "category",
    "has cultural category",
}

_TOPIC_CATEGORY_LABELS = {
    "community and identity",
    "consumer behavior",
    "cultural and environmental appreciation",
    "cultural exchange",
    "cultural traditions and festivals",
    "drinking and alcohol",
    "family dynamics",
    "family traditions and heritage",
    "finance and economy",
    "food and dining",
    "health and hygiene",
    "household and daily life",
    "lifestyles",
    "miscellaneous",
    "social interactions",
    "social norms and etiquette",
    "work-life balance",
}

_VALUE_KEYWORDS = {
    "hospitality": "hospitality",
    "guest": "hospitality",
    "host": "hospitality",
    "gratitude": "gratitude",
    "thank": "gratitude",
    "appreciation": "appreciation",
    "appreciate": "appreciation",
    "reciprocat": "reciprocity",
    "gift": "reciprocity",
    "respect": "respect",
    "polite": "politeness",
    "etiquette": "politeness",
    "generous": "generosity",
    "generosity": "generosity",
    "care": "care",
    "family": "family connection",
    "clean": "cleanliness",
    "hygien": "cleanliness",
    "comfort": "comfort",
    "welcome": "welcoming",
}


def _triple_to_compact(t: CulturalTriple) -> dict:
    return {
        "triple_id": t.triple_id,
        "subject": t.subject,
        "predicate": t.predicate,
        "object": t.object,
        "subject_layer": t.subject_layer.value,
        "predicate_layer": t.predicate_layer.value,
        "object_layer": t.object_layer.value,
        "confidence": t.confidence,
        "generated": t.generated,
        "evidence": t.evidence,
    }


def _label_key(text: str) -> str:
    return normalize_text(text).strip().upper().replace(" ", "_").replace("-", "_")


def _is_layer_placeholder(text: str) -> bool:
    return _label_key(text) in _LAYER_LABEL_TEXT


def _looks_like_topic_label(text: str) -> bool:
    return normalize_text(text).lower() in _TOPIC_CATEGORY_LABELS


def _is_invalid_label_triple(t: CulturalTriple) -> bool:
    """Return True when subject/object is a schema layer label, not an entity."""
    return _is_layer_placeholder(t.subject) or _is_layer_placeholder(t.object)


def _is_topic_metadata_triple(t: CulturalTriple) -> bool:
    """Return True for dataset-topic triples miscast as cultural values."""
    predicate = normalize_text(t.predicate).lower()
    obj = normalize_text(t.object).lower()

    if predicate in _TOPIC_METADATA_PREDICATES:
        return True

    if _looks_like_topic_label(obj):
        return True

    if t.object_layer == LayerLabel.LAYER_3_DEEP_VALUE and (
        "topic" in predicate or "category" in predicate
    ):
        return True

    return False


def _is_supported_value_suggestion(t: CulturalTriple) -> bool:
    """Reject critique suggestions that only restate metadata/schema labels as values."""
    if _is_invalid_label_triple(t):
        return False
    if _is_topic_metadata_triple(t):
        return False
    if t.object_layer == LayerLabel.LAYER_3_DEEP_VALUE and _looks_like_topic_label(t.object):
        return False
    if t.subject_layer == LayerLabel.LAYER_3_DEEP_VALUE and _looks_like_topic_label(t.subject):
        return False
    return True


def _build_suggested_triple(rec: dict, fields: dict) -> CulturalTriple:
    subject = normalize_text(fields.get("subject"))
    predicate = normalize_text(fields.get("predicate"))
    obj = normalize_text(fields.get("object"))
    # Critique suggestions must be generated=true and capped below 0.5 unless
    # directly supported by source text.
    conf = float(fields.get("confidence", 0.4) or 0.4)
    conf = min(conf, 0.49)
    triple_id = stable_id(
        rec.get("instance_id", ""), subject, predicate, obj, "critique",
        prefix="tc_",
    )
    return CulturalTriple(
        triple_id=triple_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        subject_layer=LayerLabel.coerce(fields.get("subject_layer")),
        predicate_layer=LayerLabel.coerce(fields.get("predicate_layer")),
        object_layer=LayerLabel.coerce(fields.get("object_layer")),
        confidence=conf,
        source_text=rec.get("source_text") or "",
        dataset_id=rec.get("dataset_id", "culturebank"),
        instance_id=rec.get("instance_id", ""),
        raw_fields=rec.get("raw", {}) or {},
        extraction_method="critique_suggested",
        generated=True,
        evidence=fields.get("evidence"),
        notes=fields.get("notes") or "Cautious critique suggestion (interpretation).",
    )


def _infer_value_from_record(record: dict, triples: list[CulturalTriple]) -> Optional[str]:
    """Very conservative fallback value cue from record/triple text."""
    parts = [
        record.get("context"),
        record.get("goal"),
        record.get("actor_behavior"),
        record.get("recipient_behavior"),
        record.get("source_text"),
    ]
    parts.extend(f"{t.subject} {t.predicate} {t.object}" for t in triples)
    blob = normalize_text(" ".join(p for p in parts if p)).lower()
    for key, value in _VALUE_KEYWORDS.items():
        if key in blob:
            return value
    return None


def _best_surface_triple(triples: list[CulturalTriple]) -> Optional[CulturalTriple]:
    for t in triples:
        if t.object_layer == LayerLabel.LAYER_1_SURFACE and not _is_topic_metadata_triple(t):
            return t
    for t in triples:
        if t.subject_layer == LayerLabel.LAYER_1_SURFACE and not _is_topic_metadata_triple(t):
            return t
    return None


# --- LLM critique (MAIN) ---------------------------------------------------

def _format_critique_prompt(record: dict, triples: list[CulturalTriple]) -> str:
    record_json = truncate(json.dumps(record, ensure_ascii=False, default=str), 2500)
    triples_json = json.dumps([_triple_to_compact(t) for t in triples],
                              ensure_ascii=False)
    return prompts.CRITIQUE_USER.format(
        record_json=record_json,
        triples_json=truncate(triples_json, 4000),
    )


def _parse_critique_response(record: dict, triples: list[CulturalTriple],
                             parsed: dict) -> CritiqueResult:
    rec_id = record.get("instance_id", "")
    issues = []
    for it in parsed.get("issues", []) or []:
        if isinstance(it, dict):
            issues.append(CritiqueIssue(
                issue_type=normalize_text(it.get("issue_type")) or "unspecified",
                severity=normalize_text(it.get("severity")) or "low",
                description=normalize_text(it.get("description")),
                affected_triple_ids=[str(x) for x in (it.get("affected_triple_ids") or [])],
            ))

    flagged = []
    flagged_ids: set[str] = set()
    for ft in parsed.get("triples_to_flag", []) or []:
        if isinstance(ft, dict) and ft.get("triple_id"):
            tid = str(ft.get("triple_id"))
            flagged_ids.add(tid)
            flagged.append(FlaggedTriple(
                triple_id=tid,
                reason=normalize_text(ft.get("reason")),
            ))

    for t in triples:
        if _is_topic_metadata_triple(t) and t.triple_id not in flagged_ids:
            flagged_ids.add(t.triple_id)
            flagged.append(FlaggedTriple(
                triple_id=t.triple_id,
                reason="Dataset topic/category metadata is not a real L3 deep value.",
            ))
            issues.append(CritiqueIssue(
                issue_type="topic_metadata_as_value",
                severity="high",
                description="Topic/category metadata was labeled as a deep value.",
                affected_triple_ids=[t.triple_id],
            ))
        if _is_invalid_label_triple(t) and t.triple_id not in flagged_ids:
            flagged_ids.add(t.triple_id)
            flagged.append(FlaggedTriple(
                triple_id=t.triple_id,
                reason="Subject/object is a schema layer label, not a cultural entity.",
            ))
            issues.append(CritiqueIssue(
                issue_type="layer_label_as_entity",
                severity="high",
                description="A schema layer label was used as subject/object text.",
                affected_triple_ids=[t.triple_id],
            ))

    suggested = []
    for st in parsed.get("suggested_triples", []) or []:
        if isinstance(st, dict) and normalize_text(st.get("subject")) and normalize_text(st.get("object")):
            st = validate_triple_layers(st)
            candidate = _build_suggested_triple(record, st)
            if not _is_supported_value_suggestion(candidate):
                continue
            suggested.append(candidate)
            if len(suggested) >= config.MAX_CRITIQUE_SUGGESTIONS:
                break

    return CritiqueResult(
        record_id=rec_id,
        original_triple_ids=[t.triple_id for t in triples],
        issues=issues,
        suggested_triples=suggested,
        triples_to_flag=flagged,
        summary=normalize_text(parsed.get("summary")),
        method="llm_based",
    )


def critique_record_llm(record: dict,
                        triples: list[CulturalTriple],
                        *,
                        model: Optional[str] = None) -> CritiqueResult:
    """Critique one record's triples using the local LLM.

    Raises local_llm.LLMUnavailableError if the server is unreachable.
    """
    user_prompt = _format_critique_prompt(record, triples)
    parsed = local_llm.generate_json(
        user_prompt,
        system_prompt=prompts.CRITIQUE_SYSTEM,
        model=model,
    )
    rec_id = record.get("instance_id", "")
    if not parsed or not isinstance(parsed, dict):
        LOGGER.warning("Critique LLM returned no valid JSON for %s", rec_id)
        return CritiqueResult(record_id=rec_id,
                              original_triple_ids=[t.triple_id for t in triples],
                              summary="No valid critique JSON returned.",
                              method="llm_based")
    return _parse_critique_response(record, triples, parsed)


def critique_record_llm_backend(record: dict,
                                triples: list[CulturalTriple],
                                backend) -> CritiqueResult:
    """Critique one record using a pluggable LLM backend."""
    user_prompt = _format_critique_prompt(record, triples)
    text = backend.generate_one(user_prompt, system_prompt=prompts.CRITIQUE_SYSTEM)
    parsed = local_llm.safe_parse_json(text)
    rec_id = record.get("instance_id", "")
    if not parsed or not isinstance(parsed, dict):
        LOGGER.warning("Critique backend returned no valid JSON for %s", rec_id)
        return CritiqueResult(record_id=rec_id,
                              original_triple_ids=[t.triple_id for t in triples],
                              summary="No valid critique JSON returned.",
                              method="llm_based")
    return _parse_critique_response(record, triples, parsed)


# --- Rule-based critique (FALLBACK only) -----------------------------------

def critique_record_rule_based(record: dict,
                               triples: list[CulturalTriple]) -> CritiqueResult:
    """Heuristic critique used only when the LLM is unavailable."""
    rec_id = record.get("instance_id", "")
    issues: list[CritiqueIssue] = []
    suggested: list[CulturalTriple] = []
    flagged: list[FlaggedTriple] = []

    usable_triples = [
        t for t in triples
        if not _is_topic_metadata_triple(t) and not _is_invalid_label_triple(t)
    ]

    for t in triples:
        if _is_topic_metadata_triple(t):
            flagged.append(FlaggedTriple(
                triple_id=t.triple_id,
                reason="Dataset topic/category metadata is not a real L3 deep value.",
            ))
            issues.append(CritiqueIssue(
                issue_type="topic_metadata_as_value",
                severity="high",
                description="Topic/category metadata was labeled as a deep value.",
                affected_triple_ids=[t.triple_id],
            ))
        if _is_invalid_label_triple(t):
            flagged.append(FlaggedTriple(
                triple_id=t.triple_id,
                reason="Subject/object is a schema layer label, not a cultural entity.",
            ))
            issues.append(CritiqueIssue(
                issue_type="layer_label_as_entity",
                severity="high",
                description="A schema layer label was used as subject/object text.",
                affected_triple_ids=[t.triple_id],
            ))

    layers_present = set()
    for t in usable_triples:
        layers_present.update({t.subject_layer, t.predicate_layer, t.object_layer})

    cg = normalize_text(record.get("cultural_group"))

    if LayerLabel.LAYER_2_SOCIAL not in layers_present:
        issues.append(CritiqueIssue(
            issue_type="missing_layer_2_social",
            severity="medium",
            description="No Layer 2 social-expectation triple present.",
            affected_triple_ids=[t.triple_id for t in usable_triples],
        ))
    if LayerLabel.LAYER_3_DEEP_VALUE not in layers_present:
        issues.append(CritiqueIssue(
            issue_type="missing_layer_3_value",
            severity="medium",
            description="No real Layer 3 deep-value triple present.",
            affected_triple_ids=[t.triple_id for t in usable_triples],
        ))
        surface = _best_surface_triple(usable_triples)
        value = _infer_value_from_record(record, usable_triples)
        if surface and value:
            subject = surface.object if surface.object_layer == LayerLabel.LAYER_1_SURFACE else surface.subject
            suggested.append(_build_suggested_triple(record, {
                "subject": subject,
                "predicate": "may express",
                "object": value,
                "subject_layer": "LAYER_1_SURFACE",
                "predicate_layer": "LAYER_3_DEEP_VALUE",
                "object_layer": "LAYER_3_DEEP_VALUE",
                "confidence": 0.35,
                "evidence": surface.triple_id,
                "notes": "Cautious value bridge inferred from record wording; not from Topic metadata.",
            }))

    # Surface practice without a social/value bridge.
    has_surface = any(LayerLabel.LAYER_1_SURFACE in
                      {t.subject_layer, t.object_layer} for t in usable_triples)
    has_bridge = any(LayerLabel.LAYER_2_SOCIAL in {t.subject_layer, t.predicate_layer, t.object_layer}
                     or LayerLabel.LAYER_3_DEEP_VALUE in {t.subject_layer, t.predicate_layer, t.object_layer}
                     for t in usable_triples)
    if has_surface and not has_bridge:
        issues.append(CritiqueIssue(
            issue_type="surface_without_bridge",
            severity="high",
            description="Surface practices present without social/value mediation.",
            affected_triple_ids=[t.triple_id for t in usable_triples],
        ))

    if not cg:
        issues.append(CritiqueIssue(
            issue_type="behavior_without_cultural_group",
            severity="medium",
            description="Behaviors lack an associated cultural group.",
        ))

    return CritiqueResult(
        record_id=rec_id,
        original_triple_ids=[t.triple_id for t in triples],
        issues=issues,
        suggested_triples=suggested,
        triples_to_flag=flagged,
        summary="Rule-based fallback critique.",
        method="rule_based",
    )


# --- Dispatcher ------------------------------------------------------------

def critique_record(record: dict,
                    triples: list[CulturalTriple],
                    mode: str = "llm_based",
                    *,
                    allow_fallback: bool = False,
                    model: Optional[str] = None) -> CritiqueResult:
    if mode == "rule_based":
        return critique_record_rule_based(record, triples)
    try:
        return critique_record_llm(record, triples, model=model)
    except local_llm.LLMUnavailableError:
        if not allow_fallback:
            raise
        LOGGER.warning("Critique LLM unavailable; rule-based fallback for %s",
                       record.get("instance_id"))
        return critique_record_rule_based(record, triples)


# --- Refined triple builder ------------------------------------------------

def build_refined_triples(records_by_id: dict[str, dict],
                          triples_by_record: dict[str, list[CulturalTriple]],
                          critiques: list[CritiqueResult]) -> list[CulturalTriple]:
    """Combine original valid triples with critique suggestions.

    - Keep original non-metadata triples (annotate flagged ones in notes).
    - Drop dataset topic/category triples that were miscast as values.
    - Drop triples whose subject/object is a schema layer label.
    - Add suggested triples (generated=true) if they are not topic metadata.
    - Deduplicate exact (subject, predicate, object) per instance, keeping the
      grounded (non-generated) version when there is a conflict.
    """
    _ = records_by_id  # Kept for API compatibility and future record-aware checks.
    flagged_map: dict[str, str] = {}
    suggestions_by_record: dict[str, list[CulturalTriple]] = {}
    for c in critiques:
        for fl in c.triples_to_flag:
            flagged_map[fl.triple_id] = fl.reason
        suggestions_by_record.setdefault(c.record_id, []).extend(c.suggested_triples)

    refined: dict[str, CulturalTriple] = {}

    def _key(t: CulturalTriple) -> str:
        return stable_id(t.instance_id, t.subject.lower(),
                         t.predicate.lower(), t.object.lower())

    # Original triples first (grounded take priority), but remove topic/category
    # metadata triples and schema-label nodes because they corrupt pathways.
    for rec_id, triples in triples_by_record.items():
        for t in triples:
            t2 = t.model_copy(deep=True)
            if _is_topic_metadata_triple(t2):
                LOGGER.debug("Dropping topic metadata triple from refined KG: %s", t2.triple_id)
                continue
            if _is_invalid_label_triple(t2):
                LOGGER.debug("Dropping layer-placeholder triple from refined KG: %s", t2.triple_id)
                continue
            if t2.triple_id in flagged_map:
                reason = flagged_map[t2.triple_id]
                note = f"[FLAGGED: {reason}]"
                t2.notes = (t2.notes + " " + note) if t2.notes else note
            refined[_key(t2)] = t2

    # Critique suggestions (do not overwrite grounded triples).
    for rec_id, suggestions in suggestions_by_record.items():
        for s in suggestions:
            if not _is_supported_value_suggestion(s):
                continue
            k = _key(s)
            if k not in refined:
                refined[k] = s

    return list(refined.values())


def _chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def critique_records_batched_resumable(
    records: list[dict],
    triples_by_record: dict[str, list[CulturalTriple]],
    backend,
    *,
    state_path: str,
    critiques_output: str,
    cache_dir: str,
    batch_size: int = 32,
    allow_fallback: bool = False,
    resume: bool = True,
) -> tuple[list[CritiqueResult], list[CulturalTriple]]:
    """Resumable batched critique; returns critiques and refined triple list."""
    from .storage import (
        append_jsonl,
        generate_with_cache,
        mark_completed,
        prepare_stage_state,
        read_completed_ids,
        save_corpus_state,
    )

    state = prepare_stage_state(state_path, "critique", resume=resume)
    records_by_id = {r.get("instance_id", ""): r for r in records}
    completed = read_completed_ids(state, "critique")
    pending = [r for r in records if r.get("instance_id") and r["instance_id"] not in completed]

    all_critiques: list[CritiqueResult] = []
    if resume and critiques_output:
        from .storage import read_jsonl_list
        for row in read_jsonl_list(critiques_output):
            try:
                all_critiques.append(CritiqueResult(**row))
            except Exception:  # noqa: BLE001
                continue

    if not pending:
        LOGGER.info("Critique: all %d records already complete.", len(records))
        refined = build_refined_triples(records_by_id, triples_by_record, all_critiques)
        return all_critiques, refined

    critiques_path = Path(critiques_output)
    if not resume or not critiques_path.exists():
        critiques_path.parent.mkdir(parents=True, exist_ok=True)
        critiques_path.write_text("", encoding="utf-8")

    LOGGER.info("Critique: %d pending / %d total.", len(pending), len(records))

    for batch in _chunked(pending, batch_size):
        user_prompts = []
        batch_recs = []
        batch_triples = []
        for rec in batch:
            iid = rec.get("instance_id", "")
            triples = triples_by_record.get(iid, [])
            user_prompts.append(_format_critique_prompt(rec, triples))
            batch_recs.append(rec)
            batch_triples.append(triples)

        outputs = generate_with_cache(
            backend, "critique", user_prompts, prompts.CRITIQUE_SYSTEM, cache_dir)

        batch_critiques: list[CritiqueResult] = []
        done_ids: list[str] = []
        for rec, triples, text in zip(batch_recs, batch_triples, outputs):
            parsed = local_llm.safe_parse_json(text)
            if parsed and isinstance(parsed, dict):
                cr = _parse_critique_response(rec, triples, parsed)
            elif allow_fallback:
                cr = critique_record_rule_based(rec, triples)
            else:
                cr = CritiqueResult(
                    record_id=rec.get("instance_id", ""),
                    original_triple_ids=[t.triple_id for t in triples],
                    summary="Critique parse failed.",
                    method="llm_based",
                )
            batch_critiques.append(cr)
            done_ids.append(rec.get("instance_id", ""))

        append_jsonl(critiques_path, [c.model_dump() for c in batch_critiques])
        all_critiques.extend(batch_critiques)
        mark_completed(state, "critique", done_ids)
        save_corpus_state(state_path, state)

    refined = build_refined_triples(records_by_id, triples_by_record, all_critiques)
    return all_critiques, refined
