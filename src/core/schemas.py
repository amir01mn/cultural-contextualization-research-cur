"""Pydantic data models shared by every pipeline stage.

Every triple preserves provenance: dataset_id, instance_id, source_text, raw
fields, extraction method, confidence, and a generated flag. This is essential
for the research question because we score answers in part on whether they make
use of grounded provenance vs. generated (inferred) triples.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class LayerLabel(str, Enum):
    """Cultural abstraction layers used throughout the graph."""

    LAYER_1_SURFACE = "LAYER_1_SURFACE"
    LAYER_2_SOCIAL = "LAYER_2_SOCIAL"
    LAYER_3_DEEP_VALUE = "LAYER_3_DEEP_VALUE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def coerce(cls, value: Any) -> "LayerLabel":
        """Best-effort coercion of an arbitrary value to a LayerLabel."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        key = str(value).strip().upper().replace(" ", "_").replace("-", "_")
        for member in cls:
            if key == member.value:
                return member
        # Accept a few common shorthands.
        aliases = {
            "L1": cls.LAYER_1_SURFACE,
            "L2": cls.LAYER_2_SOCIAL,
            "L3": cls.LAYER_3_DEEP_VALUE,
            "SURFACE": cls.LAYER_1_SURFACE,
            "SOCIAL": cls.LAYER_2_SOCIAL,
            "DEEP_VALUE": cls.LAYER_3_DEEP_VALUE,
            "VALUE": cls.LAYER_3_DEEP_VALUE,
        }
        return aliases.get(key, cls.UNKNOWN)


class CulturalRecord(BaseModel):
    """Normalized CultureBank-style record."""

    dataset_id: str = "culturebank"
    instance_id: str
    cultural_group: Optional[str] = None
    context: Optional[str] = None
    goal: Optional[str] = None
    actor: Optional[str] = None
    recipient: Optional[str] = None
    relation: Optional[str] = None
    actor_behavior: Optional[str] = None
    recipient_behavior: Optional[str] = None
    topic: Optional[str] = None
    agreement: Optional[str] = None
    source_text: Optional[str] = None
    raw: dict = Field(default_factory=dict)


class TripleElement(BaseModel):
    """A single subject/object element with its assigned layer."""

    text: str
    layer: LayerLabel = LayerLabel.UNKNOWN

    @field_validator("layer", mode="before")
    @classmethod
    def _coerce_layer(cls, v: Any) -> LayerLabel:
        return LayerLabel.coerce(v)


class CulturalTriple(BaseModel):
    """A provenance-preserving cultural knowledge graph triple."""

    triple_id: str
    subject: str
    predicate: str
    object: str
    subject_layer: LayerLabel = LayerLabel.UNKNOWN
    predicate_layer: LayerLabel = LayerLabel.UNKNOWN
    object_layer: LayerLabel = LayerLabel.UNKNOWN
    confidence: float = 0.5
    source_text: str = ""
    dataset_id: str = "culturebank"
    instance_id: str = ""
    raw_fields: dict = Field(default_factory=dict)
    extraction_method: str = "llm_based"
    generated: bool = False
    evidence: Optional[str] = None
    notes: Optional[str] = None
    # True when LLM extraction failed for this record and deterministic schema
    # triples were substituted. Such triples stay in the flat retrieval corpus
    # (so S1/S2 remain comparable) but are excluded from entity resolution and
    # pathway construction, which require real LLM-derived layer structure.
    llm_failed: bool = False

    @field_validator("subject_layer", "predicate_layer", "object_layer", mode="before")
    @classmethod
    def _coerce_layers(cls, v: Any) -> LayerLabel:
        return LayerLabel.coerce(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_conf(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, f))

    def text_blob(self) -> str:
        """Concatenated searchable text used by the retriever."""
        parts = [
            self.subject,
            self.predicate,
            self.object,
            self.evidence or "",
            self.notes or "",
            self.source_text,
        ]
        return " ".join(p for p in parts if p)


class CritiqueIssue(BaseModel):
    issue_type: str
    severity: str = "low"
    description: str = ""
    affected_triple_ids: list[str] = Field(default_factory=list)


class FlaggedTriple(BaseModel):
    triple_id: str
    reason: str = ""


class CritiqueResult(BaseModel):
    """Output of the critique agent for a single record."""

    record_id: str
    original_triple_ids: list[str] = Field(default_factory=list)
    issues: list[CritiqueIssue] = Field(default_factory=list)
    suggested_triples: list[CulturalTriple] = Field(default_factory=list)
    triples_to_flag: list[FlaggedTriple] = Field(default_factory=list)
    summary: str = ""
    method: str = "llm_based"


class EvaluationExample(BaseModel):
    """A generated cultural scenario question used for evaluation."""

    example_id: str
    source_instance_id: str
    question: str
    cultural_group: Optional[str] = None
    expected_keywords: list[str] = Field(default_factory=list)
    avoid_keywords: list[str] = Field(default_factory=list)
    target_layers: list[LayerLabel] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("target_layers", mode="before")
    @classmethod
    def _coerce_target_layers(cls, v: Any) -> list[LayerLabel]:
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        return [LayerLabel.coerce(x) for x in v]


class SystemAnswer(BaseModel):
    """One system's answer to one scenario question, with its score."""

    system: str  # baseline_prompt_only | raw_kg_augmented | critique_refined_kg_augmented
    answer: str = ""
    score: float = 0.0
    score_breakdown: dict = Field(default_factory=dict)
    retrieved_triple_ids: list[str] = Field(default_factory=list)
    used_generated_ratio: float = 0.0


class EvaluationResult(BaseModel):
    """All systems' answers + scores for a single scenario example."""

    example_id: str
    question: str
    cultural_group: Optional[str] = None
    answers: dict[str, SystemAnswer] = Field(default_factory=dict)


# ===========================================================================
# CulturalBench multiple-choice evaluation models
# ===========================================================================

class MCQVariant(str, Enum):
    """CulturalBench has two formats with different scoring."""

    EASY = "easy"  # 4-way single-correct MCQ
    HARD = "hard"  # per-option True/False; a group is correct iff all items are


class MCQQuestion(BaseModel):
    """A normalized CulturalBench item.

    Easy: a full 4-way question (``options`` A-D, ``answer_key`` is the correct
    letter).
    Hard: a single (question, option) judgment. Multiple Hard items share a
    ``group_id`` (the underlying question); the group counts as correct only if
    every item's True/False judgment is right. ``option_text`` is the candidate
    answer and ``answer_bool`` is its gold truth value.
    """

    question_id: str
    variant: MCQVariant
    group_id: str
    question: str
    cultural_group: Optional[str] = None
    region: Optional[str] = None

    # Easy fields:
    options: dict[str, str] = Field(default_factory=dict)
    answer_key: Optional[str] = None

    # Hard fields:
    option_text: Optional[str] = None
    answer_bool: Optional[bool] = None

    raw: dict = Field(default_factory=dict)

    @field_validator("variant", mode="before")
    @classmethod
    def _coerce_variant(cls, v: Any) -> "MCQVariant":
        if isinstance(v, MCQVariant):
            return v
        key = str(v).strip().lower()
        if "hard" in key:
            return MCQVariant.HARD
        return MCQVariant.EASY


class MCQSystemAnswer(BaseModel):
    """One system's answer to one MCQ item."""

    system: str  # S0_no_kg | S1_raw_text_rag | S2_flat_kg | S3_layered_kg | S4_*
    raw_output: str = ""
    predicted: Optional[str] = None  # "A".."D" (easy) or "true"/"false" (hard)
    correct: bool = False
    retrieved_ids: list[str] = Field(default_factory=list)
    completeness: Optional[str] = None  # PathwayCompleteness bucket for mechanism analysis
    completeness_score: Optional[float] = None
    shared_node_available: Optional[bool] = None


class MCQItemResult(BaseModel):
    """All systems' answers for a single MCQ item."""

    question_id: str
    group_id: str
    variant: MCQVariant
    cultural_group: Optional[str] = None
    answers: dict[str, MCQSystemAnswer] = Field(default_factory=dict)


# ===========================================================================
# Phase 2: entity resolution, alignment, pathways
# ===========================================================================

class PathwayCompleteness(str, Enum):
    """Pathway completeness bucket per RESEARCH_DESIGN.md §3.3."""

    FULL = "FULL"
    SOCIAL = "SOCIAL"
    SURFACE = "SURFACE"
    NONE = "NONE"


class CanonicalEntity(BaseModel):
    """A canonical graph node after entity resolution."""

    entity_id: str
    canonical_label: str
    layer: LayerLabel = LayerLabel.UNKNOWN
    cultures: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    source_instance_ids: list[str] = Field(default_factory=list)
    is_shared_abstraction: bool = False

    @field_validator("layer", mode="before")
    @classmethod
    def _coerce_layer(cls, v: Any) -> LayerLabel:
        return LayerLabel.coerce(v)


class CrossCulturalAlignment(BaseModel):
    """Links refinements from two cultures under a shared abstraction pivot."""

    alignment_id: str
    abstraction_node_id: str
    culture_a: str
    culture_b: str
    entity_ids: list[str] = Field(default_factory=list)
    entity_label_a: str = ""
    entity_label_b: str = ""
    similarity: float = 0.0
    verified: bool = False
    verification_rationale: str = ""

    @field_validator("similarity", mode="before")
    @classmethod
    def _clamp_sim(cls, v: Any) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))


class PathwayNodeRef(BaseModel):
    entity_id: str
    label: str
    layer: LayerLabel = LayerLabel.UNKNOWN

    @field_validator("layer", mode="before")
    @classmethod
    def _coerce_layer(cls, v: Any) -> LayerLabel:
        return LayerLabel.coerce(v)


class CulturalPathway(BaseModel):
    """An abstraction→refinement pathway (L3→L2→L1 when complete)."""

    pathway_id: str
    culture: str
    nodes: list[PathwayNodeRef] = Field(default_factory=list)
    triple_ids: list[str] = Field(default_factory=list)
    completeness: PathwayCompleteness = PathwayCompleteness.NONE
    has_shared_pivot: bool = False
    shared_pivot_id: Optional[str] = None
    # Relation label for each hop (len == len(nodes) - 1). Without these a
    # rendered pathway is just a list of nouns ("Japanese food culture ->
    # specific brands"), which is unusable as MCQ evidence.
    predicates: list[str] = Field(default_factory=list)
    # Whether any contributing triple was inferred rather than dataset-grounded.
    generated: bool = False

    @field_validator("completeness", mode="before")
    @classmethod
    def _coerce_completeness(cls, v: Any) -> PathwayCompleteness:
        if isinstance(v, PathwayCompleteness):
            return v
        key = str(v).strip().upper()
        for member in PathwayCompleteness:
            if key == member.value:
                return member
        return PathwayCompleteness.NONE


class CorpusBuildState(BaseModel):
    """Checkpoint state for resumable corpus construction."""

    extraction_completed: list[str] = Field(default_factory=list)
    critique_completed: list[str] = Field(default_factory=list)
    updated_at: Optional[str] = None
    # Records whose LLM output could not be parsed. Tracked separately from
    # *_completed so a resumed job retries them instead of treating a silent
    # failure as success.
    extraction_failed: list[str] = Field(default_factory=list)
    critique_failed: list[str] = Field(default_factory=list)
