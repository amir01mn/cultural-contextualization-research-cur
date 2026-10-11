"""The CulturalBench answering systems (S0-S4).

All systems share the same base LLM (via the pluggable backend), the same prompt
scaffold, and the same retrieval budget (top_k). The ONLY thing that differs is
the injected cultural evidence, so any accuracy difference is attributable to the
evidence/structure:

    S0 No-KG          : question + options only (closed-book).
    S1 Raw-text RAG   : retrieved raw CultureBank passages.
    S2 Flat KG        : retrieved triples as a bag, layer labels stripped.
    S3 Layered KG     : retrieved triples WITH abstraction-layer labels.
    S4 Pathway KG     : retrieved abstraction→refinement pathways (Phase 3).
    S4 ablations      : triple-retrieval on refined corpus; no shared pivots.

Prompts are built for every question up front and sent to ``backend.generate`` in
one batch, so vLLM can process them efficiently.
"""

from __future__ import annotations

import re
from typing import Optional

from ..core import config
from ..llm import prompts
from ..retrieval.pathway_retriever import PathwayRetriever
from ..retrieval.retrieval_ranking import culture_match_score
from ..retrieval.retriever import TextRetriever, TripleRetriever
from ..core.schemas import (
    CulturalPathway,
    CulturalTriple,
    LayerLabel,
    MCQQuestion,
    MCQSystemAnswer,
    MCQVariant,
    PathwayCompleteness,
)
from ..core.utils import get_logger, normalize_text, truncate

LOGGER = get_logger("mcq_systems")

SYSTEM_S0 = "S0_no_kg"
SYSTEM_S1 = "S1_raw_text_rag"
SYSTEM_S2 = "S2_flat_kg"
SYSTEM_S3 = "S3_layered_kg"
SYSTEM_S4 = "S4_abstraction_refinement_kg"
SYSTEM_S4_TRIPLE = "S4_triple_retrieval"
SYSTEM_S4_NO_SHARED = "S4_no_shared_pivot"

ALL_SYSTEMS = [
    SYSTEM_S0, SYSTEM_S1, SYSTEM_S2, SYSTEM_S3,
    SYSTEM_S4, SYSTEM_S4_TRIPLE, SYSTEM_S4_NO_SHARED,
]
KG_SYSTEMS = {SYSTEM_S2, SYSTEM_S3, SYSTEM_S4, SYSTEM_S4_TRIPLE, SYSTEM_S4_NO_SHARED}
S4_SYSTEMS = {SYSTEM_S4, SYSTEM_S4_TRIPLE, SYSTEM_S4_NO_SHARED}

_COMPLETENESS_WEIGHT = {
    PathwayCompleteness.FULL: 1.0,
    PathwayCompleteness.SOCIAL: 0.85,
    PathwayCompleteness.SURFACE: 0.7,
    PathwayCompleteness.NONE: 0.5,
}

_LAYER_SCORE = {
    LayerLabel.LAYER_1_SURFACE: 1,
    LayerLabel.LAYER_2_SOCIAL: 2,
    LayerLabel.LAYER_3_DEEP_VALUE: 3,
}


class RetrievalContext:
    """Holds the corpus indexes shared by the KG/RAG systems."""

    def __init__(self,
                 passages: Optional[list[dict]] = None,
                 triples: Optional[list[CulturalTriple]] = None,
                 refined_triples: Optional[list[CulturalTriple]] = None,
                 pathway_retriever: Optional[PathwayRetriever] = None):
        self.passage_retriever = TextRetriever(passages or []) if passages else None
        self.triple_retriever = TripleRetriever(triples or []) if triples else None
        self.refined_triple_retriever = (
            TripleRetriever(refined_triples or []) if refined_triples else None
        )
        self.pathway_retriever = pathway_retriever
        LOGGER.info(
            "RetrievalContext: passages=%s raw_triples=%s refined_triples=%s pathways=%d",
            self.passage_retriever.backend if self.passage_retriever else "none",
            self.triple_retriever.backend if self.triple_retriever else "none",
            self.refined_triple_retriever.backend if self.refined_triple_retriever else "none",
            len(pathway_retriever.pathways) if pathway_retriever else 0,
        )
        # Optional evidence masking (eval v2 held-out). None = no masking, i.e.
        # the original behaviour. Set via ``set_mask``.
        self.masked_instance_ids: Optional[frozenset] = None
        self.passage_mask = None
        self.triple_mask = None
        self.refined_triple_mask = None
        self.pathway_mask = None
        self._triple_to_instance: Optional[dict[str, str]] = None

    def triple_to_instance(self) -> dict[str, str]:
        """triple_id -> instance_id over the raw and refined triple indexes."""
        if self._triple_to_instance is None:
            m: dict[str, str] = {}
            for r in (self.triple_retriever, self.refined_triple_retriever):
                if r is not None:
                    for t in r.triples:
                        m.setdefault(t.triple_id, t.instance_id)
            self._triple_to_instance = m
        return self._triple_to_instance

    def set_mask(self, instance_ids=None) -> None:
        """Exclude every item derived from these CultureBank ``instance_id``s
        from all evidence (passages, raw/refined triples, and any pathway with
        a masked triple). ``None`` or empty clears the mask."""
        if not instance_ids:
            self.masked_instance_ids = None
            self.passage_mask = self.triple_mask = None
            self.refined_triple_mask = self.pathway_mask = None
            return
        ids = frozenset(instance_ids)
        self.masked_instance_ids = ids
        self.passage_mask = (self.passage_retriever.mask_for_instance_ids(ids)
                             if self.passage_retriever else None)
        self.triple_mask = (self.triple_retriever.mask_for_instance_ids(ids)
                            if self.triple_retriever else None)
        self.refined_triple_mask = (
            self.refined_triple_retriever.mask_for_instance_ids(ids)
            if self.refined_triple_retriever else None)
        self.pathway_mask = (
            self.pathway_retriever.mask_for_instance_ids(ids, self.triple_to_instance())
            if self.pathway_retriever and self.pathway_retriever.pathways else None)
        LOGGER.info(
            "Evidence mask set: %d instance_ids -> passages=%s triples=%s refined=%s pathways=%s",
            len(ids),
            *(int(m.sum()) if m is not None else 0 for m in (
                self.passage_mask, self.triple_mask,
                self.refined_triple_mask, self.pathway_mask)),
        )

    def mask_for_retriever(self, retriever):
        """The active exclude mask for ``retriever`` (None when unmasked)."""
        if self.masked_instance_ids is None or retriever is None:
            return None
        if retriever is self.passage_retriever:
            return self.passage_mask
        if retriever is self.triple_retriever:
            return self.triple_mask
        if retriever is self.refined_triple_retriever:
            return self.refined_triple_mask
        if retriever is self.pathway_retriever:
            return self.pathway_mask
        return None


def _mask_kw(ctx: "RetrievalContext", retriever) -> dict:
    """``exclude_mask`` kwarg only when a mask is active (keeps old call shape)."""
    mask = getattr(ctx, "mask_for_retriever", lambda _r: None)(retriever)
    return {} if mask is None else {"exclude_mask": mask}


def _query_for(q: MCQQuestion) -> str:
    """Build the retrieval query for a question.

    Answer options are excluded by default. Concatenating all four options meant
    three of them were distractors, so retrieval was steered by text asserting
    things that are false - surfacing evidence that supports a wrong option.
    """
    parts = [q.question]
    if q.cultural_group:
        parts.append(q.cultural_group)
    if config.RETRIEVAL_QUERY_INCLUDES_OPTIONS:
        if q.variant == MCQVariant.EASY:
            parts.extend(q.options.values())
        elif q.option_text:
            parts.append(q.option_text)
    return " ".join(p for p in parts if p)


def _rank_pathways(hits: list[tuple[CulturalPathway, float]],
                   question_culture: Optional[str],
                   *,
                   exclude_shared: bool = False) -> list[tuple[CulturalPathway, float]]:
    ranked: list[tuple[CulturalPathway, float, float]] = []
    for pw, score in hits:
        if exclude_shared and pw.has_shared_pivot:
            continue
        w = _COMPLETENESS_WEIGHT.get(pw.completeness, 0.5)
        cm = culture_match_score(pw.culture, question_culture)
        if pw.has_shared_pivot and cm < 0.9:
            cm = max(cm, config.RETRIEVAL_SHARED_PIVOT_CULTURE_SCORE)
        elif cm < 0.9 and question_culture and config.RETRIEVAL_STRICT_CULTURE:
            continue
        bonus = 1.1 if pw.has_shared_pivot else 1.0
        ranked.append((pw, score * w * cm * bonus, cm))
    ranked.sort(key=lambda x: x[1], reverse=True)

    if question_culture:
        matched = [x for x in ranked if x[2] >= 0.9]
        cross = [x for x in ranked if x[2] < 0.9 and x[0].has_shared_pivot]
        rest = [x for x in ranked if x[2] < 0.9 and not x[0].has_shared_pivot]
        ranked = matched + cross + rest

    return [(pw, s) for pw, s, _ in ranked]


def _completeness_from_triples(triples: list[CulturalTriple]) -> tuple[str, float, bool]:
    """Layer coverage of a retrieved triple bag (NOT pathway completeness).

    Retrieved triples are an unconnected bag, so "all three layers appear
    somewhere in the bag" says nothing about whether a value -> norm -> practice
    chain was actually available. Reporting that as FULL is what made the
    mechanism analysis claim 461 SOCIAL-completeness questions for S4 when the
    entire corpus held only 19 SOCIAL pathways.

    Connectivity is therefore required: layers only count toward a bucket when a
    single triple links them, so this measures real linkage rather than
    co-occurrence. H2 cannot be tested against a co-occurrence metric.
    """
    linked_pairs: set[tuple[int, int]] = set()
    layer_values: set[int] = set()
    for triple in triples:
        endpoints = [
            _LAYER_SCORE.get(triple.subject_layer),
            _LAYER_SCORE.get(triple.object_layer),
        ]
        present = [v for v in endpoints if v is not None]
        layer_values.update(present)
        mid = _LAYER_SCORE.get(triple.predicate_layer)
        if mid is not None:
            layer_values.add(mid)
        if len(present) == 2 and present[0] != present[1]:
            linked_pairs.add((min(present), max(present)))
        # A triple whose predicate mediates between its endpoints links both.
        if mid is not None:
            for v in present:
                if v != mid:
                    linked_pairs.add((min(v, mid), max(v, mid)))

    has_32 = (2, 3) in linked_pairs
    has_21 = (1, 2) in linked_pairs
    if has_32 and has_21:
        bucket = PathwayCompleteness.FULL.value
    elif has_21 or has_32:
        bucket = PathwayCompleteness.SOCIAL.value
    elif 1 in layer_values:
        bucket = PathwayCompleteness.SURFACE.value
    else:
        bucket = PathwayCompleteness.NONE.value
    score = len(layer_values) / 3.0 if layer_values else 0.0
    return bucket, score, False


def _completeness_from_pathways(pathways: list[CulturalPathway]) -> tuple[str, float, bool]:
    if not pathways:
        return PathwayCompleteness.NONE.value, 0.0, False
    rank = {
        PathwayCompleteness.FULL: 4,
        PathwayCompleteness.SOCIAL: 3,
        PathwayCompleteness.SURFACE: 2,
        PathwayCompleteness.NONE: 1,
    }
    best = PathwayCompleteness.NONE
    shared = False
    for pw in pathways:
        if rank.get(pw.completeness, 0) > rank.get(best, 0):
            best = pw.completeness
        if pw.has_shared_pivot:
            shared = True
    score = rank.get(best, 1) / 4.0
    return best.value, score, shared


_LAYER_ROLE = {
    LayerLabel.LAYER_3_DEEP_VALUE: "value",
    LayerLabel.LAYER_2_SOCIAL: "social norm",
    LayerLabel.LAYER_1_SURFACE: "practice",
}


def _format_pathway(pw: CulturalPathway, index: int) -> str:
    """Render a pathway as a readable value -> norm -> practice chain.

    Includes the relation label on each hop. Emitting bare node labels turned a
    pathway into a list of nouns ("Japanese food culture -> specific brands and
    flavors"), which carries no assertion the model can reason with.
    """
    lines = [f"[{index}] Pathway ({pw.completeness.value}, {pw.culture}):"]
    for i, node in enumerate(pw.nodes):
        role = _LAYER_ROLE.get(node.layer, "context")
        lines.append(f"  {node.layer.value} ({role}): {node.label}")
        if i < len(pw.nodes) - 1:
            predicate = pw.predicates[i] if i < len(pw.predicates) else ""
            lines.append(f"    --[{predicate or 'relates to'}]-->")
    if pw.has_shared_pivot:
        lines.append("  [shared abstraction pivot across cultures]")
    if pw.generated:
        lines.append("  [inferred - treat as uncertain]")
    return "\n".join(lines)


def _options_block(q: MCQQuestion) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in q.options.items())


def _passage_evidence(ctx: RetrievalContext, query: str, top_k: int,
                      q: MCQQuestion) -> tuple[str, list[str], dict]:
    if not ctx.passage_retriever:
        return "", [], {}
    hits = ctx.passage_retriever.retrieve(
        query, top_k=top_k, question_culture=q.cultural_group,
        **_mask_kw(ctx, ctx.passage_retriever))
    lines, ids = [], []
    for i, (doc, _score) in enumerate(hits, 1):
        text = truncate(doc.get("text", ""), 300)
        if text:
            lines.append(f"[{i}] {text}")
            ids.append(str(doc.get("passage_id", "")))
    meta = {
        "completeness": PathwayCompleteness.NONE.value,
        "completeness_score": 0.0,
        "shared_node_available": False,
    }
    return "\n".join(lines), ids, meta


def _flat_triple_evidence(ctx: RetrievalContext, query: str, top_k: int,
                          q: MCQQuestion,
                          retriever: Optional[TripleRetriever] = None) -> tuple[str, list[str], dict]:
    retriever = retriever or ctx.triple_retriever
    if not retriever:
        return "", [], {}
    hits = retriever.retrieve(
        query, top_k=top_k, question_culture=q.cultural_group,
        **_mask_kw(ctx, retriever))
    triples = [t for t, _ in hits]
    lines, ids = [], []
    for i, (t, _score) in enumerate(hits, 1):
        lines.append(f"[{i}] {t.subject} - {t.predicate} - {t.object}")
        ids.append(t.triple_id)
    bucket, score, shared = _completeness_from_triples(triples)
    meta = {"completeness": bucket, "completeness_score": score, "shared_node_available": shared}
    return "\n".join(lines), ids, meta


def _layered_triple_evidence(ctx: RetrievalContext, query: str, top_k: int,
                             q: MCQQuestion,
                             retriever: Optional[TripleRetriever] = None) -> tuple[str, list[str], dict]:
    retriever = retriever or ctx.refined_triple_retriever or ctx.triple_retriever
    if not retriever:
        return "", [], {}
    hits = retriever.retrieve(
        query, top_k=top_k, question_culture=q.cultural_group,
        **_mask_kw(ctx, retriever))
    triples = [t for t, _ in hits]
    lines, ids = [], []
    for i, (t, _score) in enumerate(hits, 1):
        lines.append(
            f"[{i}] ({t.subject_layer.value}) {t.subject} "
            f"-[{t.predicate} | {t.predicate_layer.value}]-> "
            f"({t.object_layer.value}) {t.object}"
        )
        ids.append(t.triple_id)
    bucket, score, shared = _completeness_from_triples(triples)
    meta = {"completeness": bucket, "completeness_score": score, "shared_node_available": shared}
    return "\n".join(lines), ids, meta


def _pathway_evidence(ctx: RetrievalContext, query: str, top_k: int,
                      q: MCQQuestion, *,
                      exclude_shared: bool = False) -> tuple[str, list[str], dict]:
    if not ctx.pathway_retriever:
        return "", [], {
            "completeness": PathwayCompleteness.NONE.value,
            "completeness_score": 0.0,
            "shared_node_available": False,
        }
    hits = ctx.pathway_retriever.retrieve(
        query, top_k=max(top_k * config.RETRIEVAL_CANDIDATE_MULTIPLIER, top_k),
        question_culture=q.cultural_group,
        **_mask_kw(ctx, ctx.pathway_retriever))
    ranked = _rank_pathways(hits, q.cultural_group, exclude_shared=exclude_shared)[:top_k]
    lines, ids, pathways = [], [], []
    for i, (pw, _score) in enumerate(ranked, 1):
        lines.append(_format_pathway(pw, i))
        ids.append(pw.pathway_id)
        pathways.append(pw)
    bucket, score, shared = _completeness_from_pathways(pathways)
    meta = {"completeness": bucket, "completeness_score": score, "shared_node_available": shared}
    return "\n".join(lines), ids, meta


def _evidence_for(system: str, ctx: Optional[RetrievalContext],
                  q: MCQQuestion, top_k: int) -> tuple[str, list[str], dict]:
    empty_meta = {
        "completeness": PathwayCompleteness.NONE.value,
        "completeness_score": 0.0,
        "shared_node_available": False,
    }
    if system == SYSTEM_S0 or ctx is None:
        return "", [], empty_meta
    query = _query_for(q)
    if system == SYSTEM_S1:
        body, ids, meta = _passage_evidence(ctx, query, top_k, q)
    elif system == SYSTEM_S2:
        body, ids, meta = _flat_triple_evidence(ctx, query, top_k, q)
    elif system == SYSTEM_S3:
        body, ids, meta = _layered_triple_evidence(ctx, query, top_k, q)
    elif system == SYSTEM_S4:
        body, ids, meta = _pathway_evidence(ctx, query, top_k, q)
    elif system == SYSTEM_S4_NO_SHARED:
        body, ids, meta = _pathway_evidence(ctx, query, top_k, q, exclude_shared=True)
    elif system == SYSTEM_S4_TRIPLE:
        retriever = ctx.refined_triple_retriever or ctx.triple_retriever
        body, ids, meta = _layered_triple_evidence(ctx, query, top_k, q, retriever=retriever)
    else:
        return "", [], empty_meta
    if not body:
        return "", ids, meta
    header = prompts.MCQ_EVIDENCE_HEADERS.get(system, "Retrieved evidence:")
    return f"{header}\n{body}\n\n", ids, meta


def evidence_for(system: str, ctx: Optional[RetrievalContext],
                 q: MCQQuestion, top_k: int) -> tuple[str, list[str], dict]:
    """Public: (evidence_block, retrieved_ids, meta) exactly as ``run_system`` uses."""
    return _evidence_for(system, ctx, q, top_k)


def build_prompt(system: str, q: MCQQuestion,
                 ctx: Optional[RetrievalContext], top_k: int) -> tuple[str, str, list[str], dict]:
    evidence_block, ids, meta = _evidence_for(system, ctx, q, top_k)
    sp, up = _prompt_from_evidence(q, evidence_block)
    return sp, up, ids, meta


def _prompt_from_evidence(q: MCQQuestion, evidence_block: str) -> tuple[str, str]:
    if q.variant == MCQVariant.EASY:
        user = prompts.MCQ_EASY_USER.format(
            evidence_block=evidence_block,
            question=q.question,
            options_block=_options_block(q),
        )
        return prompts.MCQ_SYSTEM, user
    user = prompts.MCQ_HARD_USER.format(
        evidence_block=evidence_block,
        question=q.question,
        option_text=q.option_text or "",
    )
    return prompts.MCQ_HARD_SYSTEM, user


_ANSWER_LINE = re.compile(r"ANSWER\s*[:\-]?\s*([A-Za-z]+)", re.IGNORECASE)


def parse_easy_answer(text: str, options: dict[str, str]) -> Optional[str]:
    if not text:
        return None
    valid = set(options.keys())
    matches = _ANSWER_LINE.findall(text)
    if matches:
        cand = matches[-1].strip().upper()
        if cand and cand[0] in valid:
            return cand[0]
    letters = re.findall(r"\b([A-Z])\b", text.upper())
    for cand in reversed(letters):
        if cand in valid:
            return cand
    return None


def parse_hard_answer(text: str) -> Optional[bool]:
    if not text:
        return None
    matches = _ANSWER_LINE.findall(text)
    if matches:
        cand = matches[-1].strip().lower()
        if cand.startswith("t") or cand.startswith("y"):
            return True
        if cand.startswith("f") or cand.startswith("n"):
            return False
    low = text.lower()
    tail = low[-40:]
    if "true" in tail or "yes" in tail:
        return True
    if "false" in tail or "no" in tail:
        return False
    return None


def _score_easy(q: MCQQuestion, predicted: Optional[str]) -> bool:
    return predicted is not None and q.answer_key is not None and predicted == q.answer_key


def _score_hard(q: MCQQuestion, predicted: Optional[bool]) -> bool:
    return predicted is not None and q.answer_bool is not None and predicted == q.answer_bool


def run_system(system: str,
               questions: list[MCQQuestion],
               backend,
               ctx: Optional[RetrievalContext] = None,
               *,
               top_k: int = 8,
               evidence: Optional[list[tuple[str, list[str], dict]]] = None,
               max_tokens: Optional[int] = None) -> list[MCQSystemAnswer]:
    """Answer ``questions`` with one system.

    Optional (default-off) extras for eval v2: ``evidence`` supplies
    precomputed ``evidence_for`` results aligned with ``questions`` (so the
    same retrieval can be reused across option-order seeds), and
    ``max_tokens`` is passed explicitly to the backend when given.
    """
    if not questions:
        return []
    if evidence is not None and len(evidence) != len(questions):
        raise ValueError("evidence must align with questions")
    sys_prompts, user_prompts, ids_list, meta_list = [], [], [], []
    for qi, q in enumerate(questions):
        if evidence is None:
            sp, up, ids, meta = build_prompt(system, q, ctx, top_k)
        else:
            block, ids, meta = evidence[qi]
            sp, up = _prompt_from_evidence(q, block)
        sys_prompts.append(sp)
        user_prompts.append(up)
        ids_list.append(ids)
        meta_list.append(meta)

    answers: list[Optional[MCQSystemAnswer]] = [None] * len(questions)
    for sp in set(sys_prompts):
        idxs = [i for i, s in enumerate(sys_prompts) if s == sp]
        batch_prompts = [user_prompts[i] for i in idxs]
        if max_tokens is None:
            outputs = backend.generate(batch_prompts, system_prompt=sp)
        else:
            outputs = backend.generate(batch_prompts, system_prompt=sp,
                                       max_tokens=max_tokens)
        for i, out in zip(idxs, outputs):
            q = questions[i]
            meta = meta_list[i]
            if q.variant == MCQVariant.EASY:
                pred = parse_easy_answer(out, q.options)
                correct = _score_easy(q, pred)
                predicted = pred
            else:
                pred_bool = parse_hard_answer(out)
                correct = _score_hard(q, pred_bool)
                predicted = None if pred_bool is None else ("true" if pred_bool else "false")
            answers[i] = MCQSystemAnswer(
                system=system,
                raw_output=truncate(out, 2000),
                predicted=predicted,
                correct=correct,
                retrieved_ids=ids_list[i],
                completeness=meta.get("completeness"),
                completeness_score=meta.get("completeness_score"),
                shared_node_available=meta.get("shared_node_available"),
            )
    return [a for a in answers]  # type: ignore[return-value]
