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

from . import config, prompts
from .pathway_retriever import PathwayRetriever
from .retriever import TextRetriever, TripleRetriever
from .schemas import (
    CulturalPathway,
    CulturalTriple,
    LayerLabel,
    MCQQuestion,
    MCQSystemAnswer,
    MCQVariant,
    PathwayCompleteness,
)
from .utils import get_logger, normalize_text, truncate

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


def _query_for(q: MCQQuestion) -> str:
    parts = [q.question]
    if q.cultural_group:
        parts.append(q.cultural_group)
    if q.variant == MCQVariant.EASY:
        parts.extend(q.options.values())
    elif q.option_text:
        parts.append(q.option_text)
    return " ".join(p for p in parts if p)


def _culture_match(pathway_culture: str, question_culture: Optional[str]) -> float:
    pc = normalize_text(pathway_culture).lower()
    qc = normalize_text(question_culture or "").lower()
    if not pc or not qc:
        return 0.5
    if pc == qc or pc in qc or qc in pc:
        return 1.0
    return 0.35


def _rank_pathways(hits: list[tuple[CulturalPathway, float]],
                   question_culture: Optional[str],
                   *,
                   exclude_shared: bool = False) -> list[tuple[CulturalPathway, float]]:
    ranked: list[tuple[CulturalPathway, float]] = []
    for pw, score in hits:
        if exclude_shared and pw.has_shared_pivot:
            continue
        w = _COMPLETENESS_WEIGHT.get(pw.completeness, 0.5)
        cm = _culture_match(pw.culture, question_culture)
        bonus = 1.1 if pw.has_shared_pivot else 1.0
        ranked.append((pw, score * w * cm * bonus))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _completeness_from_triples(triples: list[CulturalTriple]) -> tuple[str, float, bool]:
    layers = {
        layer
        for triple in triples
        for layer in (triple.subject_layer, triple.predicate_layer, triple.object_layer)
        if layer != LayerLabel.UNKNOWN
    }
    layer_values = {_LAYER_SCORE[l] for l in layers if l in _LAYER_SCORE}
    if {3, 2, 1}.issubset(layer_values):
        bucket = PathwayCompleteness.FULL.value
    elif 2 in layer_values and 1 in layer_values:
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


def _format_pathway(pw: CulturalPathway, index: int) -> str:
    lines = [f"[{index}] Pathway ({pw.completeness.value}, {pw.culture}):"]
    for node in pw.nodes:
        lines.append(f"  {node.layer.value}: {node.label}")
    if pw.has_shared_pivot:
        lines.append("  [shared abstraction pivot across cultures]")
    return "\n".join(lines)


def _options_block(q: MCQQuestion) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in q.options.items())


def _passage_evidence(ctx: RetrievalContext, query: str, top_k: int) -> tuple[str, list[str], dict]:
    if not ctx.passage_retriever:
        return "", [], {}
    hits = ctx.passage_retriever.retrieve(query, top_k=top_k)
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
                          retriever: Optional[TripleRetriever] = None) -> tuple[str, list[str], dict]:
    retriever = retriever or ctx.triple_retriever
    if not retriever:
        return "", [], {}
    hits = retriever.retrieve(query, top_k=top_k)
    triples = [t for t, _ in hits]
    lines, ids = [], []
    for i, (t, _score) in enumerate(hits, 1):
        lines.append(f"[{i}] {t.subject} - {t.predicate} - {t.object}")
        ids.append(t.triple_id)
    bucket, score, shared = _completeness_from_triples(triples)
    meta = {"completeness": bucket, "completeness_score": score, "shared_node_available": shared}
    return "\n".join(lines), ids, meta


def _layered_triple_evidence(ctx: RetrievalContext, query: str, top_k: int,
                             retriever: Optional[TripleRetriever] = None) -> tuple[str, list[str], dict]:
    retriever = retriever or ctx.triple_retriever
    if not retriever:
        return "", [], {}
    hits = retriever.retrieve(query, top_k=top_k)
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
    hits = ctx.pathway_retriever.retrieve(query, top_k=max(top_k * 3, top_k))
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
        body, ids, meta = _passage_evidence(ctx, query, top_k)
    elif system == SYSTEM_S2:
        body, ids, meta = _flat_triple_evidence(ctx, query, top_k)
    elif system == SYSTEM_S3:
        body, ids, meta = _layered_triple_evidence(ctx, query, top_k)
    elif system == SYSTEM_S4:
        body, ids, meta = _pathway_evidence(ctx, query, top_k, q)
    elif system == SYSTEM_S4_NO_SHARED:
        body, ids, meta = _pathway_evidence(ctx, query, top_k, q, exclude_shared=True)
    elif system == SYSTEM_S4_TRIPLE:
        retriever = ctx.refined_triple_retriever or ctx.triple_retriever
        body, ids, meta = _layered_triple_evidence(ctx, query, top_k, retriever=retriever)
    else:
        return "", [], empty_meta
    if not body:
        return "", ids, meta
    header = prompts.MCQ_EVIDENCE_HEADERS.get(system, "Retrieved evidence:")
    return f"{header}\n{body}\n\n", ids, meta


def build_prompt(system: str, q: MCQQuestion,
                 ctx: Optional[RetrievalContext], top_k: int) -> tuple[str, str, list[str], dict]:
    evidence_block, ids, meta = _evidence_for(system, ctx, q, top_k)
    if q.variant == MCQVariant.EASY:
        user = prompts.MCQ_EASY_USER.format(
            evidence_block=evidence_block,
            question=q.question,
            options_block=_options_block(q),
        )
        return prompts.MCQ_SYSTEM, user, ids, meta
    user = prompts.MCQ_HARD_USER.format(
        evidence_block=evidence_block,
        question=q.question,
        option_text=q.option_text or "",
    )
    return prompts.MCQ_HARD_SYSTEM, user, ids, meta


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
               top_k: int = 8) -> list[MCQSystemAnswer]:
    if not questions:
        return []
    sys_prompts, user_prompts, ids_list, meta_list = [], [], [], []
    for q in questions:
        sp, up, ids, meta = build_prompt(system, q, ctx, top_k)
        sys_prompts.append(sp)
        user_prompts.append(up)
        ids_list.append(ids)
        meta_list.append(meta)

    answers: list[Optional[MCQSystemAnswer]] = [None] * len(questions)
    for sp in set(sys_prompts):
        idxs = [i for i, s in enumerate(sys_prompts) if s == sp]
        batch_prompts = [user_prompts[i] for i in idxs]
        outputs = backend.generate(batch_prompts, system_prompt=sp)
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
