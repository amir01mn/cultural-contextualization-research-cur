"""The four CulturalBench answering systems (S0-S3).

All systems share the same base LLM (via the pluggable backend), the same prompt
scaffold, and the same retrieval budget (top_k). The ONLY thing that differs is
the injected cultural evidence, so any accuracy difference is attributable to the
evidence/structure:

    S0 No-KG          : question + options only (closed-book).
    S1 Raw-text RAG   : retrieved raw CultureBank passages.
    S2 Flat KG        : retrieved triples as a bag, layer labels stripped.
    S3 Layered KG     : retrieved triples WITH abstraction-layer labels.

The S4 abstraction->refinement pathway system is a later phase and is not here.

Prompts are built for every question up front and sent to ``backend.generate`` in
one batch, so vLLM can process them efficiently.
"""

from __future__ import annotations

import re
from typing import Optional

from . import config, prompts
from .retriever import TextRetriever, TripleRetriever
from .schemas import CulturalTriple, MCQQuestion, MCQSystemAnswer, MCQVariant
from .utils import get_logger, normalize_text, truncate

LOGGER = get_logger("mcq_systems")

SYSTEM_S0 = "S0_no_kg"
SYSTEM_S1 = "S1_raw_text_rag"
SYSTEM_S2 = "S2_flat_kg"
SYSTEM_S3 = "S3_layered_kg"
ALL_SYSTEMS = [SYSTEM_S0, SYSTEM_S1, SYSTEM_S2, SYSTEM_S3]

KG_SYSTEMS = {SYSTEM_S2, SYSTEM_S3}


# ===========================================================================
# Retrieval context
# ===========================================================================

class RetrievalContext:
    """Holds the corpus indexes shared by the KG/RAG systems.

    Building the indexes once and reusing them across systems keeps the
    comparison fair and avoids recomputing embeddings per system.
    """

    def __init__(self,
                 passages: Optional[list[dict]] = None,
                 triples: Optional[list[CulturalTriple]] = None):
        self.passage_retriever = TextRetriever(passages or []) if passages else None
        self.triple_retriever = TripleRetriever(triples or []) if triples else None
        LOGGER.info("RetrievalContext: passages=%s triples=%s",
                    self.passage_retriever.backend if self.passage_retriever else "none",
                    self.triple_retriever.backend if self.triple_retriever else "none")


# ===========================================================================
# Query + evidence construction
# ===========================================================================

def _query_for(q: MCQQuestion) -> str:
    """Retrieval query string for a question."""
    parts = [q.question]
    if q.cultural_group:
        parts.append(q.cultural_group)
    if q.variant == MCQVariant.EASY:
        parts.extend(q.options.values())
    elif q.option_text:
        parts.append(q.option_text)
    return " ".join(p for p in parts if p)


def _options_block(q: MCQQuestion) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in q.options.items())


def _passage_evidence(ctx: RetrievalContext, query: str, top_k: int) -> tuple[str, list[str]]:
    if not ctx.passage_retriever:
        return "", []
    hits = ctx.passage_retriever.retrieve(query, top_k=top_k)
    lines, ids = [], []
    for i, (doc, _score) in enumerate(hits, 1):
        text = truncate(doc.get("text", ""), 300)
        if text:
            lines.append(f"[{i}] {text}")
            ids.append(str(doc.get("passage_id", "")))
    return "\n".join(lines), ids


def _flat_triple_evidence(ctx: RetrievalContext, query: str, top_k: int) -> tuple[str, list[str]]:
    if not ctx.triple_retriever:
        return "", []
    hits = ctx.triple_retriever.retrieve(query, top_k=top_k)
    lines, ids = [], []
    for i, (t, _score) in enumerate(hits, 1):
        lines.append(f"[{i}] {t.subject} - {t.predicate} - {t.object}")
        ids.append(t.triple_id)
    return "\n".join(lines), ids


def _layered_triple_evidence(ctx: RetrievalContext, query: str, top_k: int) -> tuple[str, list[str]]:
    if not ctx.triple_retriever:
        return "", []
    hits = ctx.triple_retriever.retrieve(query, top_k=top_k)
    lines, ids = [], []
    for i, (t, _score) in enumerate(hits, 1):
        lines.append(
            f"[{i}] ({t.subject_layer.value}) {t.subject} "
            f"-[{t.predicate} | {t.predicate_layer.value}]-> "
            f"({t.object_layer.value}) {t.object}"
        )
        ids.append(t.triple_id)
    return "\n".join(lines), ids


def _evidence_for(system: str, ctx: Optional[RetrievalContext],
                  q: MCQQuestion, top_k: int) -> tuple[str, list[str]]:
    """Return (evidence_block_text, retrieved_ids) for a system."""
    if system == SYSTEM_S0 or ctx is None:
        return "", []
    query = _query_for(q)
    if system == SYSTEM_S1:
        body, ids = _passage_evidence(ctx, query, top_k)
    elif system == SYSTEM_S2:
        body, ids = _flat_triple_evidence(ctx, query, top_k)
    elif system == SYSTEM_S3:
        body, ids = _layered_triple_evidence(ctx, query, top_k)
    else:
        return "", []
    if not body:
        return "", ids
    header = prompts.MCQ_EVIDENCE_HEADERS.get(system, "Retrieved evidence:")
    return f"{header}\n{body}\n\n", ids


def build_prompt(system: str, q: MCQQuestion,
                 ctx: Optional[RetrievalContext], top_k: int) -> tuple[str, str, list[str]]:
    """Return (system_prompt, user_prompt, retrieved_ids) for one question."""
    evidence_block, ids = _evidence_for(system, ctx, q, top_k)
    if q.variant == MCQVariant.EASY:
        user = prompts.MCQ_EASY_USER.format(
            evidence_block=evidence_block,
            question=q.question,
            options_block=_options_block(q),
        )
        return prompts.MCQ_SYSTEM, user, ids
    user = prompts.MCQ_HARD_USER.format(
        evidence_block=evidence_block,
        question=q.question,
        option_text=q.option_text or "",
    )
    return prompts.MCQ_HARD_SYSTEM, user, ids


# ===========================================================================
# Answer parsing
# ===========================================================================

_ANSWER_LINE = re.compile(r"ANSWER\s*[:\-]?\s*([A-Za-z]+)", re.IGNORECASE)


def parse_easy_answer(text: str, options: dict[str, str]) -> Optional[str]:
    """Extract an option letter from a model response."""
    if not text:
        return None
    valid = set(options.keys())
    matches = _ANSWER_LINE.findall(text)
    if matches:
        cand = matches[-1].strip().upper()
        if cand and cand[0] in valid:
            return cand[0]
    # Fallback: last standalone valid letter in the text.
    letters = re.findall(r"\b([A-Z])\b", text.upper())
    for cand in reversed(letters):
        if cand in valid:
            return cand
    return None


def parse_hard_answer(text: str) -> Optional[bool]:
    """Extract a True/False judgment from a model response."""
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
    # Fallback: look near the end.
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


# ===========================================================================
# Running a system
# ===========================================================================

def run_system(system: str,
               questions: list[MCQQuestion],
               backend,
               ctx: Optional[RetrievalContext] = None,
               *,
               top_k: int = 8) -> list[MCQSystemAnswer]:
    """Run one system over all questions; returns answers aligned to ``questions``.

    All prompts are built first and generated in a single batched backend call.
    """
    if not questions:
        return []
    sys_prompts, user_prompts, ids_list = [], [], []
    for q in questions:
        sp, up, ids = build_prompt(system, q, ctx, top_k)
        sys_prompts.append(sp)
        user_prompts.append(up)
        ids_list.append(ids)

    # Easy and Hard use different system prompts; split the batch by system prompt
    # so each batch is internally consistent.
    answers: list[Optional[MCQSystemAnswer]] = [None] * len(questions)
    for sp in set(sys_prompts):
        idxs = [i for i, s in enumerate(sys_prompts) if s == sp]
        batch_prompts = [user_prompts[i] for i in idxs]
        outputs = backend.generate(batch_prompts, system_prompt=sp)
        for i, out in zip(idxs, outputs):
            q = questions[i]
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
            )
    return [a for a in answers]  # type: ignore[return-value]
