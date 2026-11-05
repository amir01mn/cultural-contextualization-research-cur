"""Matched Qwen3-32B V2 MCQ evaluation with provenance-preserving evidence.

Only the user-side evidence block differs across S1-S4R. The question,
option order, assistant Answer: continuation, token scoring and model revision
remain identical to closed-book V2.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from ..loaders.kazculture_loader import CHOICES, KazCultureQuestion
from ..retrieval.kazculture_index import render_evidence, render_evidence_with_hits
from .kazculture_hf_scorer_v2 import (
    HFQwenLetterScorerV2, candidate_token_ids, user_question,
)
from .kazculture_phase1 import choose_highest


EVIDENCE_HEADER = ("Retrieved cultural evidence (may be incomplete; answer only "
                   "from supported facts):\n")
QWEN3_TEMPLATE_KWARGS = {"enable_thinking": False}


def evidence_prompt(tokenizer, q: KazCultureQuestion, evidence: str, *,
                    template_kwargs: dict | None = None) -> str:
    """The single answering template; only the evidence block varies.

    Empty evidence gives exactly the closed-book V2 prompt. ``template_kwargs``
    defaults to Qwen3 non-thinking; other models (OLMo-2) pass ``{}``.
    """
    kwargs = QWEN3_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
    user = user_question(q)
    if evidence.strip():
        user = EVIDENCE_HEADER + evidence + "\n\n" + user
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}], tokenize=False,
        add_generation_prompt=True, **kwargs
    ) + "Answer:"


class V2EvidenceScorer(HFQwenLetterScorerV2):
    template_kwargs: dict | None = None  # None -> Qwen3 enable_thinking=False

    def render(self, q: KazCultureQuestion, evidence: str) -> str:
        return evidence_prompt(self.tokenizer, q, evidence,
                               template_kwargs=self.template_kwargs)

    def count_tokens(self, rendered: str) -> int:
        return len(self.tokenizer.encode(rendered, add_special_tokens=False))

    def score_rendered(self, rendered: str) -> dict[str, float]:
        torch = self.torch
        ids = candidate_token_ids(self.tokenizer, rendered)
        inputs = self.tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        )
        if inputs["input_ids"].shape[-1] > self.max_context_tokens:
            raise ValueError("Evidence exceeds frozen model context budget")
        inputs = {k: t.to(next(self.model.parameters()).device)
                  for k, t in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)
        return {label: float(logprobs[token].item()) for label, token in ids.items()}

    def score_with_evidence(self, q: KazCultureQuestion, evidence: str) -> dict[str, float]:
        return self.score_rendered(self.render(q, evidence))


def selected_evidence(index, question: KazCultureQuestion, system: str, *,
                      top_k: int, min_similarity: float,
                      exclude_passage_ids: set[str] | None = None) -> tuple[str, list[dict]]:
    combined = system == "S4R_combined"
    base_system = "S4R" if combined else system
    direct = index.search(
        question.question, system=base_system, k=top_k,
        min_similarity=min_similarity,
        exclude_passage_ids=exclude_passage_ids, combined=False
    )
    if not combined:
        return render_evidence_with_hits(direct)
    # Shared abstract values are the ONLY allowed alignment pivot. Every
    # CultureBank analog is explicitly indirect and has its own source ID.
    kaz_values = {v for hit in direct for v in hit.get("values", [])}
    if not kaz_values:
        return render_evidence_with_hits(direct)
    candidates = index.search(
        question.question, system="S4R", k=top_k * 6,
        min_similarity=min_similarity,
        exclude_passage_ids=exclude_passage_ids, combined=True
    )
    analogs = [
        hit for hit in candidates
        if hit["dataset"] == "CultureBank"
        and bool(set(hit.get("values", [])) & kaz_values)
    ][:min(2, max(0, top_k - 1))]
    hits = (direct[:max(1, top_k - len(analogs))] + analogs)[:top_k]
    return render_evidence_with_hits(hits)


def prediction(q: KazCultureQuestion, scores: dict, hits: list[dict],
               evidence: str, system: str) -> dict:
    predicted = choose_highest(scores)
    # Gold label only enters after the model call, never the retrieval query.
    return {
        "question_id": q.question_id, "category": q.category,
        "system": system, "prediction": predicted,
        "answer_key": q.answer_key, "correct": predicted == q.answer_key,
        "letter_logprobs": scores,
        "retrieved_passage_ids": [hit["passage_id"] for hit in hits],
        "retrieved_unit_ids": [hit["unit_id"] for hit in hits],
        "retrieved_datasets": [hit["dataset"] for hit in hits],
        "retrieval_scores": [hit["similarity"] for hit in hits],
        "evidence_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
        "retrieval_mode": "no_evidence" if not hits else
                          "combined" if system == "S4R_combined" else "direct",
    }


def paired_statistics(baseline: list[dict], comparison: list[dict],
                      *, iterations: int = 2000, seed: int = 42) -> dict:
    b = {r["question_id"]: bool(r["correct"]) for r in baseline}
    c = {r["question_id"]: bool(r["correct"]) for r in comparison}
    if set(b) != set(c) or not b:
        raise ValueError("Paired evaluation requires exactly matching question IDs")
    ids = sorted(b)
    n = len(ids)
    bc = [int(b[x]) for x in ids]
    cc = [int(c[x]) for x in ids]
    rescued = sum(not u and v for u, v in zip(bc, cc))
    harmed = sum(u and not v for u, v in zip(bc, cc))
    delta = (sum(cc) - sum(bc)) / n
    rng = random.Random(seed)
    paired = [cc[i] - bc[i] for i in range(n)]
    boot = []
    for _ in range(iterations):
        boot.append(sum(paired[rng.randrange(n)] for __ in range(n)) / n)
    boot.sort()
    discordant = rescued + harmed
    # Exact two-sided McNemar p-value is the two-sided binomial test with
    # p=0.5 on discordant pairs. Integer combinations avoid scipy dependency
    # and the numerical underflow of explicitly computing 0.5**discordant.
    # Symmetry makes the two-sided probability twice the smaller tail.
    if discordant:
        smaller = min(rescued, harmed)
        tail_count = sum(math.comb(discordant, k) for k in range(smaller + 1))
        pval = min(1.0, (2 * tail_count) / (1 << discordant))
    else:
        pval = 1.0
    return {
        "baseline_accuracy": sum(bc) / n,
        "system_accuracy": sum(cc) / n,
        "difference_pp": round(100 * delta, 4),
        "paired_bootstrap_95_delta_pp": [
            round(100 * boot[int(0.025 * (iterations - 1))], 4),
            round(100 * boot[int(0.975 * (iterations - 1))], 4),
        ],
        "mcnemar_exact_p": pval,
        "evidence_rescued_n": rescued,
        "evidence_induced_errors_n": harmed,
        "total_questions": n,
    }


def category_report(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    return {
        k: {"n": len(v), "correct": sum(bool(x["correct"]) for x in v),
            "accuracy": sum(bool(x["correct"]) for x in v) / len(v)}
        for k, v in sorted(groups.items())
    }


def retrieval_recall(rows: list[dict], test_source_id_by_question: dict[str, str],
                     eligible_ids: set[str]) -> dict:
    """Recall@k among test records whose exact passage appears in train index.

    This is intentionally undefined for source-disjoint examples; do not
    label their zero possible-hit rate as evidence of poor retrieval.
    """
    possible = [r for r in rows if test_source_id_by_question.get(r["question_id"])
                in eligible_ids]
    if not possible:
        return {"eligible_source_overlap_n": 0, "recall_at_k": None}
    success = sum(test_source_id_by_question[r["question_id"]] in
                  r["retrieved_passage_ids"] for r in possible)
    return {
        "eligible_source_overlap_n": len(possible),
        "recalled_n": success, "recall_at_k": success / len(possible),
        "definition": "exact source passage hit on overlapping train/test passages only",
    }
