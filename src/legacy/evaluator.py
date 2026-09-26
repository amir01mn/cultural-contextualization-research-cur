"""LEGACY open-ended evaluator from the original MVP (not a research metric).

Kept only for ``scripts/legacy/`` and historical comparison. The research
metric is CulturalBench exact match in ``src/evaluation/``.

Scenario generation, answer generation (3 systems), and scoring.

Systems compared:
    A. baseline_prompt_only           - LLM with only the question
    B. raw_kg_augmented               - LLM + retrieved raw triples
    C. critique_refined_kg_augmented  - LLM + retrieved refined triples

All three use the SAME local LLM so differences are attributable to the KG.

Automatic scoring (out of 100):
    keyword coverage            : 30
    cultural group match        : 15
    provenance use              : 20
    layer coverage              : 20
    answer specificity          : 10
    avoids bad keywords         : 5
    generated dominance penalty : up to -15
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..core import config
from ..llm import local_llm, prompts
from ..retrieval.retriever import TripleRetriever
from ..core.schemas import (
    CulturalTriple,
    EvaluationExample,
    EvaluationResult,
    LayerLabel,
    SystemAnswer,
)
from ..core.utils import get_logger, normalize_text, stable_id, truncate

LOGGER = get_logger("evaluator")

SYSTEM_BASELINE = "baseline_prompt_only"
SYSTEM_RAW = "raw_kg_augmented"
SYSTEM_REFINED = "critique_refined_kg_augmented"


# ===========================================================================
# Scenario generation
# ===========================================================================

def _record_for_prompt(rec: dict) -> str:
    return truncate(json.dumps(rec, ensure_ascii=False, default=str), 2000)


def generate_scenario_llm(record: dict,
                          *,
                          model: Optional[str] = None) -> Optional[EvaluationExample]:
    """Generate one scenario question for a record via the local LLM."""
    user_prompt = prompts.SCENARIO_USER.format(record_json=_record_for_prompt(record))
    parsed = local_llm.generate_json(
        user_prompt, system_prompt=prompts.SCENARIO_SYSTEM, model=model
    )
    if not parsed or not isinstance(parsed, dict):
        return None
    examples = parsed.get("examples")
    item = None
    if isinstance(examples, list) and examples:
        item = examples[0]
    elif "question" in parsed:
        item = parsed
    if not isinstance(item, dict) or not normalize_text(item.get("question")):
        return None
    return _build_example(record, item)


def _build_example(record: dict, item: dict) -> EvaluationExample:
    instance_id = record.get("instance_id", "")
    example_id = normalize_text(item.get("example_id")) or stable_id(
        instance_id, item.get("question", ""), prefix="ex_"
    )
    return EvaluationExample(
        example_id=example_id,
        source_instance_id=normalize_text(item.get("source_instance_id")) or instance_id,
        question=normalize_text(item.get("question")),
        cultural_group=normalize_text(item.get("cultural_group")) or record.get("cultural_group"),
        expected_keywords=[normalize_text(k) for k in (item.get("expected_keywords") or []) if normalize_text(k)],
        avoid_keywords=[normalize_text(k) for k in (item.get("avoid_keywords") or []) if normalize_text(k)],
        target_layers=item.get("target_layers") or [],
        notes=normalize_text(item.get("notes")) or None,
    )


def generate_scenario_fallback(record: dict) -> EvaluationExample:
    """Deterministic scenario question built from record fields (fallback)."""
    cg = normalize_text(record.get("cultural_group")) or "this culture"
    context = normalize_text(record.get("context")) or "this situation"
    behavior = normalize_text(record.get("actor_behavior"))
    topic = normalize_text(record.get("topic"))
    question = (
        f"You are in a {cg} cultural setting: {context}. "
        f"What should you do, avoid, or understand to act respectfully?"
    )
    expected = [w for w in [cg, context, behavior, topic] if w]
    expected_keywords = []
    for phrase in expected:
        expected_keywords.extend([t for t in re.split(r"\W+", phrase) if len(t) > 3])
    return EvaluationExample(
        example_id=stable_id(record.get("instance_id", ""), question, prefix="ex_"),
        source_instance_id=record.get("instance_id", ""),
        question=question,
        cultural_group=record.get("cultural_group"),
        expected_keywords=list(dict.fromkeys(expected_keywords))[:12],
        avoid_keywords=["weird", "primitive", "strange", "backward", "all of them always"],
        target_layers=[LayerLabel.LAYER_1_SURFACE, LayerLabel.LAYER_2_SOCIAL,
                       LayerLabel.LAYER_3_DEEP_VALUE],
        notes="Fallback-generated scenario.",
    )


def generate_scenarios(records: list[dict],
                       num_scenarios: int = 25,
                       *,
                       use_llm: bool = True,
                       allow_fallback: bool = False,
                       model: Optional[str] = None) -> list[EvaluationExample]:
    """Generate ~num_scenarios scenario questions from the records."""
    selected = records[:num_scenarios]
    examples: list[EvaluationExample] = []

    iterator = selected
    try:
        from tqdm import tqdm
        iterator = tqdm(selected, desc="Generating scenarios")
    except Exception:  # noqa: BLE001
        pass

    for rec in iterator:
        ex = None
        if use_llm:
            try:
                ex = generate_scenario_llm(rec, model=model)
            except local_llm.LLMUnavailableError:
                if not allow_fallback:
                    raise
                use_llm = False  # stop trying the LLM for the rest
        if ex is None:
            if use_llm and not allow_fallback:
                # LLM was available but produced nothing parseable; skip.
                continue
            ex = generate_scenario_fallback(rec)
        examples.append(ex)
    return examples


# ===========================================================================
# Answer generation
# ===========================================================================

def _triples_to_json(triples: list[CulturalTriple]) -> str:
    payload = []
    for t in triples:
        payload.append({
            "subject": t.subject,
            "predicate": t.predicate,
            "object": t.object,
            "subject_layer": t.subject_layer.value,
            "predicate_layer": t.predicate_layer.value,
            "object_layer": t.object_layer.value,
            "confidence": round(t.confidence, 3),
            "generated": t.generated,
            "evidence": t.evidence,
            "source_text": truncate(t.source_text, 200),
            "dataset_id": t.dataset_id,
            "instance_id": t.instance_id,
        })
    return json.dumps(payload, ensure_ascii=False)


def answer_baseline(question: str, *, model: Optional[str] = None) -> str:
    prompt = prompts.BASELINE_USER.format(question=question)
    return local_llm.generate_text(prompt, system_prompt=prompts.ANSWER_SYSTEM, model=model)


def answer_with_kg(question: str,
                   triples: list[CulturalTriple],
                   *,
                   refined: bool,
                   model: Optional[str] = None) -> str:
    triples_json = _triples_to_json(triples)
    if refined:
        prompt = prompts.REFINED_KG_USER.format(
            question=question, retrieved_refined_triples_json=triples_json)
    else:
        prompt = prompts.RAW_KG_USER.format(
            question=question, retrieved_raw_triples_json=triples_json)
    return local_llm.generate_text(prompt, system_prompt=prompts.ANSWER_SYSTEM, model=model)


# ===========================================================================
# Scoring
# ===========================================================================

_LAYER_TERMS = {
    LayerLabel.LAYER_2_SOCIAL: [
        "expect", "role", "host", "guest", "elder", "polite", "hierarchy",
        "relationship", "norm", "obligation", "respectful", "etiquette", "social",
    ],
    LayerLabel.LAYER_3_DEEP_VALUE: [
        "respect", "harmony", "hospitality", "honor", "dignity", "value",
        "belief", "identity", "modesty", "family", "trust", "care", "spiritual",
    ],
}

_SPECIFIC_HINTS = [
    "in this context", "based on the provided", "a respectful approach",
    "for example", "specifically", "you might", "you could", "consider",
]


def _count_keyword_hits(answer_lc: str, keywords: list[str]) -> int:
    hits = 0
    for kw in keywords:
        kw = normalize_text(kw).lower()
        if kw and kw in answer_lc:
            hits += 1
    return hits


def score_answer(answer: str,
                 example: EvaluationExample,
                 retrieved: list[CulturalTriple]) -> tuple[float, dict]:
    """Score one answer out of 100 with a detailed breakdown."""
    answer_lc = normalize_text(answer).lower()
    breakdown: dict[str, float] = {}

    # 1. Keyword coverage (30)
    exp = example.expected_keywords or []
    if exp:
        coverage = _count_keyword_hits(answer_lc, exp) / len(exp)
    else:
        coverage = 0.0
    breakdown["keyword_coverage"] = round(30.0 * coverage, 2)

    # 2. Cultural group match (15)
    cg = normalize_text(example.cultural_group).lower()
    cg_score = 0.0
    if cg:
        # match on the group or any of its tokens.
        tokens = [t for t in re.split(r"\W+", cg) if len(t) > 2]
        if cg in answer_lc or any(t in answer_lc for t in tokens):
            cg_score = 15.0
    breakdown["cultural_group_match"] = cg_score

    # 3. Provenance use (20): does the answer reflect retrieved triple content?
    prov_score = 0.0
    if retrieved:
        grounded = [t for t in retrieved if not t.generated]
        ref_pool = grounded or retrieved
        hits = 0
        considered = 0
        for t in ref_pool:
            terms = [normalize_text(t.subject).lower(), normalize_text(t.object).lower()]
            for term in terms:
                toks = [w for w in re.split(r"\W+", term) if len(w) > 3]
                if not toks:
                    continue
                considered += 1
                if any(w in answer_lc for w in toks):
                    hits += 1
        if considered:
            prov_score = 20.0 * min(1.0, hits / considered)
    breakdown["provenance_use"] = round(prov_score, 2)

    # 4. Layer coverage (20): rewards explaining social + value layers.
    layer_score = 0.0
    for layer in (LayerLabel.LAYER_2_SOCIAL, LayerLabel.LAYER_3_DEEP_VALUE):
        if any(term in answer_lc for term in _LAYER_TERMS[layer]):
            layer_score += 10.0
    breakdown["layer_coverage"] = layer_score

    # 5. Answer specificity (10): length + cautious framing hints.
    words = len(answer_lc.split())
    length_component = min(1.0, words / 120.0) * 6.0
    hint_component = min(1.0, sum(1 for h in _SPECIFIC_HINTS if h in answer_lc) / 2.0) * 4.0
    breakdown["answer_specificity"] = round(length_component + hint_component, 2)

    # 6. Avoids bad keywords (5)
    avoid = example.avoid_keywords or []
    bad_hits = _count_keyword_hits(answer_lc, avoid)
    breakdown["avoids_bad_keywords"] = 5.0 if bad_hits == 0 else 0.0

    # 7. Generated dominance penalty (up to -15)
    used_generated_ratio = 0.0
    if retrieved:
        gen = sum(1 for t in retrieved if t.generated)
        used_generated_ratio = gen / len(retrieved)
    penalty = -15.0 * used_generated_ratio
    breakdown["generated_dominance_penalty"] = round(penalty, 2)

    total = sum(breakdown.values())
    total = max(0.0, min(100.0, total))
    breakdown["used_generated_ratio"] = round(used_generated_ratio, 3)
    return round(total, 2), breakdown


# ===========================================================================
# Orchestration
# ===========================================================================

def evaluate_examples(examples: list[EvaluationExample],
                      raw_triples: list[CulturalTriple],
                      refined_triples: list[CulturalTriple],
                      *,
                      top_k: int = 8,
                      model: Optional[str] = None,
                      use_llm_answers: bool = True,
                      allow_fallback: bool = False) -> tuple[list[EvaluationResult], dict]:
    """Run all three systems on every example and score them."""
    raw_retriever = TripleRetriever(raw_triples)
    refined_retriever = TripleRetriever(refined_triples)
    LOGGER.info("Retriever backends: raw=%s refined=%s",
                raw_retriever.backend, refined_retriever.backend)

    results: list[EvaluationResult] = []
    retrieval_debug: list[dict] = []

    iterator = examples
    try:
        from tqdm import tqdm
        iterator = tqdm(examples, desc="Evaluating systems")
    except Exception:  # noqa: BLE001
        pass

    for ex in iterator:
        raw_hits = raw_retriever.retrieve(ex.question, top_k=top_k)
        ref_hits = refined_retriever.retrieve(ex.question, top_k=top_k)
        raw_top = [t for t, _ in raw_hits]
        ref_top = [t for t, _ in ref_hits]

        retrieval_debug.append({
            "example_id": ex.example_id,
            "question": ex.question,
            "raw_retrieved": [
                {"triple_id": t.triple_id, "subject": t.subject,
                 "predicate": t.predicate, "object": t.object,
                 "generated": t.generated, "score": round(s, 4)}
                for t, s in raw_hits
            ],
            "refined_retrieved": [
                {"triple_id": t.triple_id, "subject": t.subject,
                 "predicate": t.predicate, "object": t.object,
                 "generated": t.generated, "score": round(s, 4)}
                for t, s in ref_hits
            ],
        })

        answers: dict[str, SystemAnswer] = {}

        def _run(system: str, text_fn) -> None:
            try:
                ans = text_fn() if use_llm_answers else ""
            except local_llm.LLMUnavailableError:
                if not allow_fallback:
                    raise
                ans = ""
            if system == SYSTEM_BASELINE:
                retr: list[CulturalTriple] = []
            elif system == SYSTEM_RAW:
                retr = raw_top
            else:
                retr = ref_top
            score, breakdown = score_answer(ans, ex, retr)
            answers[system] = SystemAnswer(
                system=system,
                answer=ans,
                score=score,
                score_breakdown=breakdown,
                retrieved_triple_ids=[t.triple_id for t in retr],
                used_generated_ratio=breakdown.get("used_generated_ratio", 0.0),
            )

        _run(SYSTEM_BASELINE, lambda: answer_baseline(ex.question, model=model))
        _run(SYSTEM_RAW, lambda: answer_with_kg(ex.question, raw_top, refined=False, model=model))
        _run(SYSTEM_REFINED, lambda: answer_with_kg(ex.question, ref_top, refined=True, model=model))

        results.append(EvaluationResult(
            example_id=ex.example_id,
            question=ex.question,
            cultural_group=ex.cultural_group,
            answers=answers,
        ))

    summary = summarize_results(results)
    return results, {"retrieval_debug": retrieval_debug, "summary": summary}


# ===========================================================================
# Aggregation, verdict, and persistence
# ===========================================================================

def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def summarize_results(results: list[EvaluationResult]) -> dict:
    baseline = [r.answers[SYSTEM_BASELINE].score for r in results if SYSTEM_BASELINE in r.answers]
    raw = [r.answers[SYSTEM_RAW].score for r in results if SYSTEM_RAW in r.answers]
    refined = [r.answers[SYSTEM_REFINED].score for r in results if SYSTEM_REFINED in r.answers]

    avg_baseline = _avg(baseline)
    avg_raw = _avg(raw)
    avg_refined = _avg(refined)

    raw_minus_baseline = round(avg_raw - avg_baseline, 2)
    refined_minus_raw = round(avg_refined - avg_raw, 2)

    gen_ratios = [r.answers[SYSTEM_REFINED].used_generated_ratio
                  for r in results if SYSTEM_REFINED in r.answers]
    generated_support_ratio = _avg([g * 100 for g in gen_ratios])

    # Layer coverage measured on the refined system answers.
    def layer_cov(layer_key: str) -> float:
        vals = []
        for r in results:
            sa = r.answers.get(SYSTEM_REFINED)
            if sa:
                # 10 points max per layer in breakdown; layer_coverage holds both.
                vals.append(1.0 if _answer_has_layer(sa, layer_key) else 0.0)
        return round(_avg([v * 100 for v in vals]), 2)

    summary = {
        "num_examples": len(results),
        "average_baseline_score": avg_baseline,
        "average_raw_kg_score": avg_raw,
        "average_refined_kg_score": avg_refined,
        "raw_kg_improvement_over_baseline": raw_minus_baseline,
        "refined_kg_improvement_over_raw": refined_minus_raw,
        "generated_support_ratio_percent": round(generated_support_ratio, 2),
        "layer_2_coverage_percent": layer_cov("LAYER_2_SOCIAL"),
        "layer_3_coverage_percent": layer_cov("LAYER_3_DEEP_VALUE"),
    }
    summary["verdict"] = compute_verdict(summary)
    return summary


def _answer_has_layer(sa: SystemAnswer, layer_key: str) -> bool:
    layer = LayerLabel(layer_key)
    text_lc = normalize_text(sa.answer).lower()
    return any(term in text_lc for term in _LAYER_TERMS[layer])


def compute_verdict(summary: dict) -> str:
    raw_imp = summary["raw_kg_improvement_over_baseline"]
    ref_imp = summary["refined_kg_improvement_over_raw"]
    gen_ratio = summary["generated_support_ratio_percent"]

    raw_ok = raw_imp >= 5.0
    ref_ok = ref_imp >= 3.0
    gen_ok = gen_ratio <= 25.0

    if raw_ok and ref_ok and gen_ok:
        return "PASS"
    if raw_imp <= 0.0 or ref_imp < 0.0:
        return "FAIL"
    # Raw improves baseline but refined improvement small, or refined improves
    # but generated dominance too high.
    if raw_imp > 0.0 and (not ref_ok or not gen_ok):
        return "PARTIAL PASS"
    return "PARTIAL PASS"


def save_evaluation_outputs(examples: list[EvaluationExample],
                            results: list[EvaluationResult],
                            extras: dict,
                            *,
                            outputs_dir=None,
                            run_meta: Optional[dict] = None) -> dict:
    """Persist all evaluation artifacts and return the file paths."""
    import pandas as pd

    from ..core.utils import write_json

    outputs_dir = outputs_dir or config.OUTPUTS_DIR
    summary = extras.get("summary") or summarize_results(results)

    paths = {
        "examples": outputs_dir / "evaluation_examples_100.json",
        "results": outputs_dir / "evaluation_results_100.json",
        "csv": outputs_dir / "evaluation_results_100.csv",
        "summary_md": outputs_dir / "evaluation_summary_100.md",
        "plot": outputs_dir / "evaluation_average_scores_100.png",
        "retrieval_debug": outputs_dir / "retrieval_debug_100.json",
    }

    write_json(paths["examples"], [e.model_dump() for e in examples])
    write_json(paths["results"], {
        "summary": summary,
        "run_meta": run_meta or {},
        "results": [r.model_dump() for r in results],
    })
    write_json(paths["retrieval_debug"], extras.get("retrieval_debug", []))

    # Flat CSV: one row per (example, system).
    rows = []
    for r in results:
        for system, sa in r.answers.items():
            row = {
                "example_id": r.example_id,
                "system": system,
                "score": sa.score,
                "cultural_group": r.cultural_group,
                "used_generated_ratio": sa.used_generated_ratio,
            }
            row.update({f"score_{k}": v for k, v in sa.score_breakdown.items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(paths["csv"], index=False)

    _write_summary_md(paths["summary_md"], summary, run_meta or {})
    _save_plot(paths["plot"], summary)

    return {k: str(v) for k, v in paths.items()}


def _write_summary_md(path, summary: dict, run_meta: dict) -> None:
    lines = [
        "# Cultural KG MVP - Evaluation Summary (100-sample local LLM experiment)",
        "",
        f"- Dataset: {run_meta.get('dataset', 'CultureBank')}",
        f"- Records loaded: {run_meta.get('records_loaded', 'n/a')}",
        f"- LLM provider: {run_meta.get('llm_provider', config.LLM_PROVIDER)}",
        f"- Model: {run_meta.get('model', config.OLLAMA_MODEL)}",
        f"- Scenario examples evaluated: {summary.get('num_examples')}",
        "",
        "## Average scores (out of 100)",
        "",
        "| System | Average score |",
        "|---|---|",
        f"| A. Baseline (no KG) | {summary['average_baseline_score']} |",
        f"| B. Raw KG | {summary['average_raw_kg_score']} |",
        f"| C. Refined KG | {summary['average_refined_kg_score']} |",
        "",
        "## Improvements",
        "",
        f"- Raw KG - Baseline: **{summary['raw_kg_improvement_over_baseline']:+}**",
        f"- Refined KG - Raw KG: **{summary['refined_kg_improvement_over_raw']:+}**",
        "",
        "## KG quality indicators",
        "",
        f"- Generated support ratio (refined answers): {summary['generated_support_ratio_percent']}%",
        f"- Layer 2 coverage: {summary['layer_2_coverage_percent']}%",
        f"- Layer 3 coverage: {summary['layer_3_coverage_percent']}%",
        "",
        "## Result",
        "",
        f"### {summary.get('verdict', 'N/A')}",
        "",
        "PASS criteria:",
        "- Raw KG average score is at least 5 points higher than baseline.",
        "- Refined KG average score is at least 3 points higher than raw KG.",
        "- Refined KG generated support ratio is not above 25%.",
        "",
        "PARTIAL PASS: Raw KG improves baseline but refined improvement is small,",
        "or refined KG improves but generated dominance is too high.",
        "",
        "FAIL: Raw KG does not improve baseline, or refined KG performs worse than raw KG.",
    ]
    from ..core.utils import ensure_dir
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_plot(path, summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("matplotlib unavailable; skipping plot: %s", exc)
        return

    systems = ["Baseline", "Raw KG", "Refined KG"]
    scores = [
        summary["average_baseline_score"],
        summary["average_raw_kg_score"],
        summary["average_refined_kg_score"],
    ]
    colors = ["#999999", "#3A7CA5", "#A24C9F"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(systems, scores, color=colors)
    ax.set_ylabel("Average score (out of 100)")
    ax.set_title("Cultural KG MVP: average scores by system")
    ax.set_ylim(0, 100)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{score}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
