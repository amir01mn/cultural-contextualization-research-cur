"""Step 7: evaluate systems S0-S4 on CulturalBench and save results.

Loads the held-out CulturalBench benchmark, loads the CultureBank retrieval
corpus (built by scripts/06_build_retrieval_corpus.py), runs each system with a
shared backend, scores accuracy with bootstrap CIs + McNemar, and writes outputs.

Examples:
    # Local smoke test (Ollama), tiny:
    python scripts/07_eval_culturalbench.py --provider ollama --limit 6 --systems S0_no_kg,S4_abstraction_refinement_kg

    # Cluster (vLLM), full:
    python scripts/07_eval_culturalbench.py --provider vllm --limit 250

    # Controlled selected-question eval:
    python scripts/07_eval_culturalbench.py --provider vllm --questions_jsonl data/controlled/food_hospitality_culturalbench_questions.jsonl --limit 40
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.culturalbench_loader import load_culturalbench  # noqa: E402
from src.llm_backend import get_backend  # noqa: E402
from src.mcq_evaluator import save_outputs, summarize  # noqa: E402
from src.mcq_systems import (  # noqa: E402
    ALL_SYSTEMS,
    S4_SYSTEMS,
    SYSTEM_S2,
    SYSTEM_S3,
    SYSTEM_S4,
    RetrievalContext,
    run_system,
)
from src.pathway_retriever import PathwayRetriever  # noqa: E402
from src.schemas import CulturalTriple, MCQQuestion  # noqa: E402
from src.utils import get_logger, read_jsonl  # noqa: E402

LOGGER = get_logger("07_eval_culturalbench")


def _variant_name(q: MCQQuestion) -> str:
    value = q.variant.value if hasattr(q.variant, "value") else q.variant
    return str(value).lower()


def _limit_by_group(questions: list[MCQQuestion], limit: int | None) -> list[MCQQuestion]:
    """Limit by question group so Hard true/false groups are not split."""
    if not limit:
        return questions

    selected: list[MCQQuestion] = []
    seen_groups: set[str] = set()
    for q in questions:
        group_id = q.group_id or q.question_id
        if group_id not in seen_groups:
            if len(seen_groups) >= limit:
                continue
            seen_groups.add(group_id)
        selected.append(q)
    return selected


def _load_questions_jsonl(path: Path,
                          variants: tuple[str, ...],
                          limit: int | None) -> list[MCQQuestion]:
    """Load preselected normalized CulturalBench questions from JSONL.

    This is used for controlled subset evaluations, such as the selected
    food/hospitality CulturalBench questions. The JSONL rows must already match
    the normalized MCQQuestion schema produced by the CulturalBench loader.
    """
    if not path.exists():
        raise FileNotFoundError(f"Selected question file not found: {path}")

    wanted = {v.lower() for v in variants}
    questions: list[MCQQuestion] = []

    for row in read_jsonl(path):
        try:
            q = MCQQuestion(**row)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Skipping invalid question row in %s: %s", path, exc)
            continue
        if wanted and _variant_name(q) not in wanted:
            continue
        questions.append(q)

    questions = _limit_by_group(questions, limit)
    LOGGER.info("Loaded %d selected CulturalBench items from %s.",
                len(questions), path)
    return questions


def _load_corpus(passages_path: Path,
                 triples_path: Path,
                 refined_path: Path,
                 pathway_index_dir: Path,
                 need_passages: bool,
                 need_triples: bool,
                 need_refined: bool,
                 need_pathways: bool):
    passages = None
    triples = None
    refined = None
    pathway_retriever = None

    if need_passages and passages_path.exists():
        passages = list(read_jsonl(passages_path))
        LOGGER.info("Loaded %d retrieval passages.", len(passages))
    elif need_passages:
        LOGGER.warning("Passages file not found at %s; S1 will have no evidence.", passages_path)

    if need_triples and triples_path.exists():
        triples = [CulturalTriple(**t) for t in read_jsonl(triples_path)]
        LOGGER.info("Loaded %d raw corpus triples.", len(triples))
    elif need_triples:
        LOGGER.warning("Triples file not found at %s; S2/S3 will have no evidence.", triples_path)

    if need_refined and refined_path.exists():
        refined = [CulturalTriple(**t) for t in read_jsonl(refined_path)]
        LOGGER.info("Loaded %d refined corpus triples.", len(refined))
    elif need_refined:
        LOGGER.warning("Refined triples not found at %s; S4 triple ablation empty.", refined_path)

    if need_pathways:
        pathway_retriever = PathwayRetriever(pathway_index_dir)
        LOGGER.info("Pathway index: %d pathways.", len(pathway_retriever.pathways))

    return passages, triples, refined, pathway_retriever


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate S0-S4 on CulturalBench.")
    p.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument("--variants", default="easy,hard")
    p.add_argument("--limit", type=int, default=250,
                   help="Cap on Easy questions and Hard groups.")
    p.add_argument(
        "--questions_jsonl",
        default=None,
        help="Optional JSONL file of preselected normalized CulturalBench questions.",
    )
    p.add_argument("--allow_fallback", action="store_true", default=False,
                   help="Use tiny built-in sample if HF load fails (dev/tests only).")
    p.add_argument("--top_k", type=int, default=config.RETRIEVAL_TOP_K)
    p.add_argument(
        "--systems",
        default=",".join(ALL_SYSTEMS),
        help="Comma-separated subset of: " + ",".join(ALL_SYSTEMS),
    )
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--passages",
                   default=str(config.PROCESSED_DIR / "corpus_passages.jsonl"))
    p.add_argument("--triples",
                   default=str(config.PROCESSED_DIR / "corpus_triples.jsonl"))
    p.add_argument("--refined_triples", default=str(config.CORPUS_TRIPLES_REFINED_PATH))
    p.add_argument("--pathway_index", default=str(config.PATHWAY_INDEX_DIR))
    args = p.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    invalid = [s for s in systems if s not in ALL_SYSTEMS]
    if invalid:
        LOGGER.error("Unknown systems: %s. Valid: %s", invalid, ALL_SYSTEMS)
        sys.exit(2)

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    if args.questions_jsonl:
        questions = _load_questions_jsonl(
            Path(args.questions_jsonl),
            variants=variants,
            limit=args.limit,
        )
    else:
        questions = load_culturalbench(variants=variants, limit=args.limit,
                                       allow_fallback=args.allow_fallback)
        if any(q.question_id.endswith("_fb1") or "_fb" in q.question_id for q in questions):
            LOGGER.error("Loaded fallback CulturalBench IDs; refusing to evaluate.")
            sys.exit(2)
        LOGGER.info("Loaded %d CulturalBench items.", len(questions))

    if not questions:
        LOGGER.error("No CulturalBench questions loaded. Check --variants, --limit, and --questions_jsonl.")
        sys.exit(2)

    need_passages = "S1_raw_text_rag" in systems
    need_triples = any(s in {SYSTEM_S2, SYSTEM_S3} for s in systems) or bool(
        set(systems) & S4_SYSTEMS
    )
    need_refined = (
        SYSTEM_S3 in systems
        or "S4_triple_retrieval" in systems
        or SYSTEM_S4 in systems
        or "S4_no_shared_pivot" in systems
    )
    need_pathways = bool(set(systems) & {s for s in S4_SYSTEMS if s != "S4_triple_retrieval"})

    passages, triples, refined, pathway_retriever = _load_corpus(
        Path(args.passages),
        Path(args.triples),
        Path(args.refined_triples),
        Path(args.pathway_index),
        need_passages,
        need_triples,
        need_refined,
        need_pathways,
    )
    ctx = RetrievalContext(
        passages=passages,
        triples=triples,
        refined_triples=refined,
        pathway_retriever=pathway_retriever,
    )

    backend = get_backend(args.provider, model=args.model)
    LOGGER.info("Answering backend: %s (model=%s)", backend.name, backend.model)
    status = backend.status()
    LOGGER.info("Backend status: %s", status.get("message", ""))
    if not status.get("available"):
        LOGGER.error("Backend unavailable: %s", status.get("message"))
        LOGGER.error("Start Ollama (local) or run on a GPU node with vLLM.")
        sys.exit(2)
    if hasattr(backend, "warmup"):
        backend.warmup()

    answers_by_system = {}
    for system in systems:
        LOGGER.info("Running system %s ...", system)
        answers_by_system[system] = run_system(
            system, questions, backend, ctx, top_k=args.top_k)

    summary, units_meta, correctness = summarize(
        questions, answers_by_system, n_boot=args.n_boot, seed=args.seed)

    run_meta = {
        "provider": args.provider,
        "model": backend.model,
        "top_k": args.top_k,
        "seed": args.seed,
        "variants": list(variants),
        "limit": args.limit,
        "questions_jsonl": args.questions_jsonl,
        "systems": systems,
        "corpus_passages": len(passages) if passages else 0,
        "corpus_triples": len(triples) if triples else 0,
        "corpus_triples_refined": len(refined) if refined else 0,
        "corpus_pathways": len(pathway_retriever.pathways) if pathway_retriever else 0,
    }
    paths = save_outputs(questions, answers_by_system, summary, units_meta,
                         correctness, run_meta=run_meta)

    _print_summary(summary, run_meta)
    for name, path in paths.items():
        LOGGER.info("  %s -> %s", name, path)


def _print_summary(summary: dict, run_meta: dict) -> None:
    combined = summary["accuracy"]["combined"]
    print("\n===== CULTURALBENCH EVALUATION (S0-S4) =====\n")
    print(f"Provider: {run_meta['provider']}   Model: {run_meta['model']}")
    if run_meta.get("questions_jsonl"):
        print(f"Questions: {run_meta['questions_jsonl']}")
    print(f"Units: {summary['num_units_total']} "
          f"(easy={summary['num_units_easy']}, hard={summary['num_units_hard']})")
    print(f"Retrieval top-k: {run_meta['top_k']}   "
          f"corpus triples: {run_meta['corpus_triples']}   "
          f"pathways: {run_meta.get('corpus_pathways', 0)}\n")
    print("Combined accuracy (95% CI):")
    for s in summary["systems"]:
        e = combined.get(s, {})
        if e:
            print(f"  {s:30s}: {e['accuracy']*100:5.1f}%  "
                  f"[{e['ci_low']*100:.1f}, {e['ci_high']*100:.1f}]")
    print("\nKey McNemar comparisons:")
    for key, m in summary.get("mcnemar_combined", {}).items():
        print(f"  {key.replace('__vs__',' vs '):40s} "
              f"delta={m['delta_accuracy']*100:+5.1f}%  p={m.get('p_value')}")
    print("\nOutputs saved to: data/outputs/\n")


if __name__ == "__main__":
    main()
