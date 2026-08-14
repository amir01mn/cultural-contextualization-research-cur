"""Step 7: evaluate systems S0-S4 on CulturalBench and save results.

Loads the held-out CulturalBench benchmark, loads the CultureBank retrieval
corpus (built by scripts/06_build_retrieval_corpus.py), runs each system with a
shared backend, scores accuracy with bootstrap CIs + McNemar, and writes outputs.

Examples:
    # Local smoke test (Ollama), tiny:
    python scripts/07_eval_culturalbench.py --provider ollama --limit 6 --systems S0_no_kg,S4_abstraction_refinement_kg

    # Cluster (vLLM), full:
    python scripts/07_eval_culturalbench.py --provider vllm --limit 250
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
from src.schemas import CulturalTriple  # noqa: E402
from src.utils import get_logger, read_jsonl  # noqa: E402

LOGGER = get_logger("07_eval_culturalbench")


def _read_rows(path: Path) -> list[dict]:
    """Read triple rows, preferring a sibling Parquet file when present.

    The JSONL stores are ~430 MB and ~525 MB; their Parquet equivalents are
    ~10x smaller and read columnar in one pass. On a shared Lustre filesystem
    the difference is large - parsing the JSONL line by line took ~35 minutes of
    an evaluation job's wall clock.
    """
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        from src.storage import read_parquet

        rows = read_parquet(parquet_path)
        LOGGER.info("Read %d rows from %s (parquet).", len(rows), parquet_path.name)
        return rows
    return list(read_jsonl(path))


def _clean_row(row: dict) -> dict:
    """Normalize a Parquet-decoded row back to JSON-like Python values.

    Parquet has no None for a float column, so optional strings that were null
    come back as NaN and fail validation. Numpy scalars and arrays likewise need
    unwrapping before pydantic sees them.
    """
    cleaned: dict = {}
    for key, value in row.items():
        if value is None:
            cleaned[key] = None
        elif isinstance(value, float) and value != value:  # NaN
            cleaned[key] = None
        elif hasattr(value, "item") and getattr(value, "ndim", None) == 0:
            cleaned[key] = value.item()
        elif hasattr(value, "tolist"):
            cleaned[key] = value.tolist()
        else:
            cleaned[key] = value
    return cleaned


def _to_triples(rows: list[dict]) -> list[CulturalTriple]:
    return [CulturalTriple(**_clean_row(r)) for r in rows if isinstance(r, dict)]


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
        triples = _to_triples(_read_rows(triples_path))
        LOGGER.info("Loaded %d raw corpus triples.", len(triples))
    elif need_triples:
        LOGGER.warning("Triples file not found at %s; S2/S3 will have no evidence.", triples_path)

    if need_refined and refined_path.exists():
        refined = _to_triples(_read_rows(refined_path))
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
    p.add_argument("--min_score", type=float, default=None,
                   help="Retrieval relevance floor; below it no evidence is "
                        "injected and the system answers closed-book.")
    p.add_argument("--sweep_min_score", default=None,
                   help="Comma-separated floors to sweep after the main run "
                        "(e.g. 0.0,0.1,0.2,0.3). Reuses the loaded model and "
                        "corpus, so each extra point costs only inference.")
    p.add_argument("--sweep_systems", default=None,
                   help="Systems to include in the sweep (default: the KG "
                        "systems being evaluated, plus S0 as the reference).")
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

    if args.min_score is not None:
        config.RETRIEVAL_MIN_SCORE = args.min_score
        LOGGER.info("Retrieval relevance floor: %.3f", config.RETRIEVAL_MIN_SCORE)

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    questions = load_culturalbench(variants=variants, limit=args.limit,
                                   allow_fallback=args.allow_fallback)
    if any(q.question_id.endswith("_fb1") or "_fb" in q.question_id for q in questions):
        LOGGER.error("Loaded fallback CulturalBench IDs; refusing to evaluate.")
        sys.exit(2)
    LOGGER.info("Loaded %d CulturalBench items.", len(questions))

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

    if args.sweep_min_score:
        _sweep_min_score(args, questions, backend, ctx, systems)


def _sweep_min_score(args, questions, backend, ctx, systems) -> None:
    """Re-evaluate at several relevance floors, reusing the loaded model.

    The floor governs when retrieved evidence is injected at all, which is the
    main precision control: on the previous corpus the KG recovered ~20% of the
    baseline's errors but corrupted ~10-15% of its correct answers, so where the
    floor sits decides whether augmentation is a net gain or a net loss.
    """
    from src.mcq_evaluator import summarize as _summarize

    values = []
    for raw in args.sweep_min_score.split(","):
        raw = raw.strip()
        if raw:
            try:
                values.append(float(raw))
            except ValueError:
                LOGGER.warning("Skipping non-numeric sweep value %r", raw)
    if not values:
        return

    if args.sweep_systems:
        sweep_systems = [s.strip() for s in args.sweep_systems.split(",") if s.strip()]
    else:
        sweep_systems = [s for s in systems if s != "S0_no_kg"]
    reference = "S0_no_kg" if "S0_no_kg" in systems else None

    original = config.RETRIEVAL_MIN_SCORE
    rows: list[tuple[float, dict]] = []
    try:
        for value in values:
            config.RETRIEVAL_MIN_SCORE = value
            LOGGER.info("Sweep: relevance floor = %.3f", value)
            answers = {
                s: run_system(s, questions, backend, ctx, top_k=args.top_k)
                for s in sweep_systems
            }
            summary, _meta, _corr = _summarize(
                questions, answers, n_boot=200, seed=args.seed)
            rows.append((value, summary["accuracy"]["combined"]))
    finally:
        config.RETRIEVAL_MIN_SCORE = original

    print("\n===== RELEVANCE-FLOOR SWEEP =====\n")
    header = "  floor  | " + " | ".join(f"{s[:22]:>22s}" for s in sweep_systems)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for value, combined in rows:
        cells = " | ".join(
            f"{combined.get(s, {}).get('accuracy', 0) * 100:21.1f}%"
            for s in sweep_systems
        )
        print(f"  {value:5.2f}  | {cells}")
    if reference:
        print(f"\n  (reference) {reference}: see main summary above")
    print()


def _print_summary(summary: dict, run_meta: dict) -> None:
    combined = summary["accuracy"]["combined"]
    print("\n===== CULTURALBENCH EVALUATION (S0-S4) =====\n")
    print(f"Provider: {run_meta['provider']}   Model: {run_meta['model']}")
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
