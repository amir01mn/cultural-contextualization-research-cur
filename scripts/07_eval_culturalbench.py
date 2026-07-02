"""Step 7: evaluate systems S0-S3 on CulturalBench and save results.

Loads the held-out CulturalBench benchmark, loads the CultureBank retrieval
corpus (built by scripts/06_build_retrieval_corpus.py), runs each system with a
shared backend, scores accuracy with bootstrap CIs + McNemar, and writes outputs.

Examples:
    # Local smoke test (Ollama), tiny:
    python scripts/07_eval_culturalbench.py --provider ollama --limit 6 --systems S0_no_kg,S3_layered_kg

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
from src.mcq_systems import ALL_SYSTEMS, KG_SYSTEMS, RetrievalContext, run_system  # noqa: E402
from src.schemas import CulturalTriple  # noqa: E402
from src.utils import get_logger, read_jsonl  # noqa: E402

LOGGER = get_logger("07_eval_culturalbench")


def _load_corpus(passages_path: Path, triples_path: Path, need_passages: bool,
                 need_triples: bool):
    passages = None
    triples = None
    if need_passages and passages_path.exists():
        passages = list(read_jsonl(passages_path))
        LOGGER.info("Loaded %d retrieval passages.", len(passages))
    elif need_passages:
        LOGGER.warning("Passages file not found at %s; S1 will have no evidence.",
                       passages_path)
    if need_triples and triples_path.exists():
        triples = [CulturalTriple(**t) for t in read_jsonl(triples_path)]
        LOGGER.info("Loaded %d corpus triples.", len(triples))
    elif need_triples:
        LOGGER.warning("Triples file not found at %s; S2/S3 will have no evidence. "
                       "Run scripts/06_build_retrieval_corpus.py first.", triples_path)
    return passages, triples


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate S0-S3 on CulturalBench.")
    p.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument("--variants", default="easy,hard")
    p.add_argument("--limit", type=int, default=250,
                   help="Cap on Easy questions and Hard groups.")
    p.add_argument("--allow_fallback", action="store_true", default=False,
                   help="Use tiny built-in sample if HF load fails (dev/tests only).")
    p.add_argument("--top_k", type=int, default=config.RETRIEVAL_TOP_K)
    p.add_argument("--systems", default=",".join(ALL_SYSTEMS),
                   help="Comma-separated subset of S0_no_kg,S1_raw_text_rag,S2_flat_kg,S3_layered_kg")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--passages",
                   default=str(config.PROCESSED_DIR / "corpus_passages.jsonl"))
    p.add_argument("--triples",
                   default=str(config.PROCESSED_DIR / "corpus_triples.jsonl"))
    args = p.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    invalid = [s for s in systems if s not in ALL_SYSTEMS]
    if invalid:
        LOGGER.error("Unknown systems: %s. Valid: %s", invalid, ALL_SYSTEMS)
        sys.exit(2)

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    questions = load_culturalbench(variants=variants, limit=args.limit,
                                   allow_fallback=args.allow_fallback)
    if any(q.question_id.endswith("_fb1") or "_fb" in q.question_id for q in questions):
        LOGGER.error("Loaded fallback CulturalBench IDs; refusing to evaluate.")
        sys.exit(2)
    LOGGER.info("Loaded %d CulturalBench items.", len(questions))

    need_passages = "S1_raw_text_rag" in systems
    need_triples = any(s in KG_SYSTEMS for s in systems)
    passages, triples = _load_corpus(Path(args.passages), Path(args.triples),
                                     need_passages, need_triples)
    ctx = RetrievalContext(passages=passages, triples=triples)

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
    }
    paths = save_outputs(questions, answers_by_system, summary, units_meta,
                         correctness, run_meta=run_meta)

    _print_summary(summary, run_meta)
    for name, path in paths.items():
        LOGGER.info("  %s -> %s", name, path)


def _print_summary(summary: dict, run_meta: dict) -> None:
    combined = summary["accuracy"]["combined"]
    print("\n===== CULTURALBENCH EVALUATION (Phase 1: S0-S3) =====\n")
    print(f"Provider: {run_meta['provider']}   Model: {run_meta['model']}")
    print(f"Units: {summary['num_units_total']} "
          f"(easy={summary['num_units_easy']}, hard={summary['num_units_hard']})")
    print(f"Retrieval top-k: {run_meta['top_k']}   "
          f"corpus triples: {run_meta['corpus_triples']}\n")
    print("Combined accuracy (95% CI):")
    for s in summary["systems"]:
        e = combined.get(s, {})
        if e:
            print(f"  {s:18s}: {e['accuracy']*100:5.1f}%  "
                  f"[{e['ci_low']*100:.1f}, {e['ci_high']*100:.1f}]")
    print("\nKey McNemar comparisons:")
    for key, m in summary.get("mcnemar_combined", {}).items():
        print(f"  {key.replace('__vs__',' vs '):40s} "
              f"delta={m['delta_accuracy']*100:+5.1f}%  p={m.get('p_value')}")
    print("\nOutputs saved to: data/outputs/\n")


if __name__ == "__main__":
    main()
