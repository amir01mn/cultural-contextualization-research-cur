"""Step 5: generate scenarios, run 3 systems, score, and save outputs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, local_llm  # noqa: E402
from src.evaluator import (  # noqa: E402
    evaluate_examples,
    generate_scenarios,
    save_evaluation_outputs,
)
from src.schemas import CulturalTriple  # noqa: E402
from src.utils import get_logger, read_jsonl  # noqa: E402

LOGGER = get_logger("05_evaluate")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Evaluate the three systems.")
    p.add_argument("--records",
                   default=str(config.PROCESSED_DIR / "culturebank_sample_100.jsonl"))
    p.add_argument("--raw_triples",
                   default=str(config.PROCESSED_DIR / "raw_triples_100.jsonl"))
    p.add_argument("--refined_triples",
                   default=str(config.PROCESSED_DIR / "refined_triples_100.jsonl"))
    p.add_argument("--num_scenarios", type=int, default=config.NUM_SCENARIOS)
    p.add_argument("--top_k", type=int, default=config.RETRIEVAL_TOP_K)
    p.add_argument("--ollama_model", default=config.OLLAMA_MODEL)
    p.add_argument("--allow_fallback", action="store_true", default=False)
    args = p.parse_args(argv)

    status = local_llm.check_ollama_available(model=args.ollama_model)
    use_llm = status["available"] and status["model_present"]
    if not use_llm and not args.allow_fallback:
        LOGGER.error("Local LLM unavailable: %s", status["message"])
        LOGGER.error("Re-run with --allow_fallback to score empty/fallback answers.")
        sys.exit(2)

    records = list(read_jsonl(args.records))
    raw_triples = [CulturalTriple(**t) for t in read_jsonl(args.raw_triples)]
    refined_triples = [CulturalTriple(**t) for t in read_jsonl(args.refined_triples)]

    examples = generate_scenarios(
        records, args.num_scenarios, use_llm=use_llm,
        allow_fallback=args.allow_fallback, model=args.ollama_model)
    LOGGER.info("Generated %d scenarios.", len(examples))

    results, extras = evaluate_examples(
        examples, raw_triples, refined_triples,
        top_k=args.top_k, model=args.ollama_model,
        use_llm_answers=use_llm, allow_fallback=args.allow_fallback)

    run_meta = {
        "dataset": "culturebank",
        "records_loaded": len(records),
        "llm_provider": config.LLM_PROVIDER,
        "model": args.ollama_model,
        "used_real_llm": use_llm,
        "raw_triples": len(raw_triples),
        "refined_triples": len(refined_triples),
    }
    paths = save_evaluation_outputs(examples, results, extras, run_meta=run_meta)
    for name, path in paths.items():
        LOGGER.info("  %s -> %s", name, path)

    s = extras["summary"]
    LOGGER.info("Verdict: %s | baseline=%.2f raw=%.2f refined=%.2f",
                s["verdict"], s["average_baseline_score"],
                s["average_raw_kg_score"], s["average_refined_kg_score"])


if __name__ == "__main__":
    main()
