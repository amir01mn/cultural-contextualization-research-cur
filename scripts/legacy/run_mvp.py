"""LEGACY end-to-end MVP runner (100-sample local LLM experiment).

Example:
    python scripts/legacy/run_mvp.py \
        --dataset culturebank --split train --limit 100 \
        --extraction_mode llm_based --critic llm_based \
        --llm_provider ollama --ollama_model llama3.1:8b

By default this requires a running local Ollama LLM (no paid APIs). If the LLM
is unavailable and --allow_fallback is NOT set, the run stops with instructions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/legacy/run_mvp.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import config  # noqa: E402
from src.llm import local_llm  # noqa: E402
from src.construction.critique_agent import build_refined_triples, critique_record  # noqa: E402
from src.loaders.culturebank_loader import load_culturebank  # noqa: E402
from src.legacy.evaluator import (  # noqa: E402
    evaluate_examples,
    generate_scenarios,
    save_evaluation_outputs,
)
from src.construction.graph_builder import build_graph, export_all  # noqa: E402
from src.construction.triple_extractor import extract_triples  # noqa: E402
from src.core.utils import get_logger, write_jsonl, write_json  # noqa: E402

LOGGER = get_logger("run_mvp")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the cultural KG MVP pipeline.")
    p.add_argument("--dataset", default=config.DATASET_NAME)
    p.add_argument("--split", default=config.DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    p.add_argument("--extraction_mode", default="llm_based",
                   choices=["llm_based", "schema_fallback", "both"])
    p.add_argument("--critic", default="llm_based",
                   choices=["llm_based", "rule_based"])
    p.add_argument("--llm_provider", default=config.LLM_PROVIDER)
    p.add_argument("--ollama_model", default=config.OLLAMA_MODEL)
    p.add_argument("--num_scenarios", type=int, default=config.NUM_SCENARIOS)
    p.add_argument("--top_k", type=int, default=config.RETRIEVAL_TOP_K)
    p.add_argument("--allow_fallback", action="store_true", default=False,
                   help="Use schema/rule-based fallback if the local LLM is "
                        "unavailable. Default: false (real LLM required).")
    return p.parse_args(argv)


def _preflight_llm(args) -> bool:
    """Check Ollama. Returns True if real LLM should be used.

    Stops the program if the LLM is unavailable and fallback is not allowed.
    """
    needs_llm = args.extraction_mode != "schema_fallback" or args.critic == "llm_based"
    status = local_llm.check_ollama_available(model=args.ollama_model)
    LOGGER.info("Ollama status: %s", status["message"])

    if status["available"] and status["model_present"]:
        LOGGER.info("Warming up model '%s' (first load can take a few minutes)...",
                    args.ollama_model)
        warm = local_llm.warmup_model(model=args.ollama_model)
        LOGGER.info("Warmup: %s", warm["message"])
        return True

    # LLM not fully available.
    if args.allow_fallback:
        LOGGER.warning("Local LLM unavailable/model missing; running in FALLBACK mode "
                       "(rule/schema-based). Results are NOT a true LLM experiment.")
        return False

    # Hard stop with helpful instructions.
    print("\n" + "=" * 70)
    print("LOCAL LLM NOT AVAILABLE")
    print("=" * 70)
    print(status["message"])
    print("\nThis experiment requires real local LLM extraction. To proceed:")
    print("  1. Install Ollama:  https://ollama.com/download")
    print("  2. Start it:        ollama serve")
    print(f"  3. Pull the model:  ollama pull {args.ollama_model}")
    print("\nThen re-run this command.")
    print("\nAlternatively, to run with rule/schema-based fallback (NOT a real")
    print("LLM experiment), re-run with the flag:  --allow_fallback")
    print("=" * 70 + "\n")
    sys.exit(2)


def main(argv=None) -> None:
    args = parse_args(argv)
    config.OLLAMA_MODEL = args.ollama_model  # honor CLI override at runtime

    use_llm = _preflight_llm(args)
    model = args.ollama_model

    # 1. Load records.
    LOGGER.info("Loading %d records from %s (%s)...", args.limit, args.dataset, args.split)
    records = load_culturebank(args.dataset, args.split, args.limit)

    # 2. Save normalized records.
    records_path = config.PROCESSED_DIR / f"culturebank_sample_{args.limit}.jsonl"
    write_jsonl(records_path, records)
    LOGGER.info("Saved normalized records -> %s", records_path)

    records_by_id = {r["instance_id"]: r for r in records}

    # 3. Extract triples with the local LLM (main method).
    LOGGER.info("Extracting triples (mode=%s)...", args.extraction_mode)
    triples_by_record: dict[str, list] = {}
    all_raw_triples = []
    from tqdm import tqdm
    for rec in tqdm(records, desc="Extracting triples"):
        triples = extract_triples(rec, args.extraction_mode,
                                  allow_fallback=args.allow_fallback, model=model)
        triples_by_record[rec["instance_id"]] = triples
        all_raw_triples.extend(triples)

    # 4. Save raw triples.
    raw_triples_path = config.PROCESSED_DIR / f"raw_triples_{args.limit}.jsonl"
    write_jsonl(raw_triples_path, [t.model_dump() for t in all_raw_triples])
    LOGGER.info("Extracted %d raw triples -> %s", len(all_raw_triples), raw_triples_path)

    # 5. Build raw KG.
    raw_graph = build_graph(all_raw_triples)
    export_all(raw_graph, "raw_cultural_kg")

    # 6. Critique each record.
    LOGGER.info("Running critique agent (critic=%s)...", args.critic)
    critiques = []
    for rec in tqdm(records, desc="Critiquing"):
        triples = triples_by_record.get(rec["instance_id"], [])
        if not triples:
            continue
        critiques.append(critique_record(
            rec, triples, mode=args.critic,
            allow_fallback=args.allow_fallback, model=model))

    # 7. Save critique results.
    critique_path = config.PROCESSED_DIR / f"critiques_{args.limit}.jsonl"
    write_jsonl(critique_path, [c.model_dump() for c in critiques])
    num_suggestions = sum(len(c.suggested_triples) for c in critiques)
    LOGGER.info("Critique produced %d suggested triples -> %s",
                num_suggestions, critique_path)

    # 8. Build refined triples.
    refined_triples = build_refined_triples(records_by_id, triples_by_record, critiques)
    refined_path = config.PROCESSED_DIR / f"refined_triples_{args.limit}.jsonl"
    write_jsonl(refined_path, [t.model_dump() for t in refined_triples])
    LOGGER.info("Refined triples total: %d -> %s", len(refined_triples), refined_path)

    # 9. Build refined KG.
    refined_graph = build_graph(refined_triples)
    export_all(refined_graph, "refined_cultural_kg")

    # 10. Generate scenarios.
    LOGGER.info("Generating ~%d scenario questions...", args.num_scenarios)
    examples = generate_scenarios(
        records, args.num_scenarios,
        use_llm=use_llm,
        allow_fallback=args.allow_fallback, model=model)
    LOGGER.info("Generated %d scenario examples.", len(examples))

    # 11-14. Generate baseline / raw / refined answers and score them.
    LOGGER.info("Evaluating three systems on %d examples...", len(examples))
    results, extras = evaluate_examples(
        examples, all_raw_triples, refined_triples,
        top_k=args.top_k, model=model,
        use_llm_answers=use_llm, allow_fallback=args.allow_fallback)

    # 15. Save outputs.
    run_meta = {
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "records_loaded": len(records),
        "llm_provider": args.llm_provider,
        "model": model,
        "extraction_mode": args.extraction_mode,
        "critic": args.critic,
        "used_real_llm": use_llm,
        "raw_triples": len(all_raw_triples),
        "critique_suggestions": num_suggestions,
        "refined_triples": len(refined_triples),
    }
    paths = save_evaluation_outputs(examples, results, extras, run_meta=run_meta)
    write_json(config.OUTPUTS_DIR / "run_meta.json", run_meta)

    # 16. Print summary.
    print_summary(extras["summary"], run_meta)
    LOGGER.info("All outputs saved under %s", config.OUTPUTS_DIR)


def print_summary(summary: dict, meta: dict) -> None:
    print("\n===== CULTURAL KG MVP: 100-SAMPLE LOCAL LLM EXPERIMENT =====\n")
    print(f"Dataset: CultureBank")
    print(f"Records loaded: {meta['records_loaded']}")
    print(f"LLM provider: {meta['llm_provider'].capitalize()}")
    print(f"Model: {meta['model']}")
    if not meta.get("used_real_llm", True):
        print("(NOTE: ran in FALLBACK mode - not a true local LLM experiment)")
    print()
    print(f"Raw triples extracted: {meta['raw_triples']}")
    print(f"Critique suggestions generated: {meta['critique_suggestions']}")
    print(f"Refined triples total: {meta['refined_triples']}")
    print()
    print("Average scores:")
    print(f"Baseline: {summary['average_baseline_score']}")
    print(f"Raw KG: {summary['average_raw_kg_score']}")
    print(f"Refined KG: {summary['average_refined_kg_score']}")
    print()
    print("Improvement:")
    print(f"Raw KG - Baseline: {summary['raw_kg_improvement_over_baseline']:+}")
    print(f"Refined KG - Raw KG: {summary['refined_kg_improvement_over_raw']:+}")
    print()
    print(f"Generated support ratio: {summary['generated_support_ratio_percent']}%")
    print(f"Layer 2 coverage: {summary['layer_2_coverage_percent']}%")
    print(f"Layer 3 coverage: {summary['layer_3_coverage_percent']}%")
    print()
    print(f"Result: {summary['verdict']}")
    print()
    print("Outputs saved to:")
    print("data/outputs/")
    print()


if __name__ == "__main__":
    main()
