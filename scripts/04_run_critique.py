"""Step 4: run the critique agent and build refined triples + refined KG."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, local_llm  # noqa: E402
from src.critique_agent import build_refined_triples, critique_record  # noqa: E402
from src.graph_builder import build_graph, export_all  # noqa: E402
from src.schemas import CulturalTriple  # noqa: E402
from src.utils import get_logger, read_jsonl, write_jsonl  # noqa: E402

LOGGER = get_logger("04_run_critique")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run critique agent.")
    p.add_argument("--records",
                   default=str(config.PROCESSED_DIR / "culturebank_sample_100.jsonl"))
    p.add_argument("--triples",
                   default=str(config.PROCESSED_DIR / "raw_triples_100.jsonl"))
    p.add_argument("--critic", default="llm_based",
                   choices=["llm_based", "rule_based"])
    p.add_argument("--ollama_model", default=config.OLLAMA_MODEL)
    p.add_argument("--allow_fallback", action="store_true", default=False)
    p.add_argument("--refined_output",
                   default=str(config.PROCESSED_DIR / "refined_triples_100.jsonl"))
    p.add_argument("--critique_output",
                   default=str(config.PROCESSED_DIR / "critiques_100.jsonl"))
    args = p.parse_args(argv)

    if args.critic == "llm_based":
        status = local_llm.check_ollama_available(model=args.ollama_model)
        if not (status["available"] and status["model_present"]) and not args.allow_fallback:
            LOGGER.error("Local LLM unavailable: %s", status["message"])
            LOGGER.error("Re-run with --allow_fallback or --critic rule_based.")
            sys.exit(2)

    records = list(read_jsonl(args.records))
    records_by_id = {r["instance_id"]: r for r in records}
    triples_by_record: dict[str, list] = defaultdict(list)
    for t in read_jsonl(args.triples):
        triple = CulturalTriple(**t)
        triples_by_record[triple.instance_id].append(triple)

    critiques = []
    from tqdm import tqdm
    for rec in tqdm(records, desc="Critiquing"):
        triples = triples_by_record.get(rec["instance_id"], [])
        if not triples:
            continue
        critiques.append(critique_record(
            rec, triples, mode=args.critic,
            allow_fallback=args.allow_fallback, model=args.ollama_model))

    write_jsonl(args.critique_output, [c.model_dump() for c in critiques])
    LOGGER.info("Wrote %d critiques -> %s", len(critiques), args.critique_output)

    refined = build_refined_triples(records_by_id, dict(triples_by_record), critiques)
    write_jsonl(args.refined_output, [t.model_dump() for t in refined])
    LOGGER.info("Wrote %d refined triples -> %s", len(refined), args.refined_output)

    graph = build_graph(refined)
    export_all(graph, "refined_cultural_kg")


if __name__ == "__main__":
    main()
