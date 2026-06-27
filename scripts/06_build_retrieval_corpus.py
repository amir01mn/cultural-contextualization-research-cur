"""Step 6: build the CultureBank retrieval corpus for the KG-augmented systems.

Produces the artifacts the MCQ systems retrieve over:
  - a raw-text corpus (for S1 Raw-text RAG): one record-derived passage each;
  - a triple store (for S2 Flat KG and S3 Layered KG), extracted with the local
    LLM via the pluggable backend (batched on vLLM);
  - the networkx graph export for inspection.

Entity resolution, cross-cultural alignment, pathway construction, and the S4
system are intentionally OUT OF SCOPE here (later phases). This is just the
retrieval substrate so we can get a first CulturalBench number.

Example (cluster):
    python scripts/06_build_retrieval_corpus.py --provider vllm --limit 2000
Example (local dev):
    python scripts/06_build_retrieval_corpus.py --provider ollama --limit 50
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.data_loader import load_culturebank  # noqa: E402
from src.graph_builder import build_graph, export_all  # noqa: E402
from src.llm_backend import get_backend  # noqa: E402
from src.triple_extractor import extract_triples_llm_batched  # noqa: E402
from src.utils import get_logger, normalize_text, write_jsonl  # noqa: E402

LOGGER = get_logger("06_build_corpus")


def _record_passage(rec: dict) -> dict:
    """A retrievable raw-text passage for S1 (Raw-text RAG)."""
    parts = []
    for key in ("cultural_group", "context", "actor", "recipient", "relation",
                "actor_behavior", "recipient_behavior", "topic", "source_text"):
        val = normalize_text(rec.get(key))
        if val:
            parts.append(val)
    text = ". ".join(dict.fromkeys(parts))  # dedupe, keep order
    return {
        "passage_id": rec.get("instance_id", ""),
        "instance_id": rec.get("instance_id", ""),
        "dataset_id": rec.get("dataset_id", "culturebank"),
        "cultural_group": rec.get("cultural_group"),
        "topic": rec.get("topic"),
        "text": text or normalize_text(rec.get("source_text")),
    }


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build CultureBank retrieval corpus.")
    p.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "vllm"])
    p.add_argument("--model", default=None, help="Override model id for the provider.")
    p.add_argument("--dataset", default=config.DATASET_NAME)
    p.add_argument("--split", default=config.DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=config.CORPUS_LIMIT)
    p.add_argument("--allow_fallback", action="store_true", default=False,
                   help="Use schema fallback for records the LLM can't parse.")
    p.add_argument("--passages_output",
                   default=str(config.PROCESSED_DIR / "corpus_passages.jsonl"))
    p.add_argument("--triples_output",
                   default=str(config.PROCESSED_DIR / "corpus_triples.jsonl"))
    args = p.parse_args(argv)

    LOGGER.info("Loading up to %d CultureBank records...", args.limit)
    records = load_culturebank(args.dataset, args.split, args.limit)
    LOGGER.info("Loaded %d records.", len(records))

    # Raw-text passages (S1).
    passages = [_record_passage(r) for r in records]
    write_jsonl(args.passages_output, passages)
    LOGGER.info("Wrote %d raw-text passages -> %s", len(passages), args.passages_output)

    # Batched LLM triple extraction (S2/S3).
    backend = get_backend(args.provider, model=args.model)
    LOGGER.info("Extraction backend: %s (model=%s)", backend.name, backend.model)
    status = backend.status()
    LOGGER.info("Backend status: %s", status.get("message", ""))
    if not status.get("available") and not args.allow_fallback:
        LOGGER.error("Backend unavailable and --allow_fallback not set. Aborting.")
        sys.exit(2)
    if hasattr(backend, "warmup"):
        backend.warmup()

    triples_by_record = extract_triples_llm_batched(
        records, backend, allow_fallback=args.allow_fallback)
    all_triples = [t for ts in triples_by_record.values() for t in ts]
    write_jsonl(args.triples_output, [t.model_dump() for t in all_triples])
    LOGGER.info("Extracted %d triples from %d records -> %s",
                len(all_triples), len(records), args.triples_output)

    # Export the graph for inspection.
    graph = build_graph(all_triples)
    export_all(graph, "corpus_cultural_kg")
    LOGGER.info("Corpus build complete.")


if __name__ == "__main__":
    main()
