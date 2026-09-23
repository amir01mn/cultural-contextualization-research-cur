"""Step 6: build the CultureBank retrieval corpus (Phase 1 + Phase 2 pipeline).

Produces:
  - raw-text passages (S1)
  - raw + refined triple stores (S2/S3)
  - entity map, alignments, pathways, pathway index
  - resolved graph exports

Example (cluster):
    python scripts/06_build_retrieval_corpus.py --provider vllm --limit 2000 --resume
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.cross_cultural_aligner import align_cross_cultural  # noqa: E402
from src.critique_agent import critique_records_batched_resumable  # noqa: E402
from src.data_loader import load_culturebank  # noqa: E402
from src.entity_resolver import resolve_entities  # noqa: E402
from src.graph_builder import build_graph, build_graph_resolved, export_all  # noqa: E402
from src.llm_backend import get_backend  # noqa: E402
from src.pathway_builder import build_pathways  # noqa: E402
from src.pathway_retriever import build_pathway_index  # noqa: E402
from src.schemas import CulturalTriple  # noqa: E402
from src.storage import write_parquet  # noqa: E402
from src.triple_extractor import extract_triples_llm_batched_resumable  # noqa: E402
from src.utils import get_logger, normalize_text, read_jsonl, write_jsonl  # noqa: E402

LOGGER = get_logger("06_build_corpus")


def _record_passage(rec: dict) -> dict:
    parts = []
    for key in ("cultural_group", "context", "actor", "recipient", "relation",
                "actor_behavior", "recipient_behavior", "topic", "source_text"):
        val = normalize_text(rec.get(key))
        if val:
            parts.append(val)
    text = ". ".join(dict.fromkeys(parts))
    return {
        "passage_id": rec.get("instance_id", ""),
        "instance_id": rec.get("instance_id", ""),
        "dataset_id": rec.get("dataset_id", "culturebank"),
        "cultural_group": rec.get("cultural_group"),
        "topic": rec.get("topic"),
        "text": text or normalize_text(rec.get("source_text")),
    }


def _load_triples_by_record(path: Path) -> dict[str, list[CulturalTriple]]:
    by_record: dict[str, list[CulturalTriple]] = defaultdict(list)
    if not path.exists():
        return by_record
    for row in read_jsonl(path):
        t = CulturalTriple(**row)
        by_record[t.instance_id].append(t)
    return dict(by_record)


def _load_triples_flat(path: Path) -> list[CulturalTriple]:
    if not path.exists():
        return []
    return [CulturalTriple(**row) for row in read_jsonl(path)]


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build CultureBank retrieval corpus (Phase 2).")
    p.add_argument("--provider", default=config.LLM_PROVIDER, choices=["ollama", "vllm"])
    p.add_argument("--model", default=None)
    p.add_argument("--dataset", default=config.DATASET_NAME)
    p.add_argument("--split", default=config.DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=config.CORPUS_LIMIT)
    p.add_argument("--batch_size", type=int, default=config.CORPUS_BATCH_SIZE)
    p.add_argument("--allow_fallback", action="store_true", default=False)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", action="store_true", default=False,
                   help="Disable checkpoint resume.")
    p.add_argument("--skip_extraction", action="store_true", default=False)
    p.add_argument("--skip_critique", action="store_true", default=False)
    p.add_argument("--skip_alignment", action="store_true", default=False)
    p.add_argument("--skip_alignment_verify", action="store_true", default=False)
    p.add_argument("--passages_output",
                   default=str(config.PROCESSED_DIR / "corpus_passages.jsonl"))
    p.add_argument("--triples_output",
                   default=str(config.PROCESSED_DIR / "corpus_triples.jsonl"))
    p.add_argument("--refined_output", default=str(config.CORPUS_TRIPLES_REFINED_PATH))
    p.add_argument("--state_path", default=str(config.CORPUS_STATE_PATH))
    p.add_argument("--cache_dir", default=str(config.LLM_CACHE_DIR))
    p.add_argument("--input_records", default=None)
    args = p.parse_args(argv)

    resume = args.resume and not args.no_resume

    LOGGER.info("Loading up to %d CultureBank records...", args.limit)

    if args.input_records:
        LOGGER.info("Loading records from %s...", args.input_records)
        records = list(read_jsonl(args.input_records))
        if args.limit:
            records = records[:args.limit]
    else:
        LOGGER.info("Loading up to %d CultureBank records...", args.limit)
        records = load_culturebank(args.dataset, args.split, args.limit)

    LOGGER.info("Loaded %d records.", len(records))
    passages = [_record_passage(r) for r in records]
    write_jsonl(args.passages_output, passages)
    LOGGER.info("Wrote %d passages -> %s", len(passages), args.passages_output)

    backend = get_backend(args.provider, model=args.model)
    LOGGER.info("Backend: %s (model=%s)", backend.name, backend.model)
    status = backend.status()
    LOGGER.info("Backend status: %s", status.get("message", ""))
    if not status.get("available") and not args.allow_fallback:
        LOGGER.error("Backend unavailable and --allow_fallback not set. Aborting.")
        sys.exit(2)
    if hasattr(backend, "warmup"):
        backend.warmup()

    triples_path = Path(args.triples_output)
    if not args.skip_extraction:
        extract_triples_llm_batched_resumable(
            records,
            backend,
            state_path=args.state_path,
            output_path=str(triples_path),
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            allow_fallback=args.allow_fallback,
            resume=resume,
        )
    else:
        LOGGER.info("Skipping extraction (--skip_extraction).")

    triples_by_record = _load_triples_by_record(triples_path)
    raw_triples = _load_triples_flat(triples_path)
    LOGGER.info("Raw triples: %d from %d records.", len(raw_triples), len(triples_by_record))

    refined_path = Path(args.refined_output)
    critiques_path = config.PROCESSED_DIR / "corpus_critiques.jsonl"

    if not args.skip_critique:
        _critiques, refined_triples = critique_records_batched_resumable(
            records,
            triples_by_record,
            backend,
            state_path=args.state_path,
            critiques_output=str(critiques_path),
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            allow_fallback=args.allow_fallback,
            resume=resume,
        )
        write_jsonl(str(refined_path), [t.model_dump() for t in refined_triples])
        LOGGER.info("Refined triples: %d -> %s", len(refined_triples), refined_path)
    else:
        LOGGER.info("Skipping critique (--skip_critique).")
        refined_triples = _load_triples_flat(refined_path) or raw_triples

    # Raw inspection graph (Phase 1 parity)
    if raw_triples:
        export_all(build_graph(raw_triples), "corpus_cultural_kg")

    # Phase 2: entity resolution → alignment → pathways
    entities, label_map = resolve_entities(refined_triples, records)

    if not args.skip_alignment:
        alignments, entities = align_cross_cultural(
            entities,
            backend=backend if not args.skip_alignment_verify else None,
            verify=not args.skip_alignment_verify,
        )
    else:
        LOGGER.info("Skipping cross-cultural alignment.")
        alignments = []

    # Persist the post-alignment entity state so JSONL and Parquet artifacts
    # agree on shared-abstraction flags and alignment provenance.
    write_jsonl(str(config.ENTITY_MAP_PATH), [e.model_dump() for e in entities])
    write_jsonl(str(config.ALIGNMENTS_PATH), [a.model_dump() for a in alignments])

    resolved_graph = build_graph_resolved(refined_triples, entities, label_map)
    export_all(resolved_graph, "resolved_cultural_kg")

    pathways = build_pathways(resolved_graph, entities, alignments)
    write_jsonl(str(config.PATHWAYS_PATH), [p.model_dump() for p in pathways])

    write_parquet(str(config.PROCESSED_DIR / "corpus_triples.parquet"),
                  [t.model_dump() for t in raw_triples])
    write_parquet(str(config.PROCESSED_DIR / "corpus_triples_refined.parquet"),
                  [t.model_dump() for t in refined_triples])
    write_parquet(str(config.PROCESSED_DIR / "entity_map.parquet"),
                  [e.model_dump() for e in entities])
    write_parquet(str(config.PROCESSED_DIR / "corpus_pathways.parquet"),
                  [p.model_dump() for p in pathways])

    build_pathway_index(pathways, config.PATHWAY_INDEX_DIR)

    LOGGER.info("Phase 2 corpus build complete.")
    LOGGER.info("  entities: %d", len(entities))
    LOGGER.info("  alignments: %d", len(alignments))
    LOGGER.info("  pathways: %d", len(pathways))


if __name__ == "__main__":
    main()
