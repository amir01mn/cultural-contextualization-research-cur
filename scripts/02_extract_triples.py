"""Step 2: extract cultural triples (local LLM main; schema as fallback)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, local_llm  # noqa: E402
from src.triple_extractor import extract_triples  # noqa: E402
from src.utils import get_logger, read_jsonl, write_jsonl  # noqa: E402

LOGGER = get_logger("02_extract_triples")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Extract triples.")
    p.add_argument("--input",
                   default=str(config.PROCESSED_DIR / "culturebank_sample_100.jsonl"))
    p.add_argument("--output",
                   default=str(config.PROCESSED_DIR / "raw_triples_100.jsonl"))
    p.add_argument("--extraction_mode", default="llm_based",
                   choices=["llm_based", "schema_fallback", "both"])
    p.add_argument("--ollama_model", default=config.OLLAMA_MODEL)
    p.add_argument("--allow_fallback", action="store_true", default=False)
    args = p.parse_args(argv)

    if args.extraction_mode != "schema_fallback":
        status = local_llm.check_ollama_available(model=args.ollama_model)
        if not (status["available"] and status["model_present"]) and not args.allow_fallback:
            LOGGER.error("Local LLM unavailable: %s", status["message"])
            LOGGER.error("Re-run with --allow_fallback to use schema fallback.")
            sys.exit(2)

    records = list(read_jsonl(args.input))
    LOGGER.info("Loaded %d records.", len(records))

    all_triples = []
    from tqdm import tqdm
    for rec in tqdm(records, desc="Extracting"):
        all_triples.extend(extract_triples(
            rec, args.extraction_mode,
            allow_fallback=args.allow_fallback, model=args.ollama_model))

    count = write_jsonl(args.output, [t.model_dump() for t in all_triples])
    LOGGER.info("Wrote %d triples -> %s", count, args.output)


if __name__ == "__main__":
    main()
