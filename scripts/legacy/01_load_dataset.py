"""LEGACY MVP step 1 (not the research pipeline; see scripts/build_corpus.py): load and normalize CultureBank records to JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core import config  # noqa: E402
from src.loaders.culturebank_loader import load_culturebank  # noqa: E402
from src.core.utils import get_logger, write_jsonl  # noqa: E402

LOGGER = get_logger("01_load_dataset")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Load CultureBank.")
    p.add_argument("--dataset", default=config.DATASET_NAME)
    p.add_argument("--split", default=config.DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT)
    p.add_argument("--output",
                   default=str(config.PROCESSED_DIR / "culturebank_sample_100.jsonl"))
    args = p.parse_args(argv)

    records = load_culturebank(args.dataset, args.split, args.limit)
    count = write_jsonl(args.output, records)
    LOGGER.info("Wrote %d records -> %s", count, args.output)


if __name__ == "__main__":
    main()
