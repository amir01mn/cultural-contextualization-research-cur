"""Step 3: build and export a cultural KG from triples JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.graph_builder import build_graph, export_all  # noqa: E402
from src.schemas import CulturalTriple  # noqa: E402
from src.utils import get_logger, read_jsonl  # noqa: E402

LOGGER = get_logger("03_build_graph")


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build cultural KG.")
    p.add_argument("--input",
                   default=str(config.PROCESSED_DIR / "raw_triples_100.jsonl"))
    p.add_argument("--prefix", default="raw_cultural_kg")
    args = p.parse_args(argv)

    triples = [CulturalTriple(**t) for t in read_jsonl(args.input)]
    LOGGER.info("Loaded %d triples.", len(triples))
    graph = build_graph(triples)
    paths = export_all(graph, args.prefix)
    for fmt, path in paths.items():
        LOGGER.info("  %s -> %s", fmt, path)


if __name__ == "__main__":
    main()
