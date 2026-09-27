"""Shared helpers for the eval-v2 drivers (see docs/EVAL_V2_PLAN.md).

Loads the August KG artifacts read-only and builds one RetrievalContext, reusing
the loading code of ``eval_culturalbench.py`` so every evaluation sees exactly the
same indexes as the CulturalBench run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core import config  # noqa: E402
from src.evaluation.mcq_systems import RetrievalContext  # noqa: E402

from eval_culturalbench import _clean_row, _load_corpus, _read_rows, _to_triples  # noqa: E402,F401

EVAL_V2_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_v2"
EVAL_V2_OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "outputs" / "eval_v2"


def build_retrieval_context(*,
                            need_passages: bool = True,
                            need_triples: bool = True,
                            need_refined: bool = True,
                            need_pathways: bool = True) -> RetrievalContext:
    """Build the retrieval context over the August KG (read-only)."""
    passages, triples, refined, pathway_retriever = _load_corpus(
        config.PROCESSED_DIR / "corpus_passages.jsonl",
        config.PROCESSED_DIR / "corpus_triples.jsonl",
        Path(config.CORPUS_TRIPLES_REFINED_PATH),
        Path(config.PATHWAY_INDEX_DIR),
        need_passages, need_triples, need_refined, need_pathways,
    )
    return RetrievalContext(
        passages=passages,
        triples=triples,
        refined_triples=refined,
        pathway_retriever=pathway_retriever,
    )
