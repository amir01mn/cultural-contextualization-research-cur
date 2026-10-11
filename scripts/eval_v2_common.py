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
                            need_pathways: bool = True,
                            kg_v3_dir: Path | None = None,
                            drop_eval_leak_triples: bool = False) -> RetrievalContext:
    """Build the retrieval context over the August KG (read-only).

    ``kg_v3_dir`` additionally loads KG v3 (docs/EVAL_V3_PLAN.md): the frame KG
    (``full/``), its build-time held-out variants (``heldout_*``) and the
    record-local pathway index. ``drop_eval_leak_triples`` removes August triples
    whose wording comes from CultureBank's eval columns (``src/core/leakage.py``).
    """
    passages, triples, refined, pathway_retriever = _load_corpus(
        config.PROCESSED_DIR / "corpus_passages.jsonl",
        config.PROCESSED_DIR / "corpus_triples.jsonl",
        Path(config.CORPUS_TRIPLES_REFINED_PATH),
        Path(config.PATHWAY_INDEX_DIR),
        need_passages, need_triples, need_refined, need_pathways,
    )
    if drop_eval_leak_triples:
        from src.core.leakage import triple_uses_eval_text

        def clean(ts):
            if not ts:
                return ts
            kept = [t for t in ts if not triple_uses_eval_text(t.subject, t.object, t.raw_fields)]
            print(f"[eval_v2_common] dropped {len(ts) - len(kept)} of {len(ts)} triples "
                  "carrying eval-column wording", flush=True)
            return kept
        triples, refined = clean(triples), clean(refined)
    local_pw = frames = None
    heldout = {}
    if kg_v3_dir is not None:
        from src.retrieval.frame_retriever import FrameRetriever
        from src.retrieval.pathway_retriever import PathwayRetriever

        kg_v3_dir = Path(kg_v3_dir)
        frames = FrameRetriever(kg_v3_dir / "full")
        for sub in sorted(kg_v3_dir.glob("heldout_*")):
            heldout[sub.name.removeprefix("heldout_")] = FrameRetriever(sub)
        if (kg_v3_dir / "local_pathways_index").exists():
            local_pw = PathwayRetriever(kg_v3_dir / "local_pathways_index")
    ctx = RetrievalContext(
        passages=passages,
        triples=triples,
        refined_triples=refined,
        pathway_retriever=pathway_retriever,
        local_pathway_retriever=local_pw,
        frame_retriever=frames,
    )
    ctx.heldout_frame_retrievers = heldout
    return ctx
