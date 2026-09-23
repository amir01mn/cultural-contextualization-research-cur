"""Persistent pathway retrieval index (embedding-based)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .embedding_utils import backend_name, cosine_similarity_matrix, embed_texts
from .retrieval_ranking import rerank_hits
from .schemas import CulturalPathway
from .utils import ensure_dir, get_logger, normalize_text

LOGGER = get_logger("pathway_retriever")


def pathway_document(pathway: CulturalPathway) -> str:
    parts = [f"culture: {pathway.culture}", f"completeness: {pathway.completeness.value}"]
    for node in pathway.nodes:
        parts.append(f"{node.layer.value}: {node.label}")
    for edge in pathway.edges:
        parts.append(
            f"relation: ({edge.source_layer.value}) {edge.source_label} "
            f"-[{edge.predicate} | {edge.predicate_layer.value}]-> "
            f"({edge.target_layer.value}) {edge.target_label}"
        )
    if pathway.has_shared_pivot:
        parts.append("shared_pivot: true")
    return " | ".join(parts)


class PathwayRetriever:
    """Load a pre-built pathway embedding index from disk."""

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.pathways: list[CulturalPathway] = []
        self.documents: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.backend = "none"
        self._load()

    def _load(self) -> None:
        meta_path = self.index_dir / "pathways.jsonl"
        emb_path = self.index_dir / "embeddings.npy"
        cfg_path = self.index_dir / "config.json"
        if not meta_path.exists() or not emb_path.exists():
            LOGGER.warning("Pathway index incomplete at %s", self.index_dir)
            return

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.pathways.append(CulturalPathway(**json.loads(line)))
        self.documents = [pathway_document(p) for p in self.pathways]
        self.embeddings = np.load(emb_path)
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            self.backend = cfg.get("backend", "unknown")
        else:
            self.backend = "unknown"
        LOGGER.info("Loaded pathway index: %d pathways from %s",
                    len(self.pathways), self.index_dir)

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 *,
                 question_culture: Optional[str] = None) -> list[tuple[CulturalPathway, float]]:
        if not self.pathways or self.embeddings is None:
            return []
        candidate_k = min(len(self.pathways), max(top_k * config.RETRIEVAL_CANDIDATE_MULTIPLIER, top_k))
        if self.backend == "tfidf":
            texts = self.documents + [query]
            embs = embed_texts(texts, fit=True)
            if embs.shape[0] < 2:
                return []
            scores = embs[-1] @ embs[:-1].T
        else:
            q_emb = embed_texts([query], fit=False)
            if q_emb.shape[0] == 0:
                return []
            scores = cosine_similarity_matrix(q_emb, self.embeddings)[0]
        idx = np.argsort(scores)[::-1][:candidate_k]
        hits = [(self.pathways[i], float(scores[i])) for i in idx if scores[i] > 0]
        return rerank_hits(
            hits,
            top_k=top_k,
            question_culture=question_culture,
            get_culture=lambda p: p.culture,
            allow_cross_culture_shared=lambda p: p.has_shared_pivot,
        )


def build_pathway_index(
    pathways: list[CulturalPathway],
    index_dir: str | Path,
) -> PathwayRetriever:
    """Embed pathways and persist index to disk."""
    index_dir = Path(index_dir)
    ensure_dir(index_dir)

    documents = [pathway_document(p) for p in pathways]
    embeddings = embed_texts(documents, fit=True) if documents else np.zeros((0, 1))

    meta_path = index_dir / "pathways.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for p in pathways:
            f.write(json.dumps(p.model_dump(), ensure_ascii=False, default=str) + "\n")

    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "config.json").write_text(
        json.dumps({"backend": backend_name(), "count": len(pathways)}, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Built pathway index (%d pathways) -> %s", len(pathways), index_dir)
    return PathwayRetriever(index_dir)
