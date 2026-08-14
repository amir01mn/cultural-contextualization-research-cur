"""Retrieve the top-k most relevant triples for a question.

Uses sentence-transformers embeddings if installed; otherwise falls back to a
TF-IDF cosine similarity over a rich text view of each triple (subject,
predicate, object, evidence, source_text, cultural group, topic, notes).
"""

from __future__ import annotations

from typing import Optional

from . import config
from .embedding_utils import get_sentence_transformer
from .retrieval_ranking import passage_culture, rerank_hits, triple_culture
from .schemas import CulturalTriple
from .utils import get_logger, normalize_text

LOGGER = get_logger("retriever")


def _triple_document(t: CulturalTriple) -> str:
    """Rich searchable text for a triple, including provenance context."""
    raw = t.raw_fields or {}
    parts = [
        t.subject,
        t.predicate,
        t.object,
        t.evidence or "",
        t.notes or "",
        t.source_text or "",
        normalize_text(raw.get("cultural_group") or raw.get("cultural group") or ""),
        normalize_text(raw.get("topic") or ""),
    ]
    return " ".join(p for p in parts if p)


def _get_st_model():
    """Shared sentence-transformers handle.

    Delegates to ``embedding_utils`` so triple/passage retrieval honours the same
    ``EMBEDDING_BACKEND`` setting as entity resolution and pathway retrieval, and
    so the model is loaded at most once per process. This module previously kept
    a private loader that ignored the setting, meaning S1/S2/S3 could silently
    use a different embedding backend than S4.
    """
    return get_sentence_transformer()


class TripleRetriever:
    """Index a set of triples and retrieve the most relevant ones per query."""

    def __init__(self, triples: list[CulturalTriple]):
        self.triples = triples
        self.documents = [_triple_document(t) for t in triples]
        self.backend = "none"
        self._embeddings = None
        self._vectorizer = None
        self._matrix = None
        if self.documents:
            self._build_index()

    def _build_index(self) -> None:
        model = _get_st_model()
        if model is not None:
            try:
                self._embeddings = model.encode(
                    self.documents, normalize_embeddings=True, show_progress_bar=False
                )
                self.backend = "sentence-transformers"
                return
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Embedding failed (%s); falling back to TF-IDF.", exc)

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=8192)
        self._matrix = self._vectorizer.fit_transform(self.documents)
        self.backend = "tfidf"

    def retrieve(self,
                 question: str,
                 top_k: int = 8,
                 *,
                 question_culture: Optional[str] = None) -> list[tuple[CulturalTriple, float]]:
        """Return up to top_k (triple, score) pairs sorted by relevance."""
        if not self.triples:
            return []
        query = normalize_text(question)
        if not query:
            return []

        candidate_k = min(len(self.triples), max(top_k * config.RETRIEVAL_CANDIDATE_MULTIPLIER, top_k))
        hits: list[tuple[CulturalTriple, float]] = []

        if self.backend == "sentence-transformers":
            import numpy as np

            model = _get_st_model()
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            scores = self._embeddings @ q_emb
            order = np.argsort(-scores)[:candidate_k]
            hits = [(self.triples[i], float(scores[i])) for i in order]

        elif self.backend == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            q_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._matrix)[0]
            order = np.argsort(-scores)[:candidate_k]
            hits = [(self.triples[i], float(scores[i])) for i in order]

        return rerank_hits(
            hits,
            top_k=top_k,
            question_culture=question_culture,
            get_culture=triple_culture,
            get_generated=lambda t: bool(t.generated),
        )


def retrieve_triples(question: str,
                     triples: list[CulturalTriple],
                     top_k: int = 8) -> list[CulturalTriple]:
    """Convenience function: build an index and retrieve top-k triples."""
    retriever = TripleRetriever(triples)
    return [t for t, _ in retriever.retrieve(question, top_k=top_k)]


class TextRetriever:
    """Generic top-k retriever over arbitrary (id, text, payload) documents.

    Used for raw-text passage retrieval (S1). Shares the same sentence-
    transformers/TF-IDF backend selection as TripleRetriever.
    """

    def __init__(self, documents: list[dict], text_key: str = "text"):
        self.documents = documents
        self.text_key = text_key
        self.texts = [normalize_text(d.get(text_key, "")) for d in documents]
        self.backend = "none"
        self._embeddings = None
        self._vectorizer = None
        self._matrix = None
        if any(self.texts):
            self._build_index()

    def _build_index(self) -> None:
        model = _get_st_model()
        if model is not None:
            try:
                self._embeddings = model.encode(
                    self.texts, normalize_embeddings=True, show_progress_bar=False)
                self.backend = "sentence-transformers"
                return
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Passage embedding failed (%s); using TF-IDF.", exc)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=8192)
        self._matrix = self._vectorizer.fit_transform(self.texts)
        self.backend = "tfidf"

    def retrieve(self,
                 query: str,
                 top_k: int = 8,
                 *,
                 question_culture: Optional[str] = None) -> list[tuple[dict, float]]:
        if not self.documents:
            return []
        query = normalize_text(query)
        if not query:
            return []

        candidate_k = min(len(self.documents), max(top_k * config.RETRIEVAL_CANDIDATE_MULTIPLIER, top_k))
        hits: list[tuple[dict, float]] = []

        if self.backend == "sentence-transformers":
            import numpy as np

            model = _get_st_model()
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            scores = self._embeddings @ q_emb
            order = np.argsort(-scores)[:candidate_k]
            hits = [(self.documents[i], float(scores[i])) for i in order]

        elif self.backend == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            q_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._matrix)[0]
            order = np.argsort(-scores)[:candidate_k]
            hits = [(self.documents[i], float(scores[i])) for i in order]

        return rerank_hits(
            hits,
            top_k=top_k,
            question_culture=question_culture,
            get_culture=passage_culture,
        )
