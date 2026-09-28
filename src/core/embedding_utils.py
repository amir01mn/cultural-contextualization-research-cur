"""Text embedding helpers shared by entity resolution and pathway retrieval."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import config
from .utils import get_logger

LOGGER = get_logger("embedding_utils")

_ST_MODEL = None
_ST_TRIED = False


def _get_st_model():
    global _ST_MODEL, _ST_TRIED
    if _ST_TRIED:
        return _ST_MODEL
    _ST_TRIED = True
    if config.EMBEDDING_BACKEND == "tfidf":
        LOGGER.info("EMBEDDING_BACKEND=tfidf; skipping sentence-transformers.")
        _ST_MODEL = None
        return _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        LOGGER.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("sentence-transformers unavailable (%s); using TF-IDF.", exc)
        _ST_MODEL = None
    return _ST_MODEL


_TFIDF_VECTORIZER = None


def get_sentence_transformer():
    """Return the shared SentenceTransformer, or None when unavailable/disabled.

    Public entry point so every retriever in the pipeline resolves its embedding
    backend the same way and loads the model at most once per process.
    """
    return _get_st_model()


def embed_texts(texts: list[str], *, fit: bool = False) -> np.ndarray:
    """Return L2-normalized embedding matrix of shape (n, d).

    With the TF-IDF backend the fitted vectorizer is retained at module level so
    ``fit=False`` can project a query into the same vector space as the corpus.
    Previously each call built a fresh vectorizer, so ``fit=False`` raised
    NotFittedError - reachable whenever an index built with sentence-transformers
    is later loaded in an environment that falls back to TF-IDF.
    """
    global _TFIDF_VECTORIZER

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    model = _get_st_model()
    if model is not None:
        try:
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Embedding failed (%s); falling back to TF-IDF.", exc)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    if fit or _TFIDF_VECTORIZER is None:
        if not fit:
            LOGGER.warning(
                "TF-IDF vectorizer not fitted yet; fitting on the given texts. "
                "Query vectors may not share the corpus vector space."
            )
        _TFIDF_VECTORIZER = TfidfVectorizer(stop_words="english", max_features=4096)
        matrix = _TFIDF_VECTORIZER.fit_transform(texts)
    else:
        matrix = _TFIDF_VECTORIZER.transform(texts)
    dense = normalize(matrix, norm="l2", axis=1).toarray()
    return np.asarray(dense, dtype=np.float32)


def reset_embedding_state() -> None:
    """Clear cached model/vectorizer state (used by tests)."""
    global _ST_MODEL, _ST_TRIED, _TFIDF_VECTORIZER
    _ST_MODEL = None
    _ST_TRIED = False
    _TFIDF_VECTORIZER = None


def cosine_similarity_matrix(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    if b is None:
        b = a
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)


def backend_name() -> str:
    return "sentence-transformers" if _get_st_model() is not None else "tfidf"
