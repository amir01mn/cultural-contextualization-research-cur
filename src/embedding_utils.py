"""Text embedding helpers shared by entity resolution and pathway retrieval."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import get_logger

LOGGER = get_logger("embedding_utils")

_ST_MODEL = None
_ST_TRIED = False


def _get_st_model():
    global _ST_MODEL, _ST_TRIED
    if _ST_TRIED:
        return _ST_MODEL
    _ST_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer

        LOGGER.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("sentence-transformers unavailable (%s); using TF-IDF.", exc)
        _ST_MODEL = None
    return _ST_MODEL


def embed_texts(texts: list[str], *, fit: bool = False) -> np.ndarray:
    """Return L2-normalized embedding matrix of shape (n, d)."""
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

    vectorizer = TfidfVectorizer(stop_words="english", max_features=4096)
    if fit:
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = vectorizer.transform(texts)
    dense = normalize(matrix, norm="l2", axis=1).toarray()
    return np.asarray(dense, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    if b is None:
        b = a
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)


def backend_name() -> str:
    return "sentence-transformers" if _get_st_model() is not None else "tfidf"
