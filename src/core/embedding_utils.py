"""Text embedding helpers shared by entity resolution and pathway retrieval."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .utils import get_logger

LOGGER = get_logger("embedding_utils")

_ST_MODEL = None
_ST_TRIED = False

# BGE-M3 (KazCulture frames v3k; docs/KAZCULTURE_FRAMES_V3K.md). Pinned to the
# revision the S1 index and the pre-registration name.
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_BACKEND = "bge-m3"
_BGE_MODEL = None
_BGE_CACHE = None


def _get_st_model():
    global _ST_MODEL, _ST_TRIED
    if _ST_TRIED:
        return _ST_MODEL
    _ST_TRIED = True
    if config.EMBEDDING_BACKEND in ("tfidf", BGE_M3_BACKEND):
        LOGGER.info("EMBEDDING_BACKEND=%s; skipping MiniLM.", config.EMBEDDING_BACKEND)
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


def embed_texts(texts: list[str], *, fit: bool = False,
                backend: Optional[str] = None) -> np.ndarray:
    """Return L2-normalized embedding matrix of shape (n, d).

    ``backend`` overrides ``config.EMBEDDING_BACKEND`` for this call; with
    ``bge-m3`` the vectors come from ``embed_bge_m3`` (``fit`` is irrelevant).

    With the TF-IDF backend the fitted vectorizer is retained at module level so
    ``fit=False`` can project a query into the same vector space as the corpus.
    Previously each call built a fresh vectorizer, so ``fit=False`` raised
    NotFittedError - reachable whenever an index built with sentence-transformers
    is later loaded in an environment that falls back to TF-IDF.
    """
    global _TFIDF_VECTORIZER

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    if (backend or config.EMBEDDING_BACKEND) == BGE_M3_BACKEND:
        return embed_bge_m3(texts)

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
    global _ST_MODEL, _ST_TRIED, _TFIDF_VECTORIZER, _BGE_MODEL, _BGE_CACHE
    _ST_MODEL = None
    _ST_TRIED = False
    _TFIDF_VECTORIZER = None
    _BGE_MODEL = None
    _BGE_CACHE = None


def cosine_similarity_matrix(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    if b is None:
        b = a
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)


def backend_name() -> str:
    if config.EMBEDDING_BACKEND == BGE_M3_BACKEND:
        return f"{BGE_M3_MODEL}@{BGE_M3_REVISION}"
    return "sentence-transformers" if _get_st_model() is not None else "tfidf"


# ---------------------------------------------------------------------------
# BGE-M3 with an on-disk cache
# ---------------------------------------------------------------------------

def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Append-only vector cache for one (model, revision): ``keys.txt`` + ``vectors.f32``.

    Row i of ``vectors.f32`` (float32, ``dim`` wide) belongs to line i of
    ``keys.txt`` (sha256 of the exact text). A job killed mid-write leaves at
    most a partial tail, which is cut back to the last complete pair on open.
    """

    def __init__(self, root: Path, model: str, revision: str):
        self.dir = Path(root) / f"{model.replace('/', '--')}@{revision}"
        self.keys_path = self.dir / "keys.txt"
        self.vec_path = self.dir / "vectors.f32"
        self.dim: Optional[int] = None
        self.index: dict[str, int] = {}
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if not self.dir.exists():
            return
        import fcntl
        with (self.dir / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # never read/repair mid-append
            try:
                self._load_locked()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _load_locked(self) -> None:
        keys, dim = self._repair_locked()
        if dim is None:
            return
        self.dim = dim
        n = len(keys)
        self.matrix = (np.fromfile(self.vec_path, dtype=np.float32, count=n * dim)
                       .reshape(n, dim) if n else np.zeros((0, dim), dtype=np.float32))
        self.index = {k: i for i, k in enumerate(keys)}

    def _repair_locked(self) -> tuple[list[str], Optional[int]]:
        """Keep complete key/vector pairs, including an interrupted first write.

        Called under the shared lock before reads *and* appends: another writer
        can die after this process has already opened the cache.
        """
        raw = self.keys_path.read_bytes() if self.keys_path.exists() else b""
        complete = raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else b""
        keys = [k for k in complete.decode("ascii").split("\n") if k]
        meta = self.dir / "dim.txt"
        vector_bytes = self.vec_path.stat().st_size if self.vec_path.exists() else 0
        if not meta.exists():
            if raw or vector_bytes:
                raise ValueError(f"Embedding cache {self.dir}: missing dimension metadata")
            return [], None
        dim = int(meta.read_text().strip())
        if dim <= 0:
            raise ValueError(f"Embedding cache {self.dir}: invalid dimension {dim}")
        row_bytes = 4 * dim
        n = min(len(keys), vector_bytes // row_bytes)
        if n != len(keys) or len(complete) != len(raw) \
                or vector_bytes != n * row_bytes:
            LOGGER.warning("Embedding cache %s: repairing truncated tail (%d rows kept).",
                           self.dir, n)
            keys = keys[:n]
            self.keys_path.write_text("".join(k + "\n" for k in keys), encoding="ascii")
            with self.vec_path.open("a+b") as fh:
                fh.truncate(n * row_bytes)
        return keys, dim

    def get(self, keys: list[str]) -> list[Optional[np.ndarray]]:
        return [self.matrix[self.index[k]] if k in self.index else None for k in keys]

    def add(self, keys: list[str], vecs: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if not len(keys):
            return
        if vecs.ndim != 2 or vecs.shape[0] != len(keys) or vecs.shape[1] == 0:
            raise ValueError("Embedding cache needs one nonempty vector per key")
        dim = int(vecs.shape[1])
        if self.dim is not None and dim != self.dim:
            raise ValueError(f"Cache dim {self.dim} != {vecs.shape[1]}")
        self.dir.mkdir(parents=True, exist_ok=True)
        # Vectors first, then keys: a crash between the two leaves extra vector
        # bytes, which _load() trims; a key never points at a missing vector.
        # The exclusive lock keeps concurrent jobs from interleaving the two
        # appends (several SLURM jobs share one cache directory).
        import fcntl
        with (self.dir / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                _, stored_dim = self._repair_locked()
                if stored_dim is not None and stored_dim != dim:
                    raise ValueError(f"Cache dim {stored_dim} != {dim}")
                if stored_dim is None:
                    (self.dir / "dim.txt").write_text(str(dim))
                with self.vec_path.open("ab") as fh:
                    vecs.tofile(fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                with self.keys_path.open("a", encoding="ascii") as fh:
                    fh.write("".join(k + "\n" for k in keys))
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        self.dim = dim
        # Row number, not len(self.index): keys.txt may hold a key twice (two jobs
        # encoding the same text), and then the dict is shorter than the matrix.
        start = self.matrix.reshape(-1, self.dim).shape[0]
        self.matrix = np.vstack([self.matrix.reshape(-1, self.dim), vecs])
        for i, k in enumerate(keys):
            self.index[k] = start + i


def _get_bge_model():
    global _BGE_MODEL
    if _BGE_MODEL is None:
        from sentence_transformers import SentenceTransformer

        device = config.EMBEDDING_DEVICE
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        LOGGER.info("Loading %s@%s on %s ...", BGE_M3_MODEL, BGE_M3_REVISION, device)
        _BGE_MODEL = SentenceTransformer(BGE_M3_MODEL, revision=BGE_M3_REVISION, device=device)
    return _BGE_MODEL


def _get_bge_cache() -> EmbeddingCache:
    global _BGE_CACHE
    root = Path(config.EMBEDDING_CACHE_DIR)
    if _BGE_CACHE is None or _BGE_CACHE.dir.parent != root:
        _BGE_CACHE = EmbeddingCache(root, BGE_M3_MODEL, BGE_M3_REVISION)
    return _BGE_CACHE


def embed_bge_m3(texts: list[str], *, batch_size: Optional[int] = None,
                 chunk: int = 512) -> np.ndarray:
    """L2-normalized BGE-M3 vectors, batched, served from / written to the cache.

    Misses are encoded in chunks of ``chunk`` texts and persisted after each
    chunk, so a killed job resumes where it stopped. Raises instead of falling
    back to another backend.
    """
    cache = _get_bge_cache()
    keys = [text_key(t) for t in texts]
    found = cache.get(keys)
    todo: dict[str, str] = {}
    for k, t, v in zip(keys, texts, found):
        if v is None and k not in todo:
            todo[k] = t
    if todo:
        model = _get_bge_model()
        items = list(todo.items())
        bs = batch_size or config.EMBEDDING_BATCH_SIZE
        for start in range(0, len(items), chunk):
            part = items[start:start + chunk]
            vecs = np.asarray(model.encode(
                [t for _k, t in part], batch_size=bs, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True), dtype=np.float32)
            if vecs.shape[0] != len(part) or not np.isfinite(vecs).all():
                raise ValueError("BGE-M3 returned invalid embeddings")
            cache.add([k for k, _t in part], vecs)
            LOGGER.info("BGE-M3: encoded %d/%d new texts", min(start + chunk, len(items)),
                        len(items))
        found = cache.get(keys)
    if not texts:
        return np.zeros((0, cache.dim or 1), dtype=np.float32)
    return np.stack(found).astype(np.float32)
