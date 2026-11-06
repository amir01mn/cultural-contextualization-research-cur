"""EmbeddingCache must map new keys to the right rows even if keys.txt has duplicates."""
import numpy as np

from src.core.embedding_utils import EmbeddingCache


def test_new_keys_after_duplicate_keys_map_to_their_vectors(tmp_path):
    a = EmbeddingCache(tmp_path, "m", "r")
    a.add(["k1", "k2"], np.array([[1, 0], [0, 1]], dtype=np.float32))
    b = EmbeddingCache(tmp_path, "m", "r")
    b.add(["k2"], np.array([[0, 1]], dtype=np.float32))  # second writer, same text
    c = EmbeddingCache(tmp_path, "m", "r")               # loads a duplicate key
    assert len(c.index) == 2 and c.matrix.shape[0] == 3
    c.add(["k3"], np.array([[0.6, 0.8]], dtype=np.float32))
    np.testing.assert_allclose(c.get(["k3"])[0], [0.6, 0.8])
    d = EmbeddingCache(tmp_path, "m", "r")               # and on disk
    np.testing.assert_allclose(d.get(["k3"])[0], [0.6, 0.8])
    np.testing.assert_allclose(d.get(["k1"])[0], [1, 0])
