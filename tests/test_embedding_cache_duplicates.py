"""EmbeddingCache must map new keys to the right rows even if keys.txt has duplicates."""
import numpy as np
import pytest

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


@pytest.mark.parametrize("keys_tail", [None, b"", b"incomplete-key"])
def test_first_interrupted_write_does_not_shift_later_vectors(tmp_path, keys_tail):
    cache = EmbeddingCache(tmp_path, "m", "r")
    cache.add(["uncommitted"], np.array([[1, 0]], dtype=np.float32))
    # The first writer died after vectors were flushed but before a complete key.
    if keys_tail is None:
        cache.keys_path.unlink()
    else:
        cache.keys_path.write_bytes(keys_tail)
    restarted = EmbeddingCache(tmp_path, "m", "r")
    restarted.add(["new"], np.array([[0, 1]], dtype=np.float32))
    reloaded = EmbeddingCache(tmp_path, "m", "r")
    assert set(reloaded.index) == {"new"}
    np.testing.assert_array_equal(reloaded.get(["new"])[0], [0, 1])


def test_live_writer_repairs_another_writers_interrupted_append(tmp_path):
    first = EmbeddingCache(tmp_path, "m", "r")
    first.add(["committed"], np.array([[1, 0]], dtype=np.float32))
    waiting = EmbeddingCache(tmp_path, "m", "r")
    # A different process appended a vector but died before appending its key.
    with first.vec_path.open("ab") as fh:
        np.array([[0.6, 0.8]], dtype=np.float32).tofile(fh)
    waiting.add(["new"], np.array([[0, 1]], dtype=np.float32))
    reloaded = EmbeddingCache(tmp_path, "m", "r")
    np.testing.assert_array_equal(reloaded.get(["committed", "new"]), [[1, 0], [0, 1]])
