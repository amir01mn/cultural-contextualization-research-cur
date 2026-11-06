"""The KazCulture frame adapter must not query a KG in another embedding space."""
import json

import pytest

from src.retrieval import kazculture_frame_retriever as kfr


def test_mismatched_backend_is_refused(tmp_path, monkeypatch):
    (tmp_path / "index_config.json").write_text(json.dumps({"backend": "BAAI/bge-m3@rev"}))
    monkeypatch.setattr("src.core.embedding_utils.backend_name", lambda: "sentence-transformers")
    with pytest.raises(RuntimeError, match="EMBEDDING_BACKEND=bge-m3"):
        kfr.check_embedding_backend(tmp_path)


def test_matching_or_missing_backend_passes(tmp_path, monkeypatch):
    kfr.check_embedding_backend(tmp_path)  # no index_config.json
    (tmp_path / "index_config.json").write_text(json.dumps({"backend": "tfidf"}))
    monkeypatch.setattr("src.core.embedding_utils.backend_name", lambda: "tfidf")
    kfr.check_embedding_backend(tmp_path)
