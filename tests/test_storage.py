from src.core.schemas import CorpusBuildState
from src.core.storage import (
    append_jsonl,
    generate_with_cache,
    load_corpus_state,
    mark_completed,
    read_completed_ids,
    save_corpus_state,
)


def test_corpus_state_roundtrip(tmp_path):
    state_path = tmp_path / "state.json"
    state = CorpusBuildState()
    mark_completed(state, "extraction", ["a", "b"])
    save_corpus_state(state_path, state)
    loaded = load_corpus_state(state_path)
    assert read_completed_ids(loaded, "extraction") == {"a", "b"}


def test_llm_cache(fake_backend_factory, tmp_path):
    calls = {"n": 0}

    def responder(prompt, _system):
        calls["n"] += 1
        return f"resp:{prompt[:5]}"

    backend = fake_backend_factory(responder)
    prompts = ["prompt-one", "prompt-two"]
    out1 = generate_with_cache(backend, "test", prompts, "sys", tmp_path)
    out2 = generate_with_cache(backend, "test", prompts, "sys", tmp_path)
    assert out1 == out2
    assert calls["n"] == 2


def test_append_jsonl(tmp_path):
    path = tmp_path / "out.jsonl"
    append_jsonl(path, [{"a": 1}])
    append_jsonl(path, [{"b": 2}])
    text = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 2
