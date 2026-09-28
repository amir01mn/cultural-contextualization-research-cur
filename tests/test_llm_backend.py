import pytest

from src import llm_backend
from src.llm_backend import OllamaBackend, VLLMBackend, get_backend


def test_factory_returns_ollama_by_default():
    backend = get_backend("ollama")
    assert isinstance(backend, OllamaBackend)
    assert backend.name == "ollama"


def test_factory_returns_vllm():
    backend = get_backend("vllm")
    assert isinstance(backend, VLLMBackend)
    assert backend.name == "vllm"


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_backend("not_a_provider")


def test_ollama_backend_generate_batches(monkeypatch):
    # Mock the underlying Ollama text generation.
    def fake_generate_text(prompt, system_prompt=None, **kwargs):
        return f"echo:{prompt}"

    monkeypatch.setattr(llm_backend.local_llm, "generate_text", fake_generate_text)
    backend = OllamaBackend(max_workers=1)
    out = backend.generate(["a", "b", "c"], system_prompt="sys")
    assert out == ["echo:a", "echo:b", "echo:c"]


def test_ollama_backend_threaded_preserves_order(monkeypatch):
    def fake_generate_text(prompt, system_prompt=None, **kwargs):
        return prompt.upper()

    monkeypatch.setattr(llm_backend.local_llm, "generate_text", fake_generate_text)
    backend = OllamaBackend(max_workers=4)
    out = backend.generate([f"p{i}" for i in range(10)])
    assert out == [f"P{i}" for i in range(10)]


def test_vllm_status_reports_when_not_installed():
    backend = VLLMBackend(model="some/model")
    status = backend.status()
    # vLLM is not installed locally; status should be a dict, not raise.
    assert "available" in status and "message" in status


def test_ollama_backend_passes_max_tokens_to_local_adapter(monkeypatch):
    seen = {}

    def fake_generate_text(prompt, system_prompt=None, **kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(llm_backend.local_llm, "generate_text", fake_generate_text)
    backend = OllamaBackend(max_workers=1)
    out = backend.generate(["a"], max_tokens=777)

    assert out == ["ok"]
    assert seen["num_predict"] == 777
