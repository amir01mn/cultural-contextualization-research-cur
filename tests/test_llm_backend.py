import sys
import types

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


def test_vllm_status_reports_when_not_installed(monkeypatch):
    """status() must report cleanly when vLLM cannot be imported.

    The absent-vLLM case is simulated rather than assumed. Letting the real
    import run makes the outcome depend on the machine, and on a cluster node it
    cold-loads the entire torch/CUDA stack from shared storage - which stalled
    this single test for over 20 minutes and timed out the whole suite.
    """
    monkeypatch.setitem(sys.modules, "vllm", None)  # forces ImportError
    backend = VLLMBackend(model="some/model")
    status = backend.status()
    assert status["available"] is False
    assert "vLLM not installed" in status["message"]
    assert status["model"] == "some/model"


def test_vllm_status_reports_when_installed(monkeypatch):
    """The available branch, with a stub module so no real import happens."""
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    backend = VLLMBackend(model="some/model")
    status = backend.status()
    assert status["available"] is True
    assert "some/model" in status["message"]
