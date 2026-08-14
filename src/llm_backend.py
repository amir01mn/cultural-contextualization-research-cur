"""Pluggable LLM backend abstraction.

The MVP pipeline talks to Ollama directly via ``src/local_llm.py``. For the
scalable / publishable experiments we also need batched GPU inference on a SLURM
cluster (Narval), where Ollama is not the right tool. This module provides a
single ``LLMBackend`` interface with two implementations:

    - OllamaBackend : local development; wraps ``local_llm`` with light threading.
    - VLLMBackend   : cluster GPU; loads a model once and batches generation.

Both expose ``generate(prompts) -> list[str]`` so the evaluation code is backend
agnostic. vLLM is imported lazily so local environments without it still work.

No paid API is ever used.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Protocol, runtime_checkable

from . import config, local_llm
from .utils import get_logger

LOGGER = get_logger("llm_backend")


@runtime_checkable
class LLMBackend(Protocol):
    """Minimal interface every backend must implement."""

    name: str
    model: str

    def generate(self,
                 prompts: list[str],
                 system_prompt: Optional[str] = None,
                 *,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> list[str]:
        """Generate one completion per prompt. Output order matches input."""
        ...

    def generate_one(self,
                     prompt: str,
                     system_prompt: Optional[str] = None,
                     **kwargs) -> str:
        """Convenience single-prompt generation."""
        ...

    def status(self) -> dict:
        """Return an availability/status dict (never raises)."""
        ...


class OllamaBackend:
    """Local-dev backend that wraps the existing Ollama adapter.

    Batching is emulated with a small thread pool, since Ollama serves requests
    one (or a few) at a time. Suitable for local smoke tests, not large runs.
    """

    name = "ollama"

    def __init__(self,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 max_workers: Optional[int] = None):
        self.model = model or config.OLLAMA_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL
        self.max_workers = max_workers or config.OLLAMA_MAX_WORKERS

    def generate_one(self,
                     prompt: str,
                     system_prompt: Optional[str] = None,
                     *,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None) -> str:
        return local_llm.generate_text(
            prompt,
            system_prompt=system_prompt,
            model=self.model,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(self,
                 prompts: list[str],
                 system_prompt: Optional[str] = None,
                 *,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> list[str]:
        if not prompts:
            return []
        if self.max_workers <= 1 or len(prompts) == 1:
            return [
                self.generate_one(p, system_prompt, temperature=temperature,
                                  max_tokens=max_tokens)
                for p in prompts
            ]
        results: list[str] = [""] * len(prompts)

        def _work(idx_prompt):
            idx, prompt = idx_prompt
            return idx, self.generate_one(
                prompt, system_prompt, temperature=temperature,
                max_tokens=max_tokens)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for idx, text in pool.map(_work, list(enumerate(prompts))):
                results[idx] = text
        return results

    def status(self) -> dict:
        return local_llm.check_ollama_available(base_url=self.base_url,
                                                model=self.model)

    def warmup(self) -> dict:
        return local_llm.warmup_model(model=self.model, base_url=self.base_url)


class VLLMBackend:
    """Cluster GPU backend using vLLM for true batched offline inference.

    Loads the model weights once (from a local/offline cache on Narval) and
    generates completions for an entire list of prompts in a single batched call.
    vLLM is imported lazily so importing this module never requires vLLM.
    """

    name = "vllm"

    def __init__(self,
                 model: Optional[str] = None,
                 *,
                 dtype: Optional[str] = None,
                 max_model_len: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 tensor_parallel_size: Optional[int] = None):
        self.model = model or config.VLLM_MODEL
        self.dtype = dtype or config.VLLM_DTYPE
        self.max_model_len = max_model_len or config.VLLM_MAX_MODEL_LEN
        self.gpu_memory_utilization = (
            gpu_memory_utilization if gpu_memory_utilization is not None
            else config.VLLM_GPU_MEMORY_UTILIZATION
        )
        self.tensor_parallel_size = (
            tensor_parallel_size or config.VLLM_TENSOR_PARALLEL_SIZE
        )
        self._llm = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM  # lazy import; only available on the cluster
        except Exception as exc:  # noqa: BLE001
            raise local_llm.LLMUnavailableError(
                "vLLM is not installed in this environment. Install it on the "
                "GPU node (e.g. via the Alliance wheelhouse) or use "
                "--provider ollama for local development."
            ) from exc

        LOGGER.info("Loading vLLM model '%s' (dtype=%s, max_model_len=%s, tp=%d)...",
                    self.model, self.dtype, self.max_model_len,
                    self.tensor_parallel_size)
        self._llm = LLM(
            model=self.model,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
        )
        try:
            self._tokenizer = self._llm.get_tokenizer()
        except Exception:  # noqa: BLE001
            self._tokenizer = None

    def _build_prompt(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Apply the model's chat template if a tokenizer is available."""
        if self._tokenizer is None:
            if system_prompt:
                return f"{system_prompt}\n\n{prompt}"
            return prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001
            if system_prompt:
                return f"{system_prompt}\n\n{prompt}"
            return prompt

    def generate(self,
                 prompts: list[str],
                 system_prompt: Optional[str] = None,
                 *,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None) -> list[str]:
        if not prompts:
            return []
        self._ensure_loaded()
        from vllm import SamplingParams

        temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        max_tokens = max_tokens or config.LLM_MAX_TOKENS
        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=config.RANDOM_SEED,
        )
        built = [self._build_prompt(p, system_prompt) for p in prompts]
        outputs = self._llm.generate(built, sampling)
        # vLLM preserves input order in the returned list.
        return [o.outputs[0].text if o.outputs else "" for o in outputs]

    def generate_one(self,
                     prompt: str,
                     system_prompt: Optional[str] = None,
                     **kwargs) -> str:
        return self.generate([prompt], system_prompt, **kwargs)[0]

    def status(self) -> dict:
        status = {"available": False, "model_present": False,
                  "model": self.model, "message": ""}
        try:
            import vllm  # noqa: F401

            status["available"] = True
            status["model_present"] = True
            status["message"] = f"vLLM available; model '{self.model}' will load on first call."
        except Exception as exc:  # noqa: BLE001
            status["message"] = f"vLLM not installed: {exc}"
        return status

    def warmup(self) -> dict:
        try:
            self._ensure_loaded()
            return {"warmed": True, "message": f"vLLM model '{self.model}' loaded."}
        except local_llm.LLMUnavailableError as exc:
            return {"warmed": False, "message": str(exc)}


def get_backend(provider: Optional[str] = None,
                model: Optional[str] = None,
                **kwargs) -> LLMBackend:
    """Construct a backend from the provider name (defaults to config)."""
    provider = (provider or config.LLM_PROVIDER or "ollama").strip().lower()
    if provider == "ollama":
        return OllamaBackend(model=model, **kwargs)
    if provider == "vllm":
        return VLLMBackend(model=model, **kwargs)
    raise ValueError(
        f"Unknown LLM provider '{provider}'. Use 'ollama' or 'vllm'."
    )
