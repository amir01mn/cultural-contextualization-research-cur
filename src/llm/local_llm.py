"""Local LLM adapter for Ollama.

This is the primary engine for the pipeline. All extraction, critique, scenario
generation, and answering go through a local/open-source model served by Ollama.
No paid API is ever used.

Public API:
    check_ollama_available() -> dict
    generate_text(prompt, system_prompt=None) -> str
    generate_json(prompt, system_prompt=None) -> dict
    safe_parse_json(text) -> dict | None
    repair_json_if_needed(text) -> str
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import requests


from ..core import config
from ..core.utils import get_logger

LOGGER = get_logger("local_llm")


class LLMUnavailableError(RuntimeError):
    """Raised when the local LLM cannot be reached and fallback is not allowed."""


def check_ollama_available(base_url: Optional[str] = None,
                           model: Optional[str] = None) -> dict:
    """Probe the local Ollama server.

    Returns a status dict (never raises) with keys:
        available: bool
        model_present: bool
        models: list[str]
        message: str
    """
    base_url = base_url or config.OLLAMA_BASE_URL
    model = model or config.OLLAMA_MODEL
    status = {
        "available": False,
        "model_present": False,
        "models": [],
        "base_url": base_url,
        "model": model,
        "message": "",
    }
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        status["available"] = True
        status["models"] = models
        # Ollama tags include the :tag suffix; match exactly or by base name.
        status["model_present"] = any(
            m == model or m.split(":")[0] == model.split(":")[0] for m in models
        )
        if status["model_present"]:
            status["message"] = f"Ollama reachable; model '{model}' available."
        else:
            status["message"] = (
                f"Ollama reachable but model '{model}' not pulled. "
                f"Run: ollama pull {model}"
            )
    except Exception as exc:  # noqa: BLE001 - report any connection issue
        status["message"] = (
            f"Ollama not reachable at {base_url}: {exc}. "
            f"Start it with 'ollama serve' and 'ollama pull {model}'."
        )
    return status


def warmup_model(model: Optional[str] = None,
                 base_url: Optional[str] = None) -> dict:
    """Pre-load the model into memory (cold start) before the main loop.

    The first request after `ollama pull` loads the weights into RAM/VRAM, which
    can be slow. Doing it once up front (with a generous timeout) avoids a
    ReadTimeout on the first real extraction call. Never raises; returns status.
    """
    base_url = base_url or config.OLLAMA_BASE_URL
    model = model or config.OLLAMA_MODEL
    status = {"warmed": False, "message": ""}
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": "Reply with the single word: ready.",
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                "options": {"num_predict": 8, "temperature": 0.0},
            },
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        status["warmed"] = True
        status["message"] = f"Model '{model}' loaded and ready."
    except requests.RequestException as exc:
        status["message"] = f"Warmup failed for '{model}': {exc}"
    return status


def generate_text(prompt: str,
                  system_prompt: Optional[str] = None,
                  *,
                  model: Optional[str] = None,
                  temperature: Optional[float] = None,
                  base_url: Optional[str] = None,
                  max_tokens: Optional[int] = None,
                  json_mode: bool = False) -> str:
    """Call Ollama's /api/generate and return the raw response text.

    Raises LLMUnavailableError if the server cannot be reached.
    """
    base_url = base_url or config.OLLAMA_BASE_URL
    model = model or config.OLLAMA_MODEL
    temperature = config.OLLAMA_TEMPERATURE if temperature is None else temperature
    num_predict = max_tokens or config.OLLAMA_NUM_PREDICT

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt
    if json_mode:
        payload["format"] = "json"

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=config.OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.RequestException as exc:
        raise LLMUnavailableError(
            f"Failed to call Ollama at {base_url} with model '{model}': {exc}"
        ) from exc


def generate_json(prompt: str,
                  system_prompt: Optional[str] = None,
                  *,
                  model: Optional[str] = None,
                  base_url: Optional[str] = None,
                  retries: int = 2) -> Optional[dict]:
    """Call the LLM expecting JSON; parse and repair as needed.

    Returns a parsed dict, or None if no valid JSON could be obtained.
    Raises LLMUnavailableError if the server is unreachable.
    """
    last_text = ""
    for attempt in range(retries + 1):
        # Request JSON format on the first try; fall back to free-form on retry
        # in case the model struggles with constrained decoding.
        use_json_mode = attempt == 0
        text = generate_text(
            prompt,
            system_prompt=system_prompt,
            model=model,
            base_url=base_url,
            json_mode=use_json_mode,
        )
        last_text = text
        parsed = safe_parse_json(text)
        if parsed is not None:
            return parsed
        # Retry with an explicit nudge.
        prompt = (
            prompt
            + "\n\nYour previous output was not valid JSON. "
            "Return ONLY a single valid JSON object. No markdown, no prose."
        )
    LOGGER.warning("Could not parse JSON from LLM after %d attempts.", retries + 1)
    LOGGER.debug("Last raw LLM text: %s", last_text[:500])
    return None


def repair_json_if_needed(text: str) -> str:
    """Heuristically clean common LLM JSON formatting problems."""
    if text is None:
        return ""
    s = text.strip()

    # Strip markdown code fences.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s).strip()
        s = re.sub(r"```$", "", s).strip()

    # Extract the outermost JSON object/array if extra prose surrounds it.
    first_obj = s.find("{")
    first_arr = s.find("[")
    starts = [i for i in (first_obj, first_arr) if i != -1]
    if starts:
        start = min(starts)
        # Find matching last closing bracket of the same type.
        open_ch = s[start]
        close_ch = "}" if open_ch == "{" else "]"
        end = s.rfind(close_ch)
        if end != -1 and end > start:
            s = s[start : end + 1]

    # Remove trailing commas before } or ].
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Replace fancy quotes.
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    return s.strip()


def safe_parse_json(text: str) -> Optional[dict]:
    """Parse JSON from possibly-noisy LLM output. Returns None on failure."""
    if not text:
        return None
    # Direct parse first.
    try:
        result = json.loads(text)
        return result if isinstance(result, (dict, list)) else None
    except json.JSONDecodeError:
        pass
    # Repair then parse.
    repaired = repair_json_if_needed(text)
    if not repaired:
        return None
    try:
        result = json.loads(repaired)
        return result if isinstance(result, (dict, list)) else None
    except json.JSONDecodeError:
        return None
