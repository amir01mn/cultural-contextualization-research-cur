"""Central configuration for the cultural-kg-mvp pipeline.

Values are read from environment variables (optionally loaded from a local
``.env`` file). The defaults match ``.env.example`` so the pipeline runs locally
against Ollama with ``llama3.1:8b`` and never requires a paid API.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Dataset ---------------------------------------------------------------
DATASET_NAME = os.getenv("DATASET_NAME", "culturebank")
DEFAULT_SPLIT = os.getenv("DEFAULT_SPLIT", "train")
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "100"))

# Hugging Face dataset id used when attempting a real download.
CULTUREBANK_HF_ID = os.getenv("CULTUREBANK_HF_ID", "SALT-NLP/CultureBank")


# --- LLM (local / Ollama) --------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Request behavior for the local LLM.
# The first call after a fresh `ollama pull` must cold-load the model into
# memory before generating, which can take a few minutes on some machines. The
# timeout is generous to avoid spurious ReadTimeout failures on that first call.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "1536"))
# Keep the model resident in memory between calls (Ollama 'keep_alive').
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")


# --- Feature flags ---------------------------------------------------------
USE_LLM_EXTRACTION = _get_bool("USE_LLM_EXTRACTION", True)
USE_LLM_CRITIQUE = _get_bool("USE_LLM_CRITIQUE", True)
USE_LLM_ANSWERS = _get_bool("USE_LLM_ANSWERS", True)
USE_LLM_EVALUATION = _get_bool("USE_LLM_EVALUATION", False)


# --- Pipeline knobs --------------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
NUM_SCENARIOS = int(os.getenv("NUM_SCENARIOS", "25"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


# --- Layer distribution priors --------------------------------------------
# Maps a layer label to a [L1, L2, L3] soft distribution used on graph nodes.
LAYER_DISTRIBUTIONS = {
    "LAYER_1_SURFACE": [0.70, 0.15, 0.15],
    "LAYER_2_SOCIAL": [0.15, 0.70, 0.15],
    "LAYER_3_DEEP_VALUE": [0.15, 0.15, 0.70],
    "UNKNOWN": [0.33, 0.33, 0.33],
}


def summary() -> dict:
    """Return a serializable snapshot of the active configuration."""
    return {
        "dataset_name": DATASET_NAME,
        "default_split": DEFAULT_SPLIT,
        "default_limit": DEFAULT_LIMIT,
        "llm_provider": LLM_PROVIDER,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "use_llm_extraction": USE_LLM_EXTRACTION,
        "use_llm_critique": USE_LLM_CRITIQUE,
        "use_llm_answers": USE_LLM_ANSWERS,
        "use_llm_evaluation": USE_LLM_EVALUATION,
    }
