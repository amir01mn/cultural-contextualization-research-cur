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

# Hugging Face dataset ids used when attempting a real download.
CULTUREBANK_HF_ID = os.getenv("CULTUREBANK_HF_ID", "SALT-NLP/CultureBank")
# CulturalBench is the sole evaluation benchmark (Easy + Hard).
CULTURALBENCH_HF_ID = os.getenv("CULTURALBENCH_HF_ID", "kellycyy/CulturalBench")

# Size of the CultureBank retrieval corpus used by the KG-augmented systems.
# (The KG is built from CultureBank; CulturalBench is held out for evaluation.)
CORPUS_LIMIT = int(os.getenv("CORPUS_LIMIT", "2000"))

# Phase 2 artifact paths
CORPUS_STATE_PATH = PROCESSED_DIR / "corpus_build_state.json"
LLM_CACHE_DIR = PROCESSED_DIR / "llm_cache"
CORPUS_TRIPLES_REFINED_PATH = PROCESSED_DIR / "corpus_triples_refined.jsonl"
ENTITY_MAP_PATH = PROCESSED_DIR / "entity_map.jsonl"
ALIGNMENTS_PATH = PROCESSED_DIR / "cross_cultural_alignments.jsonl"
PATHWAYS_PATH = PROCESSED_DIR / "corpus_pathways.jsonl"
PATHWAY_INDEX_DIR = PROCESSED_DIR / "pathway_index"
PHASE2_VALIDATION_PATH = OUTPUTS_DIR / "phase2_validation.md"

# Phase 2 knobs
CORPUS_BATCH_SIZE = int(os.getenv("CORPUS_BATCH_SIZE", "32"))
ER_SIMILARITY_THRESHOLD = float(os.getenv("ER_SIMILARITY_THRESHOLD", "0.88"))
ALIGNMENT_SIMILARITY_THRESHOLD = float(os.getenv("ALIGNMENT_SIMILARITY_THRESHOLD", "0.82"))
MAX_PATHWAYS_PER_CULTURE = int(os.getenv("MAX_PATHWAYS_PER_CULTURE", "500"))


# --- LLM provider selection ------------------------------------------------
# 'ollama' for local development; 'vllm' for batched GPU inference on Narval.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Generation defaults shared across backends.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))


# --- Ollama (local dev) ----------------------------------------------------
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
# Emulated batching: number of concurrent Ollama requests.
OLLAMA_MAX_WORKERS = int(os.getenv("OLLAMA_MAX_WORKERS", "4"))


# --- vLLM (cluster GPU) ----------------------------------------------------
# Default to a non-gated instruct model to avoid Hugging Face gating friction.
# Llama-3.1-8B-Instruct is gated (requires token + license acceptance); set
# VLLM_MODEL to it once access is granted. Multi-model comparison is Phase 3.
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "auto")
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))


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


def active_model(provider: str | None = None) -> str:
    """Return the model id for the active (or given) provider."""
    provider = (provider or LLM_PROVIDER or "ollama").strip().lower()
    return VLLM_MODEL if provider == "vllm" else OLLAMA_MODEL


def summary() -> dict:
    """Return a serializable snapshot of the active configuration."""
    return {
        "dataset_name": DATASET_NAME,
        "default_split": DEFAULT_SPLIT,
        "default_limit": DEFAULT_LIMIT,
        "culturebank_hf_id": CULTUREBANK_HF_ID,
        "culturalbench_hf_id": CULTURALBENCH_HF_ID,
        "corpus_limit": CORPUS_LIMIT,
        "llm_provider": LLM_PROVIDER,
        "active_model": active_model(),
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "vllm_model": VLLM_MODEL,
        "use_llm_extraction": USE_LLM_EXTRACTION,
        "use_llm_critique": USE_LLM_CRITIQUE,
        "use_llm_answers": USE_LLM_ANSWERS,
        "use_llm_evaluation": USE_LLM_EVALUATION,
    }
