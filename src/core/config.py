"""Central configuration for the cultural-KG research pipeline.

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
# src/core/config.py -> project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
# Neighbours examined per entity during resolution. Bounds cost in place of the
# former dense n x n similarity matrix (~8 GB at 45k surface entities).
ER_MAX_NEIGHBORS = int(os.getenv("ER_MAX_NEIGHBORS", "25"))
ALIGNMENT_SIMILARITY_THRESHOLD = float(os.getenv("ALIGNMENT_SIMILARITY_THRESHOLD", "0.82"))
# Candidate cross-cultural pairs considered. The old cap of 200 yielded only 107
# verified pivots for a 23k-record corpus.
ALIGNMENT_MAX_PAIRS = int(os.getenv("ALIGNMENT_MAX_PAIRS", "2000"))
MAX_PATHWAYS_PER_CULTURE = int(os.getenv("MAX_PATHWAYS_PER_CULTURE", "500"))
# Global ceiling on the pathway index. Per-culture caps alone are unbounded when
# the corpus has a long tail of free-text cultural groups; the index is scanned
# once per evaluation query, so an unbounded pool would slow retrieval and bloat
# the on-disk embeddings. Highest-scoring pathways are kept.
MAX_TOTAL_PATHWAYS = int(os.getenv("MAX_TOTAL_PATHWAYS", "150000"))


# --- LLM provider selection ------------------------------------------------
# 'ollama' for local development; 'vllm' for batched GPU inference on Narval.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Generation defaults shared across backends.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# --- Per-stage generation budgets -----------------------------------------
# A single global 512-token budget silently truncated extraction and critique
# JSON mid-object, which made 96% of the corpus fall back to schema triples and
# 98% of critiques unparseable. Each stage now gets a budget sized to its own
# expected output: extraction emits 5-12 fully-specified triples, critique emits
# issues + suggestions, alignment answers a yes/no verdict, MCQ emits a letter.
EXTRACTION_MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", "2048"))
CRITIQUE_MAX_TOKENS = int(os.getenv("CRITIQUE_MAX_TOKENS", "1536"))
ALIGNMENT_MAX_TOKENS = int(os.getenv("ALIGNMENT_MAX_TOKENS", "256"))
MCQ_MAX_TOKENS = int(os.getenv("MCQ_MAX_TOKENS", "512"))

# Retry a JSON stage once with a larger budget before giving up.
JSON_RETRY_TOKEN_MULTIPLIER = float(os.getenv("JSON_RETRY_TOKEN_MULTIPLIER", "1.75"))

STAGE_MAX_TOKENS = {
    "extraction": EXTRACTION_MAX_TOKENS,
    "critique": CRITIQUE_MAX_TOKENS,
    "alignment": ALIGNMENT_MAX_TOKENS,
    "mcq": MCQ_MAX_TOKENS,
}


def stage_max_tokens(stage: str) -> int:
    """Return the generation budget for a named pipeline stage."""
    return STAGE_MAX_TOKENS.get(stage, LLM_MAX_TOKENS)


# --- Embeddings ------------------------------------------------------------
# 'auto' prefers sentence-transformers and falls back to TF-IDF. Force 'tfidf'
# for fast, deterministic, offline test runs (loading MiniLM from a shared
# filesystem can take minutes).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()


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
# Retrieval quality (Phase 3+ eval improvements)
RETRIEVAL_GENERATED_PENALTY = float(os.getenv("RETRIEVAL_GENERATED_PENALTY", "0.45"))
RETRIEVAL_CULTURE_MISMATCH_PENALTY = float(os.getenv("RETRIEVAL_CULTURE_MISMATCH_PENALTY", "0.15"))
RETRIEVAL_CULTURE_UNKNOWN_SCORE = float(os.getenv("RETRIEVAL_CULTURE_UNKNOWN_SCORE", "0.6"))
RETRIEVAL_SHARED_PIVOT_CULTURE_SCORE = float(os.getenv("RETRIEVAL_SHARED_PIVOT_CULTURE_SCORE", "0.75"))
RETRIEVAL_CANDIDATE_MULTIPLIER = int(os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "5"))
# Absolute relevance floor. Evidence scoring below this is dropped entirely; if
# nothing clears it the system emits an empty evidence block and answers
# closed-book. Without this, irrelevant evidence was injected unconditionally
# and corrupted answers the base model would otherwise have got right.
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.15"))
# Cap how much of one retrieval result set may come from a single culture, to
# stop one generic pathway dominating (one was returned for 239/500 questions).
RETRIEVAL_MAX_PER_CULTURE = int(os.getenv("RETRIEVAL_MAX_PER_CULTURE", "3"))
# Include answer options in the retrieval query. Off by default: distractor text
# pulls in evidence supporting wrong options.
RETRIEVAL_QUERY_INCLUDES_OPTIONS = _get_bool("RETRIEVAL_QUERY_INCLUDES_OPTIONS", False)
# Corpus build: schema fallback when LLM returns empty/unparseable (not when server down)
CORPUS_SCHEMA_FALLBACK_ON_EMPTY = _get_bool("CORPUS_SCHEMA_FALLBACK_ON_EMPTY", True)
MAX_CRITIQUE_SUGGESTIONS = int(os.getenv("MAX_CRITIQUE_SUGGESTIONS", "2"))
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
