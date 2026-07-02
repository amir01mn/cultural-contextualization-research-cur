#!/bin/bash
# Shared Hugging Face cache paths for Narval (datasets + model weights live on $SCRATCH).

setup_hf_cache() {
  export HF_HOME="${HF_HOME:-${SCRATCH:?SCRATCH must be set}/hf_cache}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE"
}

setup_hf_offline() {
  setup_hf_cache
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
}

log_hf_env() {
  echo "HF_HOME=$HF_HOME"
  echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
  echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
  echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-0}"
}
