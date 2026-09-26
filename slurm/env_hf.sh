#!/bin/bash
# Shared Hugging Face cache paths. Datasets and model weights live in the
# workspace folder that contains this repo (backed-up /project space):
#
#   <workspace>/hf_cache/      <- HF_HOME
#   <workspace>/<this repo>/
#
# Nothing here depends on $SCRATCH, which is purged. Override with
# CKG_HF_HOME=/some/path to use a different location.

_ckg_default_hf_home() {
  local repo_root
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  echo "$(dirname "$repo_root")/hf_cache"
}

setup_hf_cache() {
  # Deliberately ignore any HF_HOME inherited from the login environment (often
  # under $HOME or $SCRATCH): the workspace cache is the single source of truth.
  export HF_HOME="${CKG_HF_HOME:-$(_ckg_default_hf_home)}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
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
