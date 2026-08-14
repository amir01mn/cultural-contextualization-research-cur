#!/bin/bash
#SBATCH --job-name=ckg_corpus
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# Max on gpubase_bygpu_b1; re-submit with RESUME=1 if the job times out.
#SBATCH --output=slurm/logs/corpus_%j.out
#SBATCH --error=slurm/logs/corpus_%j.err
#
# Build the CultureBank retrieval corpus (Phase 2 pipeline) on a GPU node.
# Run prep_offline.sh first.
#
#   CORPUS_LIMIT=2000 BATCH_SIZE=32 sbatch slurm/build_corpus.sh
#   CORPUS_LIMIT=50 sbatch slurm/build_corpus.sh   # smoke test

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
# shellcheck source=load_modules.sh
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_gpu_modules

cd "$PROJECT_DIR"
mkdir -p slurm/logs

# shellcheck source=env_hf.sh
source "${PROJECT_DIR}/slurm/env_hf.sh"
setup_hf_offline
log_hf_env

source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

export LLM_PROVIDER=vllm
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CORPUS_LIMIT="${CORPUS_LIMIT:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESUME="${RESUME:-1}"
SKIP_ALIGNMENT_VERIFY="${SKIP_ALIGNMENT_VERIFY:-0}"
STRICT="${STRICT:-1}"
RUN_TAG="${RUN_TAG:-}"

# Per-run cache/state paths. The response cache is keyed on the generation
# context (model + token budget), but an explicit tag keeps a rebuild fully
# isolated from the previous checkpoint - which matters because the old
# checkpoint marks all 22,990 records complete and would otherwise cause
# extraction and critique to be skipped entirely.
if [[ -n "$RUN_TAG" ]]; then
  CACHE_DIR="${CACHE_DIR:-data/processed/llm_cache_${RUN_TAG}}"
  STATE_PATH="${STATE_PATH:-data/processed/corpus_build_state_${RUN_TAG}.json}"
else
  CACHE_DIR="${CACHE_DIR:-data/processed/llm_cache}"
  STATE_PATH="${STATE_PATH:-data/processed/corpus_build_state.json}"
fi

EXTRA_ARGS=(--provider vllm --model "$VLLM_MODEL" --limit "$CORPUS_LIMIT" \
            --batch_size "$BATCH_SIZE" --cache_dir "$CACHE_DIR" \
            --state_path "$STATE_PATH")
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
else
  EXTRA_ARGS+=(--no_resume)
fi
if [[ "$SKIP_ALIGNMENT_VERIFY" == "1" ]]; then
  EXTRA_ARGS+=(--skip_alignment_verify)
fi

echo "Building Phase 2 corpus with vLLM model: $VLLM_MODEL (limit=$CORPUS_LIMIT batch=$BATCH_SIZE)"
echo "  cache_dir=$CACHE_DIR"
echo "  state_path=$STATE_PATH"
python scripts/06_build_retrieval_corpus.py "${EXTRA_ARGS[@]}"

echo "Running Phase 2 validation..."
VALIDATE_ARGS=()
if [[ "$STRICT" == "1" ]]; then
  VALIDATE_ARGS+=(--strict)
fi
if ! python scripts/09_validate_phase2.py "${VALIDATE_ARGS[@]}"; then
  echo "QUALITY GATES FAILED - see data/outputs/phase2_validation.md" >&2
  exit 1
fi

echo "Corpus build job complete."
