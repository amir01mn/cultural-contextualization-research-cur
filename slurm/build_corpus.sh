#!/bin/bash
#SBATCH --job-name=ckg_corpus
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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

EXTRA_ARGS=(--provider vllm --model "$VLLM_MODEL" --limit "$CORPUS_LIMIT" --batch_size "$BATCH_SIZE")
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
else
  EXTRA_ARGS+=(--no_resume)
fi
if [[ "$SKIP_ALIGNMENT_VERIFY" == "1" ]]; then
  EXTRA_ARGS+=(--skip_alignment_verify)
fi

echo "Building Phase 2 corpus with vLLM model: $VLLM_MODEL (limit=$CORPUS_LIMIT batch=$BATCH_SIZE)"
python scripts/06_build_retrieval_corpus.py "${EXTRA_ARGS[@]}"

echo "Running Phase 2 validation..."
python scripts/09_validate_phase2.py

echo "Corpus build job complete."
