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
# Build the CultureBank retrieval corpus (raw-text passages + LLM-extracted
# triples) on a GPU node using the vLLM backend. Run prep_offline.sh first.
#
#   sbatch slurm/build_corpus.sh

set -euo pipefail

# SLURM copies batch scripts to spool; use submit dir, not $0 or BASH_SOURCE.
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

echo "Building corpus with vLLM model: $VLLM_MODEL (limit=$CORPUS_LIMIT)"
python scripts/06_build_retrieval_corpus.py \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --limit "$CORPUS_LIMIT"

echo "Corpus build job complete."
