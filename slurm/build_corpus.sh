#!/bin/bash
#SBATCH --job-name=ckg_corpus
#SBATCH --account=def-CHANGEME          # <-- set your Alliance allocation
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/corpus_%j.out
#SBATCH --error=slurm/logs/corpus_%j.err
#
# Build the CultureBank retrieval corpus (raw-text passages + LLM-extracted
# triples) on a GPU node using the vLLM backend. Run prep_offline.sh first.
#
#   sbatch slurm/build_corpus.sh

set -euo pipefail

module load StdEnv/2023 gcc/12.3 python/3.11 arrow/17.0.0 cuda/12.2 || true

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs

# Offline: compute nodes have no internet. Read from the staged HF cache.
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"

export LLM_PROVIDER=vllm
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CORPUS_LIMIT="${CORPUS_LIMIT:-2000}"

echo "Building corpus with vLLM model: $VLLM_MODEL (limit=$CORPUS_LIMIT)"
python scripts/06_build_retrieval_corpus.py \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --limit "$CORPUS_LIMIT"

echo "Corpus build job complete."
