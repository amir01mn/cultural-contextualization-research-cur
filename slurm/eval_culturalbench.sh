#!/bin/bash
#SBATCH --job-name=ckg_eval
#SBATCH --account=def-CHANGEME          # <-- set your Alliance allocation
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/eval_%j.out
#SBATCH --error=slurm/logs/eval_%j.err
#
# Evaluate systems S0-S3 on CulturalBench with the vLLM backend.
# Requires the retrieval corpus (sbatch slurm/build_corpus.sh) and staged data.
#
#   sbatch slurm/eval_culturalbench.sh

set -euo pipefail

module load StdEnv/2023 gcc/12.3 python/3.11 arrow/17.0.0 cuda/12.2 || true

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"

export LLM_PROVIDER=vllm
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LIMIT="${LIMIT:-250}"
TOP_K="${TOP_K:-8}"
SEED="${SEED:-42}"

echo "Evaluating S0-S3 on CulturalBench with vLLM model: $VLLM_MODEL"
python scripts/07_eval_culturalbench.py \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --variants easy,hard \
  --limit "$LIMIT" \
  --top_k "$TOP_K" \
  --seed "$SEED"

echo "Eval job complete. See data/outputs/culturalbench_summary.md"
