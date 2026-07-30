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
#   INPUT_RECORDS=data/controlled/food_hospitality_records_keep_only.jsonl CORPUS_LIMIT=10 BATCH_SIZE=8 sbatch slurm/build_corpus.sh
#   SKIP_VALIDATION=1 INPUT_RECORDS=data/controlled/food_hospitality_records_keep_only.jsonl CORPUS_LIMIT=10 BATCH_SIZE=8 sbatch slurm/build_corpus.sh

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
INPUT_RECORDS="${INPUT_RECORDS:-}"
SKIP_CRITIQUE="${SKIP_CRITIQUE:-0}"
SKIP_ALIGNMENT="${SKIP_ALIGNMENT:-0}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"

EXTRA_ARGS=(--provider vllm --model "$VLLM_MODEL" --limit "$CORPUS_LIMIT" --batch_size "$BATCH_SIZE")
if [[ "$RESUME" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
else
  EXTRA_ARGS+=(--no_resume)
fi
if [[ "$SKIP_ALIGNMENT_VERIFY" == "1" ]]; then
  EXTRA_ARGS+=(--skip_alignment_verify)
fi
if [[ -n "$INPUT_RECORDS" ]]; then
  EXTRA_ARGS+=(--input_records "$INPUT_RECORDS")
fi
if [[ "$SKIP_CRITIQUE" == "1" ]]; then
  EXTRA_ARGS+=(--skip_critique)
fi
if [[ "$SKIP_ALIGNMENT" == "1" ]]; then
  EXTRA_ARGS+=(--skip_alignment)
fi

echo "Building Phase 2 corpus with vLLM model: $VLLM_MODEL (limit=$CORPUS_LIMIT batch=$BATCH_SIZE input_records=${INPUT_RECORDS:-default CultureBank})"
python scripts/06_build_retrieval_corpus.py "${EXTRA_ARGS[@]}"

if [[ "$SKIP_VALIDATION" == "1" ]]; then
  echo "Skipping Phase 2 validation because SKIP_VALIDATION=1."
else
  echo "Running Phase 2 validation..."
  python scripts/09_validate_phase2.py
fi

echo "Corpus build job complete."