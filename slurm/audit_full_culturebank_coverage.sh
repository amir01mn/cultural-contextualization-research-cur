#!/bin/bash
#SBATCH --job-name=ckg_covfull
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/coverage_full_%j.out
#SBATCH --error=slurm/logs/coverage_full_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_gpu_modules

cd "$PROJECT_DIR"
mkdir -p slurm/logs

source "${PROJECT_DIR}/slurm/env_hf.sh"
setup_hf_offline
log_hf_env

source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

export LLM_PROVIDER=vllm
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

QUESTIONS_JSONL="${QUESTIONS_JSONL:-data/controlled/food_hospitality_culturalbench_questions.jsonl}"
RECORDS_JSONL="${RECORDS_JSONL:?Set RECORDS_JSONL to the normalized full CultureBank JSONL}"
CHUNK_SIZE="${CHUNK_SIZE:-24}"
MAX_PASSAGE_CHARS="${MAX_PASSAGE_CHARS:-500}"
OUTPUT_JSONL="${OUTPUT_JSONL:-data/outputs/coverage_full_culturebank_exhaustive.jsonl}"
OUTPUT_MD="${OUTPUT_MD:-data/outputs/coverage_full_culturebank_exhaustive.md}"

echo "Running exhaustive CulturalBench coverage audit"
echo "Questions: $QUESTIONS_JSONL"
echo "Records: $RECORDS_JSONL"
echo "Chunk size: $CHUNK_SIZE"
echo "Model: $VLLM_MODEL"

python scripts/12_audit_culturalbench_coverage_exhaustive.py \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --questions_jsonl "$QUESTIONS_JSONL" \
  --records_jsonl "$RECORDS_JSONL" \
  --chunk_size "$CHUNK_SIZE" \
  --max_passage_chars "$MAX_PASSAGE_CHARS" \
  --output_jsonl "$OUTPUT_JSONL" \
  --output_md "$OUTPUT_MD"

echo "Exhaustive coverage audit complete. See $OUTPUT_MD"
