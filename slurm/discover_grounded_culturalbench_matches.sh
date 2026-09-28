#!/bin/bash
#SBATCH --job-name=ckg_ground
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/grounded_%j.out
#SBATCH --error=slurm/logs/grounded_%j.err

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
RECORDS_JSONL="${RECORDS_JSONL:?Set RECORDS_JSONL to the frozen full CultureBank JSONL}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LIMIT_QUESTIONS="${LIMIT_QUESTIONS:-}"

STATE_ROOT="${STATE_ROOT:-${SCRATCH:-$PROJECT_DIR/data/outputs}/cultural_kg_state/grounded_culturalbench}"
mkdir -p "$STATE_ROOT"

STAGE1_STATE="${STAGE1_STATE:-$STATE_ROOT/stage1.jsonl}"
STAGE2_STATE="${STAGE2_STATE:-$STATE_ROOT/stage2.jsonl}"
OUTPUT_MATCHES="${OUTPUT_MATCHES:-data/outputs/grounded_culturalbench_matches.jsonl}"
OUTPUT_MD="${OUTPUT_MD:-data/outputs/grounded_culturalbench_matches.md}"

EXTRA_ARGS=()
if [[ -n "$LIMIT_QUESTIONS" ]]; then
  EXTRA_ARGS+=(--limit_questions "$LIMIT_QUESTIONS")
fi

echo "Running strict CultureBank-CulturalBench grounded match discovery"
echo "Questions: $QUESTIONS_JSONL"
echo "Records: $RECORDS_JSONL"
echo "Batch size: $BATCH_SIZE"
echo "Stage 1 state: $STAGE1_STATE"
echo "Stage 2 state: $STAGE2_STATE"
echo "Model: $VLLM_MODEL"

python scripts/13_discover_grounded_culturalbench_matches.py \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --questions_jsonl "$QUESTIONS_JSONL" \
  --records_jsonl "$RECORDS_JSONL" \
  --batch_size "$BATCH_SIZE" \
  --stage1_state "$STAGE1_STATE" \
  --stage2_state "$STAGE2_STATE" \
  --output_matches "$OUTPUT_MATCHES" \
  --output_md "$OUTPUT_MD" \
  "${EXTRA_ARGS[@]}"

echo "Grounded match discovery complete. See $OUTPUT_MD"
