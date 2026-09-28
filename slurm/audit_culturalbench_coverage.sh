#!/bin/bash
#SBATCH --job-name=ckg_cover
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/coverage_%j.out
#SBATCH --error=slurm/logs/coverage_%j.err

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
PASSAGES="${PASSAGES:-data/processed/corpus_passages.jsonl}"
SEMANTIC_FALLBACK_K="${SEMANTIC_FALLBACK_K:-10}"
MAX_CULTURE_RECORDS="${MAX_CULTURE_RECORDS:-80}"
MAX_PASSAGE_CHARS="${MAX_PASSAGE_CHARS:-500}"

echo "Running CulturalBench corpus coverage audit"
echo "Questions: $QUESTIONS_JSONL"
echo "Passages: $PASSAGES"
echo "Model: $VLLM_MODEL"

python scripts/10_audit_culturalbench_coverage.py   --provider vllm   --model "$VLLM_MODEL"   --questions_jsonl "$QUESTIONS_JSONL"   --passages "$PASSAGES"   --semantic_fallback_k "$SEMANTIC_FALLBACK_K"   --max_culture_records "$MAX_CULTURE_RECORDS"   --max_passage_chars "$MAX_PASSAGE_CHARS"

echo "Coverage audit complete. See data/outputs/coverage_audit.md"
