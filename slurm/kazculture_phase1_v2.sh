#!/bin/bash
#SBATCH --job-name=kazculture_v2
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kazculture_v2_%j.out
#SBATCH --error=slurm/logs/kazculture_v2_%j.err
#
# Independent assistant-prefix protocol, never overwrites V1's scores.
# Pilot (20 examples):
#   export KAZCULTURE_TEST_FILE="$PWD/data/raw/kazculture_phase1/test.parquet"
#   export KAZCULTURE_MODEL_REVISION=9216db5781bf21249d130ec9da846c4624c16137
#   export KAZCULTURE_LIMIT=20
#   sbatch --time=00:45:00 --export=ALL slurm/kazculture_phase1_v2.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs data/outputs

source slurm/load_modules.sh
load_gpu_modules
source slurm/env_hf.sh
setup_hf_offline
log_hf_env
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

: "${KAZCULTURE_TEST_FILE:?Set KAZCULTURE_TEST_FILE to staged 1334-row test.parquet}"
if [[ ! -f "$KAZCULTURE_TEST_FILE" ]]; then
    echo "Missing test data: $KAZCULTURE_TEST_FILE" >&2
    exit 2
fi

ARGS=(
    --test_file "$KAZCULTURE_TEST_FILE"
    --model "${KAZCULTURE_MODEL:-Qwen/Qwen3-32B}"
    --resume
)
if [[ -n "${KAZCULTURE_MODEL_REVISION:-}" ]]; then
    ARGS+=(--model_revision "$KAZCULTURE_MODEL_REVISION")
fi
if [[ -n "${KAZCULTURE_LIMIT:-}" ]]; then
    ARGS+=(--limit "$KAZCULTURE_LIMIT")
    OUTPUT_DIR="data/outputs/kazculture_phase1_v2_pilot_${KAZCULTURE_LIMIT}"
else
    OUTPUT_DIR="data/outputs/kazculture_phase1_v2"
fi
ARGS+=(--output "$OUTPUT_DIR")
echo "Running Qwen3-32B assistant-prefix V2, n=${KAZCULTURE_LIMIT:-1334}"
python scripts/eval_kazculture_phase1_v2.py "${ARGS[@]}"
