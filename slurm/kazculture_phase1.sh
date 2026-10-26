#!/bin/bash
#SBATCH --job-name=kazculture_p1
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kazculture_phase1_%j.out
#SBATCH --error=slurm/logs/kazculture_phase1_%j.err
#
# Narval generally uses full A100 40GB GPUs. Two 40GB cards offer ~80GB
# combined VRAM for ~64GB BF16 model weights; actual headroom may be tight.
# Verify available GPU type, partition and account with sinfo/sacctmgr.
# If memory allocation or OOM fails, request additional GPUs where permitted.
# Example:
#   export KAZCULTURE_TEST_FILE=/project/.../kazculture_test.parquet
#   sbatch slurm/kazculture_phase1.sh
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs data/outputs/kazculture_phase1

source slurm/load_modules.sh
load_gpu_modules
source slurm/env_hf.sh
setup_hf_offline
log_hf_env
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

: "${KAZCULTURE_TEST_FILE:?Stage the gated KazCulture test split and set KAZCULTURE_TEST_FILE}"
if [[ ! -f "$KAZCULTURE_TEST_FILE" ]]; then
  echo "Missing dataset: $KAZCULTURE_TEST_FILE" >&2
  exit 2
fi
# Pilot and full runs write to separate checkpoints. No test-set leakage between runs.
# Example pilot: KAZCULTURE_LIMIT=5 sbatch --time=00:45:00 --export=ALL slurm/kazculture_phase1.sh
ARGS=(
  --test_file "$KAZCULTURE_TEST_FILE"
  --model "${KAZCULTURE_MODEL:-Qwen/Qwen3-32B}"
  --resume
)
if [[ -n "${KAZCULTURE_LIMIT:-}" ]]; then
  ARGS+=(--limit "$KAZCULTURE_LIMIT")
  OUT_DIR="data/outputs/kazculture_phase1_pilot_${KAZCULTURE_LIMIT}"
else
  OUT_DIR="data/outputs/kazculture_phase1"
fi
if [[ -n "${KAZCULTURE_MODEL_REVISION:-}" ]]; then
  ARGS+=(--model_revision "$KAZCULTURE_MODEL_REVISION")
fi
ARGS+=(--output "$OUT_DIR")
echo "Evaluating model on ${KAZCULTURE_LIMIT:-all 1334} questions; outputs: $OUT_DIR"
python scripts/eval_kazculture_phase1.py "${ARGS[@]}"
