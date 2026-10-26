#!/bin/bash
#SBATCH --job-name=kazculture_p1
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/kazculture_phase1_%j.out
#SBATCH --error=slurm/logs/kazculture_phase1_%j.err
#
# Use two full A100 80GB GPUs for the Qwen3-32B BF16 weights plus overhead.
# Adjust resources to your cluster's actual GPU availability/partitions.
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
python scripts/eval_kazculture_phase1.py \
  --test_file "$KAZCULTURE_TEST_FILE" \
  --model "${KAZCULTURE_MODEL:-Qwen/Qwen3-32B}" \
  --output data/outputs/kazculture_phase1 \
  --resume
