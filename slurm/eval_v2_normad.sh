#!/bin/bash
#SBATCH --job-name=ckg_v2_normad
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/v2_normad_%j.out
#SBATCH --error=slurm/logs/v2_normad_%j.err
#
# Eval v2 (NormAd); protocol in docs/EVAL_V2_PLAN.md. One Python process so
# the model loads once. Writes only under data/outputs/eval_v2/.
#
#   sbatch slurm/eval_v2_normad.sh

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
# shellcheck source=load_modules.sh
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_gpu_modules

cd "$PROJECT_DIR"
mkdir -p slurm/logs data/outputs/eval_v2

# Duplicate submissions (full A100 + MIG slice) race for the queue; the first to
# start takes this lock and any later copy exits without touching outputs.
LOCK_DIR="data/eval_v2/LOCK_${LOCK_NAME:-normad}"
if ! mkdir -p data/eval_v2 || ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another job owns $LOCK_DIR ($(cat "$LOCK_DIR/job" 2>/dev/null)); exiting."
  exit 0
fi
echo "${SLURM_JOB_ID:-local}" > "$LOCK_DIR/job"

# shellcheck source=env_hf.sh
source "${PROJECT_DIR}/slurm/env_hf.sh"
setup_hf_offline
log_hf_env

source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

export LLM_PROVIDER=vllm
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# Queue-position hedge: the job may start before the driver code is reviewed.
# Warm the Lustre import cache while waiting for the READY flag (max WAIT_READY_MIN).
READY_FLAG="data/eval_v2/READY_normad"
if [[ ! -f "$READY_FLAG" ]]; then
  echo "Waiting for $READY_FLAG (warming imports) $(date)"
  python -c "import torch, vllm, transformers, sentence_transformers; print('imports warm')" &
  for ((i=0; i<${WAIT_READY_MIN:-70}; i++)); do
    [[ -f "$READY_FLAG" ]] && break
    sleep 60
  done
  wait
  if [[ ! -f "$READY_FLAG" ]]; then echo "READY flag never appeared; exiting" >&2; exit 3; fi
fi

echo "===== EVAL V2 NormAd: $(date) model=$VLLM_MODEL ====="
python scripts/run_normad_v2.py --provider vllm  ${EXTRA_ARGS:-}
RC=$?
echo "===== done rc=$RC: $(date) ====="
exit $RC
