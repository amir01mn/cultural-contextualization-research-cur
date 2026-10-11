#!/bin/bash
#SBATCH --job-name=ckg_v3_build
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/v3_build_%j.out
#SBATCH --error=slurm/logs/v3_build_%j.err
#
# KG v3 build (docs/EVAL_V3_PLAN.md): frame labels with Qwen2.5-7B, norm-type
# clustering, frame/held-out/local-pathway indexes (MiniLM), dev-only check,
# judge audit samples, and the hard-distractor MCQ prep. Writes only under
# data/kg_v3/ and data/eval_v3/.
#
#   sbatch slurm/build_kg_v3.sh

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_gpu_modules
cd "$PROJECT_DIR"
mkdir -p slurm/logs data/kg_v3 data/eval_v3

LOCK_DIR="data/kg_v3/LOCK_${LOCK_NAME:-build}"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another job owns $LOCK_DIR ($(cat "$LOCK_DIR/job" 2>/dev/null)); exiting."
  exit 0
fi
echo "${SLURM_JOB_ID:-local}" > "$LOCK_DIR/job"

source "${PROJECT_DIR}/slurm/env_hf.sh"
setup_hf_offline
log_hf_env
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

export LLM_PROVIDER=vllm
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export EMBEDDING_BACKEND=sentence-transformers

echo "===== KG v3 build: $(date) model=$VLLM_MODEL ====="
python scripts/build_kg_v3.py --provider vllm --stages "${STAGES:-frames,label,kg,local,check,audit}"
RC=$?
echo "===== build rc=$RC: $(date) ====="
[[ $RC -eq 0 ]] && touch data/kg_v3/READY

if [[ "${SKIP_HARD:-0}" != "1" ]]; then
  echo "===== hard-distractor MCQ prep: $(date) ====="
  python scripts/prep_culturebank_hard_v3.py --sim minilm \
    && touch data/eval_v3/culturebank_hard/READY || echo "hard prep failed" >&2
fi
echo "===== done: $(date) ====="
exit $RC
