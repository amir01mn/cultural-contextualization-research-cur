#!/bin/bash
#SBATCH --job-name=ckg_v2_culturebank
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/v2_culturebank_%j.out
#SBATCH --error=slurm/logs/v2_culturebank_%j.err
#
# Eval v2 (CultureBank grounded); protocol in docs/EVAL_V2_PLAN.md. One Python process so
# the model loads once. Writes only under data/outputs/eval_v2/.
#
#   sbatch slurm/eval_v2_culturebank.sh

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
# shellcheck source=load_modules.sh
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_gpu_modules

cd "$PROJECT_DIR"
mkdir -p slurm/logs data/outputs/eval_v2

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
READY_FLAG="data/eval_v2/READY_culturebank"
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

# Re-run prep with MiniLM on the GPU (login node was too slow; TF-IDF makes the
# pre-registered 0.75/0.85 similarity cut-offs nearly inert). Falls back to the
# TF-IDF prep if this fails.
if [[ "${PREP_MINILM:-1}" == "1" && ! -f data/eval_v2/culturebank/.minilm ]]; then
  echo "===== PREP (MiniLM) $(date) ====="
  rm -rf data/eval_v2/culturebank_minilm
  if EMBEDDING_BACKEND=sentence-transformers python scripts/prep_culturebank_eval_v2.py \
       --sim minilm --skip_verify --out_dir data/eval_v2/culturebank_minilm; then
    rm -rf data/eval_v2/culturebank_tfidf
    mv data/eval_v2/culturebank data/eval_v2/culturebank_tfidf
    mv data/eval_v2/culturebank_minilm data/eval_v2/culturebank
    touch data/eval_v2/culturebank/.minilm
  else
    echo "MiniLM prep failed; using TF-IDF prep" >&2
  fi
fi

echo "===== EVAL V2 CultureBank grounded: $(date) model=$VLLM_MODEL ====="
python scripts/run_culturebank_eval_v2.py --provider vllm --stages "${STAGES:-paraphrase,open_gen,mcq,heldout,retrieval_metrics}" ${EXTRA_ARGS:-}
RC=$?
echo "===== done rc=$RC: $(date) ====="
exit $RC
