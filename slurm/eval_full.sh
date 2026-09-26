#!/bin/bash
#SBATCH --job-name=ckg_eval_full
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/evalfull_%j.out
#SBATCH --error=slurm/logs/evalfull_%j.err
#
# Phase 8: validate the rebuilt corpus, evaluate every system on the full
# CulturalBench, sweep the retrieval relevance floor, and emit the before/after
# comparison - all in one job so the model is loaded once.
#
#   sbatch slurm/eval_full.sh

set -uo pipefail

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

# Full CulturalBench: 1,227 Easy questions + 1,227 Hard groups = 2,454 units.
LIMIT="${LIMIT:-100000}"
TOP_K="${TOP_K:-8}"
SEED="${SEED:-42}"
SYSTEMS="${SYSTEMS:-S0_no_kg,S1_raw_text_rag,S2_flat_kg,S3_layered_kg,S4_abstraction_refinement_kg,S4_triple_retrieval,S4_no_shared_pivot}"
SWEEP="${SWEEP:-0.0,0.10,0.20,0.30,0.40}"
SWEEP_SYSTEMS="${SWEEP_SYSTEMS:-S4_abstraction_refinement_kg}"

# Validation runs as its own Python process, so it pays the sentence-transformers
# import cost (~50 min from Lustre) a second time before the evaluation process
# pays it again. Skip it here by default and validate in a separate job.
if [[ "${SKIP_VALIDATE:-1}" != "1" ]]; then
  echo "===== STEP 1: validate rebuilt corpus ====="
  python scripts/validate_corpus.py --strict
  VAL_RC=$?
  if [[ $VAL_RC -ne 0 ]]; then
    echo "QUALITY GATES FAILED - see data/outputs/phase2_validation.md" >&2
    echo "Continuing to evaluation anyway so the run is not wasted; treat results with care." >&2
  fi
else
  echo "===== STEP 1: validation skipped (SKIP_VALIDATE=1) ====="
fi

echo "===== STEP 2: evaluate on full CulturalBench ====="
python scripts/eval_culturalbench.py \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --variants easy,hard \
  --limit "$LIMIT" \
  --top_k "$TOP_K" \
  --seed "$SEED" \
  --systems "$SYSTEMS" \
  --sweep_min_score "$SWEEP" \
  --sweep_systems "$SWEEP_SYSTEMS"
EVAL_RC=$?

if [[ $EVAL_RC -ne 0 ]]; then
  echo "Evaluation failed (rc=$EVAL_RC)" >&2
  exit $EVAL_RC
fi

echo "===== STEP 3: before/after comparison ====="
python scripts/compare_runs.py

echo "Phase 8 complete. See data/outputs/culturalbench_summary.md and before_after_comparison.md"
