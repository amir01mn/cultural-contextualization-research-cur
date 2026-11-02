#!/bin/bash
#SBATCH --job-name=kz_s1_eval
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_s1_eval_%j.out
#SBATCH --error=slurm/logs/kz_s1_eval_%j.err
#
# KAZCULTURE_S1_EVAL_MODE=full|strict
# Requires KAZCULTURE_DEV_CONFIG pointing to the frozen train-only config.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
source slurm/load_modules.sh
load_gpu_modules
source slurm/env_hf.sh
setup_hf_offline
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

MODE="${KAZCULTURE_S1_EVAL_MODE:?Set KAZCULTURE_S1_EVAL_MODE=full or strict}"
: "${KAZCULTURE_DEV_CONFIG:?Set KAZCULTURE_DEV_CONFIG to frozen train-only S1 config}"
case "$MODE" in
  full|strict) ;;
  *) echo "Invalid KAZCULTURE_S1_EVAL_MODE=$MODE" >&2; exit 2;;
esac

BASE=data/kazculture_pipeline
INDEX="$BASE/index/s1_$MODE"
[[ -f "$INDEX/build_manifest.json" ]] || {
  echo "Missing S1 index: $INDEX" >&2; exit 2;
}
mkdir -p slurm/logs "$BASE/evaluation"
python scripts/eval_kazculture_retrieval.py \
  --test_file data/raw/kazculture_phase1/test.parquet \
  --baseline_predictions data/outputs/kazculture_phase1_v2/predictions.jsonl \
  --index "$INDEX" \
  --out "$BASE/evaluation/s1_$MODE" \
  --systems S1 \
  --config "$KAZCULTURE_DEV_CONFIG" \
  --model_revision "${KAZCULTURE_MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}" \
  --resume
