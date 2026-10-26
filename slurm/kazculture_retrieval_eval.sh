#!/bin/bash
#SBATCH --job-name=kazculture_rag
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kazculture_rag_%j.out
#SBATCH --error=slurm/logs/kazculture_rag_%j.err
#
# KG_EVAL_MODE=full|strict|combined|combined_strict.
# Frozen S0 baseline must be the 41.3793% V2 predictions from the same Qwen3.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
source slurm/load_modules.sh
load_gpu_modules
source slurm/env_hf.sh
setup_hf_offline
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"
MODE="${KG_EVAL_MODE:-full}"
BASE="data/kazculture_pipeline"
mkdir -p slurm/logs "$BASE/evaluation"
case "$MODE" in
full|strict) SYSTEMS="S1,S2,S3,S4F,S4R";;
combined|combined_strict) SYSTEMS="S4R_combined";;
*) echo "Unexpected KG_EVAL_MODE=$MODE" >&2; exit 2;;
esac
INDEX="$BASE/index/$MODE"
[[ -f "$INDEX/build_manifest.json" ]] || { echo "Missing indexed KG: $INDEX" >&2; exit 2; }
ARGS=(
  --test_file "data/raw/kazculture_phase1/test.parquet"
  --baseline_predictions "data/outputs/kazculture_phase1_v2/predictions.jsonl"
  --index "$INDEX"
  --out "$BASE/evaluation/$MODE"
  --systems "$SYSTEMS"
  --model_revision "${KAZCULTURE_MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}"
  --resume
)
if [[ -n "${KAZCULTURE_DEV_CONFIG:-}" ]]; then
  ARGS+=(--config "$KAZCULTURE_DEV_CONFIG")
else
  echo "Set KAZCULTURE_DEV_CONFIG to a train-only frozen threshold file" >&2
  exit 2
fi
python scripts/eval_kazculture_retrieval.py "${ARGS[@]}"
