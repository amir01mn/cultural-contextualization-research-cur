#!/bin/bash
#SBATCH --job-name=kz_hout_qa
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_hout_qa_%j.out
#SBATCH --error=slurm/logs/kz_hout_qa_%j.err
#
# Exploratory overlap-only source-first QA diagnostic.
# Uses frozen S0 predictions and the already-built 150-source diagnostic indexes.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
source slurm/load_modules.sh
load_gpu_modules
source slurm/env_hf.sh
setup_hf_offline
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"
mkdir -p slurm/logs

BASE=data/kazculture_pipeline
python scripts/eval_kazculture_holdout_enriched_qa.py \
  --test_file data/raw/kazculture_phase1/test.parquet \
  --baseline_predictions data/outputs/kazculture_phase1_v2/predictions.jsonl \
  --index_v21 "$BASE/index/heldout_diagnostic/v21_diagnostic" \
  --index_v22 "$BASE/index/heldout_diagnostic/v22_diagnostic" \
  --out "$BASE/evaluation/heldout_source_first_qwen" \
  --model_revision "${KAZCULTURE_MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}" \
  --resume
