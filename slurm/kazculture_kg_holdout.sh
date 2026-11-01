#!/bin/bash
#SBATCH --job-name=kzkg_hout
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kzkg_hout_%j.out
#SBATCH --error=slurm/logs/kzkg_hout_%j.err
#
# Held-out 150 source passages at offsets [150:300). Use same Qwen3 32B
# weights and TP settings for both protocols; different versioned outputs.
# Set KG_HOLDOUT_VERSION=v21 or v22 at sbatch time. No test QA leakage.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
source slurm/load_modules.sh
load_gpu_modules
source slurm/env_hf.sh
setup_hf_offline
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"
BASE=data/kazculture_pipeline
VERSION="${KG_HOLDOUT_VERSION:?Must set KG_HOLDOUT_VERSION=v21 or v22}"
case "$VERSION" in
  v21) SCRIPT=scripts/extract_kazculture_kg_v21.py
       DEST="$BASE/kg_v21_holdout";;
  v22) SCRIPT=scripts/extract_kazculture_kg_v22.py
       DEST="$BASE/kg_v22_holdout";;
  *) echo "Invalid KG_HOLDOUT_VERSION=$VERSION; choose v21 or v22" >&2; exit 2;;
esac
mkdir -p slurm/logs "$DEST"
python "$SCRIPT" \
  --passages "$BASE/corpus/pilot_passages.jsonl" \
  --start 150 --limit 150 \
  --out "$DEST/heldout_150_frames.jsonl" \
  --audit_csv "$DEST/audit_heldout_150.csv" \
  --audit_count 60 --max_tokens 1536 --batch_size 8 \
  --tp "${VLLM_TP:-2}" \
  --model_revision "${KAZCULTURE_MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}"
