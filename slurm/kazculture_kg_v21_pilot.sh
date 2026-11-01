#!/bin/bash
#SBATCH --job-name=kazkg_v21
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kazkg_v21_%j.out
#SBATCH --error=slurm/logs/kazkg_v21_%j.err
#
# Matching the first 150 V2 source passages. Uses a new v21 output path.
# NO train/test question or answer content enters extraction.
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
mkdir -p slurm/logs "$BASE/kg_v21"
[[ -f "$BASE/corpus/pilot_passages.jsonl" ]] || {
  echo "Missing training-only pilot source file" >&2; exit 2;
}
python scripts/extract_kazculture_kg_v21.py \
  --passages "$BASE/corpus/pilot_passages.jsonl" \
  --start 0 \
  --limit 150 \
  --out "$BASE/kg_v21/pilot_150_frames.jsonl" \
  --audit_csv "$BASE/kg_v21/audit_v21_pilot_150.csv" \
  --audit_count 60 \
  --max_tokens 1536 \
  --batch_size 8 \
  --tp "${VLLM_TP:-2}" \
  --model_revision "${KAZCULTURE_MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}"
