#!/bin/bash
#SBATCH --job-name=kazculture_kg
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kazculture_kg_%j.out
#SBATCH --error=slurm/logs/kazculture_kg_%j.err
#
# KG_STAGE=pilot (750 unique passages) or full (all unique training passages).
# Both are checkpointed and use distinct paths. Never extract from test.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
source slurm/load_modules.sh
load_gpu_modules
source slurm/env_hf.sh
setup_hf_offline
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"
STAGE="${KG_STAGE:-pilot}"
mkdir -p slurm/logs data/kazculture_pipeline/kg
case "$STAGE" in
pilot)
  INPUT="data/kazculture_pipeline/corpus/pilot_passages.jsonl"
  OUT="data/kazculture_pipeline/kg/pilot_frames.jsonl"
  AUDIT=(--audit_csv data/kazculture_pipeline/kg/audit_pilot.csv --audit_count 60)
  ;;
full)
  INPUT="data/kazculture_pipeline/corpus/passages.jsonl"
  OUT="data/kazculture_pipeline/kg/full_frames.jsonl"
  # Human pilot audit gate required BEFORE full extraction.
  python -c 'from pathlib import Path; from src.construction.kazculture_kg import validate_audit; print(validate_audit(Path("data/kazculture_pipeline/kg/audit_pilot.csv")))'
  AUDIT=()
  ;;
*) echo "KG_STAGE must be pilot or full" >&2; exit 2;;
esac
[[ -f "$INPUT" ]] || { echo "Missing training passage file $INPUT" >&2; exit 2; }
python scripts/extract_kazculture_kg.py \
 --passages "$INPUT" \
 --out "$OUT" \
 --model "Qwen/Qwen3-32B" \
 --model_revision "${KAZCULTURE_MODEL_REVISION:-9216db5781bf21249d130ec9da846c4624c16137}" \
 --tp "${VLLM_TP:-2}" \
 --batch_size "${KG_BATCH_SIZE:-24}" \
 "${AUDIT[@]}"
