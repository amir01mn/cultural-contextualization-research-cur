#!/bin/bash
#SBATCH --job-name=kazculture_index
#SBATCH --account=def-enaskt_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kazculture_index_%j.out
#SBATCH --error=slurm/logs/kazculture_index_%j.err
#
# KG_INDEX_MODE=pilot|full|strict|combined|combined_strict
# Requires audited pilot CSV. If combined, set KAZCULTURE_CB_FRAMES to
# accessible CultureBank v3/full/frames.jsonl (no automatic source substitution).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
source slurm/load_modules.sh
load_login_modules
source slurm/env_hf.sh
setup_hf_offline
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
MODE="${KG_INDEX_MODE:-pilot}"
BASE="data/kazculture_pipeline"
mkdir -p slurm/logs "$BASE/index"
ARGS=(--audit_csv "$BASE/kg/audit_pilot.csv" --device cpu --batch_size 16)
case "$MODE" in
pilot)
  ARGS+=(--passages "$BASE/corpus/pilot_passages.jsonl"
         --frames "$BASE/kg/pilot_frames.jsonl")
  ;;
full|strict|combined|combined_strict)
  ARGS+=(--passages "$BASE/corpus/passages.jsonl"
         --frames "$BASE/kg/full_frames.jsonl")
  ;;
*) echo "Unexpected KG_INDEX_MODE=$MODE" >&2; exit 2;;
esac
if [[ "$MODE" == *strict* ]]; then
  ARGS+=(--strict_passage_disjoint --overlap_file "$BASE/corpus/test_overlap.json")
fi
if [[ "$MODE" == *combined* ]]; then
  : "${KAZCULTURE_CB_FRAMES:?Set explicitly to available CultureBank frame KG path}"
  ARGS+=(--culturebank_frames "$KAZCULTURE_CB_FRAMES")
fi
ARGS+=(--out "$BASE/index/$MODE")
python scripts/build_kazculture_index.py "${ARGS[@]}"
