#!/bin/bash
#SBATCH --job-name=kz_s1_index
#SBATCH --account=def-enaskt_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_s1_index_%j.out
#SBATCH --error=slurm/logs/kz_s1_index_%j.err
#
# KAZCULTURE_S1_INDEX_MODE=full|strict
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
source slurm/load_modules.sh
load_login_modules
source slurm/env_hf.sh
setup_hf_offline
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

BASE=data/kazculture_pipeline
MODE="${KAZCULTURE_S1_INDEX_MODE:?Set KAZCULTURE_S1_INDEX_MODE=full or strict}"
mkdir -p slurm/logs "$BASE/index"
ARGS=(
  --passages "$BASE/corpus/passages.jsonl"
  --out "$BASE/index/s1_$MODE"
  --device cpu
  --batch_size 16
)
case "$MODE" in
  full) ;;
  strict)
    ARGS+=(--strict_passage_disjoint --overlap_file "$BASE/corpus/test_overlap.json")
    ;;
  *) echo "Invalid KAZCULTURE_S1_INDEX_MODE=$MODE" >&2; exit 2;;
esac
python scripts/build_kazculture_s1_index.py "${ARGS[@]}"
