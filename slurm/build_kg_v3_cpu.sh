#!/bin/bash
#SBATCH --job-name=ckg_v3_kg_cpu
#SBATCH --account=def-enaskt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/v3_build_cpu_%j.out
#SBATCH --error=slurm/logs/v3_build_cpu_%j.err
#
# KG v3 stages that need no LLM (labels are cached in data/kg_v3/frame_labels.jsonl):
# norm types + profiles + indexes, local pathways, dev check, audit samples.
# Touches data/kg_v3/READY on success.
#
#   sbatch slurm/build_kg_v3_cpu.sh

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_login_modules
cd "$PROJECT_DIR"
source "${PROJECT_DIR}/slurm/env_hf.sh"
setup_hf_offline
export EMBEDDING_BACKEND=sentence-transformers
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
rm -rf data/kg_v3/full/frames.jsonl   # guard directory left to stop the GPU job's stale-parser KG
echo "===== KG v3 CPU stages: $(date) ====="
.venv/bin/python scripts/build_kg_v3.py --stages "${STAGES:-kg,local,check,audit}"
RC=$?
[[ $RC -eq 0 ]] && touch data/kg_v3/READY
echo "===== rc=$RC: $(date) ====="
exit $RC
