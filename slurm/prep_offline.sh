#!/bin/bash
# Run this on the Narval LOGIN node (it has internet). NOT an sbatch job:
# compute nodes have no internet, so staging must happen here.
#
#   bash slurm/prep_offline.sh
#
# It loads the Python module, activates the project venv, points HF_HOME at
# $SCRATCH, and pre-downloads CultureBank + CulturalBench (and optionally the
# model). After this, GPU jobs can run fully offline.

set -euo pipefail

module load StdEnv/2023 gcc/12.3 python/3.11 arrow/17.0.0 || true

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# Persistent caches live on $SCRATCH (large, not backed up but big quota).
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
mkdir -p "$HF_HOME"

# Activate the project virtualenv (see docs/NARVAL_SETUP.md to create it).
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"

echo "HF_HOME=$HF_HOME"
echo "Staging datasets (and model if VLLM_MODEL set)..."

# Pass --model to also pre-download model weights for offline vLLM.
MODEL_ARG=""
if [[ -n "${VLLM_MODEL:-}" ]]; then
  MODEL_ARG="--model ${VLLM_MODEL}"
fi

python scripts/prep_offline_data.py ${MODEL_ARG}

echo "Done. Submit GPU jobs with: sbatch slurm/build_corpus.sh && sbatch slurm/eval_culturalbench.sh"
