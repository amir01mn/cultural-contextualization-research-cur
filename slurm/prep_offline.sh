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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_modules.sh
source "${SCRIPT_DIR}/load_modules.sh"
load_login_modules

PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck source=env_hf.sh
source "${SCRIPT_DIR}/env_hf.sh"
setup_hf_cache
log_hf_env

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
