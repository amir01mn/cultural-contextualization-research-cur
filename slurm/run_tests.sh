#!/bin/bash
#SBATCH --job-name=ckg_tests
#SBATCH --account=def-enaskt
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/logs/tests_%j.out
#SBATCH --error=slurm/logs/tests_%j.err
#
# CPU-only test + eval-path smoke run.
#
# The login node is shared with 150+ users and its Lustre client stalls: a suite
# that completes in ~40s can sit for 45+ minutes at 0.3% CPU in cl_sync_io_wait.
# Running on a compute node gives a trustworthy pass/fail signal.
#
#   sbatch slurm/run_tests.sh
#   SMOKE_EVAL=1 sbatch slurm/run_tests.sh   # also exercise the eval path

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
# shellcheck source=load_modules.sh
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_login_modules

cd "$PROJECT_DIR"
mkdir -p slurm/logs

# shellcheck source=env_hf.sh
source "${PROJECT_DIR}/slurm/env_hf.sh"
setup_hf_offline

source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"

# TF-IDF keeps the suite deterministic and avoids a multi-minute MiniLM load.
# Set CONFTEST_SELFCONFIG=1 to leave it unset instead, which checks that
# tests/conftest.py supplies the offline defaults on its own.
if [[ "${CONFTEST_SELFCONFIG:-0}" == "1" ]]; then
  unset EMBEDDING_BACKEND HF_HUB_OFFLINE
  echo "CONFTEST_SELFCONFIG=1: relying on tests/conftest.py for offline defaults"
else
  export EMBEDDING_BACKEND="${EMBEDDING_BACKEND:-tfidf}"
fi

echo "===== PYTEST ====="
python -m pytest -q -p no:cacheprovider
PYTEST_RC=$?
echo "pytest exit=$PYTEST_RC"

if [[ "${SMOKE_EVAL:-0}" == "1" ]]; then
  echo "===== EVAL SMOKE (stub backend, pilot corpus) ====="
  python "${SMOKE_EVAL_SCRIPT:?set SMOKE_EVAL_SCRIPT}" || echo "smoke eval failed"
fi

exit $PYTEST_RC
