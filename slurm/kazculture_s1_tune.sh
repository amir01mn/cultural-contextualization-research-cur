#!/bin/bash
#SBATCH --job-name=kz_s1_tune
#SBATCH --account=def-enaskt_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/kz_s1_tune_%j.out
#SBATCH --error=slurm/logs/kz_s1_tune_%j.err

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

OUT=data/kazculture_pipeline/s1_full_dev_thresholds_v3.json
[[ ! -e "$OUT" ]] || {
  echo "Frozen config already exists: $OUT" >&2
  exit 2
}

python scripts/tune_kazculture_retrieval.py \
  --index data/kazculture_pipeline/index/s1_full \
  --dev data/kazculture_pipeline/corpus/train_dev_qa.jsonl \
  --max_queries 400 \
  --target_recall 0.70 \
  --batch_size 32 \
  --out "$OUT"

# The original frozen config is not accessible; the regenerated one must equal
# the documented config (top_k 8, min_sim 0.45, 5,000 chars, recall 285/400).
# Exit 3 (job fails) on any mismatch; the check JSON sits next to the config.
python scripts/check_kazculture_s1_config.py "$OUT" \
  --report "${OUT%.json}_documented_check.json"
