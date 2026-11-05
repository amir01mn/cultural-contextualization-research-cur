#!/bin/bash
#SBATCH --job-name=kz_frames_v3k_tune
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_frames_v3k_tune_%j.out
#SBATCH --error=slurm/logs/kz_frames_v3k_tune_%j.err
#
# Train-only S4R routing threshold tuning (scripts/tune_kazculture_routing.py):
# 400 seeded train questions (disjoint from the S1 tuner's 400), leave-source-out,
# objective = mean gold log-likelihood under Qwen3-32B via score_gold_loglik.
# BGE-M3 query embeddings run on CPU first, then the scorer loads on both GPUs.
# Needs frames_v3k/{full,...} from slurm/kazculture_frames_index.sh.
#
#   sbatch slurm/kazculture_frames_tune.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs
source slurm/load_modules.sh
load_gpu_modules
source slurm/env_hf.sh
setup_hf_offline
log_hf_env
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

export TOKENIZERS_PARALLELISM=false
export EMBEDDING_BACKEND=bge-m3
export EMBEDDING_DEVICE=cpu
export EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-data/kazculture_pipeline/embedding_cache}"
OUT="${FRAMES_OUT:-data/kazculture_pipeline/frames_v3k}"
test -f "$OUT/READY" || { echo "frames not built: $OUT/READY missing" >&2; exit 2; }

python scripts/tune_kazculture_routing.py --frames_dir "$OUT" --embedding_backend bge-m3
