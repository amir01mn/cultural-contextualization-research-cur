#!/bin/bash
#SBATCH --job-name=kz_frames_v3k_kg
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_frames_v3k_kg_%j.out
#SBATCH --error=slurm/logs/kz_frames_v3k_kg_%j.err
#
# KazCulture frames v3k, merge + kg stages: merge label shards, cluster norm
# types over KazCulture + CultureBank v3 frames, culture profiles, and BGE-M3
# (@5617a9f6) frame / norm-type embeddings for full/ and strict/. One GPU only
# for BGE-M3 speed (no LLM); the same command runs on CPU with
# EMBEDDING_DEVICE=cpu, just slowly (~30k long documents).
#
#   sbatch slurm/kazculture_frames_index.sh
#   sbatch --dependency=afterok:<label array job id> slurm/kazculture_frames_index.sh
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

OUT="${FRAMES_OUT:-data/kazculture_pipeline/frames_v3k}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export EMBEDDING_BACKEND=bge-m3
export EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-data/kazculture_pipeline/embedding_cache}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-16}"

echo "===== frames v3k merge+kg: $(date) ====="
python scripts/build_kazculture_frames.py --stages merge,kg --out "$OUT" \
  --n_shards "${N_SHARDS:-4}" --embedding_backend bge-m3
touch "$OUT/READY"
echo "===== done: $(date) ====="
