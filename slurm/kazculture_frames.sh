#!/bin/bash
#SBATCH --job-name=kz_frames_v3k
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm/logs/kz_frames_v3k_%A_%a.out
#SBATCH --error=slurm/logs/kz_frames_v3k_%A_%a.err
#
# KazCulture frames v3k, label stage (docs/KAZCULTURE_FRAMES_V3K.md): Qwen3-32B
# @9216db57 (bf16, non-thinking) with vLLM tensor parallel 2 on 2x A100-40GB.
# One array task per round-robin shard; each task appends to its own per-passage
# checkpoint, so a resubmitted task resumes. N_SHARDS must match --array.
#
#   sbatch slurm/kazculture_frames.sh                       # 4 shards (0-3)
#   sbatch --array=2 slurm/kazculture_frames.sh             # rerun shard 2 only
#   N_SHARDS=8 sbatch --array=0-7 slurm/kazculture_frames.sh
# Then: sbatch slurm/kazculture_frames_index.sh  (merge + clustering + index)
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

N_SHARDS="${N_SHARDS:-4}"
SHARD="${SLURM_ARRAY_TASK_ID:?run as an array job}"
OUT="${FRAMES_OUT:-data/kazculture_pipeline/frames_v3k}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "===== frames v3k label shard ${SHARD}/${N_SHARDS}: $(date) ====="
python scripts/build_kazculture_frames.py --stages label --shard "${SHARD}/${N_SHARDS}" \
  --out "$OUT" \
  --model Qwen/Qwen3-32B --revision 9216db5781bf21249d130ec9da846c4624c16137 \
  --tensor_parallel_size 2 --max_model_len "${MAX_MODEL_LEN:-6144}" \
  --gpu_memory_utilization "${GPU_MEM_UTIL:-0.92}"
touch "$OUT/labels/DONE_${SHARD}_of_${N_SHARDS}"
echo "===== done: $(date) ====="
