#!/bin/bash
# Build the v2 dev MCQ from held-out train questions (Qwen3-32B via vLLM).
#SBATCH --job-name=kz_dev_mcq
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/kz_dev_mcq_%j.out
#SBATCH --error=slurm/logs/kz_dev_mcq_%j.err
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python scripts/build_kazculture_dev_mcq.py --tensor_parallel_size "${SLURM_GPUS_ON_NODE:-2}"
