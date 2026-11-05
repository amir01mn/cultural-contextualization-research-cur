#!/bin/bash
#SBATCH --job-name=kz_final_strict_b
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_final_strict_b_%j.out
#SBATCH --error=slurm/logs/kz_final_strict_b_%j.err
#
# Stage B (Qwen3-32B, strict): S1+L, CTRL-LEN, CTRL-SHUF. Needs stage A's
# reproduction files (gate); submit with --dependency=afterok:<stage A job id>.
# Estimate: load ~12 min + ~25 + ~35 (longer prompts) + ~25 min = ~100 min.
# Resumable: if the 165-min guard stops it, resubmit the same script.
# Submit from the repo root (PROJECT_DIR defaults to the submit directory).
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
kz_run_final strict "S1+L,CTRL-LEN,CTRL-SHUF"
