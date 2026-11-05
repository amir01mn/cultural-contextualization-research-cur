#!/bin/bash
#SBATCH --job-name=kz_final_strict_a
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_final_strict_a_%j.out
#SBATCH --error=slurm/logs/kz_final_strict_a_%j.err
#
# Stage A (Qwen3-32B, strict): S0 and S1 reproduction (exact match vs the frozen files, or
# the documented 552/1334 and 831/1334, else exit 3), then primary S4R and S4R-noabs.
# Estimate: load ~12 min + S0 ~10 + 3 x ~20-25 min = ~95 min.
# Resumable: if the 165-min guard stops it, resubmit the same script.
# Submit from the repo root (PROJECT_DIR defaults to the submit directory).
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
kz_run_final strict "S0,S1,S4R,S4R-noabs"
