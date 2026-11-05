#!/bin/bash
#SBATCH --job-name=kz_final_full_c
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_final_full_c_%j.out
#SBATCH --error=slurm/logs/kz_final_full_c_%j.err
#
# Stage C (Qwen3-32B, full setting; secondary family): S1 reproduction
# (1221/1334) then S4R, S4R-noabs, S1+L. S0 comes from stage A (gate needs it:
# submit with --dependency=afterok:<stage A job id>).
# Estimate: load ~12 min + 4 x ~22 min = ~100 min.
# Resumable: if the 165-min guard stops it, resubmit the same script.
# Submit from the repo root (PROJECT_DIR defaults to the submit directory).
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
kz_run_final full "S1,S4R,S4R-noabs,S1+L"
