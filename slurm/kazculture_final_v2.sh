#!/bin/bash
# KazCulture v2 test run (docs/KAZCULTURE_PREREGISTRATION_V2.md), Qwen3-32B.
#   KZ_V2_SETTINGS=strict KZ_V2_CONDITIONS=S1-OG,S1-OGpo,S4R-OGF,S1-OG+L \
#     slurm/submit_kz.sh slurm/kazculture_final_v2.sh
# S0/S1 + reproduction files are copied from final_v1b into KZ_FINAL_OUT first
# (scripts/prepare_kazculture_final_v2.sh). Resumable: resubmit on the time guard.
#SBATCH --job-name=kz_final_v2
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_final_v2_%j.out
#SBATCH --error=slurm/logs/kz_final_v2_%j.err
export KZ_FINAL_OUT="${KZ_FINAL_OUT:-data/kazculture_pipeline/evaluation/final_v2}"
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
: "${KZ_V2_SETTINGS:?set KZ_V2_SETTINGS}" "${KZ_V2_CONDITIONS:?set KZ_V2_CONDITIONS}"
kz_run_final "$KZ_V2_SETTINGS" "$KZ_V2_CONDITIONS"
