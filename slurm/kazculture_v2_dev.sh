#!/bin/bash
# KazCulture v2 development run on held-out train MCQ (never the test set).
#   KZ_DEV_OUT=data/kazculture_pipeline/evaluation/v2_dev/a1 KZ_DEV_CONDITIONS=S0,S1,S1+L \
#     slurm/submit_kz.sh slurm/kazculture_v2_dev.sh
#SBATCH --job-name=kz_v2_dev
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_v2_dev_%j.out
#SBATCH --error=slurm/logs/kz_v2_dev_%j.err
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
: "${KZ_DEV_OUT:?set KZ_DEV_OUT}"
python scripts/eval_kazculture_v2_dev.py --out "$KZ_DEV_OUT" \
  --conditions "${KZ_DEV_CONDITIONS:-S0,S1,S1+L,S4R,S4R-PR}" \
  --n "${KZ_DEV_N:-1000}" --seed "${KZ_DEV_SEED:-1}" \
  --max_minutes "${KZ_MAX_MINUTES:-165}" ${KZ_DEV_EXTRA:-}
