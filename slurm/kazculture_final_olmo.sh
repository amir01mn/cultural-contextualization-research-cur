#!/bin/bash
#SBATCH --job-name=kz_final_olmo
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_final_olmo_%j.out
#SBATCH --error=slurm/logs/kz_final_olmo_%j.err
#
# OLMo-2-1124-7B-Instruct replication (criterion 5): S0, S1, S4R strict only.
# 4,096-token cap; evidence truncated by whole blocks, counted in *_summary.json.
# No reproduction references exist for OLMo. Estimate: ~5 min load + 3 x ~10 min.
# Resumable: if the 165-min guard stops it, resubmit the same script.
# Submit from the repo root (PROJECT_DIR defaults to the submit directory).
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
kz_require "$KZ_TEST" "$KZ_CONFIG" "$KZ_INDEX_STRICT/build_manifest.json"
kz_check_adapter
mapfile -t ADAPTER_ARGS < <(kz_adapter_args)
python scripts/eval_kazculture_final.py \
  --model_family olmo2 \
  --test_file "$KZ_TEST" --s1_config "$KZ_CONFIG" \
  --index_strict "$KZ_INDEX_STRICT" \
  --settings strict --conditions "S0,S1,S4R" \
  --out "$KZ_OLMO_OUT" \
  --max_minutes "${KZ_MAX_MINUTES:-165}" \
  "${ADAPTER_ARGS[@]}"
