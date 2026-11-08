#!/bin/bash
# KazCulture v2 OLMo-2 replication: strict S1-OG (S0/S1 copied from final_v1b_olmo).
#SBATCH --job-name=kz_final_v2_olmo
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_final_v2_olmo_%j.out
#SBATCH --error=slurm/logs/kz_final_v2_olmo_%j.err
export KZ_OLMO_OUT="${KZ_OLMO_OUT:-data/kazculture_pipeline/evaluation/final_v2_olmo}"
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
kz_require "$KZ_TEST" "$KZ_CONFIG" "$KZ_INDEX_STRICT/build_manifest.json"
mapfile -t ADAPTER_ARGS < <(kz_adapter_args)
python scripts/eval_kazculture_final.py \
  --model_family olmo2 \
  --test_file "$KZ_TEST" --s1_config "$KZ_CONFIG" \
  --index_strict "$KZ_INDEX_STRICT" \
  --settings strict --conditions "S1-OG" \
  --out "$KZ_OLMO_OUT" \
  --max_minutes "${KZ_MAX_MINUTES:-165}" \
  "${ADAPTER_ARGS[@]}"
