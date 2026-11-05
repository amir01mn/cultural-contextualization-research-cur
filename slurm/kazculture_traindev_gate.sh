#!/bin/bash
#SBATCH --job-name=kz_traindev_gate
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --exclude=ng10706
#SBATCH --output=slurm/logs/kz_traindev_gate_%j.out
#SBATCH --error=slurm/logs/kz_traindev_gate_%j.err
#
# Train-dev go/no-go gate (preregistration §4): 400 seeded train questions
# (S1-tuning rows excluded), full index with own source left out, mean gold
# log-likelihood under S1 / S4R / S4R-noabs / S1+L. Writes gate_decision.json.
# Optional: KZ_GATE_EXCLUDE_IDS=<json list of train_row_ids used for routing tuning>.
# Estimate: load ~12 min + 1,600 forward passes (~40 min).
# Resumable: if the 165-min guard stops it, resubmit the same script.
# Submit from the repo root (PROJECT_DIR defaults to the submit directory).
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env gpu
kz_require "$KZ_DEV" "$KZ_CONFIG" "$KZ_INDEX_FULL/build_manifest.json"
kz_check_adapter
mapfile -t ADAPTER_ARGS < <(kz_adapter_args)
EXTRA=()
[[ -n "${KZ_GATE_EXCLUDE_IDS:-}" ]] && EXTRA+=(--exclude_ids_file "$KZ_GATE_EXCLUDE_IDS")
python scripts/eval_kazculture_traindev_gate.py \
  --dev "$KZ_DEV" --index "$KZ_INDEX_FULL" --s1_config "$KZ_CONFIG" \
  --out "$KZ_GATE_OUT" --n 400 --seed 0 --exclude_first_n 400 \
  --conditions "S1,S4R,S4R-noabs,S1+L" \
  --max_minutes "${KZ_MAX_MINUTES:-165}" \
  "${ADAPTER_ARGS[@]}" "${EXTRA[@]}"
