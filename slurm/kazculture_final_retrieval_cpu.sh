#!/bin/bash
#SBATCH --job-name=kz_final_retr
#SBATCH --account=def-enaskt_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_final_retr_%j.out
#SBATCH --error=slurm/logs/kz_final_retr_%j.err
#
# CPU-only: fill the final evaluator's S1 retrieval cache (BGE-M3, pinned) for
# strict and full and write retrieval_report_<setting>.json with the number of
# test questions whose evidence the frozen first-overflow break truncates.
# KZ_RETR_CONDITIONS=S1 (default; no frame adapter needed). Add S4R,... later
# to pre-cache adapter output. Optional: the GPU stages compute the same
# retrieval themselves; S1 rows are still checked against the frozen run.
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env cpu
kz_require "$KZ_TEST" "$KZ_CONFIG" "$KZ_INDEX_STRICT/build_manifest.json" "$KZ_INDEX_FULL/build_manifest.json"
mapfile -t ADAPTER_ARGS < <(kz_adapter_args)
python scripts/eval_kazculture_final.py --retrieval_only \
  --test_file "$KZ_TEST" --s1_config "$KZ_CONFIG" \
  --index_strict "$KZ_INDEX_STRICT" --index_full "$KZ_INDEX_FULL" \
  --settings strict,full --conditions "${KZ_RETR_CONDITIONS:-S1}" \
  --out "$KZ_FINAL_OUT" "${ADAPTER_ARGS[@]}"
