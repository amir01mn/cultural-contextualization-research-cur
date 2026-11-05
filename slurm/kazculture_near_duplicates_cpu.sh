#!/bin/bash
#SBATCH --job-name=kz_neardup
#SBATCH --account=def-enaskt_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kz_neardup_%j.out
#SBATCH --error=slurm/logs/kz_neardup_%j.err
#
# Near-duplicate audit: test source passages vs the strict index (max BGE-M3
# cosine, char-5-gram containment). Writes near_dedup_list.json for the
# optional strict-near-dedup subset in analyze_kazculture_final.py.
source "${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-.}}/slurm/kazculture_final_common.sh"
kz_setup_env cpu
kz_require "$KZ_TEST" "$KZ_INDEX_STRICT/metadata.json"
python scripts/audit_kazculture_near_duplicates.py \
  --test_file "$KZ_TEST" --index_strict "$KZ_INDEX_STRICT" \
  --out "$KZ_NEARDUP_OUT" --embeddings_cache "$KZ_NEARDUP_OUT/test_passage_bge_m3.npy"
