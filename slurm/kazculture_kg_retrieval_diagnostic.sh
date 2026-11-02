#!/bin/bash
#SBATCH --job-name=kzkg_ret_diag
#SBATCH --account=def-enaskt_cpu
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/kzkg_ret_diag_%j.out
#SBATCH --error=slurm/logs/kzkg_ret_diag_%j.err
#
# Diagnostic-only experiment, uses no GPU or Qwen3 QA answers.
# KG_DIAG_STAGE=v21_index | v22_index | probe | enrichment
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$PROJECT_DIR"
source slurm/load_modules.sh
load_login_modules
source slurm/env_hf.sh
setup_hf_offline
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
BASE=data/kazculture_pipeline
INPUT="$BASE/heldout/heldout_train_passages_150.jsonl"
I21="$BASE/index/heldout_diagnostic/v21_diagnostic"
I22="$BASE/index/heldout_diagnostic/v22_diagnostic"
mkdir -p slurm/logs "$BASE/retrieval_diagnostic"

STAGE="${KG_DIAG_STAGE:?Set KG_DIAG_STAGE=v21_index, v22_index, or probe}"
case "$STAGE" in
  v21_index)
    python scripts/build_kazculture_holdout_index.py \
      --passages "$INPUT" \
      --frames "$BASE/kg_v21_holdout/heldout_150_frames.jsonl" \
      --method v21 --out "$I21" --device cpu --batch_size 16
    ;;
  v22_index)
    python scripts/build_kazculture_holdout_index.py \
      --passages "$INPUT" \
      --frames "$BASE/kg_v22_holdout/heldout_150_frames.jsonl" \
      --method v22 --out "$I22" --device cpu --batch_size 16
    ;;
  probe)
    [[ -f "$I21/diagnostic_manifest.json" && -f "$I22/diagnostic_manifest.json" ]] || {
      echo "Both indexes must complete before probing." >&2; exit 2;
    }
    python scripts/probe_kazculture_holdout_retrieval.py \
      --index_v21 "$I21" --index_v22 "$I22" \
      --train_dev_qa "$BASE/corpus/train_dev_qa.jsonl" \
      --out "$BASE/retrieval_diagnostic/heldout_v21_vs_v22_source_recall.json" \
      --max_questions 300 --per_passage 3 --device cpu --batch_size 16
    ;;
  enrichment)
    [[ -f "$I21/diagnostic_manifest.json" && -f "$I22/diagnostic_manifest.json" ]] || {
      echo "Both indexes must complete before source-first enrichment." >&2; exit 2;
    }
    python scripts/probe_kazculture_source_enrichment.py \
      --index_v21 "$I21" --index_v22 "$I22" \
      --train_dev_qa "$BASE/corpus/train_dev_qa.jsonl" \
      --out "$BASE/retrieval_diagnostic/heldout_source_first_enrichment.json" \
      --max_questions 300 --per_passage 3 --device cpu --batch_size 16
    ;;
  *)
    echo "Invalid KG_DIAG_STAGE=$STAGE (expected v21_index, v22_index, probe, or enrichment)" >&2
    exit 2
    ;;
esac
