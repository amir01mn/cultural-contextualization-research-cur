#!/bin/bash
#SBATCH --job-name=ckg_eval
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
# Max on gpubase_bygpu_b1 for this account/partition.
#SBATCH --output=slurm/logs/eval_%j.out
#SBATCH --error=slurm/logs/eval_%j.err
#
# Evaluate systems S0-S4 on CulturalBench with the vLLM backend.
# Requires the retrieval corpus (sbatch slurm/build_corpus.sh) and staged data.
#
#   sbatch slurm/eval_culturalbench.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
# shellcheck source=load_modules.sh
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_gpu_modules

cd "$PROJECT_DIR"
mkdir -p slurm/logs

# shellcheck source=env_hf.sh
source "${PROJECT_DIR}/slurm/env_hf.sh"
setup_hf_offline
log_hf_env

source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"

export LLM_PROVIDER=vllm
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
# Full CulturalBench by default: 1,227 Easy questions + 1,227 Hard groups
# (4,908 True/False items) = 2,454 scored units. The previous 250/variant run
# gave ~+/-4% bootstrap CIs, too wide to resolve a genuine 3-4 point difference.
# Batched vLLM answers ~500 items per system in ~20s, so the full set costs
# roughly 30 minutes of inference.
LIMIT="${LIMIT:-100000}"
TOP_K="${TOP_K:-8}"
SEED="${SEED:-42}"

# Includes the two S4 ablations, which isolate pathway *structure* from refined
# *content*; both exist in mcq_systems.ALL_SYSTEMS but were never run.
SYSTEMS="${SYSTEMS:-S0_no_kg,S1_raw_text_rag,S2_flat_kg,S3_layered_kg,S4_abstraction_refinement_kg,S4_triple_retrieval,S4_no_shared_pivot}"

echo "Evaluating on CulturalBench with vLLM model: $VLLM_MODEL"
python scripts/eval_culturalbench.py \
  --provider vllm \
  --model "$VLLM_MODEL" \
  --variants easy,hard \
  --limit "$LIMIT" \
  --top_k "$TOP_K" \
  --seed "$SEED" \
  --systems "$SYSTEMS"

echo "Eval job complete. See data/outputs/culturalbench_summary.md"
