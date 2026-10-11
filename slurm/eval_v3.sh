#!/bin/bash
#SBATCH --job-name=ckg_v3_eval
#SBATCH --account=def-enaskt_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/v3_eval_%x_%j.out
#SBATCH --error=slurm/logs/v3_eval_%x_%j.err
#
# Eval v3 (docs/EVAL_V3_PLAN.md). EVAL selects the evaluation:
#   culturebank | culturebank_hard | normad | judge | olmo
# Waits (warming imports) for the KG v3 build's READY flag. Writes only under
# data/outputs/eval_v3/.
#
#   sbatch --job-name=v3_cb --export=ALL,EVAL=culturebank slurm/eval_v3.sh

set -uo pipefail
EVAL="${EVAL:?set EVAL}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
source "${PROJECT_DIR}/slurm/load_modules.sh"
load_gpu_modules
# vLLM cannot parse a MIG UUID in CUDA_VISIBLE_DEVICES; the job's cgroup exposes
# only its own slice, so index 0 addresses it.
if [[ "${CUDA_VISIBLE_DEVICES:-}" == MIG-* ]]; then export CUDA_VISIBLE_DEVICES=0; fi
cd "$PROJECT_DIR"
mkdir -p slurm/logs data/outputs/eval_v3 data/eval_v3

LOCK_DIR="data/eval_v3/LOCK_${EVAL}"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "Another job owns $LOCK_DIR ($(cat "$LOCK_DIR/job" 2>/dev/null)); exiting."
  exit 0
fi
echo "${SLURM_JOB_ID:-local}" > "$LOCK_DIR/job"

source "${PROJECT_DIR}/slurm/env_hf.sh"
setup_hf_offline
log_hf_env
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"
export LLM_PROVIDER=vllm
export EMBEDDING_BACKEND=sentence-transformers

READY_FLAG="data/kg_v3/READY"
[[ "$EVAL" == "culturebank_hard" ]] && READY_FLAG="data/eval_v3/culturebank_hard/READY"
if [[ ! -f "$READY_FLAG" ]]; then
  echo "Waiting for $READY_FLAG (warming imports) $(date)"
  python -c "import torch, vllm, transformers, sentence_transformers; print('imports warm')" &
  for ((i=0; i<${WAIT_READY_MIN:-90}; i++)); do
    [[ -f "$READY_FLAG" ]] && break
    sleep 60
  done
  wait
  if [[ ! -f "$READY_FLAG" ]]; then
    echo "READY flag never appeared; releasing lock and exiting" >&2
    rm -rf "$LOCK_DIR"; exit 3
  fi
fi

V3_SYS="S0,S1,S1g,S2,S3,S3g,S4_triple_retrieval,S4,S4L,S4F,S4R,S4R_noabs"
V3_OPEN="S0,S1,S3g,S4,S4L,S4R"
NORMAD_V3="C1_country,C2_value,C3_rule_of_thumb,S1_raw_text_rag,S2_flat_kg,S3_layered_kg,S4_abstraction_refinement_kg,S4_triple_retrieval,S1g,S3g,S4L,S4F,S4R,S4R_noabs"
COMMON=(--kg_v3 data/kg_v3 --drop_eval_leak_triples)
RC=0
echo "===== EVAL v3 $EVAL: $(date) ====="
case "$EVAL" in
  culturebank)
    export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    python scripts/run_culturebank_eval_v2.py --provider vllm "${COMMON[@]}" \
      --stages paraphrase,open_gen,mcq,heldout,retrieval_metrics \
      --systems "$V3_SYS" --open_systems "$V3_OPEN" \
      --out_dir data/outputs/eval_v3/culturebank \
      --para_cache data/eval_v2/culturebank/mcq_paraphrased.jsonl \
      --title "Eval v3 - CultureBank grounded QA (Qwen2.5-7B)" || RC=$?
    ;;
  culturebank_hard)
    export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    python scripts/run_culturebank_eval_v2.py --provider vllm "${COMMON[@]}" \
      --stages paraphrase,mcq,retrieval_metrics --systems "$V3_SYS" \
      --data_dir data/eval_v3/culturebank_hard \
      --out_dir data/outputs/eval_v3/culturebank_hard \
      --title "Eval v3 - CultureBank hard-distractor MCQ (Qwen2.5-7B)" || RC=$?
    ;;
  normad)
    export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    python scripts/run_normad_v2.py --provider vllm "${COMMON[@]}" --conditions "$NORMAD_V3" \
      --out_dir data/outputs/eval_v3/normad --title "Eval v3 - NormAd (Qwen2.5-7B)" || RC=$?
    ;;
  judge)
    export VLLM_MODEL=microsoft/phi-4 VLLM_MAX_MODEL_LEN=4096
    python scripts/run_intrinsic_v2.py --provider vllm --model microsoft/phi-4 \
      --stages pathway_audit,open_judge --data_dir data/kg_v3 \
      --out_dir data/outputs/eval_v3/intrinsic \
      --pathway_files frames=data/kg_v3/audit_frames.jsonl,local=data/kg_v3/audit_local_pathways.jsonl \
      --open_answers data/outputs/eval_v3/culturebank/open_answers.jsonl \
      --open_out_dir data/outputs/eval_v3/culturebank \
      --v3_comparisons --wait_minutes "${JUDGE_WAIT_MIN:-150}" || RC=$?
    ;;
  olmo)
    export VLLM_MODEL=allenai/OLMo-2-1124-7B-Instruct VLLM_MAX_MODEL_LEN=4096
    python scripts/run_culturebank_eval_v2.py --provider vllm --model "$VLLM_MODEL" "${COMMON[@]}" \
      --stages mcq --systems S0,S1,S3g,S4,S4L,S4R,S4R_noabs \
      --out_dir data/outputs/eval_v3/culturebank_olmo \
      --para_cache data/eval_v2/culturebank/mcq_paraphrased.jsonl \
      --title "Eval v3 - CultureBank MCQ (OLMo-2-7B, second model)" || RC=$?
    python scripts/run_normad_v2.py --provider vllm --model "$VLLM_MODEL" "${COMMON[@]}" \
      --conditions C1_country,C2_value,C3_rule_of_thumb,S1g,S4R,S4R_noabs \
      --out_dir data/outputs/eval_v3/normad_olmo \
      --title "Eval v3 - NormAd (OLMo-2-7B, second model)" || RC=$?
    ;;
  *) echo "unknown EVAL=$EVAL" >&2; RC=2 ;;
esac
echo "===== done $EVAL rc=$RC: $(date) ====="
exit $RC
