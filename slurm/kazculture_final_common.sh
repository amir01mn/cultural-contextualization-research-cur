#!/bin/bash
# Shared paths/helpers for the final KazCulture stages (sourced, never run).
# Frozen inputs (docs/KAZCULTURE_PREREGISTRATION.md §2). Override any via env.
KZ_BASE="${KZ_BASE:-data/kazculture_pipeline}"
KZ_TEST="${KZ_TEST:-data/raw/kazculture_phase1/test.parquet}"
KZ_CONFIG="${KZ_CONFIG:-$KZ_BASE/s1_full_dev_thresholds_v3.json}"
KZ_INDEX_STRICT="${KZ_INDEX_STRICT:-$KZ_BASE/index/s1_strict}"
KZ_INDEX_FULL="${KZ_INDEX_FULL:-$KZ_BASE/index/s1_full}"
KZ_REF_S0="${KZ_REF_S0:-data/outputs/kazculture_phase1_v2/predictions.jsonl}"
KZ_REF_S1_STRICT="${KZ_REF_S1_STRICT:-$KZ_BASE/evaluation/s1_strict/S1.jsonl}"
KZ_REF_S1_FULL="${KZ_REF_S1_FULL:-$KZ_BASE/evaluation/s1_full/S1.jsonl}"
KZ_DEV="${KZ_DEV:-$KZ_BASE/corpus/train_dev_qa.jsonl}"
KZ_FINAL_OUT="${KZ_FINAL_OUT:-$KZ_BASE/evaluation/final_v1}"
KZ_OLMO_OUT="${KZ_OLMO_OUT:-$KZ_BASE/evaluation/final_v1_olmo}"
KZ_GATE_OUT="${KZ_GATE_OUT:-$KZ_BASE/evaluation/traindev_gate_v1}"
KZ_NEARDUP_OUT="${KZ_NEARDUP_OUT:-$KZ_BASE/audit/near_duplicates_v1}"
# Frame-KG retrieval adapter (owned by the frame-KG port). Optional JSON for
# its configure(**kwargs), e.g. frame/index directories.
KZ_FRAME_ADAPTER="${KZ_FRAME_ADAPTER:-src.retrieval.kazculture_frame_retriever}"
KZ_FRAME_ADAPTER_KWARGS="${KZ_FRAME_ADAPTER_KWARGS:-}"
KZ_ROUTE_MODES="${KZ_ROUTE_MODES:-}"

kz_setup_env() {  # $1 = gpu|cpu
  set -euo pipefail
  PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
  cd "$PROJECT_DIR"
  source slurm/load_modules.sh
  if [[ "$1" == gpu ]]; then load_gpu_modules; else load_login_modules; fi
  source slurm/env_hf.sh
  setup_hf_offline
  source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
  export TOKENIZERS_PARALLELISM=false
  if [[ "$1" == gpu ]]; then verify_torch_import "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/python"; fi
  mkdir -p slurm/logs
  echo "host=$(hostname) job=${SLURM_JOB_ID:-none} commit=$(git rev-parse HEAD)"
}

kz_require() {
  local f
  for f in "$@"; do
    [[ -e "$f" ]] || { echo "Missing required input: $f" >&2; exit 2; }
  done
}

kz_adapter_args() {  # prints adapter CLI args (one per line) for mapfile
  echo --frame_adapter; echo "$KZ_FRAME_ADAPTER"
  if [[ -n "$KZ_FRAME_ADAPTER_KWARGS" ]]; then echo --frame_adapter_kwargs; echo "$KZ_FRAME_ADAPTER_KWARGS"; fi
  if [[ -n "$KZ_ROUTE_MODES" ]]; then echo --route_modes; echo "$KZ_ROUTE_MODES"; fi
}

kz_check_adapter() {
  python -c "import importlib,sys; m=importlib.import_module(sys.argv[1]); assert hasattr(m,'retrieve_s4r'), 'adapter lacks retrieve_s4r'" "$KZ_FRAME_ADAPTER"
}

# Run the final evaluator for one stage: $1 = settings, $2 = conditions.
kz_run_final() {
  local settings="$1" conditions="$2"
  kz_require "$KZ_TEST" "$KZ_CONFIG" "$KZ_REF_S0"
  [[ "$settings" == *strict* ]] && kz_require "$KZ_INDEX_STRICT/build_manifest.json" "$KZ_REF_S1_STRICT"
  [[ "$settings" == *full* ]] && kz_require "$KZ_INDEX_FULL/build_manifest.json" "$KZ_REF_S1_FULL"
  if [[ "$conditions" == *S4R* || "$conditions" == *CTRL* || "$conditions" == *S1+L* ]]; then
    kz_check_adapter
  fi
  local adapter_args
  mapfile -t adapter_args < <(kz_adapter_args)
  python scripts/eval_kazculture_final.py \
    --test_file "$KZ_TEST" --s1_config "$KZ_CONFIG" \
    --index_strict "$KZ_INDEX_STRICT" --index_full "$KZ_INDEX_FULL" \
    --settings "$settings" --conditions "$conditions" \
    --out "$KZ_FINAL_OUT" \
    --reference_s0 "$KZ_REF_S0" \
    --reference_s1_strict "$KZ_REF_S1_STRICT" --reference_s1_full "$KZ_REF_S1_FULL" \
    --require_reproduction \
    --max_minutes "${KZ_MAX_MINUTES:-165}" \
    "${adapter_args[@]}"
}
