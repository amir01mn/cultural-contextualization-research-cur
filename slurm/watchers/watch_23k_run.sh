#!/bin/bash
# Babysit full CultureBank corpus (~23k) + eval overnight.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs slurm/watch

LOG="slurm/watch/watch_23k_$(date +%Y%m%d_%H%M%S).log"
CORPUS_LIMIT=22990
BATCH_SIZE=32
EVAL_LIMIT=250
MAX_CORPUS_ATTEMPTS=15

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

corpus_complete() {
  python3 - <<'PY'
import json
from pathlib import Path
state = json.loads(Path("data/processed/corpus_build_state.json").read_text())
if len(state.get("extraction_completed") or []) >= 22990 and len(state.get("critique_completed") or []) >= 22990:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

submit_corpus() {
  CORPUS_LIMIT=$CORPUS_LIMIT BATCH_SIZE=$BATCH_SIZE RESUME=1 \
    sbatch --parsable slurm/build_corpus.sh
}

submit_eval() {
  LIMIT=$EVAL_LIMIT TOP_K=8 SEED=42 \
    SYSTEMS='S0_no_kg,S1_raw_text_rag,S2_flat_kg,S3_layered_kg,S4_abstraction_refinement_kg' \
    sbatch --parsable slurm/eval_culturalbench.sh
}

wait_job() {
  local jid=$1
  log "Watching job $jid ..."
  while true; do
    local st ec
    st=$(sacct -j "$jid" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
    case "$st" in
      COMPLETED)
        ec=$(sacct -j "$jid" --format=ExitCode -n -X 2>/dev/null | head -1 | tr -d ' ')
        log "Job $jid COMPLETED (ExitCode=$ec)"
        [[ "$ec" == "0:0" ]]
        return
        ;;
      FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
        log "Job $jid ended: $st"
        return 1
        ;;
      *) sleep 90 ;;
    esac
  done
}

log "=== Starting 23k babysit ==="
attempt=1
CORPUS_JOB=$(submit_corpus)
log "Corpus job: $CORPUS_JOB"

while (( attempt <= MAX_CORPUS_ATTEMPTS )); do
  wait_job "$CORPUS_JOB" || true
  if corpus_complete; then
    log "Corpus checkpoints complete (>=22990)."
    break
  fi
  if (( attempt >= MAX_CORPUS_ATTEMPTS )); then
    log "ERROR: corpus incomplete after $MAX_CORPUS_ATTEMPTS attempts."
    exit 1
  fi
  CORPUS_JOB=$(submit_corpus)
  log "Resubmitted corpus job: $CORPUS_JOB (attempt $attempt)"
  attempt=$((attempt + 1))
done

# Ensure downstream stages ran (pathways/index) even if validation failed before
if ! python3 scripts/validate_corpus.py >/dev/null 2>&1; then
  log "Running validation (may refresh report)..."
  python3 scripts/validate_corpus.py || true
fi

EVAL_JOB=$(submit_eval)
log "Eval job: $EVAL_JOB"
wait_job "$EVAL_JOB" || { EVAL_JOB=$(submit_eval); wait_job "$EVAL_JOB"; }
log "=== 23k pipeline complete ==="
