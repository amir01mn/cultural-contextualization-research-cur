#!/bin/bash
# Babysit the 2000-sample corpus + eval pipeline overnight.
# Resubmits corpus on timeout; submits eval when corpus checkpoints are complete.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs slurm/watch

LOG="slurm/watch/watch_2000_$(date +%Y%m%d_%H%M%S).log"
CORPUS_LIMIT=2000
BATCH_SIZE=32
EVAL_LIMIT=250
TOP_K=8
SEED=42
SYSTEMS='S0_no_kg,S1_raw_text_rag,S2_flat_kg,S3_layered_kg,S4_abstraction_refinement_kg'
MAX_CORPUS_ATTEMPTS=10

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

corpus_complete() {
  python3 - <<'PY'
import json
from pathlib import Path
state_path = Path("data/processed/corpus_build_state.json")
if not state_path.exists():
    raise SystemExit(1)
state = json.loads(state_path.read_text())
ext = len(state.get("extraction_completed") or [])
crit = len(state.get("critique_completed") or [])
if ext >= 2000 and crit >= 2000:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

submit_corpus() {
  CORPUS_LIMIT=$CORPUS_LIMIT BATCH_SIZE=$BATCH_SIZE RESUME=1 \
    sbatch --parsable slurm/build_corpus.sh
}

submit_eval() {
  LIMIT=$EVAL_LIMIT TOP_K=$TOP_K SEED=$SEED SYSTEMS="$SYSTEMS" \
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
      RUNNING|PENDING|CONFIGURING|COMPLETING)
        sleep 90
        ;;
      "")
        sleep 30
        ;;
      *)
        log "Job $jid state=$st; sleeping..."
        sleep 90
        ;;
    esac
  done
}

log "=== Starting 2000-sample babysit ==="
log "Project: $PROJECT_DIR"

CORPUS_JOB="${1:-64880536}"
EVAL_JOB="${2:-64880609}"

log "Initial corpus job: $CORPUS_JOB"
log "Initial eval job: $EVAL_JOB (will replace if needed)"

attempt=1
while (( attempt <= MAX_CORPUS_ATTEMPTS )); do
  st=$(sacct -j "$CORPUS_JOB" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ' || true)
  if [[ "$st" == "RUNNING" || "$st" == "PENDING" || "$st" == "CONFIGURING" ]]; then
    wait_job "$CORPUS_JOB" || true
  fi

  if corpus_complete; then
    log "Corpus checkpoint complete (>=2000 extraction + critique IDs)."
    break
  fi

  log "Corpus not complete after job $CORPUS_JOB (attempt $attempt/$MAX_CORPUS_ATTEMPTS)."
  if (( attempt >= MAX_CORPUS_ATTEMPTS )); then
    log "ERROR: corpus still incomplete after $MAX_CORPUS_ATTEMPTS attempts."
    exit 1
  fi

  CORPUS_JOB=$(submit_corpus)
  log "Resubmitted corpus job: $CORPUS_JOB"
  attempt=$((attempt + 1))
  wait_job "$CORPUS_JOB" || true
done

# Run final corpus stage once more if last job timed out mid-pipeline (paths/index)
st=$(sacct -j "$CORPUS_JOB" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ' || true)
ec=$(sacct -j "$CORPUS_JOB" --format=ExitCode -n -X 2>/dev/null | head -1 | tr -d ' ' || true)
if [[ "$ec" != "0:0" ]] && corpus_complete; then
  log "Checkpoints complete but last job exit=$ec; running one more corpus job for downstream stages..."
  CORPUS_JOB=$(submit_corpus)
  wait_job "$CORPUS_JOB" || true
fi

if sacct -j "$EVAL_JOB" --format=State -n -X &>/dev/null; then
  est=$(sacct -j "$EVAL_JOB" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
  if [[ "$est" == "COMPLETED" ]]; then
    eec=$(sacct -j "$EVAL_JOB" --format=ExitCode -n -X 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$eec" == "0:0" ]] && [[ -f data/outputs/culturalbench_summary.md ]]; then
      mtime=$(stat -c %Y data/outputs/culturalbench_summary.md 2>/dev/null || echo 0)
      now=$(date +%s)
      if (( now - mtime < 86400 )); then
        log "Eval already completed recently. Done."
        exit 0
      fi
    fi
  fi
  if [[ "$est" == "PENDING" || "$est" == "RUNNING" ]]; then
    log "Cancelling stale eval job $EVAL_JOB ..."
    scancel "$EVAL_JOB" 2>/dev/null || true
    sleep 5
  fi
fi

EVAL_JOB=$(submit_eval)
log "Eval job submitted: $EVAL_JOB"
if wait_job "$EVAL_JOB"; then
  log "=== 2000-sample pipeline complete ==="
  log "Outputs: data/outputs/culturalbench_summary.md, mechanism_analysis.md"
  exit 0
fi

log "Eval failed; submitting one retry..."
EVAL_JOB=$(submit_eval)
wait_job "$EVAL_JOB" || exit 1
log "=== 2000-sample pipeline complete (after eval retry) ==="
