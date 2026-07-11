#!/bin/bash
# Fresh full CultureBank rebuild (~23k) + CulturalBench eval overnight.
# Chains 3h SLURM corpus jobs until checkpoints complete, then runs eval.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"
mkdir -p slurm/logs slurm/watch

LOG="slurm/watch/watch_full_$(date +%Y%m%d_%H%M%S).log"
CORPUS_LIMIT=22990
BATCH_SIZE=32
EVAL_LIMIT=250
MAX_CORPUS_ATTEMPTS=30

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

corpus_complete() {
  python3 - <<PY
import json
from pathlib import Path
target = $CORPUS_LIMIT
state = json.loads(Path("data/processed/corpus_build_state.json").read_text())
if len(state.get("extraction_completed") or []) >= target and len(state.get("critique_completed") or []) >= target:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

pathways_built() {
  python3 - <<'PY'
from pathlib import Path
p = Path("data/processed/corpus_pathways.jsonl")
if not p.exists():
    raise SystemExit(1)
n = sum(1 for line in p.open() if line.strip())
raise SystemExit(0 if n > 0 else 1)
PY
}

submit_corpus() {
  local resume_flag=$1
  CORPUS_LIMIT=$CORPUS_LIMIT BATCH_SIZE=$BATCH_SIZE RESUME=$resume_flag \
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
      RUNNING|PENDING|CONFIGURING|COMPLETING)
        sleep 90
        ;;
      *)
        sleep 45
        ;;
    esac
  done
}

log "=== Full pipeline babysit (fresh rebuild + eval) ==="
log "Project: $PROJECT_DIR  CORPUS_LIMIT=$CORPUS_LIMIT"

# --- corpus loop ---
attempt=1
initial_resume=0
if python3 - <<PY
import json
from pathlib import Path
target = $CORPUS_LIMIT
state = json.loads(Path("data/processed/corpus_build_state.json").read_text())
if len(state.get("extraction_completed") or []) >= target:
    raise SystemExit(0)
raise SystemExit(1)
PY
then
  initial_resume=1
  log "Extraction already complete; resuming from critique (RESUME=1)."
else
  log "Starting fresh corpus build (RESUME=0)."
fi
CORPUS_JOB=$(submit_corpus "$initial_resume")
log "Initial corpus job (RESUME=$initial_resume): $CORPUS_JOB"

while (( attempt <= MAX_CORPUS_ATTEMPTS )); do
  wait_job "$CORPUS_JOB" || true

  if corpus_complete; then
    log "Checkpoints: extraction+critique >= $CORPUS_LIMIT"
    if ! pathways_built; then
      log "Pathways missing; running one more corpus job for downstream stages..."
      CORPUS_JOB=$(submit_corpus 1)
      wait_job "$CORPUS_JOB" || true
    fi
    break
  fi

  if (( attempt >= MAX_CORPUS_ATTEMPTS )); then
    log "ERROR: corpus incomplete after $MAX_CORPUS_ATTEMPTS attempts."
    exit 1
  fi

  CORPUS_JOB=$(submit_corpus 1)
  log "Resubmitted corpus job (RESUME=1): $CORPUS_JOB (attempt $attempt)"
  attempt=$((attempt + 1))
done

python3 scripts/09_validate_phase2.py >>"$LOG" 2>&1 || log "Validation script warning (non-fatal)"

# --- eval ---
EVAL_JOB=$(submit_eval)
log "Eval job: $EVAL_JOB"
if ! wait_job "$EVAL_JOB"; then
  log "Eval failed; retrying once..."
  EVAL_JOB=$(submit_eval)
  wait_job "$EVAL_JOB" || exit 1
fi

log "=== Pipeline complete ==="
log "See data/outputs/culturalbench_summary.md and mechanism_analysis.md"
