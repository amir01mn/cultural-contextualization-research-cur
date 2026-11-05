#!/bin/bash
# Submit a KazCulture SLURM script with cluster-appropriate flags (Narval/Nibi).
#
#   slurm/submit_kz.sh [sbatch flags...] slurm/<script>.sh [script args...]
#
# Adds --gres=gpu:<a100|h100>:N (N from the script's #SBATCH --gres line, or
# KZ_NGPU) and, on Narval, --exclude=ng10706. Flags you pass yourself win: an
# explicit --gres or --exclude disables the corresponding default. Runs from
# the repo root so the scripts' relative log paths and SLURM_SUBMIT_DIR work.
#
#   KZ_NGPU=1 slurm/submit_kz.sh slurm/kazculture_final_strict_a.sh
#   KZ_DRY_RUN=1 slurm/submit_kz.sh --time=00:30:00 slurm/kazculture_phase1_v2.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=cluster_profile.sh
source "$REPO_ROOT/slurm/cluster_profile.sh"

flags=() script=""
while (($#)); do
  if [[ "$1" == *.sh && -f "$REPO_ROOT/$1" || "$1" == *.sh && -f "$1" ]]; then
    script="$1"; shift; break
  fi
  flags+=("$1"); shift
done
[[ -n "$script" ]] || { echo "usage: $0 [sbatch flags...] slurm/<script>.sh [args...]" >&2; exit 2; }
[[ -f "$script" ]] || script="$REPO_ROOT/$script"

user_has() { local f; for f in "${flags[@]}"; do [[ "$f" == "$1"* ]] && return 0; done; return 1; }

extra=()
gres_line="$(grep -m1 -E '^#SBATCH[[:space:]]+--gres=gpu:' "$script" || true)"
if [[ -n "$gres_line" ]] && ! user_has --gres; then
  ngpu="${KZ_NGPU:-${gres_line##*:}}"
  ngpu="${ngpu%%[!0-9]*}"
  [[ -n "$KZ_GPU_TYPE" ]] || { echo "Unknown cluster '$KZ_CLUSTER': pass --gres yourself" >&2; exit 2; }
  extra+=("--gres=gpu:${KZ_GPU_TYPE}:${ngpu}")
fi
if [[ -n "$gres_line" && -n "$KZ_SBATCH_EXCLUDE" ]] && ! user_has --exclude; then
  extra+=("--exclude=$KZ_SBATCH_EXCLUDE")
fi

cmd=(sbatch "${extra[@]}" "${flags[@]}" "$script" "$@")
cd "$REPO_ROOT"
echo "[$KZ_CLUSTER] ${cmd[*]}"
[[ "${KZ_DRY_RUN:-0}" == 1 ]] && exit 0
"${cmd[@]}"
