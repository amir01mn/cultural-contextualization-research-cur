#!/bin/bash
# Cluster-specific SLURM settings for Narval and Nibi (sourced, never run).
#
# The #SBATCH headers in slurm/kazculture_*.sh are written for Narval (A100).
# slurm/submit_kz.sh sources this file and adds the flags below on the sbatch
# command line, where they override the #SBATCH values. Both clusters use the
# same accounts (def-enaskt_gpu / def-enaskt_cpu), the same partition names
# (gpubase_bygpu_b1 = 3 h, b2 = 12 h, ...) and the same StdEnv/2023 module
# stack, so only the GPU type and the node exclusions differ.
#
#   Narval: 4x A100-40GB per node, gres gpu:a100:N; skip ng10706 (bad node).
#   Nibi:   8x H100-80GB per node (14 cores, ~250 GB RAM per GPU),
#           gres gpu:h100:N; MIG slices exist as
#           gpu:nvidia_h100_80gb_hbm3_{1g.10gb,2g.20gb,3g.40gb}.

kz_detect_cluster() {
  local c="${CC_CLUSTER:-}"
  if [[ -z "$c" ]]; then
    case "$(hostname -f 2>/dev/null || hostname)" in
      *narval*|ng[0-9]*|nc[0-9]*|nl[0-9]*) c=narval ;;
      *nibi*) c=nibi ;;
      *) c=unknown ;;
    esac
  fi
  echo "$c"
}

KZ_CLUSTER="${KZ_CLUSTER:-$(kz_detect_cluster)}"
case "$KZ_CLUSTER" in
  narval)
    KZ_GPU_TYPE="${KZ_GPU_TYPE:-a100}"
    KZ_SBATCH_EXCLUDE="${KZ_SBATCH_EXCLUDE-ng10706}"
    ;;
  nibi)
    KZ_GPU_TYPE="${KZ_GPU_TYPE:-h100}"
    KZ_SBATCH_EXCLUDE="${KZ_SBATCH_EXCLUDE-}"
    ;;
  *)
    KZ_GPU_TYPE="${KZ_GPU_TYPE:-}"
    KZ_SBATCH_EXCLUDE="${KZ_SBATCH_EXCLUDE-}"
    ;;
esac
export KZ_CLUSTER KZ_GPU_TYPE KZ_SBATCH_EXCLUDE
