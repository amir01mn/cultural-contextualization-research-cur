#!/bin/bash
# Shared Narval module setup for GPU jobs (vLLM/torch need consistent CUDA libs).
#
# Alliance note: load arrow + opencv + cuda BEFORE activating the venv, and
# purge first so stale cudacore/12.2 paths do not break torch import.

load_gpu_modules() {
  module --force purge 2>/dev/null || true
  # Clear inherited login-shell paths that pin old cudacore nvJitLink.
  unset LD_LIBRARY_PATH
  module load StdEnv/2023 gcc/12.3 python/3.11 arrow/21.0.0 opencv/4.13.0 cuda/12.9

  # Prefer the cudacore bundled with cuda/12.9 (matches torch/vLLM wheels).
  if [[ -n "${EBROOTCUDACORE:-}" ]]; then
    export LD_LIBRARY_PATH="${EBROOTCUDACORE}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

load_login_modules() {
  module --force purge 2>/dev/null || true
  unset LD_LIBRARY_PATH
  module load StdEnv/2023 gcc/12.3 python/3.11 arrow/21.0.0 opencv/4.13.0
}

verify_torch_import() {
  local venv_python="${1:?venv python path}"
  "${venv_python}" - <<'PY'
import torch
print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()}")
PY
}
