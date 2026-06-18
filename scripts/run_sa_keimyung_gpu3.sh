#!/bin/bash
# Run ONLY the sa_stratval Site_E fold on GPU3, in parallel with the GPU1
# chain. GPU3-only. The GPU1 driver no-ops Site_E via the result.json guard
# in train_loso_sa_genbio.py. Matches the driver's sa invocation exactly
# (--val_split stratified) + determinism env for parity.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WT}"
PY=python3
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RR="${WT}/results/rerun"
LOGS="${RR}/logs"
mkdir -p "${LOGS}"
out="${RR}/sa_stratval"

echo ">>> sa_stratval(gpu3) Site_E start $(date +%T)"
CUDA_VISIBLE_DEVICES=3 "${PY}" "${WT}/train_loso_sa_genbio.py" \
  --test_hospital Site_E --output_dir "${out}" --val_split stratified \
  > "${LOGS}/sa_stratval_gpu3_Site_E.log" 2>&1
echo "<<< sa_stratval(gpu3) Site_E done $(date +%T) rc=$?"
