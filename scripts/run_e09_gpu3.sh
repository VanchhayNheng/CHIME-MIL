#!/bin/bash
# Run the e09 ablation row (5 folds) on GPU3 in parallel with the GPU1 chain.
# GPU3-only. Never touches GPU0 (hwlee) / GPU1 (meancenter+sa_stratval driver)
# / GPU2 (other user). The GPU1 driver no-ops e09 when it reaches it via the
# result.json skip-guard in train_loso_e09_genbio.py. Matches the GPU1 driver's
# exact invocation (no extra trainer args) + determinism env for parity.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${WT}"
PY=python3
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RR="${WT}/results/rerun"
LOGS="${RR}/logs"
mkdir -p "${LOGS}"
HOSP=(Site_A Site_B Site_C Site_D Site_E)
out="${RR}/e09"

for H in "${HOSP[@]}"; do
  echo ">>> e09(gpu3) ${H} start $(date +%T)"
  CUDA_VISIBLE_DEVICES=3 "${PY}" "${WT}/train_loso_e09_genbio.py" \
    --test_hospital "${H}" --output_dir "${out}" \
    > "${LOGS}/e09_gpu3_${H}.log" 2>&1
  echo "<<< e09(gpu3) ${H} done $(date +%T) rc=$?"
done
echo "=== e09(gpu3) ALL FOLDS DONE $(date +%T) ==="
