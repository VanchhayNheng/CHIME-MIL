#!/bin/bash
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
HOSPITALS=(Site_A Site_B Site_C Site_D Site_E)
LOGS=${WT}/results/logs_meancenter_rerun_$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOGS}"
for H in "${HOSPITALS[@]}"; do
  CUDA_VISIBLE_DEVICES=2 "$PY" "${WT}/train_loso_meancenter_genbio.py" --test_hospital "$H" \
    >> "${LOGS}/meancenter_${H}.log" 2>&1
done
echo "meancenter 5 folds done"
