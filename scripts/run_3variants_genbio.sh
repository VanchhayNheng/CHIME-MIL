#!/bin/bash
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
HOSPITALS=(Site_A Site_B Site_C Site_D Site_E)
TS=$(date +%Y%m%d_%H%M%S)
LOGS=${WT}/results/logs_3var_${TS}
mkdir -p "${LOGS}"
run_variant() {
  local GPU=$1; local TAG=$2; local SCRIPT=$3
  for H in "${HOSPITALS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU} "$PY" "${WT}/${SCRIPT}" --test_hospital "$H" \
      >> "${LOGS}/${TAG}_${H}.log" 2>&1
  done
}
run_variant 0 soft_assign train_loso_sa_genbio.py &
echo "soft_assign GPU0 PID=$!"
run_variant 1 e09_fixed   train_loso_e09_genbio.py &
echo "e09_fixed   GPU1 PID=$!"
run_variant 2 meancenter  train_loso_meancenter_genbio.py &
echo "meancenter  GPU2 PID=$!"
wait
echo "all 3 variants done"
