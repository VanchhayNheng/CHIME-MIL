#!/bin/bash
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
SCRIPT=${WT}/train_loso_genbio.py
LOGS=${WT}/results/logs_genbio_$(date +%Y%m%d_%H%M%S)
mkdir -p "${LOGS}"
HOSPITALS=(Site_A Site_B Site_C Site_D Site_E)
for i in 0 1 2 3; do
  H=${HOSPITALS[$i]}
  CUDA_VISIBLE_DEVICES=$i nohup "$PY" "$SCRIPT" --test_hospital "$H" >> "${LOGS}/fold_$((i+1))_${H}.log" 2>&1 &
  echo "fold $((i+1)) ${H} GPU${i} PID=$!"
done
wait %1
H=${HOSPITALS[4]}
CUDA_VISIBLE_DEVICES=0 nohup "$PY" "$SCRIPT" --test_hospital "$H" >> "${LOGS}/fold_5_${H}.log" 2>&1 &
echo "fold 5 ${H} GPU0 PID=$!"
wait
echo "all folds done"
