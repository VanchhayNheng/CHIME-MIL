#!/bin/bash
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
TS=$(date +%Y%m%d_%H%M%S)
OUT=${WT}/results/results_loso_mc_sa_warmup3_genbio_20260421
LOGS=${WT}/results/logs_mc_sa_warmup3_${TS}
mkdir -p "${LOGS}" "${OUT}"
SCRIPT="${WT}/train_loso_mc_sa_genbio.py"
COMMON="--val_split stratified --warmup_epochs 3 --output_dir ${OUT}"
echo "OUT=${OUT}"
echo "LOGS=${LOGS}"
CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" --test_hospital Site_A  $COMMON >> "${LOGS}/Site_A.log" 2>&1 &
PID0=$!; echo "Site_A GPU0 PID=$PID0"
CUDA_VISIBLE_DEVICES=1 "$PY" "$SCRIPT" --test_hospital Site_B    $COMMON >> "${LOGS}/Site_B.log" 2>&1 &
echo "Site_B GPU1 PID=$!"
CUDA_VISIBLE_DEVICES=2 "$PY" "$SCRIPT" --test_hospital Site_C $COMMON >> "${LOGS}/Site_C.log" 2>&1 &
echo "Site_C GPU2 PID=$!"
CUDA_VISIBLE_DEVICES=3 "$PY" "$SCRIPT" --test_hospital Site_D    $COMMON >> "${LOGS}/Site_D.log" 2>&1 &
echo "Site_D GPU3 PID=$!"
wait $PID0
CUDA_VISIBLE_DEVICES=0 "$PY" "$SCRIPT" --test_hospital Site_E $COMMON >> "${LOGS}/Site_E.log" 2>&1 &
echo "Site_E GPU0 PID=$! (after Site_A)"
wait
echo "all 5 folds done"
