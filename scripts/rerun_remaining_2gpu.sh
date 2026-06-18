#!/bin/bash
# Deterministic re-run of the remaining ablation rows, sharing GPUs 1 & 3 with
# the headline run. All four are 256-feature -> share the 256 page cache.
# Hospital order is REVERSED vs the headline run to avoid two jobs cold-reading
# the same fold's train set simultaneously (the shared-SSD same-fold I/O stall).
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RR="${WT}/results/rerun"
LOGS="${RR}/logs"
mkdir -p "${LOGS}"
HOSP=(Site_E Site_D Site_C Site_B Site_A)

run_track() {  # $1=gpu  $2=tag  $3=script ; rest = extra trainer args
  local gpu=$1 tag=$2 script=$3; shift 3
  local out="${RR}/${tag}"
  for H in "${HOSP[@]}"; do
    echo ">>> ${tag} ${H} (gpu ${gpu}) start $(date +%T)"
    CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${WT}/${script}" \
      --test_hospital "${H}" --output_dir "${out}" "$@" \
      > "${LOGS}/${tag}_${H}.log" 2>&1
    echo "<<< ${tag} ${H} done $(date +%T) rc=$?"
  done
  echo "=== ${tag} ALL FOLDS DONE $(date +%T) ==="
}

# GPU1: baseline then meancenter (256, random val)
( run_track 1 baseline   train_loso_genbio.py
  run_track 1 meancenter train_loso_meancenter_genbio.py
) > "${LOGS}/_track_gpu1_rest.log" 2>&1 &
P1=$!
# GPU3: sa_stratval (stratified val) then e09 (256)
( run_track 3 sa_stratval train_loso_sa_genbio.py --val_split stratified
  run_track 3 e09         train_loso_e09_genbio.py
) > "${LOGS}/_track_gpu3_rest.log" 2>&1 &
P3=$!

echo "remaining GPU1 PID=$P1 ; GPU3 PID=$P3"
wait $P1 $P3
echo "REMAINING ROWS COMPLETE $(date +%T)"
