#!/bin/bash
# Resume the four PAUSED ablation rows on GPU3 only (GPU3 freed when the
# 392 headline track finished). All four are 256-feature, run sequentially.
# Forward hospital order (Site_A first) so this never cold-reads the same
# fold the 256 headline track is finishing (Site_E) at launch time.
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RR="${WT}/results/rerun"
LOGS="${RR}/logs"
mkdir -p "${LOGS}"
HOSP=(Site_A Site_B Site_C Site_D Site_E)

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

# All four rows, sequential, GPU3 only.
run_track 3 baseline    train_loso_genbio.py
run_track 3 meancenter  train_loso_meancenter_genbio.py
run_track 3 sa_stratval train_loso_sa_genbio.py --val_split stratified
run_track 3 e09         train_loso_e09_genbio.py

echo "REMAINING(GPU3) ROWS COMPLETE $(date +%T)"
