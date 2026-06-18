#!/bin/bash
# Deterministic HEADLINE re-run: mc_sa@256 (GPU1) + mc_sa@392 (GPU3).
# Each GPU runs its 5 folds sequentially so its feature set stays warm in
# page cache across folds (256 and 392 are ~331 GB each; running both sets
# concurrently per-GPU would thrash the 415 GB cache).
set -u
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=python3
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

RR="${WT}/results/rerun"
LOGS="${RR}/logs"
mkdir -p "${LOGS}"
HOSP=(Site_A Site_B Site_C Site_D Site_E)

run_track() {  # $1=gpu  $2=tag  $3...=extra trainer args
  local gpu=$1 tag=$2; shift 2
  local out="${RR}/${tag}"
  local i=0
  for H in "${HOSP[@]}"; do
    i=$((i + 1))
    echo ">>> ${tag} fold ${i} ${H} (gpu ${gpu}) start $(date +%T)"
    CUDA_VISIBLE_DEVICES=${gpu} "${PY}" "${WT}/train_loso_mc_sa_genbio.py" \
      --test_hospital "${H}" --val_split stratified --output_dir "${out}" "$@" \
      > "${LOGS}/${tag}_${H}.log" 2>&1
    echo "<<< ${tag} fold ${i} ${H} done $(date +%T) rc=$?"
  done
  echo "=== ${tag} ALL 5 FOLDS DONE $(date +%T) ==="
}

# GPU1: 256 (default config + default 256 site means)
run_track 1 mc_sa_256 > "${LOGS}/_track_256.log" 2>&1 &
P1=$!
# GPU3: 392 (392 config + 392 site means)
run_track 3 mc_sa_392 --config "${WT}/configs/config_genbio_392.yaml" \
  --site_means_path "${WT}/fusion/site_means_genbio_392.npz" \
  > "${LOGS}/_track_392.log" 2>&1 &
P3=$!

echo "mc_sa_256 track PID=$P1 (gpu1) ; mc_sa_392 track PID=$P3 (gpu3)"
wait $P1 $P3
echo "HEADLINE RERUN COMPLETE $(date +%T)"
