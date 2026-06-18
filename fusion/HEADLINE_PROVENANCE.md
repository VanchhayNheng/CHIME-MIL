# Headline provenance -- multi-scale late fusion (deterministic re-run)

Branch: `release/paper`. Recorded 2026-06-01. All numbers reproduce exactly from
the committed `result.json` files below (verified 2026-06-01, GPUs masked, CPU-only).

## What the headline is

Per-slide flat fusion of two single-scale mc_sa LOSO tracks:
`fused_prob = 0.5 * (prob_256 + prob_392)`, `thr = 0.5 * (thr_256 + thr_392)`.
Lead headline = **per-fold-mean fused AUC = 0.8434** (5-fold LOSO).

## Prediction sources (pinned)

| Track | Pred dir | 5-fold mean test AUC |
|-------|----------|----------------------|
| mc_sa @256 | `results/rerun/mc_sa_256/fold_{1..5}_{Site A,Site B,Site C,Site D,Site E}/result.json` | 0.8341 |
| mc_sa @392 | `results/rerun/mc_sa_392/fold_{1..5}_...` | 0.8326 |

Both tracks were produced by `scripts/rerun_headline_2gpu.sh` (256 on GPU1, 392 on
GPU3), with `PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`--val_split stratified`. 392 track also uses `configs/config_genbio_392.yaml` +
`fusion/site_means_genbio_392.npz`. Python: `python3`.

## Reproduce

```bash
PY=python3
# per-fold-mean gate
$PY fusion/late_fusion_256_392.py \
  --pred_dir_256 results/rerun/mc_sa_256 --pred_dir_392 results/rerun/mc_sa_392
# pooled-LOSO bootstrap CI + paired significance
$PY fusion/fused_headline_ci.py \
  --pred_dir_256 results/rerun/mc_sa_256 --pred_dir_392 results/rerun/mc_sa_392 \
  --n_boot 10000 --seed 42
```

## Verified numbers

Per-fold-mean gate (`late_fusion_256_392.py`):

| Fold | Hospital | AUC256 | AUC392 | AUCfus | dAUC |
|------|----------|--------|--------|--------|------|
| 1 | Site A | 0.7893 | 0.7790 | 0.7929 | +0.0036 |
| 2 | Site B | 0.8620 | 0.8657 | 0.8704 | +0.0047 |
| 3 | Site C | 0.8306 | 0.8229 | 0.8354 | +0.0048 |
| 4 | Site D | 0.8278 | 0.8274 | 0.8398 | +0.0120 |
| 5 | Site E | 0.8609 | 0.8682 | 0.8783 | +0.0101 |
| **Mean** | | **0.8341** | **0.8326** | **0.8434** | **+0.0092** |

Gate: gain +0.0092 over best single-scale (256 @ 0.8341) >= 0.005 -> **ESCALATE**
(multi-scale carries real complementary signal). Per-fold dAUC all positive.

Pooled-LOSO bootstrap (`fused_headline_ci.py`, seed 42, 10k resamples, N=5036 slides):

- Pooled point AUC: 256 = 0.8087, 392 = 0.8058, **fused = 0.8174 [95% CI 0.8047, 0.8295]**
- Fused - best(256), paired on same resamples: **+0.0088 [95% CI +0.0048, +0.0127]**, two-sided p < 0.0001 -> **significant**

Note: pooled point AUC (0.8174) mixes all hospitals' slides and is lower than the
per-fold-mean (0.8434); this matches the trainer's own pooled-LOSO bootstrap
convention. Paper leads with per-fold-mean 0.8434; CI / significance from pooled.
A hierarchical (resample-folds-then-slides) bootstrap is a listed future refinement.

## Secondary metrics (ACC / BalAcc) -- reconciled 2026-06-03

Threshold: per-fold selected_threshold (val-optimized for balanced_acc), fused as
thr = 0.5*(thr_256 + thr_392); metrics on fused_prob >= thr. From pinned dirs above:

| Metric | Fold-mean | pop std |
|--------|-----------|---------|
| ACC    | 0.7529    | 0.042   |
| BalAcc | 0.7707    | 0.030   |

These SUPERSEDE the pre-rerun paper values (ACC 0.7685, BalAcc 0.7665), which did
NOT reproduce from this commit under any threshold rule (paper had ACC>BalAcc,
impossible for a balanced_acc-optimized threshold). results.tex updated 2026-06-03
to these values; population std throughout. AUC 0.8434 / pooled 0.8174 unchanged.
