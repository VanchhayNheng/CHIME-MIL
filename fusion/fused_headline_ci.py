"""Pooled-LOSO bootstrap CI + paired significance for the multi-scale late-fusion
headline (256 + 392 meancenter LOSO runs).

Companion to late_fusion_256_392.py: that script reports the per-fold-mean gate
(+ESCALATE if fused mean AUC beats best single-scale by >=0.005). This one adds
the paper-facing uncertainty:
  - pooled-LOSO point AUC (all test slides concatenated across folds, matching the
    trainer's own "Bootstrap 95% CI (pooled LOSO)" convention),
  - bootstrap 95% CI on fused pooled AUC,
  - paired bootstrap of (fused - best_single_scale) AUC on the SAME resampled
    slides -> two-sided p-value.

Note: this is a slide-level pooled bootstrap (ignores hospital/fold structure),
identical to the existing per-run trainer bootstrap. A hierarchical (resample
folds, then slides) variant is listed as a future refinement in the punchlist.
"""
import argparse, json, os
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

_DEF = Path(__file__).resolve().parents[1] / 'results'
ap = argparse.ArgumentParser()
ap.add_argument('--pred_dir_256', default=str(_DEF / 'rerun' / 'mc_sa_256'))
ap.add_argument('--pred_dir_392', default=str(_DEF / 'rerun' / 'mc_sa_392'))
ap.add_argument('--n_boot', type=int, default=10000)
ap.add_argument('--seed', type=int, default=42)
ARGS = ap.parse_args()
HOSP = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E"]


def to1d(p):
    p = np.asarray(p, dtype=np.float64)
    return p[:, 1] if p.ndim == 2 else p


lab_all, p256_all, p392_all = [], [], []
for i, h in enumerate(HOSP, 1):
    f256 = f"{ARGS.pred_dir_256}/fold_{i}_{h}/result.json"
    f392 = f"{ARGS.pred_dir_392}/fold_{i}_{h}/result.json"
    assert os.path.exists(f256) and os.path.exists(f392), f"missing fold {i} {h}"
    d256, d392 = json.load(open(f256)), json.load(open(f392))
    l256 = np.asarray(d256['labels'], dtype=np.int64)
    l392 = np.asarray(d392['labels'], dtype=np.int64)
    assert l256.shape == l392.shape and (l256 == l392).all(), f"label mismatch fold {i} {h}"
    lab_all.append(l256)
    p256_all.append(to1d(d256['probs']))
    p392_all.append(to1d(d392['probs']))

lab = np.concatenate(lab_all)
p256 = np.concatenate(p256_all)
p392 = np.concatenate(p392_all)
fused = 0.5 * (p256 + p392)
N = lab.shape[0]

auc256 = roc_auc_score(lab, p256)
auc392 = roc_auc_score(lab, p392)
aucfus = roc_auc_score(lab, fused)
best_name = '256' if auc256 >= auc392 else '392'
best = p256 if auc256 >= auc392 else p392
auc_best = max(auc256, auc392)

rng = np.random.default_rng(ARGS.seed)
boot_fus, boot_diff = [], []
for _ in range(ARGS.n_boot):
    idx = rng.integers(0, N, N)
    yl = lab[idx]
    if yl.min() == yl.max():
        continue
    af = roc_auc_score(yl, fused[idx])
    ab = roc_auc_score(yl, best[idx])
    boot_fus.append(af)
    boot_diff.append(af - ab)

boot_fus = np.asarray(boot_fus)
boot_diff = np.asarray(boot_diff)
ci_fus = np.percentile(boot_fus, [2.5, 97.5])
ci_diff = np.percentile(boot_diff, [2.5, 97.5])
# two-sided bootstrap p-value for H0: diff == 0
p_two = 2.0 * min((boot_diff <= 0).mean(), (boot_diff >= 0).mean())
p_two = min(p_two, 1.0)

print(f"pred_dir_256 = {ARGS.pred_dir_256}")
print(f"pred_dir_392 = {ARGS.pred_dir_392}")
print(f"n_boot = {ARGS.n_boot}  seed = {ARGS.seed}  N_slides = {N}  (used {len(boot_fus)} valid resamples)")
print("-" * 64)
print(f"Pooled-LOSO point AUC:  256={auc256:.4f}  392={auc392:.4f}  fused={aucfus:.4f}")
print(f"Best single-scale: {best_name} (AUC {auc_best:.4f})")
print(f"Fused pooled AUC      : {aucfus:.4f}  95% CI [{ci_fus[0]:.4f}, {ci_fus[1]:.4f}]")
print(f"Fused - best (paired) : {aucfus - auc_best:+.4f}  95% CI [{ci_diff[0]:+.4f}, {ci_diff[1]:+.4f}]")
print(f"Two-sided bootstrap p : {p_two:.4f}")
