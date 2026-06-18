"""Deterministic re-bootstrap: headline (mc_sa 256+392 fused) vs GenBio baselines.

Models the convention of fusion/fused_headline_ci.py exactly:
pooled-LOSO slide-level bootstrap, paired on the same resampled indices,
B=10000, seed 42, two-sided p = 2*min(tail fractions). BH-adjusts p across
the baseline set. Aligns headline<->baseline per fold by verifying the label
sequence matches (same test slides, same order); aborts a model if it doesn't.
"""
import csv, json, os
import numpy as np
from sklearn.metrics import roc_auc_score

from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1])
R256 = f"{REPO}/results/rerun/mc_sa_256"
R392 = f"{REPO}/results/rerun/mc_sa_392"
HOSP = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E"]
NBOOT, SEED = 10000, 42

BASE = {
    "ABMIL":      "runs/baselines/abmil_20260503_155656",
    "CLAM-SB":    "runs/baselines/clam_sb_20260503_163532",
    "CLAM-MB":    "runs/baselines/clam_mb_20260503_173749",
    "TransMIL":   "runs/baselines/transmil_20260505_154849",
    "DSMIL":      "runs/baselines/dsmil_20260503_185352",
    "RoFormer":   "runs/baselines/roformer_20260514_181808",
    "ACMIL":      "runs/baselines/acmil_20260514_221525",
    "MaxPool256": "runs/baselines/maxpool_20260503_150348",
    "MaxPool392": "runs/baselines/maxpool_392_20260504_110143",
}


def to1d(p):
    p = np.asarray(p, dtype=np.float64)
    return p[:, 1] if p.ndim == 2 else p


# ---- headline fused, per fold ----
head_fold = []  # list of (labels, fused_probs)
for i, h in enumerate(HOSP, 1):
    d256 = json.load(open(f"{R256}/fold_{i}_{h}/result.json"))
    d392 = json.load(open(f"{R392}/fold_{i}_{h}/result.json"))
    l = np.asarray(d256["labels"], dtype=np.int64)
    assert (l == np.asarray(d392["labels"], dtype=np.int64)).all(), f"256/392 label mismatch {h}"
    fused = 0.5 * (to1d(d256["probs"]) + to1d(d392["probs"]))
    head_fold.append((l, fused))


def load_baseline(d):
    """Return list of (labels, probs) per fold in HOSP order, or None if a fold missing."""
    out = []
    for i, h in enumerate(HOSP, 1):
        p = f"{REPO}/{d}/fold_{i}_{h}/test_predictions.csv"
        if not os.path.exists(p):
            return None
        labs, probs = [], []
        with open(p) as fh:
            for row in csv.DictReader(fh):
                labs.append(int(row["label"]))
                probs.append(float(row["prob_class1"]))
        out.append((np.array(labs, dtype=np.int64), np.array(probs, dtype=np.float64)))
    return out


def foldmean_auc(folds):
    return float(np.mean([roc_auc_score(l, p) for l, p in folds]))


# pooled headline
hl = np.concatenate([l for l, _ in head_fold])
hp = np.concatenate([p for _, p in head_fold])
N = hl.shape[0]
head_pool = roc_auc_score(hl, hp)
head_fm = foldmean_auc(head_fold)
print(f"HEADLINE  fold-mean={head_fm:.4f}  pooled={head_pool:.4f}  N={N}")
print("=" * 92)
print(f"{'Baseline':10s} {'fold-AUC':>8s} {'pool-AUC':>8s} {'dPool':>8s} {'95% CI':>20s} {'p_raw':>8s} {'p_BH':>8s} align")
print("-" * 92)

rng = np.random.default_rng(SEED)
results = []
for name, d in BASE.items():
    b = load_baseline(d)
    if b is None:
        print(f"{name:10s}  (missing folds -> skip)")
        continue
    # alignment: per-fold label sequence must match headline
    aligned = all(
        head_fold[k][0].shape == b[k][0].shape and (head_fold[k][0] == b[k][0]).all()
        for k in range(5)
    )
    if not aligned:
        # try: same multiset per fold? report counts to diagnose
        diag = [(int(head_fold[k][0].sum()), int(b[k][0].sum()),
                 head_fold[k][0].shape[0], b[k][0].shape[0]) for k in range(5)]
        print(f"{name:10s}  NOT ALIGNED (head_pos,base_pos,head_n,base_n per fold): {diag}")
        continue
    bp = np.concatenate([p for _, p in b])
    bl = np.concatenate([l for l, _ in b])
    base_pool = roc_auc_score(bl, bp)
    base_fm = foldmean_auc(b)
    # paired pooled bootstrap on same resampled indices
    diffs = []
    rng2 = np.random.default_rng(SEED)  # same RNG stream per baseline for reproducibility
    for _ in range(NBOOT):
        idx = rng2.integers(0, N, N)
        yl = hl[idx]
        if yl.min() == yl.max():
            continue
        diffs.append(roc_auc_score(yl, hp[idx]) - roc_auc_score(yl, bp[idx]))
    diffs = np.asarray(diffs)
    dpool = head_pool - base_pool
    ci = np.percentile(diffs, [2.5, 97.5])
    p_raw = min(1.0, 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    results.append([name, base_fm, base_pool, dpool, ci[0], ci[1], p_raw])

# BH correction across baselines
ps = np.array([r[6] for r in results])
order = np.argsort(ps)
m = len(ps)
bh = np.empty(m)
prev = 1.0
for rank in range(m - 1, -1, -1):
    i = order[rank]
    val = ps[i] * m / (rank + 1)
    prev = min(prev, val)
    bh[i] = min(prev, 1.0)

for r, q in zip(results, bh):
    name, base_fm, base_pool, dpool, lo, hi, p_raw = r
    print(f"{name:10s} {base_fm:8.4f} {base_pool:8.4f} {dpool:+8.4f} "
          f"[{lo:+.4f},{hi:+.4f}] {p_raw:8.4f} {q:8.4f}  OK")
