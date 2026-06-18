"""Paired-bootstrap CI and significance for the GenBio Top-4 ensemble vs the
best single GenBio member (`mc` = meancenter, mean LOSO AUC 0.8362).

For each of N_BOOT iterations:
  - For each fold, resample slides with replacement (same indices applied to
    BOTH ensemble and reference -> paired comparison).
  - Compute AUC/ACC/BalAcc/F1 per model on the resampled slides.
  - Average across 5 folds -> one iteration estimate per model.

95% CI = [2.5, 97.5] percentile of the iterations. Paired delta CI uses
ensemble - reference. p(delta <= 0) is the empirical fraction of iterations
where the ensemble did not beat the reference.
"""
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)

BASE = '/path/to/data/CHIME_MIL_genbio'
FOLDS = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']
N_BOOT = 1000
SEED = 42

ENSEMBLE_MEMBERS = [
    'results_loso_meancenter_genbio_20260421',     # mc
    'results_loso_mc_sa_genbio_20260421',          # mc_sa
    'results_loso_sa_stratified_genbio_20260421',  # sa_strat
    'results_loso_mc_sa_warmup3_genbio_20260421',  # mc_sa_w3
]
REFERENCE_DIR = 'results_loso_meancenter_genbio_20260421'  # mc, best single


def _probs_cls1(r):
    p = np.array(r['probs'], dtype=float)
    return p[:, 1] if p.ndim == 2 else p


def _load_fold(dir_rel, hospital):
    for i, h in enumerate(FOLDS, start=1):
        if h == hospital:
            r = json.load(open(f'{BASE}/{dir_rel}/fold_{i}_{h}/result.json'))
            return np.array(r['labels']), _probs_cls1(r)
    raise ValueError(hospital)


def fold_tensors_ensemble(member_dirs, hospital):
    rs = [json.load(open(f'{BASE}/{d}/fold_{i}_{hospital}/result.json'))
          for i in range(1, len(FOLDS) + 1) for d in [None] if False]  # noop
    # simple loader
    out_y, out_ps = None, []
    for d in member_dirs:
        y, p = _load_fold(d, hospital)
        if out_y is None:
            out_y = y
        else:
            assert np.array_equal(out_y, y), f'label mismatch {hospital} {d}'
        out_ps.append(p)
    return out_y, np.stack(out_ps, 0).mean(0)


def main():
    by_model = {
        'genbio_top4': {h: fold_tensors_ensemble(ENSEMBLE_MEMBERS, h)
                        for h in FOLDS},
        'mc_single':   {h: _load_fold(REFERENCE_DIR, h) for h in FOLDS},
    }

    def point_metrics(model):
        a, ac, ba, fs = [], [], [], []
        for h in FOLDS:
            y, p = by_model[model][h]
            pr = (p >= 0.5).astype(int)
            a.append(roc_auc_score(y, p))
            ac.append(accuracy_score(y, pr))
            ba.append(balanced_accuracy_score(y, pr))
            fs.append(f1_score(y, pr))
        return np.mean(a), np.mean(ac), np.mean(ba), np.mean(fs)

    print('Point estimates (mean over 5 folds):')
    for m in ('genbio_top4', 'mc_single'):
        a, ac, ba, fs = point_metrics(m)
        print(f'  {m:14s} AUC={a:.4f}  ACC={ac:.4f}  '
              f'BalAcc={ba:.4f}  F1={fs:.4f}')

    rng = np.random.default_rng(SEED)
    metric_names = ['AUC', 'ACC', 'BalAcc', 'F1']
    cols = ['genbio_top4', 'mc_single']
    boot_vals = {m: {k: [] for k in metric_names} for m in cols}

    for _ in range(N_BOOT):
        per_iter = {m: {k: [] for k in metric_names} for m in cols}
        for h in FOLDS:
            y_ref = by_model['genbio_top4'][h][0]
            n = len(y_ref)
            idx = rng.integers(0, n, size=n)
            ok = True
            for m in cols:
                y_full, p_full = by_model[m][h]
                y, p = y_full[idx], p_full[idx]
                if len(set(y)) < 2:
                    ok = False
                    break
                pr = (p >= 0.5).astype(int)
                per_iter[m]['AUC'].append(roc_auc_score(y, p))
                per_iter[m]['ACC'].append(accuracy_score(y, pr))
                per_iter[m]['BalAcc'].append(balanced_accuracy_score(y, pr))
                per_iter[m]['F1'].append(f1_score(y, pr))
            if not ok:
                # discard this iter to keep paired structure clean
                for m in cols:
                    for k in metric_names:
                        if len(per_iter[m][k]) > 0:
                            per_iter[m][k].pop()
        # only keep iters that produced 5 folds for both
        if all(len(per_iter[m][k]) == len(FOLDS)
               for m in cols for k in metric_names):
            for m in cols:
                for k in metric_names:
                    boot_vals[m][k].append(np.mean(per_iter[m][k]))

    n_kept = len(boot_vals['genbio_top4']['AUC'])
    print(f'\nBootstrap iterations kept: {n_kept}/{N_BOOT}')

    def ci(vals):
        arr = np.array(vals)
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5),
                float(np.mean(arr)))

    print('\n=== 95% CIs (1000 iters, paired slide resample) ===')
    print(f'{"model":14s} {"metric":7s}  {"point":>7s}  {"95% CI":>22s}')
    pts = {m: dict(zip(metric_names, point_metrics(m))) for m in cols}
    for m in cols:
        for k in metric_names:
            lo, hi, mean = ci(boot_vals[m][k])
            print(f'  {m:14s} {k:7s}  {pts[m][k]:7.4f}  '
                  f'[{lo:.4f}, {hi:.4f}]')

    print('\n=== Paired delta: genbio_top4 - mc_single ===')
    print(f'{"metric":7s}  {"delta":>9s}  {"95% CI":>22s}  {"p(delta<=0)":>12s}')
    for k in metric_names:
        diffs = (np.array(boot_vals['genbio_top4'][k])
                 - np.array(boot_vals['mc_single'][k]))
        lo = np.percentile(diffs, 2.5); hi = np.percentile(diffs, 97.5)
        pval = float((diffs <= 0).mean())
        print(f'{k:7s}  {diffs.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  '
              f'{pval:12.4f}')

    print('\n=== Per-site AUC (point): top4 - mc_single ===')
    for h in FOLDS:
        y, p_top = by_model['genbio_top4'][h]
        _, p_ref = by_model['mc_single'][h]
        a_top = roc_auc_score(y, p_top); a_ref = roc_auc_score(y, p_ref)
        print(f'  {h:12s} top4={a_top:.4f}  mc={a_ref:.4f}  '
              f'delta={a_top - a_ref:+.4f}')


if __name__ == '__main__':
    main()
