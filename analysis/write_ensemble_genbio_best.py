"""Write the best GenBio Top-N ensemble (winner of `ensemble_search_genbio.py`)
to disk, mirroring the per-fold result.json schema produced by
`write_ensemble_dir.py` so downstream reporting works unchanged.

Winner (from ensemble_search_genbio_results.csv): mc+mc_sa+sa_strat+mc_sa_w3
Mean LOSO AUC: 0.8425 (vs best single 0.8362).
"""
import json
import os
import statistics as st
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

BASE = '/path/to/data/CHIME_MIL_genbio'
FOLDS = ['fold_1_Site_A', 'fold_2_Site_B', 'fold_3_Site_C',
         'fold_4_Site_D', 'fold_5_Site_E']

MEMBERS = [
    ('mc',       'results_loso_meancenter_genbio_20260421'),
    ('mc_sa',    'results_loso_mc_sa_genbio_20260421'),
    ('sa_strat', 'results_loso_sa_stratified_genbio_20260421'),
    ('mc_sa_w3', 'results_loso_mc_sa_warmup3_genbio_20260421'),
]

STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT = f'{BASE}/results_ensemble_genbio_top4_{STAMP}'


def _probs_cls1(r):
    p = np.array(r['probs'], dtype=float)
    return p[:, 1] if p.ndim == 2 else p


def main():
    os.makedirs(OUT, exist_ok=True)
    summary = {
        'method': 'ensemble_equal_avg',
        'members': [tag for tag, _ in MEMBERS],
        'source_dirs': {tag: d for tag, d in MEMBERS},
        'feature_backbone': 'GenBio-PathFM (4608-d)',
        'folds': {},
    }

    aucs, accs, bals, f1s = [], [], [], []
    for fold in FOLDS:
        rs = [json.load(open(f'{BASE}/{d}/{fold}/result.json'))
              for _, d in MEMBERS]
        ys = [np.array(r['labels']) for r in rs]
        for r, y in zip(rs[1:], ys[1:]):
            assert np.array_equal(y, ys[0]), f'label mismatch on {fold}'
        y = ys[0]
        ps = np.stack([_probs_cls1(r) for r in rs], axis=0)
        pE = ps.mean(axis=0)
        pred = (pE >= 0.5).astype(int)

        auc = roc_auc_score(y, pE)
        acc = accuracy_score(y, pred)
        bal = balanced_accuracy_score(y, pred)
        f1 = f1_score(y, pred)
        ppv = precision_score(y, pred, zero_division=0)
        sens = recall_score(y, pred)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        aucs.append(auc); accs.append(acc); bals.append(bal); f1s.append(f1)

        fold_out = f'{OUT}/{fold}'
        os.makedirs(fold_out, exist_ok=True)
        json.dump({
            'fold': rs[0]['fold'],
            'test_hospital': rs[0]['test_hospital'],
            'method': 'ensemble_equal_avg',
            'members': [tag for tag, _ in MEMBERS],
            'n_test': int(len(y)),
            'threshold': 0.5,
            'test_auc': float(auc),
            'test_acc': float(acc),
            'test_balanced_acc': float(bal),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'f1': float(f1),
            'ppv': float(ppv),
            'labels': y.tolist(),
            'probs': pE.tolist(),
            'preds': pred.tolist(),
            'member_aucs': {tag: float(r['test_auc'])
                            for (tag, _), r in zip(MEMBERS, rs)},
        }, open(f'{fold_out}/result.json', 'w'), indent=2)

        summary['folds'][fold] = {
            'test_auc': float(auc), 'test_acc': float(acc),
            'test_balanced_acc': float(bal), 'f1': float(f1),
            'n_test': int(len(y)),
        }
        print(f'{fold:22s} AUC={auc:.4f}  ACC={acc:.4f}  '
              f'BalAcc={bal:.4f}  F1={f1:.4f}')

    summary['mean'] = {
        'test_auc': round(st.mean(aucs), 4),
        'test_acc': round(st.mean(accs), 4),
        'test_balanced_acc': round(st.mean(bals), 4),
        'f1': round(st.mean(f1s), 4),
    }
    json.dump(summary, open(f'{OUT}/ensemble_summary.json', 'w'), indent=2)
    print()
    print(f'MEAN  AUC = {summary["mean"]["test_auc"]:.4f}')
    print(f'MEAN  ACC = {summary["mean"]["test_acc"]:.4f}')
    print(f'MEAN  BAL = {summary["mean"]["test_balanced_acc"]:.4f}')
    print(f'MEAN  F1  = {summary["mean"]["f1"]:.4f}')
    print(f'Wrote: {OUT}')


if __name__ == '__main__':
    main()
