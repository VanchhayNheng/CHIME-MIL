"""Search equal-weight ensemble subsets over the 6 GenBio-PathFM LOSO runs.

For each non-empty subset of members, average per-fold P(class=1) probs and
compute per-fold test AUC, then mean over folds. Ranks subsets and writes a
CSV for downstream reporting (analog of `ensemble_search.py` for the UNI
side, plus a CSV instead of stdout-only).
"""
import csv
import itertools
import json
import os
from sklearn.metrics import roc_auc_score
import numpy as np

BASE = '/path/to/data/CHIME_MIL_genbio'
FOLDS = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']
OUT_CSV = f'{BASE}/ensemble_search_genbio_results.csv'

# (tag, result_dir). The duplicate run results_loso_genbio_20260420
# (bit-identical per-fold AUCs to soft_assign_genbio_20260421) is excluded.
MEMBERS = [
    ('mc',        'results_loso_meancenter_genbio_20260421'),
    ('mc_sa',     'results_loso_mc_sa_genbio_20260421'),
    ('sa_strat',  'results_loso_sa_stratified_genbio_20260421'),
    ('mc_sa_w3',  'results_loso_mc_sa_warmup3_genbio_20260421'),
    ('e09',       'results_loso_e09_fixed_genbio_20260421'),
    ('sa',        'results_loso_soft_assign_genbio_20260421'),
]


def _probs_cls1(r):
    p = np.array(r['probs'], dtype=float)
    return p[:, 1] if p.ndim == 2 else p


def _load_member(dir_rel):
    """Return {hospital: (labels, probs_cls1)} for a member dir."""
    out = {}
    for i, h in enumerate(FOLDS, start=1):
        path = f'{BASE}/{dir_rel}/fold_{i}_{h}/result.json'
        r = json.load(open(path))
        out[h] = (np.array(r['labels']), _probs_cls1(r))
    return out


def main():
    cache = {tag: _load_member(d) for tag, d in MEMBERS}

    # Sanity: labels must agree across members for each fold.
    for h in FOLDS:
        ref = cache[MEMBERS[0][0]][h][0]
        for tag, _ in MEMBERS[1:]:
            if not np.array_equal(cache[tag][h][0], ref):
                raise SystemExit(f'label mismatch on fold {h} for member {tag}')

    # Singleton AUC sanity print.
    print('Singleton (single-member) mean LOSO AUCs:')
    for tag, d in MEMBERS:
        aucs = [roc_auc_score(*cache[tag][h]) for h in FOLDS]
        print(f'  {tag:10s} ({d:50s}) mean={np.mean(aucs):.4f}  '
              f'per_fold={[round(a,4) for a in aucs]}')
    print()

    # Enumerate all 2^N - 1 subsets.
    tags = [t for t, _ in MEMBERS]
    rows = []
    for k in range(1, len(tags) + 1):
        for subset in itertools.combinations(tags, k):
            fold_aucs = []
            for h in FOLDS:
                y = cache[subset[0]][h][0]
                p = np.mean(np.stack([cache[t][h][1] for t in subset], 0), 0)
                fold_aucs.append(roc_auc_score(y, p))
            rows.append({
                'subset': '+'.join(subset),
                'n_members': len(subset),
                'mean_auc': float(np.mean(fold_aucs)),
                **{f'auc_{h}': float(a) for h, a in zip(FOLDS, fold_aucs)},
            })

    rows.sort(key=lambda r: r['mean_auc'], reverse=True)

    print('Top 15 subsets by mean LOSO AUC:')
    print(f'{"rank":>4s}  {"n":>2s}  {"mean":>7s}  '
          f'{"Sev":>6s} {"Dan":>6s} {"Bun":>6s} {"Mok":>6s} {"Kei":>6s}  subset')
    for i, r in enumerate(rows[:15], start=1):
        print(f'{i:>4d}  {r["n_members"]:>2d}  {r["mean_auc"]:7.4f}  '
              f'{r["auc_Site_A"]:6.4f} {r["auc_Site_B"]:6.4f} '
              f'{r["auc_Site_C"]:6.4f} {r["auc_Site_D"]:6.4f} '
              f'{r["auc_Site_E"]:6.4f}  {r["subset"]}')

    print('\nBottom 5:')
    for r in rows[-5:]:
        print(f'  n={r["n_members"]} mean={r["mean_auc"]:.4f}  {r["subset"]}')

    # Write CSV
    fieldnames = ['subset', 'n_members', 'mean_auc'] + [f'auc_{h}' for h in FOLDS]
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'\nWrote: {OUT_CSV}')

    # Per-site AUC of best subset vs best single member.
    best = rows[0]
    best_single_tag = max(tags, key=lambda t: np.mean(
        [roc_auc_score(*cache[t][h]) for h in FOLDS]))
    print(f'\nBest subset:        {best["subset"]}  mean={best["mean_auc"]:.4f}')
    print(f'Best single member: {best_single_tag}  mean='
          f'{np.mean([roc_auc_score(*cache[best_single_tag][h]) for h in FOLDS]):.4f}')
    print('Per-site AUC (best_subset - best_single):')
    for h in FOLDS:
        a_subset = best[f'auc_{h}']
        a_single = roc_auc_score(*cache[best_single_tag][h])
        print(f'  {h:12s} subset={a_subset:.4f}  single={a_single:.4f}  '
              f'delta={a_subset - a_single:+.4f}')


if __name__ == '__main__':
    main()
