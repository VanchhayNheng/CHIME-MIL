"""Per-site component ablation collation for CHIME-MIL (Stanford reviewer T5).

Decomposes the alignment / selection-criterion / multi-scale axes of the
CHIME-MIL headline, reporting per-held-out-site test AUC for each condition.

Single-scale conditions (seed 42, 256 px) are read from the original
controlled ablation series under results/results_loso_*/fold_*/result.json,
so the deltas between them are clean (same seed, same code era).

The headline row (mc_sa multi-scale) reuses the exact prob-average fusion
of analyze_fusion.py: per held-out site, fuse the 256 px and 392 px legs of
each seed in {42,123,7}, recompute AUC, then average across seeds. This
reproduces the paper headline (0.8374).

Emits a CSV and a LaTeX table body to stdout.
"""
import json, glob, statistics as st
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = '.'
RES = f'{ROOT}/results'
MCMS = f'{ROOT}/seed_sweep_mc_sa_multiscale/results'

# Display order = dataset-table order
HOSPS = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']
SEEDS = [42, 123, 7]

# label -> single-scale (seed42, 256px) result dir
SINGLE = {
    'Base (no alignment, no strat-val)': 'results_loso_genbio_20260420',
    '+ Cross-site alignment (MC)': 'results_loso_meancenter_genbio_20260421',
    '+ Stratified-val selection (no MC)': 'results_loso_sa_stratified_genbio_20260421',
    '+ MC + strat-val (mc\\_sa, 256\\,px)': 'results_loso_mc_sa_genbio_20260421',
}


def per_site_single(run_dir):
    """{hospital: test_auc} from per-fold result.json in a single-scale run."""
    out = {}
    for rj in glob.glob(f'{RES}/{run_dir}/fold_*/result.json'):
        d = json.load(open(rj))
        out[d['test_hospital']] = float(d['test_auc'])
    return out


def collect_summary(d):
    """{hospital: fold_dict} deduped across the parallel-fold summary jsons."""
    folds = {}
    for s in sorted(glob.glob(f'{d}/chime_loso_summary_*.json')):
        for f in json.load(open(s)).get('folds', []):
            folds[f.get('test_hospital')] = f
    return folds


def fuse_auc(f256, f392):
    """Prob-average fusion AUC for one held-out site, or None on mismatch."""
    p256 = np.asarray(f256.get('probs', []), float)
    p392 = np.asarray(f392.get('probs', []), float)
    y256 = np.asarray(f256.get('labels', []), int)
    y392 = np.asarray(f392.get('labels', []), int)
    if len(p256) == 0 or len(p256) != len(p392) or not np.array_equal(y256, y392):
        return None
    return roc_auc_score(y256, 0.5 * (p256 + p392))


def headline_per_site():
    """{hospital: mean_fused_auc} for mc_sa multi-scale, averaged over seeds."""
    per_seed = {h: [] for h in HOSPS}
    for sd in SEEDS:
        f256 = collect_summary(f'{MCMS}/seed{sd}_scale256')
        f392 = collect_summary(f'{MCMS}/seed{sd}_scale392')
        for h in HOSPS:
            if h in f256 and h in f392:
                a = fuse_auc(f256[h], f392[h])
                if a is not None:
                    per_seed[h].append(a)
    return {h: st.mean(v) for h, v in per_seed.items() if v}


def fmt(x):
    return f'{x:.4f}' if x is not None else '--'


def main():
    rows = []
    for label, rd in SINGLE.items():
        ps = per_site_single(rd)
        m = st.mean([ps[h] for h in HOSPS if h in ps])
        rows.append((label, ps, m, False))

    hl_auc = headline_per_site()
    hl_mean = st.mean([hl_auc[h] for h in HOSPS if h in hl_auc])
    rows.append(('\\textbf{Headline (mc\\_sa, 256$\\oplus$392, seed-mean)}',
                 hl_auc, hl_mean, True))

    print('CSV')
    print('condition,' + ','.join(HOSPS) + ',Mean')
    for label, ps, m, _ in rows:
        cells = [fmt(ps.get(h)) for h in HOSPS]
        print(f'"{label}",' + ','.join(cells) + f',{m:.4f}')

    print('\nLATEX')
    for label, ps, m, hd in rows:
        cells = ' & '.join(fmt(ps.get(h)) for h in HOSPS)
        line = f'{label} & {cells} & {m:.4f} \\\\'
        if hd:
            print('\\midrule')
        print(line)

    seed_means = []
    for sd in SEEDS:
        f256 = collect_summary(f'{MCMS}/seed{sd}_scale256')
        f392 = collect_summary(f'{MCMS}/seed{sd}_scale392')
        a = [fuse_auc(f256[h], f392[h]) for h in HOSPS if h in f256 and h in f392]
        a = [x for x in a if x is not None]
        if a:
            seed_means.append(st.mean(a))
    if seed_means:
        print(f'\n% headline seed-mean AUC = {st.mean(seed_means):.4f} '
              f'+/- {st.stdev(seed_means):.4f} (n={len(seed_means)} seeds)')


if __name__ == '__main__':
    main()
