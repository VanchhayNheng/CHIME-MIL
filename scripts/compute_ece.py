"""T6: calibration of the CHIME-MIL mc_sa headline.

Reconstructs the headline fused predictions (256 + 392 prob-average, per seed
in {42,123,7}, exactly as scripts/collate_per_site_ablation.py) and reports
TWO 15-bin Expected Calibration Error definitions per held-out fold,
seed-averaged as the headline AUC is:

  * top-label ECE  (Guo et al. 2017): confidence=max(p,1-p) vs correctness
                    -- the paper's primary 'ECE' number.
  * positive-class reliability: bin by p, compare to empirical positive rate
                    -- the curve plotted in fig11_reliability.

Also writes the positive-class reliability diagram (PDF + PNG).
"""
import json, glob, statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '.'
MCMS = f'{ROOT}/seed_sweep_mc_sa_multiscale/results'
FIGDIR = f'{ROOT}/analysis'
HOSPS = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']
SEEDS = [42, 123, 7]
NB = 15


def collect(d):
    f = {}
    for s in sorted(glob.glob(f'{d}/chime_loso_summary_*.json')):
        for x in json.load(open(s)).get('folds', []):
            f[x.get('test_hospital')] = x
    return f


def fused(h, sd):
    a = collect(f'{MCMS}/seed{sd}_scale256').get(h)
    b = collect(f'{MCMS}/seed{sd}_scale392').get(h)
    if not a or not b:
        return None
    pa, pb = np.asarray(a['probs'], float), np.asarray(b['probs'], float)
    ya, yb = np.asarray(a['labels'], int), np.asarray(b['labels'], int)
    if len(pa) != len(pb) or not np.array_equal(ya, yb):
        return None
    return 0.5 * (pa + pb), ya


def _binned(vals, ref, nb=NB):
    ed = np.linspace(0, 1, nb + 1)
    e, N = 0.0, len(vals)
    for i in range(nb):
        m = (vals >= ed[i]) & (vals < ed[i + 1]) if i < nb - 1 \
            else (vals >= ed[i]) & (vals <= ed[i + 1])
        if m.sum():
            e += m.sum() / N * abs(ref[m].mean() - vals[m].mean())
    return e


def ece_toplabel(p, y):
    conf = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    return _binned(conf, correct)


def ece_binary(p, y):
    return _binned(p, y.astype(float))


def reliability(p, y, nb=NB):
    ed = np.linspace(0, 1, nb + 1)
    xs, ys, ws = [], [], []
    for i in range(nb):
        m = (p >= ed[i]) & (p < ed[i + 1]) if i < nb - 1 else (p >= ed[i]) & (p <= ed[i + 1])
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ws.append(m.sum())
    return np.array(xs), np.array(ys), np.array(ws)


pf_top, pf_bin = {}, {}
PP, YY = [], []
for h in HOSPS:
    et, eb = [], []
    for sd in SEEDS:
        r = fused(h, sd)
        if r is None:
            continue
        p, y = r
        et.append(ece_toplabel(p, y)); eb.append(ece_binary(p, y))
        if sd == 42:
            PP.append(p); YY.append(y)
    if et:
        pf_top[h] = st.mean(et); pf_bin[h] = st.mean(eb)

P, Y = np.concatenate(PP), np.concatenate(YY)
vt, vb = list(pf_top.values()), list(pf_bin.values())
print('site         topECE   posClassGap')
for h in HOSPS:
    if h in pf_top:
        print(f'{h:11s}  {pf_top[h]:.4f}   {pf_bin[h]:.4f}')
print(f'mean(5)      {st.mean(vt):.4f}+/-{st.pstdev(vt):.4f}   '
      f'{st.mean(vb):.4f}+/-{st.pstdev(vb):.4f}')
print(f'pooled seed42 N={len(Y)}: topECE={ece_toplabel(P,Y):.4f}  '
      f'posGap={ece_binary(P,Y):.4f}')

xs, ys, ws = reliability(P, Y)
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.plot([0, 1], [0, 1], '--', color='gray', lw=1, label='Perfect calibration')
ax.plot(xs, ys, 'o-', color='#1f4e79', lw=1.8, ms=5, label='CHIME-MIL (mc\\_sa)')
ax.scatter(xs, ys, s=ws / ws.max() * 120, color='#1f4e79', alpha=0.25)
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Empirical positive-class frequency')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title(f'Positive-class reliability (top-label ECE = {ece_toplabel(P, Y):.3f})')
ax.legend(loc='upper left', fontsize=8, frameon=False)
ax.set_aspect('equal'); fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{FIGDIR}/fig11_reliability.{ext}', dpi=200, bbox_inches='tight')
print(f'figure -> {FIGDIR}/fig11_reliability.{{pdf,png}}')
print(json.dumps({'topECE_mean': round(st.mean(vt), 4),
                   'topECE_std': round(st.pstdev(vt), 4),
                   'posGap_mean': round(st.mean(vb), 4),
                   'per_site_topECE': {k: round(v, 4) for k, v in pf_top.items()}}))
