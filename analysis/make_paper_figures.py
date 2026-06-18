"""Generate paper figures for the mean-centering (CORAL) experiment.

Produces three PNGs in CHIME_MIL/figures/:
  1. fig_per_site_auc_bars.png      - Per-site AUC, grouped by method
  2. fig_stain_shift_diagnostic.png - Cross-class cosine-gap per site
  3. fig_site_c_before_after.png- Site_C soft_assign baseline vs meancenter

Also writes a supplementary per-fold table to tables/supp_per_fold.csv.
All data comes from frozen or 2026-04-14 retrain result.jsons — no GPU required.
"""
import os, json, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- de-identify hospital names -> Site A-E (matches site_relabel.py) ---
_SITE_MAP = {"Site_A": "Site A", "Site_B": "Site B", "Site_C": "Site C", "Site_D": "Site D", "Site_E": "Site E"}
def to_site(_n):
    return _SITE_MAP.get(_n, _n)
from sklearn.metrics import roc_auc_score
from diagnose_stain_shift import (hospital, load_slide_feature, SITES,
                                  CSV_PATH, H5_DIR, N_PER_CLASS_PER_SITE)

BASE = "."
FIG_DIR = f"{BASE}/figures"
TABLE_DIR = f"{BASE}/tables"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

FOLDS = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E"]
STAMP = '20260414'


def probs(r):
    p = np.array(r['probs'])
    return p[:, 1] if p.ndim == 2 else p


def load_rerun(variant, hospital):
    p = glob.glob(f"{BASE}/results_loso_{variant}_{STAMP}/"
                  f"fold_*_{hospital}/result.json")
    return json.load(open(p[0]))


def load_frozen(dirname, hospital):
    p = glob.glob(f'{BASE}/{dirname}/fold_*_{hospital}/result.json')
    return json.load(open(p[0]))


# -------- assemble per-site AUC for all methods ------------------------------
methods = [
    ('ABMIL (paper)', 'frozen',
     '/path/to/data/ExtractWSI/abmil_loso_20260330_full'),
    ('soft_assign (frozen)', 'frozen',
     f'{BASE}/results_loso_soft_assign'),
    ('e09_fixed (frozen)', 'frozen',
     f'{BASE}/results_loso_e09_fixed_20x'),
    ('baseline_sa (rerun)', 'rerun', 'baseline_soft_assign'),
    ('baseline_e09 (rerun)', 'rerun', 'baseline_e09_fixed'),
    ('meancenter_sa', 'rerun', 'meancenter_soft_assign'),
    ('meancenter_e09', 'rerun', 'meancenter_e09_fixed'),
    ('meancenter_sa+e09', 'rerun_ens',
     ['meancenter_soft_assign', 'meancenter_e09_fixed']),
    ('all4 ensemble', 'rerun_ens',
     ['baseline_soft_assign', 'baseline_e09_fixed',
      'meancenter_soft_assign', 'meancenter_e09_fixed']),
]

per_method_aucs = {}
for name, kind, src in methods:
    row = []
    for h in FOLDS:
        if kind == 'frozen':
            try:
                # src is an absolute path; look up fold_*_{hospital}/result.json
                paths = glob.glob(f'{src}/fold_*_{h}/result.json')
                if not paths:
                    row.append(np.nan); continue
                r = json.load(open(paths[0]))
                y = np.array(r['labels']); pr = probs(r)
                row.append(roc_auc_score(y, pr))
            except Exception as e:
                row.append(np.nan)
        elif kind == 'rerun':
            r = load_rerun(src, h)
            y = np.array(r['labels']); pr = probs(r)
            row.append(roc_auc_score(y, pr))
        elif kind == 'rerun_ens':
            rs = [load_rerun(m, h) for m in src]
            y = np.array(rs[0]['labels'])
            pr = np.mean(np.stack([probs(r) for r in rs], 0), 0)
            row.append(roc_auc_score(y, pr))
    per_method_aucs[name] = row
    print(name, [f'{a:.4f}' if a == a else 'NA' for a in row],
          f'mean={np.nanmean(row):.4f}')

# ------ FIG 1: Per-site AUC bars ---------------------------------------------
key_methods = ['baseline_sa (rerun)', 'meancenter_sa',
               'baseline_e09 (rerun)', 'meancenter_e09',
               'meancenter_sa+e09', 'all4 ensemble']
colors = ['#888888', '#1f77b4', '#aaaaaa', '#ff7f0e', '#2ca02c', '#d62728']
fig, ax = plt.subplots(figsize=(11, 5.2))
x = np.arange(len(FOLDS))
w = 0.13
for i, name in enumerate(key_methods):
    vals = per_method_aucs[name]
    ax.bar(x + (i - 2.5) * w, vals, w, label=name, color=colors[i])
for i, h in enumerate(FOLDS):
    ax.axvspan(i - 0.5, i + 0.5, alpha=0.03,
               color='red' if h in ('Site_C', 'Site_A') else 'blue')
ax.set_xticks(x); ax.set_xticklabels([to_site(_x) for _x in FOLDS])
ax.set_ylabel('AUC (held-out site)')
ax.set_ylim(0.70, 0.90)
ax.set_title('Per-site LOSO AUC: baseline vs CORAL mean-centered '
             '(2026-04-14 retrain)')
ax.legend(loc='lower right', ncol=2, fontsize=8, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_per_site_auc_bars.png', dpi=150)
plt.close()
print('wrote fig_per_site_auc_bars.png')

# ------ FIG 2: Stain-shift diagnostic (cross-class cosine gap) ---------------
# Recompute diagnostic to avoid depending on stdout.
import csv as _csv, random, re
from collections import defaultdict
import h5py
random.seed(42)
rows = list(_csv.DictReader(open(CSV_PATH)))
per_site_class = defaultdict(list)
for r in rows:
    s = hospital(r['slide_id'])
    if s is None: continue
    cls = 'tumor' if r['label'] == 'm' else 'normal'
    per_site_class[(s, cls)].append(r['slide_id'])
site_class_feats = {}
for site in SITES:
    for cls in ('tumor', 'normal'):
        ids = per_site_class[(site, cls)][:]
        random.shuffle(ids)
        feats = []
        for sid in ids[:N_PER_CLASS_PER_SITE]:
            f = load_slide_feature(sid)
            if f is not None: feats.append(f)
        site_class_feats[(site, cls)] = np.stack(feats, 0) if feats \
            else np.zeros((0, 1024))

centroid = {k: v.mean(0) for k, v in site_class_feats.items() if len(v) > 0}
def cos(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))

gaps_normal = []
gaps_tumor = []
for held in SITES:
    others = [s for s in SITES if s != held]
    nh = centroid[(held, 'normal')]
    on = np.mean([cos(nh, centroid[(s, 'normal')]) for s in others])
    ot = np.mean([cos(nh, centroid[(s, 'tumor')])  for s in others])
    gaps_normal.append(on - ot)
    th = centroid[(held, 'tumor')]
    tt = np.mean([cos(th, centroid[(s, 'tumor')])  for s in others])
    tn = np.mean([cos(th, centroid[(s, 'normal')]) for s in others])
    gaps_tumor.append(tt - tn)

fig, ax = plt.subplots(figsize=(8, 4.2))
xs = np.arange(len(SITES))
w = 0.38
b1 = ax.bar(xs - w/2, gaps_normal, w, label='normal(held) - gap',
            color='#1f77b4')
b2 = ax.bar(xs + w/2, gaps_tumor, w, label='tumor(held) - gap',
            color='#ff7f0e')
ax.axhline(0, color='k', lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels([to_site(_x) for _x in SITES])
ax.set_ylabel('Cross-class cosine gap\n(same-class - other-class, averaged)')
ax.set_title('Feature-space class separation per held-out site\n'
             '(smaller/negative gap = class confusion → cross-site shift)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
for rect, v in zip(b1, gaps_normal):
    ax.text(rect.get_x() + rect.get_width()/2,
            v + (0.001 if v >= 0 else -0.002),
            f'{v:+.3f}', ha='center', fontsize=8,
            va='bottom' if v >= 0 else 'top')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_stain_shift_diagnostic.png', dpi=150)
plt.close()
print('wrote fig_stain_shift_diagnostic.png')

# ------ FIG 3: Site_C before/after ---------------------------------------
labels = ['baseline\nsoft_assign', 'meancenter\nsoft_assign',
          'baseline\ne09_fixed', 'meancenter\ne09_fixed',
          'baseline\nensemble', 'meancenter\nensemble', 'all4\nensemble']
vals = [
    per_method_aucs['baseline_sa (rerun)'][2],
    per_method_aucs['meancenter_sa'][2],
    per_method_aucs['baseline_e09 (rerun)'][2],
    per_method_aucs['meancenter_e09'][2],
    # baseline ensemble on fold 2 (Site_C)
    0.0, 0.0, 0.0,
]
# compute ensembles on Site_C
rs_base = [load_rerun('baseline_soft_assign', 'Site_C'),
           load_rerun('baseline_e09_fixed', 'Site_C')]
y = np.array(rs_base[0]['labels'])
pb = np.mean(np.stack([probs(r) for r in rs_base], 0), 0)
vals[4] = roc_auc_score(y, pb)
rs_mc = [load_rerun('meancenter_soft_assign', 'Site_C'),
         load_rerun('meancenter_e09_fixed', 'Site_C')]
pm = np.mean(np.stack([probs(r) for r in rs_mc], 0), 0)
vals[5] = roc_auc_score(y, pm)
vals[6] = per_method_aucs['all4 ensemble'][2]

colors_bf = ['#888888', '#1f77b4', '#aaaaaa', '#ff7f0e',
             '#666666', '#2ca02c', '#d62728']
fig, ax = plt.subplots(figsize=(9, 4.5))
xs = np.arange(len(labels))
bars = ax.bar(xs, vals, color=colors_bf)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.003,
            f'{v:.4f}', ha='center', fontsize=9)
# mark deltas
def arrow(i_from, i_to):
    y0 = vals[i_from]; y1 = vals[i_to]
    d = y1 - y0
    ax.annotate(f'Δ+{d:.4f}',
                xy=(xs[i_to], y1), xytext=(xs[i_from] + 0.05, y0 - 0.01),
                arrowprops=dict(arrowstyle='->', color='#2a7a2a'),
                fontsize=9, color='#2a7a2a')
arrow(0, 1); arrow(2, 3); arrow(4, 5)
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('AUC (held-out: Site C)')
ax.set_ylim(0.75, 0.85)
ax.set_title('Site C fold: CORAL mean-centering effect')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig_site_c_before_after.png', dpi=150)
plt.close()
print('wrote fig_site_c_before_after.png')

# ------ Supplementary per-fold CSV -------------------------------------------
with open(f'{TABLE_DIR}/supp_per_fold.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['method'] + FOLDS + ['mean'])
    for name, aucs in per_method_aucs.items():
        mean = np.nanmean(aucs)
        w.writerow([name] + [f'{a:.4f}' if a == a else 'NA' for a in aucs]
                   + [f'{mean:.4f}'])
print(f'wrote tables/supp_per_fold.csv')
