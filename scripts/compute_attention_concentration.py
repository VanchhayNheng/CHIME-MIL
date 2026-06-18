#!/usr/bin/env python3
"""Compute per-slide attention concentration (Gini, top-10 mass) for the 4
representative WSIs in fig_heatmaps, using the headline mc model checkpoints.

Output: results/attention_concentration.csv
Columns: slide_id, hospital, scale, gt_label, n_patches, pred_prob_tumor, gini, rho10
"""
import argparse, csv, os, re, sys
from pathlib import Path
import numpy as np
import torch
import h5py

ROOT = Path('.')
sys.path.insert(0, str(ROOT))
from chime_mil import CHIME_MIL

FEATURE_DIRS = {
    256: '/path/to/data/GENBIO_PATHFM_FEATURES/mag20x/h5_files',
    392: '/path/to/data/CHIME_MIL_v2_genbio/test_392_mc_sa/feature_root/mag20x/h5_files',
}
SITE_MEANS_PATH = ROOT / 'site_means_genbio.npz'

# 4 representative slides from fig_heatmaps.pdf
SLIDES = [
    ('SC-01-0977', 0, 'malignant'),  # Site_A tumor
    ('SC_02_0134', 0, 'malignant'),  # Site_E tumor
    ('SC-01-0845', 1, 'normal'),     # Site_A normal
    ('SC_02_0080', 1, 'normal'),     # Site_E normal
]

HOSPITAL_REGEX = [
    (re.compile(r'^(SC[-_]01)'),           'Site_A'),
    (re.compile(r'^(SC[-_]02)'),           'Site_E'),
    (re.compile(r'^(SC[-_]04)'),           'Site_B'),
    (re.compile(r'^(SC-3-|GC-3-|SC-7-)'), 'Site_D'),
    (re.compile(r'^(SC[-_]03)'),           'Site_C'),
]

def get_hospital(slide_id):
    for pat, name in HOSPITAL_REGEX:
        if pat.match(slide_id):
            return name
    raise ValueError(f'unknown hospital prefix for {slide_id}')

# fold ordering as used by the headline trainer
FOLD_FOR_HOSPITAL = {
    'Site_A':  1,
    'Site_B':    2,
    'Site_C': 3,
    'Site_D':    4,
    'Site_E':   5,
}

def gini(x):
    """Gini coefficient of a non-negative vector summing to 1."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0:
        return float('nan')
    s = x.sum()
    if s <= 0:
        return float('nan')
    # standard discrete Gini: G = (sum_{i=1..n} (2i - n - 1) x_i) / (n * sum x)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(((2*idx - n - 1) * x).sum() / (n * s))

def rho_top(x, frac=0.10):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return float('nan')
    k = max(1, int(np.ceil(frac * n)))
    top = np.partition(x, -k)[-k:]
    s = x.sum()
    return float(top.sum() / s) if s > 0 else float('nan')

def load_checkpoint(scale, seed, fold_idx, hospital):
    return ROOT / 'seed_sweep_mc_sa_multiscale' / 'results' / f'seed{seed}_scale{scale}' / f'fold_{fold_idx}_{hospital}' / 'best_model.pth'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scale', type=int, default=256, choices=[256, 392])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'attention_concentration.csv'))
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    site_means = np.load(SITE_MEANS_PATH)
    feat_dir = Path(FEATURE_DIRS[args.scale])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows = []
    for slide_id, label, label_name in SLIDES:
        hospital = get_hospital(slide_id)
        fold_idx = FOLD_FOR_HOSPITAL[hospital]
        ckpt_path = load_checkpoint(args.scale, args.seed, fold_idx, hospital)
        if not ckpt_path.exists():
            print(f'[MISSING ckpt] {ckpt_path}', file=sys.stderr)
            continue

        # features + coords
        h5 = h5py.File(feat_dir / f'{slide_id}.h5', 'r')
        feats = np.asarray(h5['features'], dtype=np.float32)
        coords = np.asarray(h5['coords'], dtype=np.float32)
        h5.close()

        # site-mean center
        feats = feats - site_means[hospital].astype(np.float32)

        # build model with the headline config
        model = CHIME_MIL(
            input_dim=4608, hidden_dim=256, num_classes=2,
            num_regions=16, dropout=0.5,
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()

        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0).to(device)        # [1, N, 4608]
            c = torch.from_numpy(coords).unsqueeze(0).to(device)       # [1, N, 2]
            out = model(x, c)
            attn = out['patch_importance'].squeeze(0).cpu().numpy()    # [N], sums to ~1
            logits = out['logits'].squeeze(0).cpu().numpy()
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            p_tumor = float(probs[0])  # class 0 = tumor

        g = gini(attn)
        r10 = rho_top(attn, 0.10)
        n = int(attn.size)
        print(f'{slide_id:18s}  hosp={hospital:11s}  N={n:5d}  label={label_name:9s}  p(tumor)={p_tumor:.3f}  G={g:.4f}  rho10={r10:.4f}')
        rows.append({
            'slide_id': slide_id, 'hospital': hospital,
            'scale': args.scale, 'gt_label': label_name,
            'n_patches': n, 'pred_prob_tumor': round(p_tumor, 4),
            'gini': round(g, 4), 'rho10': round(r10, 4),
        })

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {out_path}  ({len(rows)} rows)')

if __name__ == '__main__':
    main()
