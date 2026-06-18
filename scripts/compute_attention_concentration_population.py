#!/usr/bin/env python3
"""Population-level attention concentration across all 5 LOSO folds.

For each held-out site (fold), loads that fold's best_model.pth (seed 42,
scale 256, mc), runs forward passes on every test slide of that site,
computes Gini and rho10 from PatchMILHead attention, and writes per-fold
summary plus per-slide CSV.

Output:
  results/attention_concentration_population.csv  (one row per slide)
  results/attention_concentration_summary.csv     (per-fold, per-class mean+/-std)
"""
import csv, os, re, sys
from pathlib import Path
import numpy as np
import torch
import h5py
import pandas as pd

ROOT = Path('.')
sys.path.insert(0, str(ROOT))
from chime_mil import CHIME_MIL

FEATURE_DIR = Path('/path/to/data/GENBIO_PATHFM_FEATURES/mag20x/h5_files')
SITE_MEANS_PATH = ROOT / 'site_means_genbio.npz'
CSV_PATH = '/path/to/data/dataset_csv/tumor_vs_normal_dummy_clean.csv'
SEED = 42
SCALE = 256

FOLDS = [
    (1, 'Site_A'),
    (2, 'Site_B'),
    (3, 'Site_C'),
    (4, 'Site_D'),
    (5, 'Site_E'),
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
    return None

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0: return float('nan')
    s = x.sum()
    if s <= 0: return float('nan')
    idx = np.arange(1, n+1, dtype=np.float64)
    return float(((2*idx - n - 1) * x).sum() / (n * s))

def rho_top(x, frac=0.10):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0: return float('nan')
    k = max(1, int(np.ceil(frac * n)))
    top = np.partition(x, -k)[-k:]
    s = x.sum()
    return float(top.sum()/s) if s > 0 else float('nan')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    site_means = np.load(SITE_MEANS_PATH)
    df = pd.read_csv(CSV_PATH)
    label_map = {'m': 0, 'nm': 1}  # 0=malignant 1=normal per paper convention
    rows = []

    for fold_idx, host in FOLDS:
        ckpt_path = ROOT / 'seed_sweep_mc_sa_multiscale' / 'results' / f'seed{SEED}_scale{SCALE}' / f'fold_{fold_idx}_{host}' / 'best_model.pth'
        if not ckpt_path.exists():
            print(f'[MISSING] {ckpt_path}', file=sys.stderr); continue

        model = CHIME_MIL(input_dim=4608, hidden_dim=256, num_classes=2, num_regions=16, dropout=0.5).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        model.eval()
        site_mu = site_means[host].astype(np.float32)

        test_df = df[df['slide_id'].apply(get_hospital) == host]
        print(f'\n=== fold {fold_idx} {host}: {len(test_df)} test slides ===')

        n_done = 0
        for _, r in test_df.iterrows():
            sid = r['slide_id']
            lbl_str = r.get('label', None)
            if lbl_str not in label_map: continue
            lbl = label_map[lbl_str]
            p = FEATURE_DIR / f'{sid}.h5'
            if not p.exists(): continue
            try:
                with h5py.File(p, 'r') as h5:
                    feats = np.asarray(h5['features'], dtype=np.float32)
                    coords = np.asarray(h5['coords'], dtype=np.float32)
            except Exception as e:
                print(f'  bad {sid}: {e}'); continue
            if feats.shape[0] < 2: continue
            feats = feats - site_mu
            with torch.no_grad():
                x = torch.from_numpy(feats).unsqueeze(0).to(device)
                c = torch.from_numpy(coords).unsqueeze(0).to(device)
                out = model(x, c)
                attn = out['patch_importance'].squeeze(0).cpu().numpy()
            rows.append({
                'slide_id': sid, 'hospital': host, 'fold': fold_idx,
                'gt_label_int': lbl,
                'gt_label': 'malignant' if lbl == 0 else 'normal',
                'n_patches': int(attn.size),
                'gini': round(gini(attn), 5),
                'rho10': round(rho_top(attn, 0.10), 5),
            })
            n_done += 1
            if n_done % 100 == 0:
                print(f'  {n_done}/{len(test_df)} done')

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # write per-slide
    out_dir = ROOT / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    per_slide = out_dir / 'attention_concentration_population.csv'
    with open(per_slide, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f'\nwrote {per_slide}  ({len(rows)} slides)')

    # per-fold per-class summary
    pdf = pd.DataFrame(rows)
    summary = pdf.groupby(['fold','hospital','gt_label']).agg(
        n=('slide_id','count'),
        gini_mean=('gini','mean'), gini_std=('gini','std'),
        rho10_mean=('rho10','mean'), rho10_std=('rho10','std'),
    ).round(4).reset_index()
    summary.to_csv(out_dir / 'attention_concentration_summary.csv', index=False)
    print(summary.to_string(index=False))

    # corpus pooled by gt_label
    print('\n=== corpus pooled ===')
    pooled = pdf.groupby('gt_label').agg(
        n=('slide_id','count'),
        gini_mean=('gini','mean'), gini_std=('gini','std'),
        rho10_mean=('rho10','mean'), rho10_std=('rho10','std'),
    ).round(4)
    print(pooled.to_string())

if __name__ == '__main__':
    main()
