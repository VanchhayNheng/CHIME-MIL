"""Compute per-site mean GenBio-PathFM feature vectors at 392-px patch size.

Mirrors compute_site_means_genbio.py but with H5_DIR pointed at the 392-px
features and OUT_PATH at the test_392_meancenter folder.
"""
import os, re, csv, json, time
import numpy as np
import h5py

CSV_PATH = '/path/to/data/dataset_csv/tumor_vs_normal_dummy_clean.csv'
H5_DIR   = '/path/to/data/GENBIO_PATHFM_FEATURES_392SMOKE/mag20x_site_a_392/h5_files'
OUT_PATH = '/path/to/data/CHIME_MIL_genbio/test_392_meancenter/site_means_genbio_392.npz'
SITES = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']


def hospital(slide_id):
    s = slide_id
    if re.match(r'^(SC[-_]01)', s):           return 'Site_A'
    if re.match(r'^(SC[-_]02)', s):           return 'Site_E'
    if re.match(r'^(SC[-_]04)', s):           return 'Site_B'
    if re.match(r'^(SC-3-|GC-3-|SC-7-)', s):  return 'Site_D'
    if re.match(r'^(SC[-_]03)', s):           return 'Site_C'
    return None


rows = list(csv.DictReader(open(CSV_PATH)))
sum_vec = {s: np.zeros(4608, dtype=np.float64) for s in SITES}
patch_count = {s: 0 for s in SITES}
slide_count = {s: 0 for s in SITES}
skipped = 0

t0 = time.time()
for i, r in enumerate(rows):
    site = hospital(r['slide_id'])
    if site is None:
        skipped += 1
        continue
    p = os.path.join(H5_DIR, f"{r['slide_id']}.h5")
    if not os.path.exists(p):
        skipped += 1
        continue
    try:
        with h5py.File(p, 'r') as f:
            feats = np.array(f['features'], dtype=np.float32)
    except Exception as e:
        print(f'  bad h5 {r["slide_id"]}: {e}')
        skipped += 1
        continue
    if feats.ndim != 2 or feats.shape[0] == 0:
        skipped += 1
        continue
    sum_vec[site] += feats.sum(0).astype(np.float64)
    patch_count[site] += int(feats.shape[0])
    slide_count[site] += 1
    if (i + 1) % 500 == 0:
        print(f'  processed {i+1}/{len(rows)} (elapsed {time.time()-t0:.1f}s)')

means = {s: (sum_vec[s] / patch_count[s]).astype(np.float32)
         for s in SITES if patch_count[s] > 0}

print('\nPer-site stats:')
print(f'{"site":12s} {"slides":>7s} {"patches":>10s} {"mean_norm":>10s}')
for s in SITES:
    n = np.linalg.norm(means[s]) if s in means else float('nan')
    print(f'{s:12s} {slide_count[s]:7d} {patch_count[s]:10d} {n:10.3f}')
print(f'skipped rows: {skipped}')

np.savez(OUT_PATH,
         slide_counts=json.dumps(slide_count),
         patch_counts=json.dumps(patch_count),
         **means)
print(f'\nWrote {OUT_PATH}')

print('\nPatch-weighted per-site mean pairwise cosine sims:')
print(f'{"":12s}', ' '.join(f'{s[:6]:>8s}' for s in SITES))
for si in SITES:
    if si not in means:
        print(f'{si:12s}', 'no data'); continue
    row = []
    for sj in SITES:
        if sj not in means:
            row.append('     -  ')
        else:
            u, v = means[si], means[sj]
            c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
            row.append(f'{c:8.4f}')
    print(f'{si:12s}', ' '.join(row))
