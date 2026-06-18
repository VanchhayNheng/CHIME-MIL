"""Compute per-site mean UNI feature vectors (patch-weighted).

For each site, we aggregate all patches from all of that site's slides and
compute the overall mean feature vector. Site identity is derived from
slide_id prefix via the authoritative mapping in CLAUDE.md.

Saves: {out}/site_means.npz  with keys:
  - 'Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E' -> (1024,)
  - 'slide_counts' -> dict str->int
  - 'patch_counts' -> dict str->int

This is an UNSUPERVISED statistic (uses features only, not labels), so it
is legitimate to compute from the full dataset including the LOSO held-out
site for test-time domain adaptation (CORAL-style first-moment alignment).
"""
import os, re, csv, json, time
import numpy as np
import h5py

CSV_PATH = '/path/to/data/dataset_csv/tumor_vs_normal_dummy_clean.csv'
H5_DIR   = '/path/to/data/GENBIO_PATHFM_FEATURES/mag20x/h5_files'
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'site_means_genbio.npz')
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

# accumulate per-site running sum and count
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
        print(f'  processed {i+1}/{len(rows)} '
              f'(elapsed {time.time()-t0:.1f}s)')

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

# Sanity: pairwise cosine sims of the centroids (patch-weighted, not slide-weighted)
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
