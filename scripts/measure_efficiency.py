"""T7: per-component parameter counts + single-slide inference latency.

Builds CHIME_MIL with the GenBio headline config and reports trainable
parameters per submodule, then times a batch-1 forward pass on a real
cached GenBio feature bag (H100, fp32).
"""
import glob, time, json
import numpy as np
import torch
import h5py
from chime_mil import CHIME_MIL  # grid CHIME_MIL = the model actually trained (soft-assign variant is dead code, see PRODUCTION_PUNCHLIST P0-5)

CFG = dict(input_dim=4608, hidden_dim=256, num_classes=2,
           num_regions=16, dropout=0.5)

model = CHIME_MIL(**CFG).eval()


def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


comps = ['patch_head', 'region_aggregator', 'region_head',
         'graph_head', 'graph_classifier', 'fusion']
total = nparams(model)
print('=== Per-component trainable parameters (single tower) ===')
acc = 0
for c in comps:
    n = nparams(getattr(model, c))
    acc += n
    print(f'{c:20s} {n:>10,d}  ({100.0*n/total:5.2f}%)')
print(f'{"(sum)":20s} {acc:>10,d}')
print(f'{"TOTAL (1 tower)":20s} {total:>10,d}  ({total/1e6:.3f} M)')
print(f'{"mc_sa (2 towers)":20s} {2*total:>10,d}  ({2*total/1e6:.3f} M)')

# ---- latency on a real bag ----
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev)
h5s = sorted(glob.glob('/path/to/data/'
                       'GENBIO_PATHFM_FEATURES/mag20x/h5_files/*.h5'))
# pick a mid-sized bag: sample 200 files' patch counts, take the one closest to median
import random
random.seed(0)
samp = random.sample(h5s, 200)
sizes = []
for f in samp:
    with h5py.File(f, 'r') as h:
        sizes.append((h['features'].shape[0], f))
sizes.sort()
med_n, med_f = sizes[len(sizes) // 2]
with h5py.File(med_f, 'r') as h:
    feats = torch.tensor(np.asarray(h['features']), dtype=torch.float32)
    coords = torch.tensor(np.asarray(h['coords']), dtype=torch.float32)
feats = feats.unsqueeze(0).to(dev)
coords = coords.unsqueeze(0).to(dev)

with torch.no_grad():
    for _ in range(5):  # warmup
        model(feats, coords)
    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(30):
        model(feats, coords)
    if dev == 'cuda':
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 30 * 1000.0

print(f'\n=== Inference latency (batch 1, {dev}, fp32) ===')
print(f'sampled-median bag: {med_f.split("/")[-1]}  N={med_n} patches')
print(f'single tower : {dt:.2f} ms/slide')
print(f'mc_sa (2 towers, +mean-subtract O(Nd) ~negligible): ~{2*dt:.2f} ms/slide')
print(json.dumps({'total_params_1tower': total, 'ms_1tower': round(dt, 2),
                  'median_N': med_n}))
