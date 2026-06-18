"""Late-fusion gate: average per-slide probs from 256-px and 392-px
meancenter LOSO runs, recompute test AUC / BalAcc / Acc per fold and overall.

If fused_AUC vs max(256, 392) gain >= 0.005, multi-scale carries real
complementary signal and a tighter fusion (concat / two-tower) is justified.
"""
import json, os, statistics
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

import argparse
from pathlib import Path
_DEF = Path(__file__).resolve().parents[1] / 'results'
_ap = argparse.ArgumentParser()
_ap.add_argument('--pred_dir_256', default=str(_DEF / 'rerun' / 'mc_sa_256'))
_ap.add_argument('--pred_dir_392', default=str(_DEF / 'rerun' / 'mc_sa_392'))
ARGS, _ = _ap.parse_known_args()
R256 = ARGS.pred_dir_256
R392 = ARGS.pred_dir_392
HOSP = ["Site_A","Site_B","Site_C","Site_D","Site_E"]


def load(path):
    d = json.load(open(path))
    return (np.array(d['probs'], dtype=np.float64),
            np.array(d['labels'], dtype=np.int64),
            float(d['selected_threshold']))


def to1d(p):
    return p[:, 1] if p.ndim == 2 else p


rows = []
for i, h in enumerate(HOSP, 1):
    p256 = f"{R256}/fold_{i}_{h}/result.json"
    p392 = f"{R392}/fold_{i}_{h}/result.json"
    if not (os.path.exists(p256) and os.path.exists(p392)):
        print(f"fold {i} {h} MISSING"); continue
    pr256, lab256, th256 = load(p256)
    pr392, lab392, th392 = load(p392)
    assert lab256.shape == lab392.shape and (lab256 == lab392).all(), \
        f"label mismatch fold {i} {h}"
    pr256, pr392 = to1d(pr256), to1d(pr392)
    fused = 0.5 * (pr256 + pr392)
    th_fused = 0.5 * (th256 + th392)
    rows.append((i, h, lab256, pr256, pr392, fused, th256, th392, th_fused))


def metrics(lab, prob, thr):
    pred = (prob >= thr).astype(np.int64)
    return (roc_auc_score(lab, prob),
            balanced_accuracy_score(lab, pred),
            accuracy_score(lab, pred))


hdr = f"{'F':2s} {'Hospital':12s}|{'AUC256':>8s}{'AUC392':>8s}{'AUCfus':>8s}|{'BalA256':>8s}{'BalA392':>8s}{'BalAfus':>8s}|{'dAUC':>8s}"
print(hdr); print("-" * len(hdr))

a256, a392, afus, b256, b392, bfus = [], [], [], [], [], []
for i, h, lab, p256, p392, pf, t256, t392, tf in rows:
    A256, B256, _ = metrics(lab, p256, t256)
    A392, B392, _ = metrics(lab, p392, t392)
    AF, BF, _ = metrics(lab, pf, tf)
    d = AF - max(A256, A392)
    print(f"{i:2d} {h:12s}|{A256:8.4f}{A392:8.4f}{AF:8.4f}|{B256:8.4f}{B392:8.4f}{BF:8.4f}|{d:+8.4f}")
    a256.append(A256); a392.append(A392); afus.append(AF)
    b256.append(B256); b392.append(B392); bfus.append(BF)

print("-" * len(hdr))
m = lambda v: statistics.mean(v); s = lambda v: statistics.pstdev(v)
print(f"{'Mean':>15s}|{m(a256):8.4f}{m(a392):8.4f}{m(afus):8.4f}|{m(b256):8.4f}{m(b392):8.4f}{m(bfus):8.4f}|{m(afus)-max(m(a256),m(a392)):+8.4f}")
print(f"{'Std':>15s}|{s(a256):8.4f}{s(a392):8.4f}{s(afus):8.4f}|{s(b256):8.4f}{s(b392):8.4f}{s(bfus):8.4f}|")
print()
gain = m(afus) - max(m(a256), m(a392))
verdict = "ESCALATE (>= 0.005)" if gain >= 0.005 else "STOP (< 0.005, multi-scale dead weight)"
print(f"Mean fused AUC vs best single-scale: {gain:+.4f}  ->  {verdict}")
