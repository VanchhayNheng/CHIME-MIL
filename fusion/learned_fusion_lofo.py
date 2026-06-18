"""LOFO (leave-one-fold-out) learned fusion weight for mc_sa@(256+392).

For each test fold k, search w in [0,1] step 0.01 to maximize mean test AUC
on the OTHER 4 folds, then apply that w to fold k. Fold k's own test probs
never enter the search, so this is a clean held-out evaluation.

Comparison: vs flat 0.5/0.5 baseline (current headline 0.8409).

Run with the UNI env python on athar.
"""
import json, os, numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

import argparse
from pathlib import Path
_DEF = Path(__file__).resolve().parents[1] / 'results'
_ap = argparse.ArgumentParser()
_ap.add_argument('--pred_dir_256', default=str(_DEF / 'results_loso_mc_sa_genbio_20260421'))
_ap.add_argument('--pred_dir_392', default=str(_DEF / 'results_loso_mc_sa_genbio_392'))
ARGS, _ = _ap.parse_known_args()
R256 = ARGS.pred_dir_256
R392 = ARGS.pred_dir_392
HOSP = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E"]


def load(path: str):
    """Load probs/labels/threshold from result.json. Returns (probs[N], labels[N], threshold)."""
    d = json.load(open(path))
    p = np.array(d["probs"], dtype=np.float64)
    if p.ndim == 2:
        p = p[:, 1]
    lab = np.array(d["labels"], dtype=np.int64)
    return p, lab, float(d["selected_threshold"])


def main() -> None:
    """Run LOFO learned fusion sweep and print comparison vs flat 0.5/0.5."""
    folds = []
    for i, h in enumerate(HOSP, 1):
        p256, lab256, th256 = load(f"{R256}/fold_{i}_{h}/result.json")
        p392, lab392, th392 = load(f"{R392}/fold_{i}_{h}/result.json")
        assert lab256.shape == lab392.shape and (lab256 == lab392).all(), f"label mismatch fold {i}"
        folds.append({"i": i, "h": h, "p256": p256, "p392": p392,
                      "lab": lab256, "th256": th256, "th392": th392})

    grid = np.arange(0.0, 1.001, 0.01)

    print("=" * 96)
    print(f"{'Fold':<14}{'AUC(256)':>10}{'AUC(392)':>10}{'AUC(0.5)':>10}{'w*_LOFO':>10}{'AUC(w*)':>10}{'Δ vs 0.5':>10}")
    print("-" * 96)

    rows = []
    for k, f in enumerate(folds):
        # held-in: union of all OTHER folds' probs/labels for the w search
        held_in = [g for j, g in enumerate(folds) if j != k]
        p256_in = np.concatenate([g["p256"] for g in held_in])
        p392_in = np.concatenate([g["p392"] for g in held_in])
        lab_in = np.concatenate([g["lab"] for g in held_in])

        # search w that maximises AUC on held-in pool
        best_w, best_auc = 0.5, -1.0
        for w in grid:
            fused = w * p256_in + (1.0 - w) * p392_in
            a = roc_auc_score(lab_in, fused)
            if a > best_auc:
                best_auc, best_w = a, float(w)

        # apply chosen w to held-out fold k
        fused_k = best_w * f["p256"] + (1.0 - best_w) * f["p392"]
        flat_k = 0.5 * f["p256"] + 0.5 * f["p392"]

        auc_256 = roc_auc_score(f["lab"], f["p256"])
        auc_392 = roc_auc_score(f["lab"], f["p392"])
        auc_flat = roc_auc_score(f["lab"], flat_k)
        auc_lofo = roc_auc_score(f["lab"], fused_k)
        delta = auc_lofo - auc_flat

        # also compute acc/balacc for the LOFO fusion (use averaged threshold,
        # mirroring the flat-0.5 fusion convention)
        th_lofo = best_w * f["th256"] + (1.0 - best_w) * f["th392"]
        preds_lofo = (fused_k >= th_lofo).astype(int)
        acc_lofo = accuracy_score(f["lab"], preds_lofo)
        bal_lofo = balanced_accuracy_score(f["lab"], preds_lofo)

        rows.append({"fold": f["i"], "hosp": f["h"], "w": best_w,
                     "auc_flat": auc_flat, "auc_lofo": auc_lofo, "delta": delta,
                     "acc_lofo": acc_lofo, "bal_lofo": bal_lofo})

        print(f"{f['i']} {f['h']:<12}{auc_256:>10.4f}{auc_392:>10.4f}"
              f"{auc_flat:>10.4f}{best_w:>10.2f}{auc_lofo:>10.4f}{delta:>+10.4f}")

    print("-" * 96)
    mean_flat = float(np.mean([r["auc_flat"] for r in rows]))
    mean_lofo = float(np.mean([r["auc_lofo"] for r in rows]))
    std_flat = float(np.std([r["auc_flat"] for r in rows]))
    std_lofo = float(np.std([r["auc_lofo"] for r in rows]))
    mean_delta = mean_lofo - mean_flat
    print(f"{'MEAN':<14}{'':>10}{'':>10}{mean_flat:>10.4f}"
          f"{'':>10}{mean_lofo:>10.4f}{mean_delta:>+10.4f}")
    print(f"{'STD':<14}{'':>10}{'':>10}{std_flat:>10.4f}{'':>10}{std_lofo:>10.4f}")
    print("=" * 96)
    print(f"\nFlat 0.5/0.5 baseline: AUC {mean_flat:.4f} ± {std_flat:.4f}")
    print(f"LOFO learned weight:   AUC {mean_lofo:.4f} ± {std_lofo:.4f}")
    print(f"Delta:                 {mean_delta:+.4f}")
    print(f"\nDecision rule (>= +0.005 to justify two-tower): "
          f"{'PASS' if mean_delta >= 0.005 else 'FAIL'}")

    # also print the 5 chosen weights
    print(f"\nLOFO-chosen weights w* (P256 weight): {[round(r['w'], 2) for r in rows]}")


if __name__ == "__main__":
    main()
