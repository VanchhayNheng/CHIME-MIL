#!/usr/bin/env python3
"""Render fig_heatmaps.pdf: 2x2 patch-attention scatter for the 4 representative
WSIs, using the headline mc model at 256 px. Reuses the exact preprocessing of
compute_attention_concentration.py (site-mean centering, headline CHIME_MIL),
so attention matches the reported per-slide Gini/rho10 numbers.

Tracked replacement for the lost May-4 generator. Output: results/fig_heatmaps.{pdf,png}
"""
import os, sys, re
from pathlib import Path
import numpy as np, torch, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- de-identify hospital names -> Site A-E (matches site_relabel.py) ---
_SITE_MAP = {"Site_A": "Site A", "Site_B": "Site B", "Site_C": "Site C", "Site_D": "Site D", "Site_E": "Site E"}
def to_site(_n):
    return _SITE_MAP.get(_n, _n)
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm

ROOT = Path(".")
sys.path.insert(0, str(ROOT))
from chime_mil import CHIME_MIL

FEAT_DIR = Path("/path/to/data/GENBIO_PATHFM_FEATURES/mag20x/h5_files")
SITE_MEANS = np.load(ROOT / "site_means_genbio.npz")
SCALE, SEED = 256, 42

# (slide_id, gt_label_idx, label_name, hospital, fold_idx, grid_pos)  class0=tumor
SLIDES = [
    ("SC-04-00248", 0, "malignant", "Site_B",  2, (0, 0)),
    ("SC_02_0134",  0, "malignant", "Site_E", 5, (0, 1)),
    ("SC-04-0707",  1, "normal",    "Site_B",  2, (1, 0)),
    ("SC_02_0080",  1, "normal",    "Site_E", 5, (1, 1)),
]
CMAP_ROW = {0: "hot", 1: "Blues"}   # malignant row / normal row

def ckpt_for(hospital, fold_idx):
    return ROOT / "seed_sweep_mc_sa_multiscale" / "results" / f"seed{SEED}_scale{SCALE}" / f"fold_{fold_idx}_{hospital}" / "best_model.pth"

def run_slide(slide_id, hospital, fold_idx, device):
    with h5py.File(FEAT_DIR / f"{slide_id}.h5", "r") as h5:
        feats = np.asarray(h5["features"], dtype=np.float32)
        coords = np.asarray(h5["coords"], dtype=np.float32)
    feats = feats - SITE_MEANS[hospital].astype(np.float32)
    model = CHIME_MIL(input_dim=4608, hidden_dim=256, num_classes=2, num_regions=16, dropout=0.5).to(device)
    ck = torch.load(ckpt_for(hospital, fold_idx), map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state, strict=False); model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0).to(device)
        c = torch.from_numpy(coords).unsqueeze(0).to(device)
        out = model(x, c)
        attn = out["patch_importance"].squeeze(0).cpu().numpy()
        logits = out["logits"].squeeze(0).cpu().numpy()
        p = np.exp(logits - logits.max()); p = p / p.sum()
    return coords, attn, float(p[0])  # p_tumor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig = plt.figure(figsize=(9.0, 8.0))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.12, hspace=0.22,
                  left=0.07, right=0.93, top=0.90, bottom=0.06)
    row_sc = {}
    for slide_id, gt, name, hosp, fold, (r, cidx) in SLIDES:
        coords, attn, p_tumor = run_slide(slide_id, hosp, fold, device)
        a = (attn - attn.min()) / (np.ptp(attn) + 1e-12)   # normalised alpha_tilde in [0,1]
        disp = a ** 0.5                                    # perceptual lift
        cmap = plt.get_cmap(CMAP_ROW[r])
        rgba = cmap(disp); rgba[:, 3] = np.clip(disp, 0.10, 1.0)   # alpha grows with attention
        ax = fig.add_subplot(gs[r, cidx])
        ax.scatter(coords[:, 0], coords[:, 1], c="0.82", s=7, edgecolors="none", rasterized=True)  # tissue footprint
        ax.scatter(coords[:, 0], coords[:, 1], c=rgba, s=7, edgecolors="none", rasterized=True)
        sc = plt.cm.ScalarMappable(cmap=cmap, norm=PowerNorm(gamma=0.5, vmin=0, vmax=1)); sc.set_array([])
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{to_site(hosp)} — {name}", fontsize=10, pad=4)
        pred = "Tumor" if p_tumor > 0.5 else "Normal"
        conf = p_tumor if pred == "Tumor" else 1 - p_tumor
        gt_txt = "Tumor" if gt == 0 else "Normal"
        ok = "✓" if (pred == gt_txt) else "✗"
        ax.set_xlabel(f"GT: {gt_txt}  |  Pred: {pred} ({conf:.2f}) {ok}  |  N={len(attn):,} patches",
                      fontsize=8.5, labelpad=5)
        row_sc[r] = sc
    fig.text(0.04, 0.70, "Malignant (tumor)", rotation=90, va="center", fontsize=11, fontweight="bold")
    fig.text(0.04, 0.28, "Normal (non-tumor)", rotation=90, va="center", fontsize=11, fontweight="bold")
    for r in (0, 1):
        cax = fig.add_subplot(gs[r, 2])
        cb = fig.colorbar(row_sc[r], cax=cax); cb.set_label("Attn. (norm.)", fontsize=9)
    fig.suptitle("Patch-level attention heatmaps — CHIME-MIL (256 px, GenBio-PathFM)",
                 fontsize=13, fontweight="bold", y=0.965)
    out = ROOT / "results" / "fig_heatmaps"
    fig.savefig(str(out) + ".pdf", dpi=300); fig.savefig(str(out) + ".png", dpi=200)
    print("wrote", str(out) + ".pdf / .png")

if __name__ == "__main__":
    main()
