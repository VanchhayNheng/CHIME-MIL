#!/usr/bin/env python3
"""Render fig_heatmaps (HI-RES, real H&E): 2x2 attention heatmaps overlaid on the
actual WSI thumbnails for 4 representative slides (rows=tumour/normal,
cols=Site_A/Site_E), headline mc CHIME-MIL @256px. Slide-level ground truth
shown as a colored panel border (red=tumour, blue=normal) plus GT/Pred caption.
Panels cropped to tissue bbox; overlay alpha gated so only hotspots show.
Output: results/fig_heatmaps_hires.{pdf,png}
"""
import os, sys
from pathlib import Path
import numpy as np, torch, h5py, openslide
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- de-identify hospital names -> Site A-E (matches site_relabel.py) ---
_SITE_MAP = {"Site_A": "Site A", "Site_B": "Site B", "Site_C": "Site C", "Site_D": "Site D", "Site_E": "Site E"}
def to_site(_n):
    return _SITE_MAP.get(_n, _n)
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm
from scipy.ndimage import gaussian_filter

ROOT = Path(".")
sys.path.insert(0, str(ROOT)); from chime_mil import CHIME_MIL
FEAT_DIR = Path("/path/to/data/GENBIO_PATHFM_FEATURES/mag20x/h5_files")
WSI_DIR  = Path("/path/to/data/dataset5036/DATA_DIRECTORY")
SITE_MEANS = np.load(ROOT / "site_means_genbio.npz")
SCALE, SEED, THUMB_MAX = 256, 42, 2600

SLIDES = [
    ("SC-01-0977", 0, "malignant", "Site_A", 1, (0, 0)),
    ("SC_02_0134", 0, "malignant", "Site_E",  5, (0, 1)),
    ("SC-01-0845", 1, "normal",    "Site_A", 1, (1, 0)),
    ("SC_02_0080", 1, "normal",    "Site_E",  5, (1, 1)),
]
CMAP_ROW = {0: "hot", 1: "Blues"}
GT_COLOR = {0: "#c0392b", 1: "#2266aa"}

def ckpt_for(h, f):
    return ROOT / "seed_sweep_mc_sa_multiscale/results" / f"seed{SEED}_scale{SCALE}" / f"fold_{f}_{h}" / "best_model.pth"

def patch_size_lvl0(coords):
    out = []
    for ax in (0, 1):
        u = np.unique(coords[:, ax]); d = np.diff(np.sort(u)); d = d[d > 0]
        if len(d): out.append(int(np.min(d)))
    return max(out) if out else 224

def run_slide(sid, hosp, fold, device):
    with h5py.File(FEAT_DIR / f"{sid}.h5", "r") as h5:
        feats = np.asarray(h5["features"], dtype=np.float32)
        coords = np.asarray(h5["coords"], dtype=np.float32)
    feats = feats - SITE_MEANS[hosp].astype(np.float32)
    model = CHIME_MIL(input_dim=4608, hidden_dim=256, num_classes=2, num_regions=16, dropout=0.5).to(device)
    ck = torch.load(ckpt_for(hosp, fold), map_location=device, weights_only=False)
    state = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    model.load_state_dict(state, strict=False); model.eval()
    with torch.no_grad():
        x = torch.from_numpy(feats).unsqueeze(0).to(device)
        c = torch.from_numpy(coords).unsqueeze(0).to(device)
        out = model(x, c)
        attn = out["patch_importance"].squeeze(0).cpu().numpy()
        logits = out["logits"].squeeze(0).cpu().numpy()
        p = np.exp(logits - logits.max()); p = p / p.sum()
    return coords, attn, float(p[0])

def build_overlay(sid, coords, attn, cmap_name):
    sl = openslide.OpenSlide(str(WSI_DIR / f"{sid}.svs"))
    W0, H0 = sl.dimensions
    mpp_x = float(sl.properties.get("openslide.mpp-x", 0.5))
    thumb = sl.get_thumbnail((THUMB_MAX, int(THUMB_MAX * H0 / W0)))
    thumb = np.asarray(thumb.convert("RGB"))
    Hh, Ww = thumb.shape[:2]
    ds = W0 / Ww
    ps = patch_size_lvl0(coords)
    a = (attn - attn.min()) / (np.ptp(attn) + 1e-12)
    heat = np.zeros((Hh, Ww), dtype=np.float32)
    bw = max(1, int(round(ps / ds)))
    for (x0, y0), av in zip(coords, a):
        xi, yi = int(x0 / ds), int(y0 / ds)
        heat[yi:yi+bw, xi:xi+bw] = np.maximum(heat[yi:yi+bw, xi:xi+bw], av)
    heat = gaussian_filter(heat, sigma=bw * 0.5)
    if heat.max() > 0: heat /= heat.max()
    disp = heat ** 0.5
    rgba = plt.get_cmap(cmap_name)(disp)
    rgba[..., 3] = np.clip((disp - 0.20) / 0.80, 0.0, 1.0) * 0.95  # gate: clean H&E, bold hotspots
    # crop to tissue bbox + margin
    xi0 = int(coords[:, 0].min()/ds); xi1 = int(coords[:, 0].max()/ds)+bw
    yi0 = int(coords[:, 1].min()/ds); yi1 = int(coords[:, 1].max()/ds)+bw
    mx = int(0.04*(xi1-xi0)); my = int(0.04*(yi1-yi0))
    xi0, yi0 = max(0, xi0-mx), max(0, yi0-my)
    xi1, yi1 = min(Ww, xi1+mx), min(Hh, yi1+my)
    sl.close()
    return thumb[yi0:yi1, xi0:xi1], rgba[yi0:yi1, xi0:xi1], mpp_x, ds

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.045], wspace=0.10, hspace=0.20,
                  left=0.06, right=0.92, top=0.90, bottom=0.06)
    row_sc = {}
    for sid, gt, name, hosp, fold, (r, cidx) in SLIDES:
        coords, attn, p_tum = run_slide(sid, hosp, fold, device)
        thumb, rgba, mpp_x, ds_thumb = build_overlay(sid, coords, attn, CMAP_ROW[r])
        ax = fig.add_subplot(gs[r, cidx])
        ax.imshow(thumb); ax.imshow(rgba)
        ax.set_xticks([]); ax.set_yticks([])
        # 1 mm physical scale bar (bottom-left, white outline + black core, "1 mm" label)
        bar_um = 1000.0
        bar_px = bar_um / (mpp_x * ds_thumb)
        Hh, Ww = thumb.shape[:2]
        x0 = 0.04 * Ww
        x1 = x0 + bar_px
        y_bar = Hh - 0.04 * Hh
        ax.plot([x0, x1], [y_bar, y_bar], color="white", linewidth=6.5, solid_capstyle="butt", zorder=5)
        ax.plot([x0, x1], [y_bar, y_bar], color="black", linewidth=3.0, solid_capstyle="butt", zorder=6)
        ax.text(0.5 * (x0 + x1), y_bar - 0.02 * Hh, "1 mm",
                ha="center", va="bottom", fontsize=8.5, color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2),
                zorder=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GT_COLOR[gt]); sp.set_linewidth(3.4)
        ax.set_title(f"{to_site(hosp)} — {name}", fontsize=10.5, pad=5)
        pred = "Tumour" if p_tum > 0.5 else "Normal"
        conf = p_tum if pred == "Tumour" else 1 - p_tum
        gt_txt = "Tumour" if gt == 0 else "Normal"
        ok = "✓" if pred == gt_txt else "✗"
        ax.set_xlabel(f"GT: {gt_txt}  |  Pred: {pred} ({conf:.2f}) {ok}  |  N={len(attn):,} patches",
                      fontsize=9, labelpad=5, color=("#157347" if pred == gt_txt else "#b02a37"))
        sc = plt.cm.ScalarMappable(cmap=plt.get_cmap(CMAP_ROW[r]), norm=PowerNorm(0.5, 0, 1)); sc.set_array([])
        row_sc[r] = sc
    fig.text(0.035, 0.69, "Malignant (tumour)", rotation=90, va="center", fontsize=11.5, fontweight="bold")
    fig.text(0.035, 0.27, "Normal (non-tumour)", rotation=90, va="center", fontsize=11.5, fontweight="bold")
    for r in (0, 1):
        cax = fig.add_subplot(gs[r, 2]); cb = fig.colorbar(row_sc[r], cax=cax)
        cb.set_label("Attn. (norm.)", fontsize=9)
    fig.text(0.50, 0.925, "Border = ground truth:  ", ha="right", fontsize=9, color="0.3")
    fig.text(0.50, 0.925, "■ Tumour", ha="left", fontsize=9, color=GT_COLOR[0], fontweight="bold")
    fig.text(0.575, 0.925, "■ Normal", ha="left", fontsize=9, color=GT_COLOR[1], fontweight="bold")
    fig.suptitle("Patch-level attention on whole-slide H&E — CHIME-MIL (256 px, GenBio-PathFM)",
                 fontsize=13.5, fontweight="bold", y=0.965)
    out = ROOT / "results" / "fig_heatmaps_hires"
    fig.savefig(str(out) + ".pdf", dpi=300); fig.savefig(str(out) + ".png", dpi=200)
    print("wrote", str(out) + ".pdf / .png")

if __name__ == "__main__":
    main()
