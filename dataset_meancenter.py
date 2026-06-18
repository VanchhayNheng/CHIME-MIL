"""Site-conditional mean-centered dataset wrapper for CHIME_MIL.

This is a non-destructive variant that subtracts a per-site mean feature
vector from every patch before returning features. It is a first-moment
CORAL-style alignment targeted at the stain/scanner shift diagnosed for
Site_A and Site_C (see `diagnose_stain_shift.py` output).

Usage:
    from dataset_meancenter import UniCLAMDatasetMeanCenter

    ds = UniCLAMDatasetMeanCenter(
        csv_path=csv,
        feature_dir=feat_dir,
        site_means_path=".../site_means.npz",
        use_h5_features=True,
    )

Everything else matches UniCLAMDatasetFixed. The class preserves label
semantics, coordinate handling, and subsampling unchanged.
"""
import os, re, hashlib
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Authoritative slide_id -> site mapping (CLAUDE.md)
def infer_hospital(slide_id: str) -> str:
    s = str(slide_id)
    if re.match(r'^(SC[-_]01)', s):           return 'Site_A'
    if re.match(r'^(SC[-_]02)', s):           return 'Site_E'
    if re.match(r'^(SC[-_]04)', s):           return 'Site_B'
    if re.match(r'^(SC-3-|GC-3-|SC-7-)', s):  return 'Site_D'
    if re.match(r'^(SC[-_]03)', s):           return 'Site_C'
    raise ValueError(f'Unknown slide_id prefix: {slide_id}')


class UniCLAMDatasetMeanCenter(Dataset):
    """Mirror of UniCLAMDatasetFixed but subtracts per-site mean feature
    vectors from patch features at load time.

    The site-mean vectors come from `site_means.npz`, which must contain one
    (1024,) float32 array per site keyed by site name. Computation is
    unsupervised (patch-weighted mean over all slides of the site) so it is
    legitimate to include the LOSO held-out site without label leakage.
    """

    def __init__(self, csv_path, feature_dir, site_means_path,
                 use_h5_features=True, max_instances=None, sample_seed=42):
        self.feature_dir = feature_dir
        self.use_h5_features = use_h5_features
        self.max_instances = max_instances
        self.sample_seed = sample_seed

        self.df = pd.read_csv(csv_path)
        self.df['slide_id'] = self.df['slide_id'].astype(str)
        self.slide_ids = self.df['slide_id'].values

        # Label convention — identical to dataset_fixed.py
        EXPECTED_LABEL_MAP = {"m": 0, "nm": 1, 0: 0, 1: 1}
        raw_labels = self.df['label'].values
        self.label_map = {k: v for k, v in EXPECTED_LABEL_MAP.items()
                          if k in set(raw_labels)}
        unknown = set(raw_labels) - set(EXPECTED_LABEL_MAP.keys())
        if unknown:
            raise ValueError(
                f"Unknown label values {unknown} in {csv_path}. "
                f"Expected subset of {set(EXPECTED_LABEL_MAP.keys())}."
            )
        self.labels = [EXPECTED_LABEL_MAP[l] for l in raw_labels]

        # Pre-resolve each slide's site once, for fast lookup.
        self.sites = [infer_hospital(sid) for sid in self.slide_ids]

        # Load per-site means.
        npz = np.load(site_means_path, allow_pickle=True)
        self.site_means = {}
        for key in npz.files:
            arr = npz[key]
            if isinstance(arr, np.ndarray) and arr.dtype != object \
                    and arr.ndim == 1 and arr.shape[0] == 4608:
                self.site_means[str(key)] = torch.from_numpy(
                    arr.astype(np.float32))
        required = {'Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E'}
        missing = required - set(self.site_means.keys())
        if missing:
            raise ValueError(f'site_means.npz missing sites: {missing}')

        print(f"[MeanCenter] Dataset loaded: {len(self.slide_ids)} slides")
        print(f"[MeanCenter] Feature source: "
              f"{'H5' if use_h5_features else 'PT'}")
        print(f"[MeanCenter] Label map: {self.label_map}")
        print(f"[MeanCenter] Site mean norms: "
              + ', '.join(f'{k}={torch.linalg.norm(v):.2f}'
                          for k, v in self.site_means.items()))

    def _subsample_indices(self, slide_id, num_instances):
        if self.max_instances is None or num_instances <= self.max_instances:
            return None
        seed_src = f"{self.sample_seed}:{slide_id}".encode("utf-8")
        seed = int(hashlib.sha1(seed_src).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(num_instances,
                                  size=self.max_instances, replace=False))

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        site = self.sites[idx]
        label = torch.tensor(self.labels[idx]).long()

        h5_path = os.path.join(self.feature_dir, 'h5_files', f"{slide_id}.h5")
        pt_path = os.path.join(self.feature_dir, 'pt_files', f"{slide_id}.pt")

        try:
            if self.use_h5_features:
                if not os.path.exists(h5_path):
                    return torch.zeros(1, 4608), torch.zeros(1, 2), label
                with h5py.File(h5_path, 'r') as f:
                    features = torch.from_numpy(
                        np.array(f['features'])).float()
                    coords = torch.from_numpy(
                        np.array(f['coords'])).float()
            else:
                if not os.path.exists(pt_path):
                    return torch.zeros(1, 4608), torch.zeros(1, 2), label
                features = torch.load(pt_path)
                if isinstance(features, dict):
                    features = features['features']
                if os.path.exists(h5_path):
                    with h5py.File(h5_path, 'r') as f:
                        coords = torch.from_numpy(
                            np.array(f['coords'])).float()
                else:
                    coords = torch.rand(features.shape[0], 2)

            keep_idx = self._subsample_indices(slide_id, features.shape[0])
            if keep_idx is not None:
                keep_idx = torch.from_numpy(keep_idx).long()
                features = features[keep_idx]
                coords = coords[keep_idx]

            # >>> MEAN-CENTER: subtract this slide's site mean vector. <<<
            features = features - self.site_means[site]

            if coords.shape[0] > 1:
                c_min = coords.min(0, keepdim=True)[0]
                c_max = coords.max(0, keepdim=True)[0]
                denom = c_max - c_min
                denom[denom == 0] = 1.0
                coords = (coords - c_min) / denom

            return features, coords, label

        except Exception as e:
            print(f"Error loading {slide_id}: {e}")
            return torch.zeros(1, 4608), torch.zeros(1, 2), label
