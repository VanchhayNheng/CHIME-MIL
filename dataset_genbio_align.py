"""Site-conditional higher-order alignment dataset (ZCA / full CORAL) for
CHIME-MIL. Drop-in generalization of dataset_meancenter.UniCLAMDatasetMeanCenter:
applies a per-site affine  X' = X @ W_s + c_s  loaded from an alignment .npz
made by compute_site_align_genbio.py. Mean-centring is the special case
W=I, c=-mu. Unsupervised -> may include the LOSO held-out site (deployment TTA).
"""
import os, hashlib
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataset_meancenter import infer_hospital

SITES = ['Site_A', 'Site_B', 'Site_C', 'Site_D', 'Site_E']


class UniCLAMDatasetAlign(Dataset):
    """Per-site affine alignment dataset: returns X@W_s+c_s per slide.

    Generalizes UniCLAMDatasetMeanCenter; W_s/c_s from a
    compute_site_align_genbio.py npz. H5 features only.
    """

    def __init__(self, csv_path, feature_dir, align_path,
                 use_h5_features=True, max_instances=None, sample_seed=42):
        self.feature_dir = feature_dir
        assert use_h5_features, 'UniCLAMDatasetAlign supports H5 features only'
        self.use_h5_features = use_h5_features
        self.max_instances = max_instances
        self.sample_seed = sample_seed
        self.df = pd.read_csv(csv_path)
        self.df['slide_id'] = self.df['slide_id'].astype(str)
        self.slide_ids = self.df['slide_id'].values
        EXP = {"m": 0, "nm": 1, 0: 0, 1: 1}
        raw = self.df['label'].values
        self.label_map = {k: v for k, v in EXP.items() if k in set(raw)}
        if set(raw) - set(EXP.keys()):
            raise ValueError("Unknown label values in " + csv_path)
        self.labels = [EXP[l] for l in raw]
        self.sites = [infer_hospital(s) for s in self.slide_ids]
        npz = np.load(align_path, allow_pickle=True)
        self.method = str(npz['method']) if 'method' in npz.files else 'align'
        self.site_W, self.site_c = {}, {}
        for s in SITES:
            self.site_W[s] = torch.from_numpy(npz['W_%s' % s].astype(np.float32))
            self.site_c[s] = torch.from_numpy(npz['c_%s' % s].astype(np.float32))
        print('[Align:%s] %d slides; W=%s' % (self.method,
              len(self.slide_ids), tuple(self.site_W['Site_A'].shape)))

    def _subsample_indices(self, slide_id, num_instances):
        if self.max_instances is None or num_instances <= self.max_instances:
            return None
        seed_src = ("%s:%s" % (self.sample_seed, slide_id)).encode("utf-8")
        seed = int(hashlib.sha1(seed_src).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(num_instances, size=self.max_instances,
                                  replace=False))

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        site = self.sites[idx]
        label = torch.tensor(self.labels[idx]).long()
        h5p = os.path.join(self.feature_dir, 'h5_files', "%s.h5" % slide_id)
        try:
            if not os.path.exists(h5p):
                return torch.zeros(1, 4608), torch.zeros(1, 2), label
            with h5py.File(h5p, 'r') as f:
                features = torch.from_numpy(np.array(f['features'])).float()
                coords = torch.from_numpy(np.array(f['coords'])).float()
            keep = self._subsample_indices(slide_id, features.shape[0])
            if keep is not None:
                keep = torch.from_numpy(keep).long()
                features = features[keep]; coords = coords[keep]
            features = features @ self.site_W[site] + self.site_c[site]
            if coords.shape[0] > 1:
                c_min = coords.min(0, keepdim=True)[0]
                c_max = coords.max(0, keepdim=True)[0]
                denom = c_max - c_min
                denom[denom == 0] = 1.0
                coords = (coords - c_min) / denom
            return features, coords, label
        except Exception as e:
            print("Error loading %s: %s" % (slide_id, e))
            return torch.zeros(1, 4608), torch.zeros(1, 2), label
