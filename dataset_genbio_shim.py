import os
import hashlib
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class UniCLAMDatasetFixed(Dataset):
    """
    Fixed dataset class using H5 files for BOTH features AND coords.
    
    Previous bug: PT features (4015 patches) + H5 coords (1029 patches) = MISMATCH!
    Fix: Use H5 features (1029 patches) + H5 coords (1029 patches) = CONSISTENT!
    """
    def __init__(self, csv_path, feature_dir, use_h5_features=True, max_instances=None, sample_seed=42):
        self.feature_dir = feature_dir
        self.use_h5_features = use_h5_features
        self.max_instances = max_instances
        self.sample_seed = sample_seed
        
        self.df = pd.read_csv(csv_path)
        self.df['slide_id'] = self.df['slide_id'].astype(str)
        self.slide_ids = self.df['slide_id'].values
        
        # Label convention (authoritative, matches all CHIME_MIL baselines):
        #   class 0 = 'm'  = metastasis / tumor   (POSITIVE class)
        #   class 1 = 'nm' = non-metastasis / normal
        # Metrics (sens=cm[0,0]/row0, f1 pos_label=0) depend on this mapping.
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
        
        print(f"Dataset loaded: {len(self.slide_ids)} slides")
        print(f"Feature source: {'H5 (consistent)' if use_h5_features else 'PT (original)'}")
        print(f"Label map: {self.label_map}")
        print(f"Max instances: {self.max_instances}")

    def _subsample_indices(self, slide_id, num_instances):
        if self.max_instances is None or num_instances <= self.max_instances:
            return None

        seed_src = f"{self.sample_seed}:{slide_id}".encode("utf-8")
        seed = int(hashlib.sha1(seed_src).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(num_instances, size=self.max_instances, replace=False))

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = torch.tensor(self.labels[idx]).long()
        
        h5_path = os.path.join(self.feature_dir, 'h5_files', f"{slide_id}.h5")
        pt_path = os.path.join(self.feature_dir, 'pt_files', f"{slide_id}.pt")
        
        try:
            if self.use_h5_features:
                # ✅ FIXED: Use H5 for BOTH features AND coords
                if not os.path.exists(h5_path):
                    return torch.zeros(1, 4608), torch.zeros(1, 2), label
                
                with h5py.File(h5_path, 'r') as f:
                    features = torch.from_numpy(
                        np.array(f['features'])
                    ).float()                          # (N, 4608)
                    coords = torch.from_numpy(
                        np.array(f['coords'])
                    ).float()                          # (N, 2) ✅ MATCHES!
            
            else:
                # Original (buggy) approach for comparison
                if not os.path.exists(pt_path):
                    return torch.zeros(1, 4608), torch.zeros(1, 2), label
                
                features = torch.load(pt_path)
                if isinstance(features, dict):
                    features = features['features']
                
                if os.path.exists(h5_path):
                    with h5py.File(h5_path, 'r') as f:
                        coords = torch.from_numpy(
                            np.array(f['coords'])
                        ).float()
                else:
                    coords = torch.rand(features.shape[0], 2)
            
            # Normalize coordinates to [0, 1]
            keep_idx = self._subsample_indices(slide_id, features.shape[0])
            if keep_idx is not None:
                keep_idx = torch.from_numpy(keep_idx).long()
                features = features[keep_idx]
                coords = coords[keep_idx]

            # Normalize coordinates to [0, 1]
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
