"""Auto-load site_means_genbio.npz next to this file (overridable via site_means_path)."""
from pathlib import Path
from dataset_meancenter import UniCLAMDatasetMeanCenter

DEFAULT_MEANS_PATH = str(Path(__file__).resolve().parent / "site_means_genbio.npz")


class UniCLAMDatasetFixed(UniCLAMDatasetMeanCenter):
    def __init__(self, csv_path, feature_dir, use_h5_features=True,
                 max_instances=None, sample_seed=42, site_means_path=None):
        means_path = site_means_path or DEFAULT_MEANS_PATH
        super().__init__(csv_path=csv_path, feature_dir=feature_dir,
            site_means_path=means_path, use_h5_features=use_h5_features,
            max_instances=max_instances, sample_seed=sample_seed)
