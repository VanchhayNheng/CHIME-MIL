"""Common utilities for all LOSO experiments.

This module contains shared functions used across multiple baseline
experiments to avoid code duplication.
"""
import os
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any


def get_hospital(slide_id: str) -> str:
    """
    Map slide ID to hospital name.

    Args:
        slide_id: Slide identifier (e.g., "SC01-001", "SC-04-123")

    Returns:
        Hospital name or "Unknown" if not recognized

    Examples:
        >>> get_hospital("SC01-001")
        'Site_A'
        >>> get_hospital("SC-04-123")
        'Site_B'
        >>> get_hospital("UNKNOWN-001")
        'Unknown'
    """
    s = str(slide_id).upper().replace(" ", "").replace("_", "-")

    if s.startswith("SC-01") or s.startswith("SC01"):
        return "Site_A"
    elif s.startswith("SC-04") or s.startswith("SC04"):
        return "Site_B"
    elif "SC-03" in s or "SC03" in s:
        return "Site_C"
    elif any(s.startswith(x) for x in ["SC-3", "GC-3", "SC-7"]):
        return "Site_D"
    elif s.startswith("SC-02") or s.startswith("SC02"):
        return "Site_E"

    return "Unknown"


def setup_logging(
    output_dir: str,
    experiment_name: str,
    level: int = logging.DEBUG
) -> Tuple[logging.Logger, str]:
    """
    Configure logging for LOSO experiments.

    Creates both file and console handlers:
    - File handler: DEBUG level (detailed logs saved to disk)
    - Console handler: INFO level (concise output to terminal)

    Args:
        output_dir: Directory to save log files
        experiment_name: Name of the experiment (used in log filename)
        level: Logging level for file handler (default: DEBUG)

    Returns:
        logger: Configured logger instance
        timestamp: Timestamp string used in filename (YYYYMMDD_HHMMSS)

    Example:
        >>> logger, ts = setup_logging("./results", "chime_loso")
        >>> logger.info("Training started")
        >>> # Creates file: ./results/chime_loso_20260323_120000.log
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_p{os.getpid()}"  # PID suffix avoids collision when multiple folds launch in same second

    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)
    logger.handlers.clear()  # Remove any existing handlers

    # File handler - detailed logging
    log_file = os.path.join(output_dir, f"{experiment_name}_{timestamp}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler - concise output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")

    return logger, timestamp


def collate_single(batch):
    """
    Collate function for batch size = 1 (MIL setting).

    In multiple instance learning, each "batch" is actually a single
    bag (slide) containing a variable number of instances (patches).
    This collate function simply returns the batch as-is without
    attempting to stack tensors.

    Args:
        batch: List containing a single item [(features, coords, label)]

    Returns:
        The batch unchanged (list of tuples)

    Example:
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=1, collate_fn=collate_single)
    """
    return batch


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml (default: searches parent directories)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config.yaml not found

    Example:
        >>> config = load_config()
        >>> print(config['training']['learning_rate'])
        5e-05
    """
    if config_path is None:
        # Search for config.yaml in current and parent directories
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break

        if config_path is None:
            raise FileNotFoundError(
                "config.yaml not found. Please create one or specify path."
            )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_paths_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract commonly used paths from config.

    Args:
        config: Configuration dictionary from load_config()

    Returns:
        Dictionary with commonly used paths:
        - wsi_dir
        - feature_dir
        - csv_path
        - completed_ids_file

    Example:
        >>> config = load_config()
        >>> paths = get_paths_from_config(config)
        >>> print(paths['wsi_dir'])
        /path/to/wsi/files/
    """
    csv_dir = config['data']['csv_dir']
    csv_file = config['data']['tumor_normal_csv']

    return {
        'wsi_dir': config['data']['wsi_dir'],
        'feature_dir': config['data']['feature_dir'],
        'csv_path': os.path.join(csv_dir, csv_file),
        'completed_ids_file': config['data']['completed_ids'],
    }


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2h 34m 12s")

    Examples:
        >>> format_duration(3661)
        '1h 1m 1s'
        >>> format_duration(125)
        '2m 5s'
        >>> format_duration(45)
        '45s'
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)
