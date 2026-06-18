"""Canonical LOSO training entry point for CHIME_MIL.

This script is the maintained 20x-only training path for the ExtractWSI
research codebase. It centralizes configuration through ``config.yaml``,
reuses shared utilities from ``chime_mil_utils``, and keeps the
existing training/evaluation behavior intact.
"""

from __future__ import annotations

import argparse
import hashlib
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chime_mil_utils import (  # noqa: E402
    collate_single,
    get_hospital,
    get_paths_from_config,
    load_config,
    setup_logging,
)
from causal_loss import CausalLoss  # noqa: E402
from chime_mil import CHIME_MIL  # noqa: E402
from dataset_genbio_meancenter_shim import UniCLAMDatasetFixed  # meancenter+stratified variant
from dataset_genbio_align import UniCLAMDatasetAlign
from focal_loss import FocalLoss  # noqa: E402
from metrics_utils import (  # noqa: E402
    bootstrap_ci,
    compute_metrics,
    find_best_threshold,
    format_ci,
    probs_to_preds,
)

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

DEFAULT_HOSPITALS: List[str] = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "results_loso_mc_sa_genbio_20260421"
LAMBDA_PATCH = 0.2
LAMBDA_REGION = 0.2
LAMBDA_GRAPH = 0.3


def hospital_stratified_val_split(
    train_df: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Build a hospital-stratified val split with a median-based cap.

    For each training hospital, take ``val_ratio`` of its slides as val
    (label-stratified). Each hospital's val contribution is then capped at the
    median across all hospitals so that large hospitals (e.g. Site_A, ~52%
    of slides) cannot dominate the early-stopping signal. Slides removed from a
    hospital's val by the cap are returned to its training pool.
    """

    per_hospital_train: Dict[str, List[str]] = {}
    per_hospital_val: Dict[str, List[str]] = {}

    for hospital, grp in train_df.groupby("hospital"):
        slides = grp["slide_id"].tolist()
        labels = grp["label"].tolist()
        n_val = max(1, int(round(len(slides) * val_ratio)))
        if len(slides) < 4 or len(set(labels)) < 2:
            per_hospital_train[hospital] = slides
            per_hospital_val[hospital] = []
            continue
        n_val = min(n_val, len(slides) - 2)
        h_train, h_val = train_test_split(
            slides,
            test_size=n_val,
            stratify=labels,
            random_state=seed,
        )
        per_hospital_train[hospital] = h_train
        per_hospital_val[hospital] = h_val

    val_sizes = [len(v) for v in per_hospital_val.values() if len(v) > 0]
    median_cap = int(np.median(val_sizes)) if val_sizes else 0

    train_ids: List[str] = []
    val_ids: List[str] = []
    for hospital in per_hospital_val:
        h_val = per_hospital_val[hospital]
        h_train = per_hospital_train[hospital]
        if len(h_val) > median_cap:
            rng = np.random.RandomState(seed + int(hashlib.md5(hospital.encode()).hexdigest(), 16) % 10_000)
            chosen = set(rng.choice(len(h_val), size=median_cap, replace=False).tolist())
            val_ids.extend([h_val[i] for i in range(len(h_val)) if i in chosen])
            train_ids.extend(h_train + [h_val[i] for i in range(len(h_val)) if i not in chosen])
        else:
            val_ids.extend(h_val)
            train_ids.extend(h_train)

    return train_ids, val_ids


def log_val_composition(
    val_ids: Sequence[str],
    train_df: pd.DataFrame,
    logger: Any,
    fold_label: str,
) -> None:
    """Log per-hospital composition of the val set for a given fold."""

    from collections import Counter

    id_to_hospital = dict(zip(train_df["slide_id"], train_df["hospital"]))
    counts = Counter(id_to_hospital.get(sid, "Unknown") for sid in val_ids)
    total = len(val_ids)
    logger.info(f"  Val composition ({fold_label}, n={total}):")
    for hospital in sorted(counts):
        n = counts[hospital]
        pct = (100 * n / total) if total else 0.0
        logger.info(f"    {hospital:15s} {n:4d}  ({pct:.0f}%)")


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved runtime configuration for LOSO training."""

    config_path: str
    csv_path: str
    feature_dir: str
    output_dir: str
    device: torch.device
    num_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    num_regions: int
    input_dim: int
    hidden_dim: int
    dropout: float
    val_ratio: float
    val_split_strategy: str
    seed: int
    use_fft: bool
    hospitals: List[str]
    num_workers: int
    use_wandb: bool
    wandb_project: str
    deterministic: bool
    benchmark: bool
    causal_weight: float
    causal_warmup: int
    causal_mask_k: int
    warmup_epochs: int
    equal_weight_fusion: bool


class WandbLogger:
    """Small wrapper to keep optional Weights & Biases usage non-invasive."""

    def __init__(self, enabled: bool, project: str, config: Dict[str, Any]) -> None:
        self._run = None
        if not enabled:
            return
        if wandb is None:
            print("[WARN] wandb logging requested but wandb is not installed.")
            return
        self._run = wandb.init(project=project, config=config)

    @property
    def enabled(self) -> bool:
        """Return ``True`` when an active wandb run exists."""

        return self._run is not None

    def log(self, payload: Dict[str, Any]) -> None:
        """Log a metrics payload if wandb is active."""

        if self._run is not None:
            self._run.log(payload)

    def finish(self) -> None:
        """Finish the wandb run if one was created."""

        if self._run is not None:
            self._run.finish()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the canonical trainer."""

    parser = argparse.ArgumentParser(description="CHIME_MIL LOSO trainer")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "config_genbio.yaml"), help="Path to config.yaml (default: auto-discover).")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--eval_checkpoint_root", default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override config training.seed")
    parser.add_argument("--test_hospital", choices=DEFAULT_HOSPITALS, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--select_metric", choices=["auc", "bal_acc", "f1", "combo"], default="auc")
    parser.add_argument("--threshold_metric", choices=["balanced_acc", "f1", "acc"], default="balanced_acc")
    parser.add_argument("--loss_type", choices=["focal", "weighted_ce"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--equal_weight_fusion", action="store_true", help="Use uniform 1/3 fusion weights instead of the learned softmax gate.")
    parser.add_argument("--site_means_path", default=None, help="Override site_means npz path.")
    parser.add_argument("--align_path", default=None, help="Higher-order align npz (ZCA/CORAL); overrides mean-centring.")
    parser.add_argument(
        "--val_split",
        choices=["random", "stratified"],
        default=None,
        help="Validation split strategy. Overrides training.val_split in config.yaml.",
    )
    parser.add_argument("--use_wandb", action="store_true", help="Force-enable wandb logging for this run.")
    parser.add_argument("--disable_wandb", action="store_true", help="Force-disable wandb logging for this run.")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> Tuple[RuntimeConfig, Dict[str, Any]]:
    """Load ``config.yaml`` and resolve runtime values for training."""

    config = load_config(args.config)
    config_path = args.config or str((PROJECT_ROOT / "configs" / "config_genbio.yaml").resolve())
    paths = get_paths_from_config(config)
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    loso_cfg = config.get("loso", {})
    logging_cfg = config.get("logging", {})
    reproducibility_cfg = config.get("reproducibility", {})
    causal_cfg = model_cfg.get("causal_loss", {})

    feature_dir = os.path.join(paths["feature_dir"], "mag20x")
    output_dir = args.output_dir or str(DEFAULT_OUTPUT_DIR)

    use_wandb = bool(logging_cfg.get("use_wandb", False))
    if args.use_wandb:
        use_wandb = True
    if args.disable_wandb:
        use_wandb = False

    runtime = RuntimeConfig(
        config_path=config_path,
        csv_path=paths["csv_path"],
        feature_dir=feature_dir,
        output_dir=output_dir,
        device=torch.device(training_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"),
        num_epochs=int(args.num_epochs or training_cfg.get("num_epochs", 100)),
        patience=int(training_cfg.get("patience", 10)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-3)),
        num_regions=int(model_cfg.get("num_regions", 16)),
        input_dim=int(model_cfg.get("input_dim", 1024)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        val_ratio=float(training_cfg.get("val_ratio", 0.15)),
        val_split_strategy=str(args.val_split or training_cfg.get("val_split", "random")).lower(),
        seed=int(args.seed if args.seed is not None else training_cfg.get("seed", 42)),
        use_fft=False,
        hospitals=list(loso_cfg.get("hospitals", DEFAULT_HOSPITALS)),
        num_workers=int(training_cfg.get("num_workers", 4)),
        use_wandb=use_wandb,
        wandb_project=str(logging_cfg.get("wandb_project", "chime-mil-loso")),
        deterministic=bool(reproducibility_cfg.get("deterministic", True)),
        benchmark=bool(reproducibility_cfg.get("benchmark", False)),
        causal_weight=float(causal_cfg.get("weight", 0.3)),
        causal_warmup=int(causal_cfg.get("warmup_epochs", 10)),
        causal_mask_k=int(causal_cfg.get("k_mask", 3)),
        warmup_epochs=int(args.warmup_epochs),
        equal_weight_fusion=bool(args.equal_weight_fusion),
    )
    return runtime, config


def configure_reproducibility(runtime: RuntimeConfig) -> None:
    """Apply deterministic seeds and cuDNN options."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(runtime.seed)
    torch.manual_seed(runtime.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runtime.seed)
    torch.backends.cudnn.deterministic = runtime.deterministic
    torch.backends.cudnn.benchmark = runtime.benchmark
    if runtime.deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def build_criteria(
    train_labels: Sequence[int],
    device: torch.device,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
) -> Tuple[torch.nn.Module, CausalLoss]:
    """Construct the supervised and causal loss modules."""

    class_counts = np.bincount(np.asarray(train_labels), minlength=2)
    total = len(train_labels)
    class_weights = torch.FloatTensor(
        [(total / (2 * class_counts[0])) ** 2, (total / (2 * class_counts[1])) ** 2]
    ).to(device)
    class_weights = class_weights / class_weights.mean()

    if loss_type == "focal":
        criterion: torch.nn.Module = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    elif loss_type == "weighted_ce":
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    causal_fn = CausalLoss(
        weight=class_weights,
        use_focal=(loss_type == "focal"),
        gamma=focal_gamma,
    ).to(device)
    return criterion, causal_fn


def compute_total_loss(
    model: CHIME_MIL,
    model_outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    runtime: RuntimeConfig,
    causal_fn: Optional[CausalLoss] = None,
    epoch: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine main, auxiliary, and causal losses for one batch."""

    logits = model_outputs["logits"]
    patch_logits = model_outputs["patch_logits"]
    region_logits = model_outputs["region_logits"]
    graph_logits = model_outputs["graph_logits"]

    final_loss = criterion(logits, labels)
    patch_loss = criterion(patch_logits, labels)
    region_loss = criterion(region_logits, labels)
    graph_loss = criterion(graph_logits, labels)

    total_loss = final_loss
    total_loss = total_loss + LAMBDA_PATCH * patch_loss
    total_loss = total_loss + LAMBDA_REGION * region_loss
    total_loss = total_loss + LAMBDA_GRAPH * graph_loss

    aux_value = 0.0
    if causal_fn is not None and epoch >= runtime.causal_warmup:
        importance = model_outputs["graph_importance"]
        region_feats = model_outputs["region_feats"]
        region_coords = model_outputs["region_coords"]
        batch_size, num_regions = importance.shape
        k_mask = min(runtime.causal_mask_k, num_regions)
        _, topk = torch.topk(importance, k=k_mask, dim=1)
        rand = torch.randint(0, num_regions, (batch_size, k_mask), device=importance.device)
        batch_idx = torch.arange(batch_size, device=importance.device).unsqueeze(1).expand(-1, k_mask)

        factual = region_feats.clone()
        random_mask = region_feats.clone()
        factual[batch_idx, topk] = 0.0
        random_mask[batch_idx, rand] = 0.0

        logits_causal, _, _ = model.forward_graph(factual, region_coords)
        logits_random, _, _ = model.forward_graph(random_mask, region_coords)
        aux_loss, _ = causal_fn(graph_logits, logits_causal, logits_random, labels)
        total_loss = total_loss + runtime.causal_weight * aux_loss
        aux_value = float(aux_loss.item())

    loss_dict = {
        "final": float(final_loss.item()),
        "patch": float(patch_loss.item()),
        "region": float(region_loss.item()),
        "graph": float(graph_loss.item()),
        "causal": aux_value,
    }
    return total_loss, loss_dict


def train_epoch(
    model: CHIME_MIL,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    runtime: RuntimeConfig,
    causal_fn: Optional[CausalLoss] = None,
    epoch: int = 0,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float, float, float, Dict[str, float]]:
    """Train the model for one epoch and return aggregate metrics."""

    model.train()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    loss_parts = {"final": 0.0, "patch": 0.0, "region": 0.0, "graph": 0.0, "causal": 0.0}
    use_amp = scaler is not None and device.type == "cuda"

    for batch in tqdm(loader, desc="Train", leave=False):
        feats, coords, label = batch[0]
        if feats.shape[0] < 2:
            continue
        feats = feats.unsqueeze(0).to(device)
        coords = coords.unsqueeze(0).to(device)
        label = label.unsqueeze(0).to(device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=use_amp):
            outputs = model(feats, coords)
            loss, parts = compute_total_loss(
                model,
                outputs,
                label,
                criterion,
                runtime,
                causal_fn=causal_fn,
                epoch=epoch,
            )

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        for key in loss_parts:
            loss_parts[key] += parts[key]

        logits = outputs["logits"]
        all_preds.append(torch.argmax(logits, dim=1).cpu().item())
        all_labels.append(label.cpu().item())

    if not all_labels:
        return 0.0, 0.0, 0.0, 0.0, loss_parts

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    c0 = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    c1 = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    denom = max(len(loader), 1)
    return total_loss / denom, acc, c0, c1, {key: value / denom for key, value in loss_parts.items()}


def evaluate(
    model: CHIME_MIL,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float], List[int], List[int], List[float]]:
    """Evaluate a loader and return thresholded metrics plus probabilities."""

    model.eval()
    total_loss = 0.0
    all_labels: List[int] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            feats, coords, label = batch[0]
            if feats.shape[0] < 2:
                continue
            feats = feats.unsqueeze(0).to(device)
            coords = coords.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)

            outputs = model(feats, coords)
            logits = outputs["logits"]
            loss = criterion(logits, label)
            total_loss += float(loss.item())

            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_labels.append(label.cpu().item())
            all_probs.append(probs)

    if not all_labels:
        empty_metrics = {
            "acc": 0.0,
            "auc": 0.0,
            "balanced_acc": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "ppv": 0.0,
        }
        return 0.0, empty_metrics, [], [], []

    probs = np.array(all_probs)
    prob_cls1 = probs[:, 1].tolist()
    preds = probs_to_preds(prob_cls1, threshold=threshold).tolist()
    metrics = compute_metrics(all_labels, preds, prob_cls1)
    return total_loss / max(len(loader), 1), metrics, preds, all_labels, prob_cls1


def get_selection_score(metrics: Dict[str, float], select_metric: str) -> float:
    """Compute the model-selection score from validation metrics."""

    if select_metric == "auc":
        return float(metrics["auc"])
    if select_metric == "bal_acc":
        return float(metrics["balanced_acc"])
    if select_metric == "f1":
        return float(metrics["f1"])
    if select_metric == "combo":
        return 0.5 * float(metrics["auc"]) + 0.5 * float(metrics["balanced_acc"])
    raise ValueError(f"Unsupported select_metric: {select_metric}")


def build_model(runtime: RuntimeConfig) -> CHIME_MIL:
    """Instantiate the canonical CHIME_MIL model."""

    return CHIME_MIL(
        input_dim=runtime.input_dim,
        hidden_dim=runtime.hidden_dim,
        num_classes=2,
        num_regions=runtime.num_regions,
        dropout=runtime.dropout,
        use_fft=runtime.use_fft,
        equal_weight_fusion=runtime.equal_weight_fusion,
    ).to(runtime.device)


def build_loader_kwargs(runtime: RuntimeConfig) -> Dict[str, Any]:
    """Create consistent dataloader kwargs for the MIL setting."""

    return {
        "collate_fn": collate_single,
        "num_workers": runtime.num_workers,
        "pin_memory": runtime.device.type == "cuda",
    }


def evaluate_existing_fold(
    fold_idx: int,
    test_hospital: str,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    full_dataset: UniCLAMDatasetFixed,
    checkpoint_root: str,
    logger: Any,
    threshold_metric: str,
    loss_type: str,
    focal_gamma: float,
    runtime: RuntimeConfig,
) -> Dict[str, Any]:
    """Run eval-only mode using a previously saved fold checkpoint."""

    fold_dir = os.path.join(checkpoint_root, f"fold_{fold_idx}_{test_hospital}")
    checkpoint_path = os.path.join(fold_dir, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("\n" + "=" * 70)
    logger.info(f"EVAL-ONLY FOLD {fold_idx}/5 - Test: {test_hospital}")
    logger.info(f"  Reusing checkpoint: {checkpoint_path}")
    logger.info(f"  Val:{len(val_ids)} Test:{len(test_ids)}")

    id_to_index = {slide_id: idx for idx, slide_id in enumerate(full_dataset.slide_ids)}
    train_idx = [id_to_index[slide_id] for slide_id in train_ids if slide_id in id_to_index]
    val_idx = [id_to_index[slide_id] for slide_id in val_ids if slide_id in id_to_index]
    test_idx = [id_to_index[slide_id] for slide_id in test_ids if slide_id in id_to_index]

    train_labels = [full_dataset.labels[idx] for idx in train_idx]
    criterion, _ = build_criteria(train_labels, runtime.device, loss_type=loss_type, focal_gamma=focal_gamma)
    loader_kwargs = build_loader_kwargs(runtime)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1, shuffle=False, **loader_kwargs)

    model = build_model(runtime)
    checkpoint = torch.load(checkpoint_path, map_location=runtime.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_metrics_050, _, val_labels, val_probs = evaluate(model, val_loader, criterion, runtime.device, threshold=0.5)
    threshold_state = find_best_threshold(val_labels, val_probs, metric=threshold_metric)
    selected_threshold = threshold_state["threshold"]
    logger.info(
        f"    Threshold calibration on val: metric={threshold_metric} threshold={selected_threshold:.4f} "
        f"score={threshold_state['score']:.4f}"
    )
    _, test_metrics, test_preds, test_labels, test_probs = evaluate(
        model,
        test_loader,
        criterion,
        runtime.device,
        threshold=selected_threshold,
    )
    logger.info(
        f"    Acc:{test_metrics['acc']:.4f} AUC:{test_metrics['auc']:.4f} "
        f"C0:{test_metrics['sensitivity']:.4f} C1:{test_metrics['specificity']:.4f}"
    )

    result = {
        "fold": fold_idx,
        "test_hospital": test_hospital,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "best_epoch": checkpoint.get("epoch", -1) + 1,
        "best_val_auc": round(float(checkpoint.get("val_auc", 0.0)), 4),
        "best_val_acc": round(float(checkpoint.get("val_acc", 0.0)), 4),
        "best_val_bal_acc": round(float(checkpoint.get("val_bal_acc", 0.0)), 4),
        "best_val_f1": round(float(checkpoint.get("val_f1", 0.0)), 4),
        "best_val_ppv": round(float(checkpoint.get("val_ppv", 0.0)), 4),
        "select_metric": checkpoint.get("select_metric", "auc"),
        "select_score": round(float(checkpoint.get("select_score", checkpoint.get("val_auc", 0.0))), 4),
        "selected_threshold": round(float(selected_threshold), 6),
        "threshold_metric": threshold_metric,
        "val_metrics_argmax": val_metrics_050,
        "val_metrics_at_selected_threshold": threshold_state["metrics"],
        "test_acc": round(test_metrics["acc"], 4),
        "test_auc": round(test_metrics["auc"], 4),
        "test_balanced_acc": round(test_metrics["balanced_acc"], 4),
        "test_c0": round(test_metrics["sensitivity"], 4),
        "test_c1": round(test_metrics["specificity"], 4),
        "sensitivity": test_metrics["sensitivity"],
        "specificity": test_metrics["specificity"],
        "f1": test_metrics["f1"],
        "ppv": test_metrics["ppv"],
        "preds": test_preds,
        "labels": test_labels,
        "probs": test_probs,
    }
    with open(os.path.join(fold_dir, "result_eval_only.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4)
    return result


def train_fold(
    fold_idx: int,
    test_hospital: str,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    full_dataset: UniCLAMDatasetFixed,
    output_dir: str,
    logger: Any,
    runtime: RuntimeConfig,
    num_epochs: int,
    select_metric: str,
    threshold_metric: str,
    loss_type: str,
    focal_gamma: float,
    wandb_logger: Optional[WandbLogger] = None,
) -> Dict[str, Any]:
    """Train one LOSO fold and return the saved metrics payload."""

    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}_{test_hospital}")
    os.makedirs(fold_dir, exist_ok=True)

    logger.info("\n" + "=" * 70)
    logger.info(f"FOLD {fold_idx}/5 - Test: {test_hospital}")
    logger.info(f"  Train:{len(train_ids)} Val:{len(val_ids)} Test:{len(test_ids)}")

    id_to_index = {slide_id: idx for idx, slide_id in enumerate(full_dataset.slide_ids)}
    train_idx = [id_to_index[slide_id] for slide_id in train_ids if slide_id in id_to_index]
    val_idx = [id_to_index[slide_id] for slide_id in val_ids if slide_id in id_to_index]
    test_idx = [id_to_index[slide_id] for slide_id in test_ids if slide_id in id_to_index]

    train_labels = [full_dataset.labels[idx] for idx in train_idx]
    class_counts = np.bincount(np.asarray(train_labels), minlength=2)
    logger.info(f"  Class meta(0):{class_counts[0]} non-meta(1):{class_counts[1]}")

    loader_kwargs = build_loader_kwargs(runtime)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(runtime.seed), **loader_kwargs)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1, shuffle=False, **loader_kwargs)

    criterion, causal_fn = build_criteria(train_labels, runtime.device, loss_type=loss_type, focal_gamma=focal_gamma)
    model = build_model(runtime)
    optimizer = optim.AdamW(model.parameters(), lr=runtime.learning_rate, weight_decay=runtime.weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = max(runtime.warmup_epochs * len(train_loader), 0)
    if warmup_steps > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])
        logger.info(f"  LR: warmup {runtime.warmup_epochs}ep then cosine to 1e-6")
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)

    metrics_csv = os.path.join(fold_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow([
            "Ep", "TrL", "TrA", "TrC0", "TrC1", "TrFinal", "TrPatch", "TrRegion", "TrGraph", "TrCausal",
            "VlL", "VlA", "VlAUC", "VlBalA", "VlC0", "VlC1", "VlF1", "VlPPV", "SelMetric", "SelScore", "LR", "Best", "Pat",
        ])

    best_score = float("-inf")
    patience = 0
    checkpoint_path = os.path.join(fold_dir, "best_model.pth")
    scaler = GradScaler(enabled=runtime.device.type == "cuda")

    for epoch in range(num_epochs):
        logger.info(f"  Epoch [{epoch + 1}/{num_epochs}]")
        train_loss, train_acc, train_c0, train_c1, loss_parts = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            runtime.device,
            runtime,
            causal_fn=causal_fn,
            epoch=epoch,
            scheduler=scheduler,
            scaler=scaler,
        )
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, runtime.device, threshold=0.5)
        lr = optimizer.param_groups[0]["lr"]
        val_c0 = val_metrics["sensitivity"]
        val_c1 = val_metrics["specificity"]
        select_score = get_selection_score(val_metrics, select_metric)

        logger.info(
            f"    Train L:{train_loss:.4f} A:{train_acc:.4f} C0:{train_c0:.3f} C1:{train_c1:.3f} "
            f"| final:{loss_parts['final']:.4f} patch:{loss_parts['patch']:.4f} "
            f"region:{loss_parts['region']:.4f} graph:{loss_parts['graph']:.4f} causal:{loss_parts['causal']:.4f}"
        )
        logger.info(
            f"    Val   L:{val_loss:.4f} A:{val_metrics['acc']:.4f} AUC:{val_metrics['auc']:.4f} "
            f"BalA:{val_metrics['balanced_acc']:.4f} F1:{val_metrics['f1']:.4f} PPV:{val_metrics['ppv']:.4f}"
        )

        is_best = select_score > best_score
        with open(metrics_csv, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow([
                epoch + 1,
                train_loss,
                train_acc,
                train_c0,
                train_c1,
                loss_parts["final"],
                loss_parts["patch"],
                loss_parts["region"],
                loss_parts["graph"],
                loss_parts["causal"],
                val_loss,
                val_metrics["acc"],
                val_metrics["auc"],
                val_metrics["balanced_acc"],
                val_c0,
                val_c1,
                val_metrics["f1"],
                val_metrics["ppv"],
                select_metric,
                round(select_score, 4),
                lr,
                is_best,
                patience,
            ])

        if wandb_logger is not None and wandb_logger.enabled:
            wandb_logger.log({
                "fold": fold_idx,
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/c0": train_c0,
                "train/c1": train_c1,
                "train/loss_final": loss_parts["final"],
                "train/loss_patch": loss_parts["patch"],
                "train/loss_region": loss_parts["region"],
                "train/loss_graph": loss_parts["graph"],
                "train/loss_causal": loss_parts["causal"],
                "val/loss": val_loss,
                "val/acc": val_metrics["acc"],
                "val/auc": val_metrics["auc"],
                "val/balanced_acc": val_metrics["balanced_acc"],
                "val/f1": val_metrics["f1"],
                "val/ppv": val_metrics["ppv"],
                "val/select_score": select_score,
                "lr": lr,
            })

        if is_best:
            best_score = select_score
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_metrics["acc"],
                    "val_auc": val_metrics["auc"],
                    "val_bal_acc": val_metrics["balanced_acc"],
                    "val_f1": val_metrics["f1"],
                    "val_ppv": val_metrics["ppv"],
                    "select_metric": select_metric,
                    "select_score": round(float(select_score), 4),
                },
                checkpoint_path,
            )
            logger.info(f"    Best saved {select_metric}:{select_score:.4f}")
        else:
            patience += 1
            logger.debug(f"    No improve. Pat:{patience}/{runtime.patience}")

        if patience >= runtime.patience:
            logger.info(f"    Early stop ep {epoch + 1}")
            break

    logger.info(f"  --- Test: {test_hospital} ---")
    checkpoint = torch.load(checkpoint_path, map_location=runtime.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_metrics_050, _, val_labels, val_probs = evaluate(model, val_loader, criterion, runtime.device, threshold=0.5)
    threshold_state = find_best_threshold(val_labels, val_probs, metric=threshold_metric)
    selected_threshold = threshold_state["threshold"]
    logger.info(
        f"    Threshold calibration on val: metric={threshold_metric} threshold={selected_threshold:.4f} "
        f"score={threshold_state['score']:.4f}"
    )
    _, test_metrics, test_preds, test_labels, test_probs = evaluate(
        model,
        test_loader,
        criterion,
        runtime.device,
        threshold=selected_threshold,
    )
    logger.info(
        f"    Acc:{test_metrics['acc']:.4f} AUC:{test_metrics['auc']:.4f} "
        f"C0:{test_metrics['sensitivity']:.4f} C1:{test_metrics['specificity']:.4f}"
    )
    logger.info(classification_report(test_labels, test_preds, zero_division=0))
    logger.info(
        "    Sens:%s Spec:%s F1:%s PPV:%s"
        % (
            round(test_metrics["sensitivity"], 4),
            round(test_metrics["specificity"], 4),
            round(test_metrics["f1"], 4),
            round(test_metrics["ppv"], 4),
        )
    )

    result = {
        "fold": fold_idx,
        "test_hospital": test_hospital,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "best_epoch": checkpoint["epoch"] + 1,
        "best_val_auc": round(float(checkpoint["val_auc"]), 4),
        "best_val_acc": round(float(checkpoint["val_acc"]), 4),
        "best_val_bal_acc": round(float(checkpoint["val_bal_acc"]), 4),
        "best_val_f1": round(float(checkpoint["val_f1"]), 4),
        "best_val_ppv": round(float(checkpoint["val_ppv"]), 4),
        "select_metric": checkpoint["select_metric"],
        "select_score": round(float(checkpoint["select_score"]), 4),
        "selected_threshold": round(float(selected_threshold), 6),
        "threshold_metric": threshold_metric,
        "val_metrics_argmax": val_metrics_050,
        "val_metrics_at_selected_threshold": threshold_state["metrics"],
        "test_acc": round(test_metrics["acc"], 4),
        "test_auc": round(test_metrics["auc"], 4),
        "test_balanced_acc": round(test_metrics["balanced_acc"], 4),
        "test_c0": round(test_metrics["sensitivity"], 4),
        "test_c1": round(test_metrics["specificity"], 4),
        "sensitivity": test_metrics["sensitivity"],
        "specificity": test_metrics["specificity"],
        "f1": test_metrics["f1"],
        "ppv": test_metrics["ppv"],
        "preds": test_preds,
        "labels": test_labels,
        "probs": test_probs,
    }
    with open(os.path.join(fold_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=4)

    if wandb_logger is not None and wandb_logger.enabled:
        wandb_logger.log({
            "fold": fold_idx,
            "test/acc": result["test_acc"],
            "test/auc": result["test_auc"],
            "test/balanced_acc": result["test_balanced_acc"],
            "test/sensitivity": result["sensitivity"],
            "test/specificity": result["specificity"],
            "test/f1": result["f1"],
            "test/ppv": result["ppv"],
            "test/selected_threshold": result["selected_threshold"],
        })
    return result


def main() -> None:
    """Run canonical LOSO training or evaluation for CHIME_MIL."""

    args = parse_args()
    runtime, _ = build_runtime_config(args)
    os.makedirs(runtime.output_dir, exist_ok=True)
    logger, timestamp = setup_logging(runtime.output_dir, "chime_loso")
    configure_reproducibility(runtime)

    logger.info("=" * 70)
    logger.info("LOSO TRAINING - CHIME_MIL (20x only)")
    logger.info("=" * 70)
    logger.info(f"Config path: {runtime.config_path}")
    logger.info(f"Device:{runtime.device} Seed:{runtime.seed}")
    logger.info(
        f"Config: eval_only={args.eval_only} select_metric={args.select_metric} "
        f"threshold_metric={args.threshold_metric} loss_type={args.loss_type} "
        f"focal_gamma={args.focal_gamma} max_instances={args.max_instances} "
        f"sample_seed={args.sample_seed or runtime.seed} use_wandb={runtime.use_wandb}"
    )
    logger.info(
        f"Resolved paths: csv={runtime.csv_path} feature_dir={runtime.feature_dir} output_dir={runtime.output_dir}"
    )

    wandb_logger = WandbLogger(
        enabled=runtime.use_wandb,
        project=runtime.wandb_project,
        config={
            "config_path": runtime.config_path,
            "output_dir": runtime.output_dir,
            "num_epochs": runtime.num_epochs,
            "patience": runtime.patience,
            "learning_rate": runtime.learning_rate,
            "weight_decay": runtime.weight_decay,
            "num_regions": runtime.num_regions,
            "hidden_dim": runtime.hidden_dim,
            "dropout": runtime.dropout,
            "val_ratio": runtime.val_ratio,
            "val_split_strategy": runtime.val_split_strategy,
            "seed": runtime.seed,
            "select_metric": args.select_metric,
            "threshold_metric": args.threshold_metric,
            "loss_type": args.loss_type,
            "focal_gamma": args.focal_gamma,
            "max_instances": args.max_instances,
            "sample_seed": args.sample_seed or runtime.seed,
            "causal_weight": runtime.causal_weight,
            "causal_warmup": runtime.causal_warmup,
            "causal_mask_k": runtime.causal_mask_k,
        },
    )

    dataframe = pd.read_csv(runtime.csv_path)
    dataframe["slide_id"] = dataframe["slide_id"].astype(str)
    dataframe["hospital"] = dataframe["slide_id"].apply(get_hospital)
    dataframe = dataframe[dataframe["hospital"] != "Unknown"].reset_index(drop=True)
    logger.info("\n" + dataframe.groupby(["hospital", "label"]).size().unstack(fill_value=0).to_string())

    if args.align_path:
        full_dataset = UniCLAMDatasetAlign(
            runtime.csv_path, runtime.feature_dir, args.align_path,
            use_h5_features=True,
            max_instances=args.max_instances,
            sample_seed=args.sample_seed or runtime.seed,
        )
    else:
        full_dataset = UniCLAMDatasetFixed(
            runtime.csv_path, runtime.feature_dir,
            use_h5_features=True,
            max_instances=args.max_instances,
            sample_seed=args.sample_seed or runtime.seed,
            site_means_path=args.site_means_path,
        )
    valid_ids = set(dataframe["slide_id"].tolist())
    full_dataset.slide_ids = [slide_id for slide_id in full_dataset.slide_ids if slide_id in valid_ids]
    full_dataset.labels = [
        full_dataset.labels[idx]
        for idx, slide_id in enumerate(full_dataset.df["slide_id"].tolist())
        if slide_id in valid_ids
    ]
    logger.info(f"Dataset: {len(full_dataset.slide_ids)} slides")

    results: List[Dict[str, Any]] = []
    if runtime.val_split_strategy not in ("random", "stratified"):
        raise ValueError(
            f"Unsupported val_split_strategy: {runtime.val_split_strategy} (expected 'random' or 'stratified')"
        )
    logger.info(f"Val split strategy: {runtime.val_split_strategy}")

    for fold_idx, hospital in enumerate(runtime.hospitals, 1):
        test_ids = dataframe[dataframe["hospital"] == hospital]["slide_id"].tolist()
        train_df = dataframe[dataframe["hospital"] != hospital]
        if runtime.val_split_strategy == "stratified":
            train_ids, val_ids = hospital_stratified_val_split(
                train_df,
                val_ratio=runtime.val_ratio,
                seed=runtime.seed,
            )
        else:
            train_ids, val_ids = train_test_split(
                train_df["slide_id"].tolist(),
                test_size=runtime.val_ratio,
                stratify=train_df["label"].tolist(),
                random_state=runtime.seed,
            )
        if args.test_hospital is not None and hospital != args.test_hospital:
            continue
        if runtime.val_split_strategy == "stratified":
            log_val_composition(val_ids, train_df, logger, fold_label=f"test={hospital}")

        if args.eval_only:
            checkpoint_root = args.eval_checkpoint_root or runtime.output_dir
            result = evaluate_existing_fold(
                fold_idx,
                hospital,
                train_ids,
                val_ids,
                test_ids,
                full_dataset,
                checkpoint_root,
                logger,
                args.threshold_metric,
                args.loss_type,
                args.focal_gamma,
                runtime,
            )
        else:
            result = train_fold(
                fold_idx,
                hospital,
                train_ids,
                val_ids,
                test_ids,
                full_dataset,
                runtime.output_dir,
                logger,
                runtime,
                runtime.num_epochs,
                args.select_metric,
                args.threshold_metric,
                args.loss_type,
                args.focal_gamma,
                wandb_logger=wandb_logger,
            )
        results.append(result)

    if not results:
        raise ValueError("No folds were selected. Check --test_hospital.")

    logger.info("\n" + "=" * 70)
    logger.info("LOSO SUMMARY - CHIME_MIL (20x only)")
    logger.info("=" * 70)

    accs = [result["test_acc"] for result in results]
    aucs = [result["test_auc"] for result in results]
    bal_accs = [result["test_balanced_acc"] for result in results]
    all_true: List[int] = []
    all_pred: List[int] = []
    all_prob: List[float] = []
    for result in results:
        all_true += result.get("labels", [])
        all_pred += result.get("preds", [])
        all_prob += result.get("probs", [])

    if all_true:
        ci = bootstrap_ci(all_true, all_pred, all_prob)
        logger.info("Bootstrap 95% CI (pooled LOSO):")
        logger.info(format_ci(ci))

    header = "Hospital".ljust(15) + "Acc".rjust(8) + "AUC".rjust(8) + "C0".rjust(8) + "C1".rjust(8) + "N_test".rjust(8)
    logger.info("\n" + header)
    logger.info("-" * 55)
    for result in results:
        logger.info(
            f"{result['test_hospital']:<15}{result['test_acc']:>8.4f}{result['test_auc']:>8.4f}"
            f"{result['test_c0']:>8.4f}{result['test_c1']:>8.4f}{result['n_test']:>8}"
        )
    logger.info("-" * 55)
    logger.info(f"{'Mean':<15}{np.mean(accs):>8.4f}{np.mean(aucs):>8.4f}")
    logger.info(f"{'Std':<15}{np.std(accs):>8.4f}{np.std(aucs):>8.4f}")

    summary = {
        "timestamp": timestamp,
        "model": "CHIME_MIL",
        "features": "20x_only",
        "config_path": runtime.config_path,
        "mean_acc": round(float(np.mean(accs)), 4),
        "std_acc": round(float(np.std(accs)), 4),
        "mean_auc": round(float(np.mean(aucs)), 4),
        "std_auc": round(float(np.std(aucs)), 4),
        "mean_balanced_acc": round(float(np.mean(bal_accs)), 4),
        "std_balanced_acc": round(float(np.std(bal_accs)), 4),
        "select_metric": args.select_metric,
        "threshold_metric": args.threshold_metric,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
        "val_split_strategy": runtime.val_split_strategy,
        "max_instances": args.max_instances,
        "sample_seed": args.sample_seed or runtime.seed,
        "folds": results,
    }
    summary_path = os.path.join(runtime.output_dir, f"chime_loso_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)
    logger.info(f"Summary saved: {summary_path}")

    if wandb_logger.enabled:
        wandb_logger.log({
            "summary/mean_acc": summary["mean_acc"],
            "summary/mean_auc": summary["mean_auc"],
            "summary/mean_balanced_acc": summary["mean_balanced_acc"],
        })
    wandb_logger.finish()


if __name__ == "__main__":
    main()
