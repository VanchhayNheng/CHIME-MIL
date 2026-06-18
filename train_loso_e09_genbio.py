"""LOSO Training - CHIME_MIL (20x only)

Unified hierarchical MIL training:
- patch auxiliary head
- region auxiliary head
- graph auxiliary head
- fused final head
- causal regularization on graph importance
"""

import argparse

import csv
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from causal_loss import CausalLoss
from chime_mil import CHIME_MIL
from dataset_genbio_shim import UniCLAMDatasetFixed
from focal_loss import FocalLoss
from metrics_utils import bootstrap_ci, compute_metrics, find_best_threshold, format_ci, probs_to_preds


CSV_PATH = "/path/to/data/dataset_csv/tumor_vs_normal_dummy_clean.csv"
FEATURE_DIR = '/path/to/data/GENBIO_PATHFM_FEATURES/mag20x'
OUTPUT_DIR = 'results/results_loso_e09_fixed_genbio_20260421'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
PATIENCE = 10
LR = 1e-5
WEIGHT_DECAY = 1e-3
NUM_REGIONS = 16
HIDDEN_DIM = 256
DROPOUT = 0.5
VAL_RATIO = 0.15
SEED = 42
USE_FFT = False
HOSPITALS = ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E"]

LAMBDA_PATCH = 0.4
LAMBDA_REGION = 0.0
LAMBDA_GRAPH = 0.0
LAMBDA_CAUSAL = 0.0
CAUSAL_WARMUP = 10
CAUSAL_MASK_K = 3


def get_hospital(slide_id):
    slide_id = str(slide_id).upper().replace(" ", "").replace("_", "-")
    if slide_id.startswith("SC-01") or slide_id.startswith("SC01"):
        return "Site_A"
    if slide_id.startswith("SC-04") or slide_id.startswith("SC04"):
        return "Site_B"
    if "SC-03" in slide_id or "SC03" in slide_id:
        return "Site_C"
    if any(slide_id.startswith(prefix) for prefix in ["SC-3", "GC-3", "SC-7"]):
        return "Site_D"
    if slide_id.startswith("SC-02") or slide_id.startswith("SC02"):
        return "Site_E"
    return "Unknown"


def collate_single(batch):
    return batch


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("CHIME_LOSO")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    file_handler = logging.FileHandler(os.path.join(output_dir, f"chime_loso_{timestamp}.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(f"Log: chime_loso_{timestamp}.log")
    return logger, timestamp


def build_criteria(train_labels, loss_type="focal", focal_gamma=2.0):
    class_counts = np.bincount(train_labels)
    total = len(train_labels)
    class_weights = torch.FloatTensor([(total / (2 * class_counts[0])) ** 1, (total / (2 * class_counts[1])) ** 1]).to(DEVICE)
    class_weights = class_weights / class_weights.mean()
    if loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
    elif loss_type == "weighted_ce":
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    causal_fn = CausalLoss(weight=class_weights, use_focal=(loss_type == "focal"), gamma=focal_gamma).to(DEVICE)
    return criterion, causal_fn


def compute_total_loss(model, model_outputs, labels, criterion, causal_fn=None, epoch=0):
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
    if causal_fn is not None and epoch >= CAUSAL_WARMUP:
        importance = model_outputs["graph_importance"]
        # Causal masking must operate on pre-graph region features because
        # forward_graph expects feature_dim-sized region tensors.
        region_feats = model_outputs["region_feats"]
        region_coords = model_outputs["region_coords"]
        batch_size, num_regions = importance.shape
        k_mask = min(CAUSAL_MASK_K, num_regions)
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
        total_loss = total_loss + LAMBDA_CAUSAL * aux_loss
        aux_value = float(aux_loss.item())

    loss_dict = {
        "final": float(final_loss.item()),
        "patch": float(patch_loss.item()),
        "region": float(region_loss.item()),
        "graph": float(graph_loss.item()),
        "causal": aux_value,
    }
    return total_loss, loss_dict


def train_epoch(model, loader, optimizer, criterion, device, causal_fn=None, epoch=0, scheduler=None, scaler=None):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    loss_parts = {"final": 0.0, "patch": 0.0, "region": 0.0, "graph": 0.0, "causal": 0.0}
    use_amp = scaler is not None

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
            loss, parts = compute_total_loss(model, outputs, label, criterion, causal_fn=causal_fn, epoch=epoch)
        if use_amp:
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


def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

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
        return 0.0, {"acc": 0.0, "auc": 0.0, "balanced_acc": 0.0, "sensitivity": 0.0, "specificity": 0.0, "f1": 0.0, "ppv": 0.0}, [], [], []

    probs = np.array(all_probs)
    prob_cls1 = probs[:, 1].tolist()
    preds = probs_to_preds(prob_cls1, threshold=threshold).tolist()
    metrics = compute_metrics(all_labels, preds, prob_cls1)
    return total_loss / max(len(loader), 1), metrics, preds, all_labels, prob_cls1


def get_selection_score(metrics, select_metric):
    if select_metric == "auc":
        return float(metrics["auc"])
    if select_metric == "bal_acc":
        return float(metrics["balanced_acc"])
    if select_metric == "f1":
        return float(metrics["f1"])
    if select_metric == "combo":
        return 0.5 * float(metrics["auc"]) + 0.5 * float(metrics["balanced_acc"])
    raise ValueError(f"Unsupported select_metric: {select_metric}")


def evaluate_existing_fold(
    fold_idx,
    test_hospital,
    train_ids,
    val_ids,
    test_ids,
    full_dataset,
    checkpoint_root,
    logger,
    threshold_metric,
    loss_type,
    focal_gamma,
):
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
    criterion, _ = build_criteria(train_labels, loss_type=loss_type, focal_gamma=focal_gamma)
    loader_kwargs = dict(collate_fn=collate_single, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1, shuffle=False, **loader_kwargs)

    model = CHIME_MIL(
        input_dim=4608,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        num_regions=NUM_REGIONS,
        dropout=DROPOUT,
        use_fft=USE_FFT,
    )
    model = model.to(DEVICE)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_metrics_050, _, val_labels, val_probs = evaluate(model, val_loader, criterion, DEVICE, threshold=0.5)
    threshold_state = find_best_threshold(val_labels, val_probs, metric=threshold_metric)
    selected_threshold = threshold_state["threshold"]
    logger.info(
        f"    Threshold calibration on val: metric={threshold_metric} threshold={selected_threshold:.4f} "
        f"score={threshold_state['score']:.4f}"
    )
    _, test_metrics, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, DEVICE, threshold=selected_threshold
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
    with open(os.path.join(fold_dir, "result_eval_only.json"), "w") as handle:
        json.dump(result, handle, indent=4)
    return result


def train_fold(
    fold_idx,
    test_hospital,
    train_ids,
    val_ids,
    test_ids,
    full_dataset,
    output_dir,
    logger,
    num_epochs,
    select_metric,
    threshold_metric,
    loss_type,
    focal_gamma,
):
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
    class_counts = np.bincount(train_labels)
    logger.info(f"  Class meta(0):{class_counts[0]} non-meta(1):{class_counts[1]}")

    loader_kwargs = dict(collate_fn=collate_single, num_workers=4, pin_memory=True)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(SEED), **loader_kwargs)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=1, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=1, shuffle=False, **loader_kwargs)

    criterion, causal_fn = build_criteria(train_labels, loss_type=loss_type, focal_gamma=focal_gamma)
    model = CHIME_MIL(
        input_dim=4608,
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        num_regions=NUM_REGIONS,
        dropout=DROPOUT,
        use_fft=USE_FFT,
    )
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    metrics_csv = os.path.join(fold_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as handle:
        csv.writer(handle).writerow([
            "Ep", "TrL", "TrA", "TrC0", "TrC1", "TrFinal", "TrPatch", "TrRegion", "TrGraph", "TrCausal",
            "VlL", "VlA", "VlAUC", "VlBalA", "VlC0", "VlC1", "VlF1", "VlPPV", "SelMetric", "SelScore", "LR", "Best", "Pat",
        ])

    best_score = float("-inf")
    patience = 0
    checkpoint_path = os.path.join(fold_dir, "best_model.pth")

    scaler = GradScaler(enabled=True)

    for epoch in range(num_epochs):
        logger.info(f"  Epoch [{epoch + 1}/{num_epochs}]")
        train_loss, train_acc, train_c0, train_c1, loss_parts = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE, causal_fn=causal_fn, epoch=epoch, scheduler=scheduler, scaler=scaler
        )
        val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, criterion, DEVICE, threshold=0.5)
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
        with open(metrics_csv, "a", newline="") as handle:
            csv.writer(handle).writerow([
                epoch + 1, train_loss, train_acc, train_c0, train_c1,
                loss_parts["final"], loss_parts["patch"], loss_parts["region"], loss_parts["graph"], loss_parts["causal"],
                val_loss, val_metrics["acc"], val_metrics["auc"], val_metrics["balanced_acc"], val_c0, val_c1,
                val_metrics["f1"], val_metrics["ppv"], select_metric, round(select_score, 4), lr, is_best, patience,
            ])

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
            logger.debug(f"    No improve. Pat:{patience}/{PATIENCE}")

        if patience >= PATIENCE:
            logger.info(f"    Early stop ep {epoch + 1}")
            break

    logger.info(f"  --- Test: {test_hospital} ---")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, val_metrics_050, _, val_labels, val_probs = evaluate(model, val_loader, criterion, DEVICE, threshold=0.5)
    threshold_state = find_best_threshold(val_labels, val_probs, metric=threshold_metric)
    selected_threshold = threshold_state["threshold"]
    logger.info(
        f"    Threshold calibration on val: metric={threshold_metric} threshold={selected_threshold:.4f} "
        f"score={threshold_state['score']:.4f}"
    )
    _, test_metrics, test_preds, test_labels, test_probs = evaluate(
        model, test_loader, criterion, DEVICE, threshold=selected_threshold
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
    with open(os.path.join(fold_dir, "result.json"), "w") as handle:
        json.dump(result, handle, indent=4)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="CHIME_MIL LOSO trainer")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--eval_checkpoint_root", default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--sample_seed", type=int, default=SEED)
    parser.add_argument("--test_hospital", choices=HOSPITALS, default=None)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--select_metric", choices=["auc", "bal_acc", "f1", "combo"], default="auc")
    parser.add_argument("--threshold_metric", choices=["balanced_acc", "f1", "acc"], default="balanced_acc")
    parser.add_argument("--loss_type", choices=["focal", "weighted_ce"], default="focal")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    logger, timestamp = setup_logging(args.output_dir)
    logger.info("=" * 70)
    logger.info("LOSO TRAINING - CHIME_MIL (20x only)")
    logger.info("=" * 70)
    logger.info(f"Device:{DEVICE} Seed:{SEED}")
    logger.info(
        f"Config: eval_only={args.eval_only} select_metric={args.select_metric} "
        f"threshold_metric={args.threshold_metric} loss_type={args.loss_type} "
        f"focal_gamma={args.focal_gamma} max_instances={args.max_instances} sample_seed={args.sample_seed}"
    )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    dataframe = pd.read_csv(CSV_PATH)
    dataframe["slide_id"] = dataframe["slide_id"].astype(str)
    dataframe["hospital"] = dataframe["slide_id"].apply(get_hospital)
    dataframe = dataframe[dataframe["hospital"] != "Unknown"].reset_index(drop=True)
    logger.info("\n" + dataframe.groupby(["hospital", "label"]).size().unstack(fill_value=0).to_string())

    full_dataset = UniCLAMDatasetFixed(
        CSV_PATH,
        FEATURE_DIR,
        use_h5_features=True,
        max_instances=args.max_instances,
        sample_seed=args.sample_seed,
    )
    valid_ids = set(dataframe["slide_id"].tolist())
    full_dataset.slide_ids = [slide_id for slide_id in full_dataset.slide_ids if slide_id in valid_ids]
    full_dataset.labels = [full_dataset.labels[idx] for idx, slide_id in enumerate(full_dataset.df["slide_id"].tolist()) if slide_id in valid_ids]
    logger.info(f"Dataset: {len(full_dataset.slide_ids)} slides")

    results = []
    for fold_idx, hospital in enumerate(HOSPITALS, 1):
        test_ids = dataframe[dataframe["hospital"] == hospital]["slide_id"].tolist()
        train_df = dataframe[dataframe["hospital"] != hospital]
        train_ids, val_ids = train_test_split(
            train_df["slide_id"].tolist(),
            test_size=VAL_RATIO,
            stratify=train_df["label"].tolist(),
            random_state=SEED,
        )
        if args.test_hospital is not None and hospital != args.test_hospital:
            continue

        # Resume/parallel guard: skip a fold whose result.json already exists
        # so a second worker (e09 launched on GPU3) and the GPU1 driver never
        # retrain or overwrite the same fold. Absent result.json => run here.
        if not args.eval_only:
            _done = os.path.join(args.output_dir, f"fold_{fold_idx}_{hospital}", "result.json")
            if os.path.exists(_done):
                logger.info(f"SKIP fold {fold_idx}/5 {hospital}: result.json already present at {_done}")
                continue

        if args.eval_only:
            checkpoint_root = args.eval_checkpoint_root or OUTPUT_DIR
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
            )
        else:
            result = train_fold(
                fold_idx,
                hospital,
                train_ids,
                val_ids,
                test_ids,
                full_dataset,
                args.output_dir,
                logger,
                args.num_epochs,
                args.select_metric,
                args.threshold_metric,
                args.loss_type,
                args.focal_gamma,
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
    all_true, all_pred, all_prob = [], [], []
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
        "max_instances": args.max_instances,
        "sample_seed": args.sample_seed,
        "folds": results,
    }
    summary_path = os.path.join(args.output_dir, f"chime_loso_summary_{timestamp}.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=4)
    logger.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
