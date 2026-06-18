"""
Label convention (authoritative, shared with all CHIME_MIL baselines):
    class 0 = 'm'  = metastasis / tumor   (POSITIVE class)
    class 1 = 'nm' = non-metastasis / normal

All metrics below use this convention:
    - sens = cm[0, 0] / row0  (TPR for class 0 = tumor recall)
    - spec = cm[1, 1] / row1  (TNR for class 1 = normal)
    - f1, ppv use pos_label=0
    - auc = roc_auc_score(y_true, prob_cls1); AUC is symmetric, but thresholds
      are applied to prob_cls1, so threshold selection assumes class 1 convention.

`y_prob_cls1` is P(class=1) = P(normal). Predictions: pred=1 iff prob_cls1 >= threshold;
equivalently pred=0 (tumor) iff prob_cls1 < threshold.
"""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score


def probs_to_preds(y_prob_cls1, threshold=0.5):
    y_prob = np.array(y_prob_cls1, dtype=float)
    return (y_prob >= threshold).astype(int)


def compute_metrics(y_true, y_pred, y_prob_cls1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob_cls1, dtype=float)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sens = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    spec = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    ppv = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else 0.0
    bal_acc = (sens + spec) / 2.0
    return {
        "acc": round(float(acc), 4),
        "auc": round(float(auc), 4),
        "balanced_acc": round(float(bal_acc), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1), 4),
        "ppv": round(float(ppv), 4),
    }


def find_best_threshold(y_true, y_prob_cls1, metric="balanced_acc"):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob_cls1, dtype=float)
    if y_true.size == 0:
        raise ValueError("Cannot choose a threshold with empty labels.")

    candidate_thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_prob)))
    best = None
    best_score = float("-inf")

    for threshold in candidate_thresholds:
        preds = probs_to_preds(y_prob, threshold=threshold)
        metrics = compute_metrics(y_true, preds, y_prob)
        score = float(metrics[metric])
        tie_break = float(metrics["auc"])
        if score > best_score or (np.isclose(score, best_score) and best is not None and tie_break > best["metrics"]["auc"]):
            best_score = score
            best = {
                "threshold": round(float(threshold), 6),
                "metric": metric,
                "score": round(score, 4),
                "metrics": metrics,
            }

    return best


def bootstrap_ci(y_true, y_pred, y_prob_cls1, n_bootstrap=1000, ci=0.95):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob_cls1)
    n = len(y_true)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        samples.append(compute_metrics(yt, yp, ypr))
    if not samples:
        return {}
    alpha = (1.0 - ci) / 2.0
    result = {}
    for key in samples[0]:
        vals = np.array([s[key] for s in samples])
        result[key] = {
            "mean": round(float(np.mean(vals)), 4),
            "ci_low": round(float(np.percentile(vals, 100 * alpha)), 4),
            "ci_high": round(float(np.percentile(vals, 100 * (1 - alpha))), 4),
        }
    return result


def format_ci(ci_dict):
    order = ["acc", "auc", "balanced_acc", "sensitivity", "specificity", "f1", "ppv"]
    out = []
    for k in order:
        if k not in ci_dict:
            continue
        d = ci_dict[k]
        mn, lo, hi = d["mean"], d["ci_low"], d["ci_high"]
        out.append(f"  {k:12s}: {mn:.4f}  95%CI [{lo:.4f}-{hi:.4f}]")
    return chr(10).join(out)
