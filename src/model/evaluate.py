"""Evaluation metrics and analysis for attack detection."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(
    labels: list[int],
    predictions: list[int],
    probabilities: Optional[list[float]] = None,
) -> dict:
    """Compute classification metrics.

    Returns:
        dict with precision, recall, f1, accuracy, and optionally auc_roc.
    """
    labels_arr = np.array(labels)
    preds_arr = np.array(predictions)

    tp = ((preds_arr == 1) & (labels_arr == 1)).sum()
    tn = ((preds_arr == 0) & (labels_arr == 0)).sum()
    fp = ((preds_arr == 1) & (labels_arr == 0)).sum()
    fn = ((preds_arr == 0) & (labels_arr == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    if probabilities is not None:
        metrics["auc_roc"] = _compute_auc_roc(labels_arr, np.array(probabilities))

    return metrics


def _compute_auc_roc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Compute AUC-ROC using trapezoidal rule (no sklearn dependency)."""
    if len(np.unique(labels)) < 2:
        return 0.0

    # Sort by decreasing probability
    desc_indices = np.argsort(-probs)
    labels_sorted = labels[desc_indices]

    num_pos = labels.sum()
    num_neg = len(labels) - num_pos

    if num_pos == 0 or num_neg == 0:
        return 0.0

    tpr_points = [0.0]
    fpr_points = [0.0]
    tp_count = 0
    fp_count = 0

    for label in labels_sorted:
        if label == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_points.append(tp_count / num_pos)
        fpr_points.append(fp_count / num_neg)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(fpr_points)):
        auc += (fpr_points[i] - fpr_points[i - 1]) * (tpr_points[i] + tpr_points[i - 1]) / 2

    return auc


def per_class_metrics(
    labels: list[int],
    predictions: list[int],
    attack_classes: Optional[list[str]] = None,
    class_labels: Optional[list[int]] = None,
) -> dict[str, dict]:
    """Compute metrics per attack class if multi-class labels are provided.

    For binary detection, returns metrics for class 0 (normal) and 1 (attack).
    """
    results = {}

    if class_labels is None:
        class_labels = sorted(set(labels))

    for cls in class_labels:
        cls_mask = np.array(labels) == cls
        cls_preds = np.array(predictions)[cls_mask]
        cls_labels_binary = np.ones_like(cls_preds)  # all should be this class

        tp = (cls_preds == cls).sum()
        total = len(cls_preds)

        cls_name = f"class_{cls}"
        if attack_classes and cls < len(attack_classes):
            cls_name = attack_classes[cls]

        results[cls_name] = {
            "recall": float(tp / max(total, 1)),
            "support": int(total),
        }

    return results


def print_evaluation_report(
    metrics: dict,
    per_class: Optional[dict] = None,
) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}  TN={metrics['tn']}")

    if per_class:
        print(f"\n  Per-Class Recall:")
        for name, cls_metrics in per_class.items():
            print(f"    {name}: {cls_metrics['recall']:.4f} (n={cls_metrics['support']})")

    print("=" * 60)
