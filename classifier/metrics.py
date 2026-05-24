"""Robust binary-classification metrics for imbalanced capsid datasets."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
from sklearn import metrics


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def _safe_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return math.nan
    return float(metrics.roc_auc_score(y_true, probs))


def _safe_average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    if y_true.sum() == 0:
        return math.nan
    return float(metrics.average_precision_score(y_true, probs))


def _safe_mcc(y_true: np.ndarray, preds: np.ndarray) -> float:
    if np.unique(y_true).size < 2 or np.unique(preds).size < 2:
        return 0.0
    return float(metrics.matthews_corrcoef(y_true, preds))


def binary_metrics(y_true, probs, thr: float = 0.5) -> Dict[str, float]:
    """Return threshold and ranking metrics robust to class imbalance.

    The function always returns confusion-matrix counts plus class-balance
    fields. Ranking metrics that are undefined for single-class targets are
    returned as ``nan`` instead of raising.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    if y_true.shape[0] == 0:
        raise ValueError("cannot compute metrics for an empty target array")
    if y_true.shape[0] != probs.shape[0]:
        raise ValueError(f"y_true/probs length mismatch: {y_true.shape[0]} != {probs.shape[0]}")

    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    positives = int(tp + fn)
    negatives = int(tn + fp)
    pred_positives = int(tp + fp)
    pred_negatives = int(tn + fn)

    recall = _safe_divide(tp, positives)
    specificity = _safe_divide(tn, negatives)
    balanced_parts = [x for x in (recall, specificity) if not math.isnan(x)]
    balanced_acc = float(np.mean(balanced_parts)) if balanced_parts else math.nan

    return {
        "n": int(y_true.shape[0]),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": _safe_divide(positives, y_true.shape[0]),
        "predicted_positives": pred_positives,
        "predicted_negatives": pred_negatives,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "acc": float(metrics.accuracy_score(y_true, preds)),
        "balanced_acc": balanced_acc,
        "precision": float(metrics.precision_score(y_true, preds, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, preds, zero_division=0)),
        "specificity": specificity,
        "f1": float(metrics.f1_score(y_true, preds, zero_division=0)),
        "mcc": _safe_mcc(y_true, preds),
        "auroc": _safe_auroc(y_true, probs),
        "average_precision": _safe_average_precision(y_true, probs),
    }
