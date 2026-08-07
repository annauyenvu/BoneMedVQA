"""Closed-answer metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_closed_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[list[float]] | None = None,
) -> dict[str, Any]:
    if not y_true:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "n": 0,
        }
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": len(y_true),
    }
    if y_prob is not None and len(y_prob) == len(y_true):
        probs = np.asarray(y_prob, dtype=np.float64)
        try:
            if probs.shape[1] == 2:
                metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
            else:
                metrics["auroc"] = float(
                    roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
                )
        except ValueError:
            metrics["auroc"] = None
    return metrics
