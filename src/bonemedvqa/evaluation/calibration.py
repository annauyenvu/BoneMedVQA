"""Calibration metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def expected_calibration_error(
    confidences: list[float] | np.ndarray,
    correct: list[bool] | np.ndarray,
    n_bins: int = 10,
) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if not np.any(m):
            continue
        acc = correct[m].mean()
        conf = confidences[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multiclass Brier score."""
    probs = np.asarray(probs, dtype=np.float64)
    y_true = np.asarray(y_true)
    n, c = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_bins(
    confidences: list[float] | np.ndarray,
    correct: list[bool] | np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    confidences = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (confidences > bins[i]) & (confidences <= bins[i + 1])
        rows.append(
            {
                "bin_lower": float(bins[i]),
                "bin_upper": float(bins[i + 1]),
                "count": int(m.sum()),
                "confidence": float(confidences[m].mean()) if m.any() else None,
                "accuracy": float(correct[m].mean()) if m.any() else None,
            }
        )
    return {"bins": rows, "ece": expected_calibration_error(confidences, correct, n_bins)}
