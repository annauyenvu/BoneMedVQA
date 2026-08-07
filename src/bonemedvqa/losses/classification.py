"""Classification losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    kind: str = "cross_entropy",
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Compute closed-answer classification loss (ignores label < 0)."""
    valid = labels >= 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    logits = logits[valid]
    labels = labels[valid]
    kind = kind.lower()
    if kind == "cross_entropy":
        return F.cross_entropy(logits, labels)
    if kind == "focal":
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** focal_gamma * ce).mean()
    if kind in {"bce", "bce_logits"}:
        # multi-label one-hot
        num_classes = logits.size(-1)
        target = F.one_hot(labels, num_classes=num_classes).float()
        return F.binary_cross_entropy_with_logits(logits, target)
    raise ValueError(f"Unknown classification loss: {kind}")
