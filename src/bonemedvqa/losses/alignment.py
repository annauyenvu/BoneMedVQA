"""Alignment / contrastive losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_alignment_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity (mean)."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).mean()


def info_nce_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between image and text globals."""
    image_feat = F.normalize(image_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    logits = image_feat @ text_feat.T / max(temperature, 1e-6)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i + loss_t)
