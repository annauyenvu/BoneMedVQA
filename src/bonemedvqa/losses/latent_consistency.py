"""Latent consistency loss (LaPA-inspired)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_consistency_loss(
    latent_pooled: torch.Tensor,
    target_embeddings: torch.Tensor,
    kind: str = "cosine",
) -> torch.Tensor:
    """Pull latent representation toward correct answer / concept embedding."""
    kind = kind.lower()
    if kind == "cosine":
        return (1.0 - F.cosine_similarity(latent_pooled, target_embeddings, dim=-1)).mean()
    if kind == "mse":
        return F.mse_loss(latent_pooled, target_embeddings)
    if kind == "triplet":
        # target as positive; use batch shuffle as negative
        neg = target_embeddings.roll(1, dims=0)
        dist_pos = F.pairwise_distance(latent_pooled, target_embeddings)
        dist_neg = F.pairwise_distance(latent_pooled, neg)
        return F.relu(dist_pos - dist_neg + 0.2).mean()
    raise ValueError(f"Unknown latent consistency kind: {kind}")
