"""Multimodal fusion strategies."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Baseline concatenation of global features."""

    def __init__(self, dims: list[int], hidden_dim: int):
        super().__init__()
        in_dim = sum(dims)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        for f in features:
            if f.ndim != 2:
                raise ValueError(f"ConcatFusion expects (B,D), got {tuple(f.shape)}")
        x = torch.cat(features, dim=-1)
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion with residual + LayerNorm."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.memory_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(
        self,
        query_feat: torch.Tensor,
        memory_tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query_feat: (B, D) or (B, Q, D)
        memory_tokens: (B, N, D)
        """
        if query_feat.ndim == 2:
            q = query_feat.unsqueeze(1)
        else:
            q = query_feat
        if memory_tokens.size(-1) != self.hidden_dim:
            raise ValueError("memory_tokens dim mismatch")
        q = self.query_proj(q)
        mem = self.memory_proj(memory_tokens)
        attn_out, weights = self.attn(q, mem, mem, key_padding_mask=key_padding_mask, need_weights=True)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ff(q))
        pooled = self.out(q.mean(dim=1))
        return pooled, weights
