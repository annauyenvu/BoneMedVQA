"""Learnable latent prompt with cross-attention (LaPA-inspired)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """Latent queries attend to a memory sequence."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.norm_q(query)
        kv = self.norm_kv(memory)
        out, weights = self.attn(
            q,
            kv,
            kv,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        query = query + out
        query = query + self.ff(query)
        return query, weights


class LatentPromptGenerator(nn.Module):
    """Learnable latent tokens filtering visual and textual features.

    Flow:
        learnable latent tokens
        → cross-attn with visual tokens
        → cross-attn with textual tokens
        → filtered latent representation
    """

    def __init__(
        self,
        num_latent_tokens: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        concept_bank: bool = True,
        num_concepts: int = 32,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_latent_tokens = num_latent_tokens
        self.hidden_dim = hidden_dim
        self.latent_tokens = nn.Parameter(torch.randn(num_latent_tokens, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.use_concept_bank = concept_bank
        if concept_bank:
            # Anatomy / fracture / location / severity style concept embeddings
            self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, hidden_dim) * 0.02)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        textual_tokens: torch.Tensor,
        visual_padding_mask: torch.Tensor | None = None,
        textual_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        visual_tokens : (B, Nv, D)
        textual_tokens : (B, Nt, D)
        *_padding_mask : (B, N) with True for padded positions (nn.MultiheadAttention convention)
        """
        if visual_tokens.ndim != 3 or textual_tokens.ndim != 3:
            raise ValueError("visual_tokens and textual_tokens must be (B, N, D)")
        if visual_tokens.size(-1) != self.hidden_dim or textual_tokens.size(-1) != self.hidden_dim:
            raise ValueError(
                f"Token dim mismatch: expected {self.hidden_dim}, "
                f"got visual={visual_tokens.size(-1)}, text={textual_tokens.size(-1)}"
            )

        bsz = visual_tokens.size(0)
        latent = self.latent_tokens.unsqueeze(0).expand(bsz, -1, -1)
        attn_maps = []

        for i, layer in enumerate(self.layers):
            # Alternate visual / textual conditioning
            if i % 2 == 0:
                latent, w = layer(latent, visual_tokens, visual_padding_mask)
            else:
                latent, w = layer(latent, textual_tokens, textual_padding_mask)
            attn_maps.append(w)

        # Final pass over both memories if only one layer
        if len(self.layers) == 1:
            latent, w = self.layers[0](latent, textual_tokens, textual_padding_mask)
            attn_maps.append(w)

        latent = self.out_norm(latent)
        pooled = latent.mean(dim=1)

        concept_sim = None
        if self.use_concept_bank:
            # (B, C)
            concept_sim = F.normalize(pooled, dim=-1) @ F.normalize(self.concept_embeddings, dim=-1).T

        return {
            "latent_tokens": latent,
            "latent_pooled": pooled,
            "attention_maps": attn_maps,
            "concept_similarity": concept_sim,
        }
