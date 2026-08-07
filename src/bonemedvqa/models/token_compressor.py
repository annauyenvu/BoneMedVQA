"""Token compression / pooling utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCompressor(nn.Module):
    """Reduce visual token count via strided pooling on a grid."""

    def __init__(self, hidden_dim: int, ratio: int = 4):
        super().__init__()
        if ratio < 1:
            raise ValueError("ratio must be >= 1")
        self.ratio = ratio
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, tokens: torch.Tensor, grid_size: tuple[int, int] | None = None) -> torch.Tensor:
        """
        tokens: (B, N, D)
        """
        if self.ratio == 1:
            return tokens
        b, n, d = tokens.shape
        if grid_size is None:
            side = int(n ** 0.5)
            if side * side != n:
                # fallback: reshape-unaware average pool along sequence
                new_n = max(1, n // self.ratio)
                tokens = tokens[:, : new_n * self.ratio, :].reshape(b, new_n, self.ratio, d).mean(dim=2)
                return self.proj(tokens)
            grid_size = (side, side)
        h, w = grid_size
        if h * w != n:
            new_n = max(1, n // self.ratio)
            tokens = tokens[:, : new_n * self.ratio, :].reshape(b, new_n, self.ratio, d).mean(dim=2)
            return self.proj(tokens)
        x = tokens.transpose(1, 2).reshape(b, d, h, w)
        x = F.avg_pool2d(x, kernel_size=self.ratio, stride=self.ratio, ceil_mode=True)
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)
