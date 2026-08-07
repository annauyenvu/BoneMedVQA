"""Closed-answer classification head."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClosedAnswerHead(nn.Module):
    """Multi-class classifier for closed VQA answers."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        if fused.ndim != 2:
            raise ValueError(f"Expected (B,D), got {tuple(fused.shape)}")
        return self.net(fused)
