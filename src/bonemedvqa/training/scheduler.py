"""LR schedulers."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int = 1):
    name = str(cfg.get("scheduler", "cosine")).lower()
    epochs = int(cfg.get("epochs", 10))
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    return None
