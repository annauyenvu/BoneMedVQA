"""Device selection utilities."""

from __future__ import annotations

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA device if available and preferred, else CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(config: dict | None = None) -> torch.device:
    """Resolve device from config dict with keys prefer_cuda / allow_cpu."""
    cfg = config or {}
    prefer_cuda = bool(cfg.get("prefer_cuda", True))
    allow_cpu = bool(cfg.get("allow_cpu", True))
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not allow_cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by config but not available.")
    return torch.device("cpu")
