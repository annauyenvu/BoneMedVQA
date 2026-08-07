"""Attention map visualization."""

from __future__ import annotations

import numpy as np
import torch


def attention_to_heatmap(
    attn: torch.Tensor | np.ndarray,
    grid_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert attention weights to a 2D heatmap.

    attn can be (N,), (heads,N), or (Q,N). Uses mean over leading dims.
    """
    if torch.is_tensor(attn):
        arr = attn.detach().float().cpu().numpy()
    else:
        arr = np.asarray(attn, dtype=np.float32)
    while arr.ndim > 1:
        arr = arr.mean(axis=0)
    n = arr.shape[0]
    if grid_hw is None:
        side = int(np.sqrt(n))
        if side * side != n:
            # pad to square
            side = int(np.ceil(np.sqrt(n)))
            pad = side * side - n
            arr = np.pad(arr, (0, pad))
        grid_hw = (side, side)
    h, w = grid_hw
    if h * w != arr.shape[0]:
        arr = arr[: h * w]
        if arr.shape[0] < h * w:
            arr = np.pad(arr, (0, h * w - arr.shape[0]))
    heat = arr.reshape(h, w)
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat.astype(np.float32)
