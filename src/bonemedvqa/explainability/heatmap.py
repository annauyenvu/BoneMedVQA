"""Heatmap helpers (supportive signals, not definitive explanations)."""

from __future__ import annotations

import numpy as np
from PIL import Image


def gradcam_like_heatmap(feature_map: np.ndarray) -> np.ndarray:
    """Create a simple energy map from (C,H,W) or (H,W,C) features."""
    arr = np.asarray(feature_map, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[-1]:
            # assume CHW
            heat = np.mean(np.abs(arr), axis=0)
        else:
            heat = np.mean(np.abs(arr), axis=-1)
    elif arr.ndim == 2:
        heat = arr
    else:
        raise ValueError("feature_map must be 2D or 3D")
    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat.astype(np.float32)


def overlay_heatmap(
    image: Image.Image | np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    import cv2

    if isinstance(image, Image.Image):
        base = np.array(image.convert("RGB"))
    else:
        base = np.asarray(image)
        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)
    heat = cv2.resize(heatmap.astype(np.float32), (base.shape[1], base.shape[0]))
    heat_u8 = (heat * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (alpha * colored + (1 - alpha) * base).astype(np.uint8)
    return Image.fromarray(overlay)
