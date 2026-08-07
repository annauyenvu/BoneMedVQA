"""Render prediction overlays for demo / API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from bonemedvqa.explainability.heatmap import overlay_heatmap
from bonemedvqa.utils.io import ensure_dir


def render_prediction_overlay(
    image: Image.Image,
    mask: np.ndarray | None = None,
    box: list[float] | None = None,
    heatmap: np.ndarray | None = None,
    out_path: str | Path | None = None,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    if heatmap is not None:
        canvas = overlay_heatmap(canvas, heatmap, alpha=0.4)
    draw = ImageDraw.Draw(canvas)
    if box is not None and len(box) == 4 and box[2] > box[0]:
        draw.rectangle(box, outline=(255, 220, 0), width=3)
    if mask is not None:
        m = (mask > 0).astype(np.uint8)
        if m.shape[:2] != (canvas.size[1], canvas.size[0]):
            m_img = Image.fromarray(m * 255).resize(canvas.size, Image.NEAREST)
            m = (np.array(m_img) > 0).astype(np.uint8)
        overlay = np.array(canvas)
        color = np.zeros_like(overlay)
        color[m > 0] = (0, 200, 80)
        overlay = (0.7 * overlay + 0.3 * color).astype(np.uint8)
        canvas = Image.fromarray(overlay)
    if out_path is not None:
        out_path = Path(out_path)
        ensure_dir(out_path.parent)
        canvas.save(out_path)
    return canvas
