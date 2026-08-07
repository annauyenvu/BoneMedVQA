"""Generate synthetic X-ray-like images for local Gradio testing."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "assets" / "samples"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(7)


def bone_field(h: int, w: int, seed: int) -> np.ndarray:
    rs = np.random.RandomState(seed)
    base = rs.normal(40, 10, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    dist = ((yy - cy) ** 2 / (h * 0.55) ** 2) + ((xx - cx) ** 2 / (w * 0.45) ** 2)
    base += np.clip(80 * (1 - dist), 0, 80)
    base += rs.normal(0, 4, (h, w))
    return np.clip(base, 0, 255)


def make(name: str, anatomy: str, fracture: bool, size: int = 320) -> None:
    arr = bone_field(size, size, rng.randint(0, 10_000))
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    if anatomy == "wrist":
        draw.ellipse([size * 0.35, size * 0.15, size * 0.65, size * 0.85], outline=(200, 200, 200), width=3)
        draw.rectangle([size * 0.42, size * 0.55, size * 0.58, size * 0.95], outline=(190, 190, 190), width=2)
    elif anatomy == "elbow":
        draw.ellipse([size * 0.25, size * 0.35, size * 0.75, size * 0.7], outline=(200, 200, 200), width=3)
        draw.line([(size * 0.5, size * 0.1), (size * 0.5, size * 0.9)], fill=(185, 185, 185), width=8)
    elif anatomy == "ankle":
        draw.ellipse([size * 0.3, size * 0.25, size * 0.7, size * 0.65], outline=(200, 200, 200), width=3)
        draw.rectangle([size * 0.4, size * 0.55, size * 0.6, size * 0.95], outline=(190, 190, 190), width=2)
    else:
        for ox in (-0.18, -0.09, 0.0, 0.09, 0.18):
            x = size * (0.5 + ox)
            draw.line([(x, size * 0.55), (x, size * 0.2)], fill=(185, 185, 185), width=5)
        draw.ellipse([size * 0.3, size * 0.5, size * 0.7, size * 0.85], outline=(200, 200, 200), width=2)
    if fracture:
        x1, y1 = int(size * 0.35), int(size * 0.4)
        x2, y2 = int(size * 0.62), int(size * 0.52)
        draw.line([(x1, y1), (x2, y2)], fill=(30, 30, 30), width=3)
        draw.ellipse([x1 - 5, y1 - 5, x1 + 10, y1 + 10], fill=(230, 230, 230))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = ImageEnhance.Contrast(img).enhance(1.25)
    path = OUT / name
    img.save(path)
    print(f"wrote {path}")


def main() -> None:
    specs = [
        ("synthetic_wrist_fracture.png", "wrist", True),
        ("synthetic_wrist_normal.png", "wrist", False),
        ("synthetic_elbow_fracture.png", "elbow", True),
        ("synthetic_ankle_fracture.png", "ankle", True),
        ("synthetic_hand_normal.png", "hand", False),
    ]
    for name, anatomy, fracture in specs:
        make(name, anatomy, fracture)
    print("done")


if __name__ == "__main__":
    main()
