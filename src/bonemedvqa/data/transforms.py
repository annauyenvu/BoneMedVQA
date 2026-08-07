"""Image loading and augmentation (anatomy-preserving)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


def _strip_dicom_phi(ds: Any) -> None:
    """Remove common PHI tags after pixel extraction (in-memory only)."""
    phi_tags = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAddress",
        "InstitutionName",
        "ReferringPhysicianName",
        "OperatorsName",
    ]
    for tag in phi_tags:
        if hasattr(ds, tag):
            try:
                setattr(ds, tag, "")
            except Exception:
                pass


def load_image(path: str | Path, apply_clahe: bool = False) -> Image.Image:
    """Load PNG/JPG/JPEG/DICOM as grayscale RGB PIL image (3-channel)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".dcm":
        if pydicom is None:
            raise ImportError("pydicom is required to read DICOM files")
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        _strip_dicom_phi(ds)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255.0).astype(np.uint8)
        img = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        img = Image.open(path).convert("RGB")

    if apply_clahe:
        img = apply_clahe_rgb(img)
    return img


def apply_clahe_rgb(img: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """Optional CLAHE on luminance channel."""
    import cv2

    rgb = np.array(img)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def letterbox_resize(
    image: Image.Image,
    size: int = 224,
    fill: int = 0,
) -> tuple[Image.Image, dict[str, float]]:
    """Resize keeping aspect ratio and pad to square."""
    w, h = image.size
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=(fill, fill, fill))
    pad_x = (size - nw) // 2
    pad_y = (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    meta = {
        "scale": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "orig_w": float(w),
        "orig_h": float(h),
    }
    return canvas, meta


def build_transforms(
    image_size: int = 224,
    train: bool = False,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    apply_clahe: bool = False,
):
    """Build torchvision-style transform pipeline.

    Notes
    -----
    Horizontal flip is intentionally disabled to avoid L/R anatomy errors.
    """
    import torch
    from torchvision import transforms as T

    class ToTensorNormalize:
        def __init__(self, mean, std, image_size, apply_clahe):
            self.mean = mean
            self.std = std
            self.image_size = image_size
            self.apply_clahe = apply_clahe

        def __call__(self, img: Image.Image) -> dict[str, Any]:
            if self.apply_clahe:
                img = apply_clahe_rgb(img)
            img, meta = letterbox_resize(img, self.image_size)
            tensor = T.ToTensor()(img)
            tensor = T.Normalize(self.mean, self.std)(tensor)
            return {"pixel_values": tensor, "resize_meta": meta}

    class TrainAugment:
        def __init__(self, base):
            self.base = base
            self.color = T.ColorJitter(brightness=0.15, contrast=0.15)
            self.affine = T.RandomAffine(degrees=8, translate=(0.05, 0.05))

        def __call__(self, img: Image.Image) -> dict[str, Any]:
            img = self.color(img)
            img = self.affine(img)
            # Mild Gaussian noise after tensorization
            out = self.base(img)
            noise = torch.randn_like(out["pixel_values"]) * 0.01
            out["pixel_values"] = out["pixel_values"] + noise
            return out

    base = ToTensorNormalize(mean, std, image_size, apply_clahe)
    return TrainAugment(base) if train else base
