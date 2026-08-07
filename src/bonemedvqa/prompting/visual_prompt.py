"""Visual prompt generation (point/box/mask → localization views)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np
from PIL import Image


@dataclass
class VisualPromptBundle:
    """Normalized visual prompt outputs."""

    mask: np.ndarray
    box: list[float]
    contour: np.ndarray
    masked_image: np.ndarray
    blurred_background_image: np.ndarray
    unicolor_image: np.ndarray
    multicolor_image: np.ndarray
    prompt_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mask": self.mask,
            "box": self.box,
            "contour": self.contour,
            "masked_image": self.masked_image,
            "blurred_background_image": self.blurred_background_image,
            "unicolor_image": self.unicolor_image,
            "multicolor_image": self.multicolor_image,
            "prompt_metadata": self.prompt_metadata,
        }


class SegmentorProtocol:
    """Interface for interchangeable segmentation backends."""

    def segment_from_point(self, image: np.ndarray, point: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def segment_from_box(self, image: np.ndarray, box: Sequence[float]) -> np.ndarray:
        raise NotImplementedError

    def segment_automatic(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class HeuristicSegmentor(SegmentorProtocol):
    """Intensity + morphology fallback when SAM-Med2D weights are unavailable."""

    def segment_from_point(self, image: np.ndarray, point: Sequence[float]) -> np.ndarray:
        h, w = image.shape[:2]
        x, y = int(point[0]), int(point[1])
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        seed = gray[y, x]
        mask = ((gray >= max(0, int(seed) - 35)) & (gray <= min(255, int(seed) + 35))).astype(np.uint8)
        mask = self._refine(mask)
        # Keep connected component containing the point
        num, labels = cv2.connectedComponents(mask)
        if num > 1:
            keep = labels == labels[y, x]
            mask = keep.astype(np.uint8)
        if mask.sum() == 0:
            # fallback circle
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), radius=max(8, min(h, w) // 12), color=1, thickness=-1)
        return mask

    def segment_from_box(self, image: np.ndarray, box: Sequence[float]) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        mask = np.zeros((h, w), dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
            roi = gray[y1:y2, x1:x2]
            thr = max(1, int(np.percentile(roi, 40)))
            local = (roi >= thr).astype(np.uint8)
            local = self._refine(local)
            mask[y1:y2, x1:x2] = local
            if mask.sum() == 0:
                mask[y1:y2, x1:x2] = 1
        return mask

    def segment_automatic(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._refine(mask.astype(np.uint8))

    @staticmethod
    def _refine(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return (mask > 0).astype(np.uint8)


class GroundTruthMaskSegmentor(SegmentorProtocol):
    """Use a provided GT mask; ignore interactive prompts except for metadata."""

    def __init__(self, mask: np.ndarray):
        self.mask = (mask > 0).astype(np.uint8)

    def segment_from_point(self, image: np.ndarray, point: Sequence[float]) -> np.ndarray:
        return self.mask.copy()

    def segment_from_box(self, image: np.ndarray, box: Sequence[float]) -> np.ndarray:
        return self.mask.copy()

    def segment_automatic(self, image: np.ndarray) -> np.ndarray:
        return self.mask.copy()


def build_segmentor(name: str = "heuristic", **kwargs: Any) -> SegmentorProtocol:
    """Factory for segmentation backends (pluggable)."""
    name = name.lower()
    if name in {"heuristic", "otsu", "fallback"}:
        return HeuristicSegmentor()
    if name in {"gt", "gt_mask", "ground_truth"}:
        mask = kwargs.get("mask")
        if mask is None:
            raise ValueError("gt_mask segmentor requires mask=")
        return GroundTruthMaskSegmentor(mask)
    if name in {"sam_med2d", "medsam", "sam"}:
        # Lazy optional adapter — falls back if weights unavailable.
        try:
            from bonemedvqa.segmentation.adapters import build_sam_like_segmentor

            return build_sam_like_segmentor(name, **kwargs)
        except Exception as exc:  # pragma: no cover
            import warnings

            warnings.warn(
                f"Segmentor '{name}' unavailable ({exc}); using heuristic fallback.",
                RuntimeWarning,
            )
            return HeuristicSegmentor()
    raise ValueError(f"Unknown segmentor: {name}")


class VisualPromptGenerator:
    """Generate localization-lens style visual prompt views."""

    def __init__(self, segmentor: SegmentorProtocol | None = None, blur_kernel: int = 21):
        self.segmentor = segmentor or HeuristicSegmentor()
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1

    def generate_from_point(self, image: np.ndarray | Image.Image, point: Sequence[float]) -> VisualPromptBundle:
        img = self._as_rgb(image)
        mask = self.segmentor.segment_from_point(img, point)
        return self.build_prompt_views(img, mask, metadata={"type": "point", "point": list(map(float, point))})

    def generate_from_box(self, image: np.ndarray | Image.Image, box: Sequence[float]) -> VisualPromptBundle:
        img = self._as_rgb(image)
        mask = self.segmentor.segment_from_box(img, box)
        return self.build_prompt_views(img, mask, metadata={"type": "box", "box": list(map(float, box))})

    def generate_from_mask(self, image: np.ndarray | Image.Image, mask: np.ndarray) -> VisualPromptBundle:
        img = self._as_rgb(image)
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        return self.build_prompt_views(img, mask_u8, metadata={"type": "mask"})

    def generate_automatic(self, image: np.ndarray | Image.Image) -> VisualPromptBundle:
        img = self._as_rgb(image)
        mask = self.segmentor.segment_automatic(img)
        return self.build_prompt_views(img, mask, metadata={"type": "automatic"})

    def build_prompt_views(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> VisualPromptBundle:
        img = self._as_rgb(image)
        mask_u8 = (mask > 0).astype(np.uint8)
        if mask_u8.shape[:2] != img.shape[:2]:
            mask_u8 = cv2.resize(mask_u8, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        box = self.mask_to_box(mask_u8)
        contour = self.mask_to_contour(mask_u8)

        masked = img.copy()
        masked[mask_u8 == 0] = 0

        blurred = cv2.GaussianBlur(img, (self.blur_kernel, self.blur_kernel), 0)
        blurred_bg = blurred.copy()
        blurred_bg[mask_u8 > 0] = img[mask_u8 > 0]

        unicolor = np.zeros_like(img)
        unicolor[mask_u8 > 0] = (0, 255, 0)

        # Multicolor: overlay contour + filled ROI in different colors
        multicolor = img.copy()
        color_mask = np.zeros_like(img)
        color_mask[mask_u8 > 0] = (255, 64, 64)
        multicolor = cv2.addWeighted(multicolor, 0.65, color_mask, 0.35, 0)
        if contour.size > 0:
            cv2.drawContours(multicolor, [contour], -1, (0, 255, 255), 2)
        if box[2] > box[0] and box[3] > box[1]:
            cv2.rectangle(
                multicolor,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (255, 255, 0),
                2,
            )

        meta = {
            "mask_area": int(mask_u8.sum()),
            "image_shape": list(img.shape),
            **(metadata or {}),
        }
        return VisualPromptBundle(
            mask=mask_u8,
            box=box,
            contour=contour,
            masked_image=masked,
            blurred_background_image=blurred_bg,
            unicolor_image=unicolor,
            multicolor_image=multicolor,
            prompt_metadata=meta,
        )

    @staticmethod
    def mask_to_box(mask: np.ndarray) -> list[float]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

    @staticmethod
    def mask_to_contour(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros((0, 1, 2), dtype=np.int32)
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _as_rgb(image: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[-1] == 4:
                arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr
