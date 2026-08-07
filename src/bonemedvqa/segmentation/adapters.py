"""Pluggable segmentation adapters (SAM-Med2D / MedSAM / SAM)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from bonemedvqa.prompting.visual_prompt import HeuristicSegmentor, SegmentorProtocol


class SAMLikeAdapter(SegmentorProtocol):
    """Placeholder adapter that documents integration points.

    Real SAM-Med2D / MedSAM weights are large and license-gated by users.
    This adapter attempts to load an optional user-provided callable, otherwise
    falls back to HeuristicSegmentor so the pipeline remains runnable.
    """

    def __init__(self, backend_name: str = "sam_med2d", predictor: Any | None = None, **kwargs: Any):
        self.backend_name = backend_name
        self.predictor = predictor
        self.fallback = HeuristicSegmentor()
        self.kwargs = kwargs

    def segment_from_point(self, image: np.ndarray, point: Sequence[float]) -> np.ndarray:
        if self.predictor is None:
            return self.fallback.segment_from_point(image, point)
        return self.predictor(image=image, point=point)

    def segment_from_box(self, image: np.ndarray, box: Sequence[float]) -> np.ndarray:
        if self.predictor is None:
            return self.fallback.segment_from_box(image, box)
        return self.predictor(image=image, box=box)

    def segment_automatic(self, image: np.ndarray) -> np.ndarray:
        if self.predictor is None:
            return self.fallback.segment_automatic(image)
        return self.predictor(image=image, automatic=True)


def build_sam_like_segmentor(name: str, **kwargs: Any) -> SegmentorProtocol:
    """Build a SAM-family adapter.

    Expected optional kwargs:
      - checkpoint: path to weights
      - predictor: callable implementing (image, point/box) -> mask
    """
    predictor = kwargs.get("predictor")
    checkpoint = kwargs.get("checkpoint")
    if predictor is None and checkpoint:
        # Users can plug their own loader here without hard dependency.
        raise FileNotFoundError(
            f"Checkpoint provided for {name} but no predictor loader is registered. "
            "Pass predictor=callable or use segmentor=heuristic / gt_mask."
        )
    return SAMLikeAdapter(backend_name=name, predictor=predictor, **kwargs)
