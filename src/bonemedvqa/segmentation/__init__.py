"""Segmentation backends for visual prompts."""

from .adapters import SAMLikeAdapter, build_sam_like_segmentor

__all__ = ["SAMLikeAdapter", "build_sam_like_segmentor"]
