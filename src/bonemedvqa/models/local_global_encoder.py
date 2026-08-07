"""Local-global feature extraction (FAVP-inspired ROI branch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalEncoder(nn.Module):
    """Combine global image features with ROI-cropped local features."""

    def __init__(self, hidden_dim: int = 256, roi_size: int = 112):
        super().__init__()
        self.roi_size = roi_size
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def crop_roi(self, pixel_values: torch.Tensor, boxes: list | None) -> torch.Tensor:
        """Crop ROI from normalized tensors; fallback to center crop."""
        b, c, h, w = pixel_values.shape
        crops = []
        for i in range(b):
            if boxes is not None and boxes[i] is not None:
                x1, y1, x2, y2 = boxes[i]
                # boxes assumed in original pixel space — use relative center crop heuristic
                # When resize already applied, use center region if coords look absolute large.
                if max(x2, y2) > max(h, w):
                    # map roughly into feature image space by clamping ratio
                    x1, x2 = 0, w
                    y1, y2 = 0, h
                x1 = int(max(0, min(w - 1, x1)))
                x2 = int(max(x1 + 1, min(w, x2)))
                y1 = int(max(0, min(h - 1, y1)))
                y2 = int(max(y1 + 1, min(h, y2)))
                crop = pixel_values[i : i + 1, :, y1:y2, x1:x2]
            else:
                # center crop
                ch = min(h, max(8, h // 2))
                cw = min(w, max(8, w // 2))
                y1 = (h - ch) // 2
                x1 = (w - cw) // 2
                crop = pixel_values[i : i + 1, :, y1 : y1 + ch, x1 : x1 + cw]
            crop = F.interpolate(crop, size=(self.roi_size, self.roi_size), mode="bilinear", align_corners=False)
            crops.append(crop)
        return torch.cat(crops, dim=0)

    def forward(
        self,
        global_feat: torch.Tensor,
        local_feat: torch.Tensor,
    ) -> torch.Tensor:
        if global_feat.shape != local_feat.shape:
            raise ValueError(
                f"Shape mismatch global={tuple(global_feat.shape)} local={tuple(local_feat.shape)}"
            )
        return self.fuse(torch.cat([global_feat, local_feat], dim=-1))
