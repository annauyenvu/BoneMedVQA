from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.models.multimodal_fusion import ConcatFusion, CrossAttentionFusion


def test_concat_and_cross():
    c = ConcatFusion([16, 16, 16], 16)
    x = c([torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 16)])
    assert x.shape == (2, 16)
    f = CrossAttentionFusion(16, 4)
    y, _ = f(torch.randn(2, 16), torch.randn(2, 5, 16))
    assert y.shape == (2, 16)
