"""Additional latent / fusion named tests for coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.prompting.latent_prompt import LatentPromptGenerator
from bonemedvqa.models.multimodal_fusion import CrossAttentionFusion


def test_latent_prompt_shapes_alias():
    m = LatentPromptGenerator(4, 32, 4, 1, concept_bank=True)
    out = m(torch.randn(1, 9, 32), torch.randn(1, 5, 32))
    assert out["concept_similarity"].shape[0] == 1


def test_cross_attention_output_shape():
    f = CrossAttentionFusion(32, num_heads=4)
    q = torch.randn(3, 32)
    mem = torch.randn(3, 7, 32)
    out, w = f(q, mem)
    assert out.shape == (3, 32)
