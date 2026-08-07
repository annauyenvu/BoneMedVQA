"""Visual / textual / latent prompt tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.prompting.latent_prompt import LatentPromptGenerator
from bonemedvqa.prompting.textual_prompt import TextualPromptBuilder
from bonemedvqa.prompting.visual_prompt import VisualPromptGenerator


def test_visual_prompt_from_box():
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    gen = VisualPromptGenerator()
    bundle = gen.generate_from_box(img, [20, 20, 80, 90])
    assert bundle.mask.shape == (128, 128)
    assert bundle.mask.sum() > 0
    assert len(bundle.box) == 4
    assert bundle.masked_image.shape == img.shape


def test_visual_prompt_from_point():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 200
    gen = VisualPromptGenerator()
    bundle = gen.generate_from_point(img, [50, 50])
    assert bundle.mask[50, 50] == 1


def test_textual_prompt_builder():
    b = TextualPromptBuilder()
    text = b.build(
        question="Is there evidence of a fracture?",
        question_type="auto",
        anatomy="wrist",
    )
    assert "Task:" in text
    assert "Question:" in text
    assert "abstain" in text.lower() or "Abstain" in text
    assert b.classify_question_type("Describe the abnormal findings.") == "open"


def test_latent_token_shapes():
    m = LatentPromptGenerator(num_latent_tokens=8, hidden_dim=64, num_heads=4, num_layers=2)
    B, Nv, Nt, D = 2, 16, 10, 64
    vis = torch.randn(B, Nv, D)
    txt = torch.randn(B, Nt, D)
    out = m(vis, txt)
    assert out["latent_tokens"].shape == (B, 8, D)
    assert out["latent_pooled"].shape == (B, D)
