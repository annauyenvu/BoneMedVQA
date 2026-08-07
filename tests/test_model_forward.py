"""Fusion and model forward smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.losses.combined_loss import CombinedLoss
from bonemedvqa.models.closed_answer_head import ClosedAnswerHead
from bonemedvqa.models.model_factory import build_model
from bonemedvqa.models.multimodal_fusion import ConcatFusion, CrossAttentionFusion
from bonemedvqa.models.open_answer_head import OpenAnswerHead
from bonemedvqa.utils.io import load_yaml


def test_fusion_shapes():
    concat = ConcatFusion([32, 32], hidden_dim=32)
    a = torch.randn(4, 32)
    b = torch.randn(4, 32)
    out = concat([a, b])
    assert out.shape == (4, 32)

    cross = CrossAttentionFusion(hidden_dim=32, num_heads=4)
    mem = torch.randn(4, 8, 32)
    fused, w = cross(a, mem)
    assert fused.shape == (4, 32)


def test_closed_and_open_heads():
    head = ClosedAnswerHead(64, num_classes=3)
    logits = head(torch.randn(2, 64))
    assert logits.shape == (2, 3)
    open_head = OpenAnswerHead(64, mode="template", id_to_label={0: "no", 1: "yes", 2: "wrist"})
    out = open_head(torch.randn(2, 64), closed_logits=logits)
    assert len(out["texts"]) == 2


def test_full_model_forward_and_loss():
    cfg = load_yaml(ROOT / "configs" / "lightweight.yaml")
    model = build_model(cfg, num_classes=2, id_to_label={0: "no", 1: "yes"})
    B = 2
    batch = {
        "pixel_values": torch.randn(B, 3, 224, 224),
        "input_ids": torch.randint(1, 100, (B, 64)),
        "attention_mask": torch.ones(B, 64, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
        "bboxes": [[10, 10, 50, 50], None],
    }
    out = model(batch)
    assert out["logits"].shape == (B, 2)
    assert out["confidence"].shape == (B,)
    assert "warning" in out
    crit = CombinedLoss(cfg, answer_embedding=torch.nn.Embedding(2, model.hidden_dim))
    losses = crit(out, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
