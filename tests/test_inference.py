"""Inference confidence / abstention tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.inference.confidence import apply_abstention


def test_abstention_below_threshold():
    out = apply_abstention("yes", confidence=0.2, threshold=0.55, language="vi")
    assert out["abstained"] is True
    assert "Không đủ độ tin cậy" in out["answer"]
    assert "warning" in out


def test_no_abstention_above_threshold():
    out = apply_abstention("yes", confidence=0.9, threshold=0.55)
    assert out["abstained"] is False
    assert out["answer"] == "yes"
