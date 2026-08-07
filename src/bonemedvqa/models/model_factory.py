"""Model factory."""

from __future__ import annotations

from typing import Any

from bonemedvqa.models.bonemedvqa_model import BoneMedVQAModel


def build_model(
    cfg: dict[str, Any],
    num_classes: int,
    id_to_label: dict[int, str] | None = None,
) -> BoneMedVQAModel:
    """Build BoneMedVQAModel from config."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    model = BoneMedVQAModel(cfg=cfg, num_classes=num_classes, id_to_label=id_to_label)
    return model
