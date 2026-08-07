"""Weighted combination of task losses."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bonemedvqa.losses.alignment import cosine_alignment_loss, info_nce_loss
from bonemedvqa.losses.classification import classification_loss
from bonemedvqa.losses.latent_consistency import latent_consistency_loss


class CombinedLoss(nn.Module):
    """L_total = Σ λ_i L_i with coefficients from config."""

    def __init__(self, cfg: dict[str, Any], answer_embedding: nn.Embedding | None = None):
        super().__init__()
        self.cfg = cfg
        self.answer_embedding = answer_embedding

    def forward(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        loss_cfg = self.cfg.get("loss", {})
        lambdas = {
            "classification": float(loss_cfg.get("lambda_classification", 1.0)),
            "generation": float(loss_cfg.get("lambda_generation", 0.0)),
            "alignment": float(loss_cfg.get("lambda_alignment", 0.0)),
            "latent": float(loss_cfg.get("lambda_latent", 0.0)),
            "localization": float(loss_cfg.get("lambda_localization", 0.0)),
        }

        losses: dict[str, torch.Tensor] = {}
        logits = outputs["logits"]
        labels = batch["labels"]
        losses["classification"] = classification_loss(
            logits,
            labels,
            kind=str(loss_cfg.get("classification", "cross_entropy")),
            focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        )

        if lambdas["alignment"] > 0:
            losses["alignment"] = info_nce_loss(outputs["visual_global"], outputs["text_global"])
        else:
            losses["alignment"] = logits.new_tensor(0.0)

        if lambdas["latent"] > 0 and outputs.get("latent") is not None and self.answer_embedding is not None:
            valid = labels >= 0
            if valid.any():
                lat = outputs["latent"]["latent_pooled"][valid]
                tgt = self.answer_embedding(labels[valid])
                losses["latent"] = latent_consistency_loss(
                    lat,
                    tgt,
                    kind=str(self.cfg.get("model", {}).get("latent_prompt", {}).get("consistency_loss", "cosine")),
                )
            else:
                losses["latent"] = logits.new_tensor(0.0)
        else:
            losses["latent"] = logits.new_tensor(0.0)

        # Optional localization: encourage confidence on samples with bbox/mask
        if lambdas["localization"] > 0:
            conf = outputs["confidence"]
            has_loc = torch.tensor(
                [1.0 if b is not None else 0.0 for b in batch.get("bboxes", [None] * conf.size(0))],
                device=conf.device,
            )
            if has_loc.sum() > 0:
                losses["localization"] = ((1.0 - conf) * has_loc).sum() / has_loc.sum()
            else:
                losses["localization"] = logits.new_tensor(0.0)
        else:
            losses["localization"] = logits.new_tensor(0.0)

        if lambdas["generation"] > 0 and outputs.get("open") and outputs["open"].get("lm_loss") is not None:
            losses["generation"] = outputs["open"]["lm_loss"]
        else:
            losses["generation"] = logits.new_tensor(0.0)

        total = (
            lambdas["classification"] * losses["classification"]
            + lambdas["generation"] * losses["generation"]
            + lambdas["alignment"] * losses["alignment"]
            + lambdas["latent"] * losses["latent"]
            + lambdas["localization"] * losses["localization"]
        )
        if torch.isnan(total):
            raise RuntimeError("NaN loss detected")
        losses["total"] = total
        return losses
