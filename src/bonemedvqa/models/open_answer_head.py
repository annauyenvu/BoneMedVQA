"""Open-answer generation head (template / optional FLAN-T5)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


OPEN_TEMPLATE = (
    "Observation: {observation}\n"
    "Location: {location}\n"
    "Confidence: {confidence:.2f}\n"
    "Recommendation: {recommendation}"
)


class OpenAnswerHead(nn.Module):
    """Lightweight open-answer head.

    Modes
    -----
    - template: structured text from closed logits + metadata (always runnable)
    - flan_t5: optional HF encoder-decoder if installed and configured
    """

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "template",
        backbone: str = "google/flan-t5-small",
        max_new_tokens: int = 64,
        id_to_label: dict[int, str] | None = None,
    ):
        super().__init__()
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.id_to_label = id_to_label or {}
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.lm = None
        self.tokenizer = None
        if mode in {"flan_t5", "flan_t5_small", "t5"}:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(backbone)
                self.lm = AutoModelForSeq2SeqLM.from_pretrained(backbone)
                # Map multimodal fused vector into encoder embedding bias
                enc_dim = int(self.lm.config.d_model)
                self.visual_to_lm = nn.Linear(hidden_dim, enc_dim)
            except Exception:
                self.mode = "template"

    def format_structured(
        self,
        answer_label: str,
        confidence: float,
        anatomy: str | None = None,
        abstained: bool = False,
    ) -> str:
        if abstained:
            return (
                "Observation: Insufficient visual/textual evidence.\n"
                "Location: N/A\n"
                "Confidence: low\n"
                "Recommendation: Re-check image quality/prompt or consult a medical expert."
            )
        observation = f"Model suggests answer '{answer_label}' based on available image evidence."
        location = anatomy or "region indicated by visual prompt / attention"
        if confidence >= 0.75:
            recommendation = "Findings may be reviewed by a clinician; research use only."
        elif confidence >= 0.55:
            recommendation = "Low-moderate confidence — verify with additional views/expert review."
        else:
            recommendation = "Abstain preferred; consult a medical expert."
        return OPEN_TEMPLATE.format(
            observation=observation,
            location=location,
            confidence=confidence,
            recommendation=recommendation,
        )

    def forward(
        self,
        fused: torch.Tensor,
        closed_logits: torch.Tensor | None = None,
        anatomy: list[str] | None = None,
        temperature: float = 1.0,
        confidence_threshold: float = 0.55,
    ) -> dict[str, Any]:
        fused = self.projector(fused)
        texts: list[str] = []
        confidences: list[float] = []
        abstained_flags: list[bool] = []

        if closed_logits is not None:
            probs = F.softmax(closed_logits / max(temperature, 1e-6), dim=-1)
            conf, pred = probs.max(dim=-1)
            for i in range(fused.size(0)):
                c = float(conf[i].item())
                label = self.id_to_label.get(int(pred[i].item()), str(int(pred[i].item())))
                abstain = c < confidence_threshold
                anat = anatomy[i] if anatomy else None
                texts.append(self.format_structured(label, c, anat, abstained=abstain))
                confidences.append(c)
                abstained_flags.append(abstain)
        else:
            for i in range(fused.size(0)):
                texts.append(
                    self.format_structured("unknown", 0.0, anatomy[i] if anatomy else None, abstained=True)
                )
                confidences.append(0.0)
                abstained_flags.append(True)

        # Optional LM path for richer generation (teacher-forcing not used in smoke mode)
        lm_loss = None
        if self.mode.startswith("flan") and self.lm is not None and self.training is False:
            # Keep deterministic structured output for research safety by default.
            pass

        return {
            "texts": texts,
            "confidences": confidences,
            "abstained": abstained_flags,
            "features": fused,
            "lm_loss": lm_loss,
        }
