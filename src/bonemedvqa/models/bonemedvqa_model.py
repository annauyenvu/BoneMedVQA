"""Full BoneMedVQA model combining V/T/L prompts."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bonemedvqa.models.closed_answer_head import ClosedAnswerHead
from bonemedvqa.models.local_global_encoder import LocalGlobalEncoder
from bonemedvqa.models.multimodal_fusion import ConcatFusion, CrossAttentionFusion
from bonemedvqa.models.open_answer_head import OpenAnswerHead
from bonemedvqa.models.text_encoder import TextEncoder
from bonemedvqa.models.token_compressor import TokenCompressor
from bonemedvqa.models.visual_encoder import VisualEncoder
from bonemedvqa.prompting.latent_prompt import LatentPromptGenerator


class BoneMedVQAModel(nn.Module):
    """Multimodal Med-VQA model with optional visual/textual/latent prompts."""

    MEDICAL_WARNING = (
        "For research use only. Not a medical diagnosis. "
        "Results do not replace a clinician's judgment."
    )

    def __init__(self, cfg: dict[str, Any], num_classes: int, id_to_label: dict[int, str] | None = None):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg.get("model", cfg)
        self.use_visual_prompt = bool(model_cfg.get("use_visual_prompt", False))
        self.use_textual_prompt = bool(model_cfg.get("use_textual_prompt", True))
        self.use_latent_prompt = bool(model_cfg.get("use_latent_prompt", False))
        self.fusion_type = str(model_cfg.get("fusion", "concat")).lower()

        v_cfg = model_cfg.get("visual_encoder", {})
        t_cfg = model_cfg.get("text_encoder", {})
        hidden_dim = int(v_cfg.get("hidden_dim", 256))
        # Ensure text hidden matches visual
        t_cfg = {**t_cfg, "hidden_dim": hidden_dim}
        v_cfg = {**v_cfg, "hidden_dim": hidden_dim}

        self.visual_encoder = VisualEncoder(v_cfg)
        self.text_encoder = TextEncoder(t_cfg)
        self.hidden_dim = hidden_dim

        self.visual_prompt_emb = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.local_global = None
        if model_cfg.get("local_global", {}).get("enabled", False):
            self.local_global = LocalGlobalEncoder(
                hidden_dim=hidden_dim,
                roi_size=int(model_cfg.get("local_global", {}).get("roi_size", 112)),
            )

        self.token_compressor = None
        if model_cfg.get("token_compression", {}).get("enabled", False):
            self.token_compressor = TokenCompressor(
                hidden_dim=hidden_dim,
                ratio=int(model_cfg.get("token_compression", {}).get("ratio", 4)),
            )

        lp_cfg = model_cfg.get("latent_prompt", {})
        self.latent_prompt = None
        if self.use_latent_prompt and lp_cfg.get("enabled", True):
            self.latent_prompt = LatentPromptGenerator(
                num_latent_tokens=int(lp_cfg.get("num_tokens", 8)),
                hidden_dim=hidden_dim,
                num_heads=int(lp_cfg.get("num_heads", 4)),
                num_layers=int(lp_cfg.get("num_layers", 2)),
                dropout=float(lp_cfg.get("dropout", 0.1)),
                concept_bank=bool(lp_cfg.get("concept_bank", True)),
            )

        if self.fusion_type == "concat":
            dims = [hidden_dim, hidden_dim]
            if self.use_latent_prompt:
                dims.append(hidden_dim)
            if self.use_visual_prompt:
                dims.append(hidden_dim)
            self.fusion = ConcatFusion(dims=dims, hidden_dim=hidden_dim)
            self.cross_fusion = None
        else:
            self.fusion = None
            self.cross_fusion = CrossAttentionFusion(hidden_dim=hidden_dim, num_heads=4)

        closed_cfg = model_cfg.get("closed_head", {})
        self.closed_head = ClosedAnswerHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=float(closed_cfg.get("dropout", 0.1)),
        )

        open_cfg = model_cfg.get("open_head", {})
        self.open_enabled = bool(open_cfg.get("enabled", False))
        self.open_head = None
        if self.open_enabled:
            self.open_head = OpenAnswerHead(
                hidden_dim=hidden_dim,
                mode=str(open_cfg.get("mode", "template")),
                backbone=str(open_cfg.get("backbone", "google/flan-t5-small")),
                max_new_tokens=int(open_cfg.get("max_new_tokens", 64)),
                id_to_label=id_to_label or {},
            )

        self.id_to_label = id_to_label or {}
        self.temperature = float(cfg.get("calibration", {}).get("temperature", 1.0))
        self.confidence_threshold = float(
            cfg.get("calibration", {}).get("confidence_threshold", 0.55)
        )

    def encode_visual_prompt_feature(
        self,
        pixel_values: torch.Tensor,
        prompt_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode masked/prompt view; fall back to zeros if disabled/missing."""
        if not self.use_visual_prompt:
            return torch.zeros(pixel_values.size(0), self.hidden_dim, device=pixel_values.device)
        view = prompt_images if prompt_images is not None else pixel_values
        feats = self.visual_encoder(view)["global"]
        return self.visual_prompt_emb(feats)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        vis = self.visual_encoder(pixel_values)
        visual_tokens = vis["tokens"]
        visual_global = vis["global"]

        if self.token_compressor is not None:
            visual_tokens = self.token_compressor(visual_tokens, vis.get("grid_size"))

        if self.local_global is not None:
            local_pixels = self.local_global.crop_roi(pixel_values, batch.get("bboxes"))
            local_global = self.visual_encoder(local_pixels)["global"]
            visual_global = self.local_global(visual_global, local_global)

        txt = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_tokens = txt["tokens"]
        text_global = txt["global"]

        prompt_feat = self.encode_visual_prompt_feature(
            pixel_values, batch.get("prompt_pixel_values")
        )

        latent_out = None
        latent_pooled = None
        if self.latent_prompt is not None:
            text_pad = (attention_mask == 0) if attention_mask is not None else None
            latent_out = self.latent_prompt(
                visual_tokens=visual_tokens,
                textual_tokens=text_tokens,
                textual_padding_mask=text_pad,
            )
            latent_pooled = latent_out["latent_pooled"]

        attn_weights = None
        if self.fusion_type == "concat":
            feats = [visual_global, text_global]
            if latent_pooled is not None:
                feats.append(latent_pooled)
            if self.use_visual_prompt:
                feats.append(prompt_feat)
            fused = self.fusion(feats)
        else:
            # Build memory from available modalities
            memories = [visual_tokens, text_tokens]
            if latent_pooled is not None:
                memories.append(latent_out["latent_tokens"])
            if self.use_visual_prompt:
                memories.append(prompt_feat.unsqueeze(1))
            memory = torch.cat(memories, dim=1)
            query = visual_global + text_global
            if latent_pooled is not None:
                query = query + latent_pooled
            if self.use_visual_prompt:
                query = query + prompt_feat
            fused, attn_weights = self.cross_fusion(query, memory)

        logits = self.closed_head(fused)
        probs = F.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        confidence, pred = probs.max(dim=-1)
        abstained = confidence < self.confidence_threshold

        open_out = None
        if self.open_head is not None:
            open_out = self.open_head(
                fused,
                closed_logits=logits,
                anatomy=batch.get("anatomies"),
                temperature=self.temperature,
                confidence_threshold=self.confidence_threshold,
            )

        return {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "confidence": confidence,
            "abstained": abstained,
            "fused": fused,
            "visual_global": visual_global,
            "text_global": text_global,
            "latent": latent_out,
            "attention_weights": attn_weights,
            "open": open_out,
            "warning": self.MEDICAL_WARNING,
        }

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
