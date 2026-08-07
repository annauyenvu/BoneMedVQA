"""Visual encoders with freeze / optional LoRA hooks."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCNNEncoder(nn.Module):
    """Lightweight CNN for CPU/smoke tests (no pretrained download)."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.hidden_dim = hidden_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.stem(pixel_values)  # (B, D, H', W')
        b, d, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H'*W', D)
        global_feat = self.pool(feat).flatten(1)
        return {
            "tokens": tokens,
            "global": global_feat,
            "grid_size": (h, w),
        }


class TimmVisualEncoder(nn.Module):
    """timm backbone wrapper with projection to hidden_dim."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        hidden_dim: int = 256,
        freeze: bool = False,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        # Infer feature channels with a dummy forward shape assumption
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone.forward_features(dummy)
            if feat.ndim == 4:
                in_dim = feat.shape[1]
            else:
                in_dim = feat.shape[-1]
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone.forward_features(pixel_values)
        if feat.ndim == 4:
            b, c, h, w = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)
            tokens = self.proj(tokens)
            global_feat = self.proj(self.pool(feat).flatten(1))
            return {"tokens": tokens, "global": global_feat, "grid_size": (h, w)}
        # (B, N, C) or (B, C)
        if feat.ndim == 2:
            global_feat = self.proj(feat)
            tokens = global_feat.unsqueeze(1)
            return {"tokens": tokens, "global": global_feat, "grid_size": (1, 1)}
        tokens = self.proj(feat)
        global_feat = tokens.mean(dim=1)
        return {"tokens": tokens, "global": global_feat, "grid_size": (tokens.size(1), 1)}


class VisualEncoder(nn.Module):
    """Configurable visual encoder facade."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        backbone = str(cfg.get("backbone", "tiny_cnn")).lower()
        hidden_dim = int(cfg.get("hidden_dim", 256))
        pretrained = bool(cfg.get("pretrained", False))
        freeze = bool(cfg.get("freeze", False))

        if backbone in {"tiny_cnn", "tiny"}:
            self.encoder = TinyCNNEncoder(hidden_dim=hidden_dim)
        elif backbone.startswith("timm_") or backbone in {"resnet18", "resnet34", "efficientnet_b0"}:
            name = backbone.replace("timm_", "") if backbone.startswith("timm_") else backbone
            self.encoder = TimmVisualEncoder(
                backbone=name,
                pretrained=pretrained,
                hidden_dim=hidden_dim,
                freeze=freeze,
            )
        elif backbone in {"vit_tiny", "vit"}:
            # Use tiny CNN as stand-in if timm vit not desired; prefer timm
            try:
                self.encoder = TimmVisualEncoder(
                    backbone="vit_tiny_patch16_224",
                    pretrained=pretrained,
                    hidden_dim=hidden_dim,
                    freeze=freeze,
                )
            except Exception:
                self.encoder = TinyCNNEncoder(hidden_dim=hidden_dim)
        else:
            # Default safe path for smoke tests
            self.encoder = TinyCNNEncoder(hidden_dim=hidden_dim)

        self.hidden_dim = hidden_dim
        if freeze and not isinstance(self.encoder, TimmVisualEncoder):
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Optional LoRA for linear layers (lightweight PEFT without hard peft dep)
        if cfg.get("lora"):
            self._inject_lora(int(cfg.get("lora_r", 8)), int(cfg.get("lora_alpha", 16)))

    def _inject_lora(self, r: int, alpha: int) -> None:
        """Attach simple LoRA adapters to Linear layers in projection heads."""
        scale = alpha / r

        class LoRALinear(nn.Module):
            def __init__(self, base: nn.Linear):
                super().__init__()
                self.base = base
                self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
                self.B = nn.Parameter(torch.zeros(base.out_features, r))
                self.scale = scale
                for p in self.base.parameters():
                    p.requires_grad = False

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

        if hasattr(self.encoder, "proj") and isinstance(self.encoder.proj, nn.Linear):
            self.encoder.proj = LoRALinear(self.encoder.proj)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected (B,3,H,W), got {tuple(pixel_values.shape)}")
        out = self.encoder(pixel_values)
        if torch.isnan(out["global"]).any():
            raise RuntimeError("NaN detected in visual encoder output")
        return out
