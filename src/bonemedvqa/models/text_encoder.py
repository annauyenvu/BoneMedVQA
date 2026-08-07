"""Text encoders (tiny bag-of-char / HF transformers)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class TinyTextEncoder(nn.Module):
    """Embedding + mean-pool encoder for smoke tests (no HF download)."""

    def __init__(self, vocab_size: int = 1001, hidden_dim: int = 256, max_length: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos = nn.Embedding(max_length, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
                dropout=0.1,
            ),
            num_layers=2,
        )
        self.hidden_dim = hidden_dim
        self.max_length = max_length

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, n = input_ids.shape
        pos_ids = torch.arange(n, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = self.emb(input_ids) + self.pos(pos_ids)
        key_padding = None
        if attention_mask is not None:
            key_padding = attention_mask == 0
        tokens = self.encoder(x, src_key_padding_mask=key_padding)
        if attention_mask is None:
            pooled = tokens.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return {"tokens": tokens, "global": pooled}


class HFTextEncoder(nn.Module):
    """Hugging Face encoder wrapper with projection."""

    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 256,
        freeze: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name) if pretrained else AutoModel.from_config(
            __import__("transformers").AutoConfig.from_pretrained(model_name)
        )
        in_dim = int(self.model.config.hidden_size)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tokens = self.proj(out.last_hidden_state)
        if attention_mask is None:
            pooled = tokens.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return {"tokens": tokens, "global": pooled}


BACKBONE_MAP = {
    "tiny_text": None,
    "roberta_base": "roberta-base",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
}


class TextEncoder(nn.Module):
    """Configurable text encoder facade."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        backbone = str(cfg.get("backbone", "tiny_text")).lower()
        hidden_dim = int(cfg.get("hidden_dim", 256))
        freeze = bool(cfg.get("freeze", False))
        pretrained = bool(cfg.get("pretrained", False))
        max_length = int(cfg.get("max_length", 64))
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.tokenizer = None

        if backbone in {"tiny_text", "tiny"}:
            self.encoder = TinyTextEncoder(hidden_dim=hidden_dim, max_length=max_length)
        else:
            model_name = BACKBONE_MAP.get(backbone, backbone)
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.encoder = HFTextEncoder(
                    model_name=model_name,
                    hidden_dim=hidden_dim,
                    freeze=freeze,
                    pretrained=pretrained,
                )
            except Exception:
                # Offline / no-download fallback
                self.encoder = TinyTextEncoder(hidden_dim=hidden_dim, max_length=max_length)

        if freeze and isinstance(self.encoder, TinyTextEncoder):
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)
