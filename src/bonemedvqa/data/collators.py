"""Batch collators for BoneMedVQA."""

from __future__ import annotations

from typing import Any, Optional

import torch


class BoneMedVQACollator:
    """Collate samples; optionally tokenize text with a provided tokenizer."""

    def __init__(
        self,
        tokenizer: Any | None = None,
        max_length: int = 64,
        pad_token_id: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = torch.stack([b["label"] for b in batch], dim=0)
        texts = [b["prompt_text"] for b in batch]

        out: dict[str, Any] = {
            "pixel_values": pixel_values,
            "labels": labels,
            "questions": [b["question"] for b in batch],
            "prompt_texts": texts,
            "answer_texts": [b["answer_text"] for b in batch],
            "sample_ids": [b["sample_id"] for b in batch],
            "patient_ids": [b["patient_id"] for b in batch],
            "question_types": [b["question_type"] for b in batch],
            "bboxes": [b["bbox"] for b in batch],
            "mask_paths": [b["mask_path"] for b in batch],
            "image_paths": [b["image_path"] for b in batch],
        }

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            out["input_ids"] = encoded["input_ids"]
            out["attention_mask"] = encoded["attention_mask"]
        else:
            # Fallback simple char-hash bag for tiny_text encoder
            ids = []
            masks = []
            for t in texts:
                tokens = [((ord(c) * 17) % 1000) + 1 for c in t[: self.max_length]]
                attn = [1] * len(tokens)
                pad = self.max_length - len(tokens)
                tokens = tokens + [self.pad_token_id] * pad
                attn = attn + [0] * pad
                ids.append(tokens)
                masks.append(attn)
            out["input_ids"] = torch.tensor(ids, dtype=torch.long)
            out["attention_mask"] = torch.tensor(masks, dtype=torch.long)

        return out
