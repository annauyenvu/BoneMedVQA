#!/usr/bin/env python
"""Train BoneMedVQA baseline (no visual/latent prompt)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.collators import BoneMedVQACollator
from bonemedvqa.data.datasets import BoneMedVQADataset, LabelVocab, patient_level_split
from bonemedvqa.models.model_factory import build_model
from bonemedvqa.prompting.textual_prompt import TextualPromptBuilder
from bonemedvqa.training.trainer import Trainer
from bonemedvqa.utils.device import resolve_device
from bonemedvqa.utils.io import load_yaml, read_jsonl, write_jsonl, ensure_dir
from bonemedvqa.utils.logger import get_logger
from bonemedvqa.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    logger = get_logger("train_baseline")
    cfg = load_yaml(ROOT / args.config)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    device = resolve_device(cfg.get("device"))

    ann_path = ROOT / cfg["data"]["annotation_path"]
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing {ann_path}. Run: python scripts/generate_qa.py")

    rows = read_jsonl(ann_path)
    rows = patient_level_split(rows)
    # persist split if missing
    write_jsonl(ann_path, rows)

    vocab = LabelVocab.from_rows(rows)
    if len(vocab) < 2:
        raise RuntimeError("Need at least 2 answer classes")

    prompt_builder = TextualPromptBuilder() if cfg.get("model", {}).get("use_textual_prompt", True) else None
    image_root = ROOT / cfg["data"].get("image_root", "data/images")
    image_size = int(cfg.get("model", {}).get("visual_encoder", {}).get("image_size", 224))

    train_ds = BoneMedVQADataset(
        rows, image_root=image_root, split="train", vocab=vocab, image_size=image_size, textual_prompt_builder=prompt_builder
    )
    val_ds = BoneMedVQADataset(
        rows, image_root=image_root, split="val", vocab=vocab, image_size=image_size, textual_prompt_builder=prompt_builder
    )
    collator = BoneMedVQACollator(max_length=int(cfg.get("model", {}).get("text_encoder", {}).get("max_length", 64)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.get("train", {}).get("batch_size", 8)),
        shuffle=False,
        collate_fn=collator,
        num_workers=int(cfg["data"].get("num_workers", 0)),
    )

    model = build_model(cfg, num_classes=len(vocab), id_to_label=vocab.id_to_label)
    logger.info("Trainable params: %s | device=%s", model.count_trainable_parameters(), device)
    trainer = Trainer(model, cfg, device, id_to_label=vocab.id_to_label)
    result = trainer.fit(train_loader, val_loader)
    logger.info("Done. Best metric=%s", result["best_metric"])


if __name__ == "__main__":
    main()
