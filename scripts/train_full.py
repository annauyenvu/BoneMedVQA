#!/usr/bin/env python
"""Train full / lightweight BoneMedVQA configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.collators import BoneMedVQACollator
from bonemedvqa.data.datasets import BoneMedVQADataset, LabelVocab, patient_level_split
from bonemedvqa.models.model_factory import build_model
from bonemedvqa.prompting.textual_prompt import TextualPromptBuilder
from bonemedvqa.training.trainer import Trainer
from bonemedvqa.utils.device import resolve_device
from bonemedvqa.utils.io import load_yaml, read_jsonl, write_jsonl
from bonemedvqa.utils.logger import get_logger
from bonemedvqa.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lightweight.yaml")
    args = parser.parse_args()
    logger = get_logger("train_full")
    cfg = load_yaml(ROOT / args.config)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    device = resolve_device(cfg.get("device"))

    ann_path = ROOT / cfg["data"]["annotation_path"]
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing {ann_path}. Run scripts/generate_qa.py first.")
    rows = patient_level_split(read_jsonl(ann_path))
    write_jsonl(ann_path, rows)
    vocab = LabelVocab.from_rows(rows)

    prompt_builder = TextualPromptBuilder(**(cfg.get("model", {}).get("textual_prompt") or {}))
    image_root = ROOT / cfg["data"].get("image_root", "data/images")
    image_size = int(cfg.get("model", {}).get("visual_encoder", {}).get("image_size", 224))
    train_ds = BoneMedVQADataset(
        rows, image_root=image_root, split="train", vocab=vocab, image_size=image_size, textual_prompt_builder=prompt_builder
    )
    val_ds = BoneMedVQADataset(
        rows, image_root=image_root, split="val", vocab=vocab, image_size=image_size, textual_prompt_builder=prompt_builder
    )
    collator = BoneMedVQACollator(max_length=int(cfg.get("model", {}).get("text_encoder", {}).get("max_length", 64)))
    bs = int(cfg.get("train", {}).get("batch_size", 4))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collator)

    model = build_model(cfg, num_classes=len(vocab), id_to_label=vocab.id_to_label)
    logger.info(
        "Profile=%s trainable=%s device=%s V=%s T=%s L=%s",
        cfg.get("experiment", {}).get("profile"),
        model.count_trainable_parameters(),
        device,
        model.use_visual_prompt,
        model.use_textual_prompt,
        model.use_latent_prompt,
    )
    trainer = Trainer(model, cfg, device, id_to_label=vocab.id_to_label)
    result = trainer.fit(train_loader, val_loader)
    logger.info("Finished. best=%s", result["best_metric"])


if __name__ == "__main__":
    main()
