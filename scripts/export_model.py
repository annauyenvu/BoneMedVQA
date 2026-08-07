#!/usr/bin/env python
"""Export model checkpoint to a deployable bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.models.model_factory import build_model
from bonemedvqa.training.checkpoint import load_checkpoint
from bonemedvqa.utils.io import ensure_dir, load_yaml, save_json
from bonemedvqa.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--out", default="outputs/export")
    parser.add_argument("--num-classes", type=int, default=2)
    args = parser.parse_args()
    logger = get_logger("export_model")
    cfg = load_yaml(ROOT / args.config)
    model = build_model(cfg, num_classes=args.num_classes)
    ckpt = ROOT / args.checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    payload = load_checkpoint(ckpt, model, map_location="cpu")
    out = ensure_dir(ROOT / args.out)
    torch.save({"model_state": model.state_dict(), "extra": payload.get("extra", {})}, out / "model.pt")
    save_json(out / "export_meta.json", {
        "config": args.config,
        "checkpoint": str(ckpt),
        "trainable_params": model.count_trainable_parameters(),
        "warning": "Research use only. Not a medical device.",
    })
    logger.info("Exported to %s", out)


if __name__ == "__main__":
    main()
