#!/usr/bin/env python
"""Generate a synthetic bone-X-ray-like demo dataset for smoke tests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.datasets import patient_level_split
from bonemedvqa.utils.io import ensure_dir, load_yaml, write_jsonl
from bonemedvqa.utils.seed import set_seed


def synthesize_xray(rng: random.Random, abnormal: bool, size: int = 256) -> tuple[Image.Image, list[int] | None]:
    """Create a grayscale-like synthetic radiograph with optional 'lesion' blob."""
    arr = rng.randint(30, 60) + np.random.randn(size, size) * 8
    # bone-like bright structure
    yy, xx = np.mgrid[0:size, 0:size]
    bone = ((xx - size / 2) ** 2 / (size * 0.12) ** 2 + (yy - size / 2) ** 2 / (size * 0.35) ** 2) < 1
    arr = np.clip(arr + bone.astype(np.float32) * 90, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    bbox = None
    if abnormal:
        draw = ImageDraw.Draw(img)
        x1 = rng.randint(size // 3, size // 2)
        y1 = rng.randint(size // 3, size // 2)
        x2 = x1 + rng.randint(20, 45)
        y2 = y1 + rng.randint(12, 30)
        draw.ellipse([x1, y1, x2, y2], fill=(220, 220, 220))
        # dark fracture line
        draw.line([(x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2)], fill=(40, 40, 40), width=2)
        bbox = [x1, y1, x2, y2]
    return img, bbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/synthetic_demo.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = load_yaml(ROOT / args.config)
    qa = cfg.get("qa", {})
    n_patients = int(qa.get("n_patients", 20))
    images_per_patient = int(qa.get("images_per_patient", 2))

    img_dir = ensure_dir(ROOT / cfg["paths"]["image_out"])
    ann_path = ROOT / cfg["paths"]["annotation_out"]
    ensure_dir(ann_path.parent)

    rng = random.Random(args.seed)
    rows = []
    anatomies = ["wrist", "elbow", "hand", "forearm", "ankle"]
    sample_idx = 0
    for p in range(n_patients):
        patient_id = f"patient_{p:03d}"
        for j in range(images_per_patient):
            abnormal = rng.random() < 0.5
            img, bbox = synthesize_xray(rng, abnormal=abnormal)
            fname = f"{patient_id}_img{j}.png"
            img.save(img_dir / fname)
            anatomy = rng.choice(anatomies)
            answer = "yes" if abnormal else "no"
            rows.append(
                {
                    "sample_id": f"sample_{sample_idx:04d}",
                    "patient_id": patient_id,
                    "image_path": str(Path("data/images") / fname).replace("\\", "/"),
                    "question": "Is there evidence of a fracture?",
                    "answer": answer,
                    "question_type": "closed",
                    "answer_type": "yes_no",
                    "anatomy": anatomy,
                    "abnormality": "fracture" if abnormal else "normal",
                    "mask_path": None,
                    "bbox": bbox,
                    "synthetic_qa": True,
                    "template": "Is there evidence of a fracture?",
                }
            )
            # second QA: anatomy
            sample_idx += 1
            rows.append(
                {
                    "sample_id": f"sample_{sample_idx:04d}",
                    "patient_id": patient_id,
                    "image_path": str(Path("data/images") / fname).replace("\\", "/"),
                    "question": "What body part is shown?",
                    "answer": anatomy,
                    "question_type": "closed",
                    "answer_type": "anatomy",
                    "anatomy": anatomy,
                    "abnormality": "fracture" if abnormal else "normal",
                    "mask_path": None,
                    "bbox": bbox,
                    "synthetic_qa": True,
                    "template": "What body part is shown?",
                }
            )
            sample_idx += 1

    split_cfg = cfg.get("split", {})
    rows = patient_level_split(
        rows,
        train_ratio=float(split_cfg.get("train_ratio", 0.6)),
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        test_ratio=float(split_cfg.get("test_ratio", 0.2)),
        seed=int(split_cfg.get("seed", args.seed)),
    )
    write_jsonl(ann_path, rows)
    print(f"Wrote {len(rows)} samples to {ann_path}")
    print(f"Images in {img_dir}")


if __name__ == "__main__":
    main()
