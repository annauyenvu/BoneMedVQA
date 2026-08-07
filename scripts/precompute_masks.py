#!/usr/bin/env python
"""Precompute visual prompt masks offline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.transforms import load_image
from bonemedvqa.prompting.prompt_cache import PromptCache
from bonemedvqa.prompting.visual_prompt import VisualPromptGenerator, build_segmentor
from bonemedvqa.utils.io import ensure_dir, load_yaml, read_jsonl
from bonemedvqa.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lightweight.yaml")
    parser.add_argument("--annotations", default="data/processed/annotations.jsonl")
    args = parser.parse_args()
    logger = get_logger("precompute_masks")
    cfg = load_yaml(ROOT / args.config)
    vp = cfg.get("model", {}).get("visual_prompt", {})
    cache_dir = ROOT / vp.get("cache_dir", "data/cache/masks")
    cache = PromptCache(cache_dir)
    segmentor = build_segmentor(str(vp.get("segmentor", "heuristic")))
    gen = VisualPromptGenerator(segmentor=segmentor)

    rows = read_jsonl(ROOT / args.annotations)
    n = 0
    for row in rows:
        img_path = ROOT / row["image_path"]
        if not img_path.exists():
            img_path = ROOT / "data/images" / Path(row["image_path"]).name
        if not img_path.exists():
            logger.warning("Missing image for %s", row.get("sample_id"))
            continue
        image = np.array(load_image(img_path))
        if row.get("bbox"):
            bundle = gen.generate_from_box(image, row["bbox"])
            payload = {"type": "box", "box": row["bbox"]}
        else:
            bundle = gen.generate_automatic(image)
            payload = {"type": "automatic"}
        cache.set(
            row["sample_id"],
            payload,
            {
                "mask": bundle.mask,
                "box": np.asarray(bundle.box, dtype=np.float32),
                "masked_image": bundle.masked_image,
            },
        )
        n += 1
    logger.info("Cached visual prompts for %s samples -> %s", n, cache_dir)


if __name__ == "__main__":
    main()
