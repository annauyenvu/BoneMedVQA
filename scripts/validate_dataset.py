#!/usr/bin/env python
"""Validate dataset annotations and emit a report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.validators import validate_annotations
from bonemedvqa.utils.io import ensure_dir, load_json, load_yaml, read_jsonl, save_json
from bonemedvqa.utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/synthetic_demo.yaml")
    parser.add_argument("--annotations", default=None, help="Override annotation JSONL path")
    parser.add_argument("--out", default="outputs/data_validation_report.json")
    args = parser.parse_args()
    logger = get_logger("validate_dataset")

    cfg_path = ROOT / args.config
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    ann = Path(args.annotations) if args.annotations else ROOT / cfg.get("paths", {}).get(
        "annotation_out", "data/processed/annotations.jsonl"
    )
    if not ann.exists():
        raise FileNotFoundError(
            f"Annotations not found: {ann}. Run scripts/generate_qa.py first for synthetic demo."
        )
    rows = read_jsonl(ann)
    image_root = ROOT / "data/images"
    report = validate_annotations(rows, image_root=image_root, check_images=True)
    out = ROOT / args.out
    ensure_dir(out.parent)
    save_json(out, report.to_dict())
    logger.info("Validation ok=%s | samples=%s patients=%s", report.ok, report.n_samples, report.n_patients)
    logger.info("Report saved to %s", out)
    if not report.ok:
        logger.error("Validation failed: %s", {k: v for k, v in report.to_dict().items() if v})
        sys.exit(1)


if __name__ == "__main__":
    main()
