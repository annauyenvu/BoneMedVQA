#!/usr/bin/env python
"""Prepare / convert external datasets into unified JSONL (adapters).

Does NOT auto-download datasets. Users must place raw files after accepting licenses.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.datasets import patient_level_split
from bonemedvqa.utils.io import ensure_dir, load_yaml, write_jsonl
from bonemedvqa.utils.logger import get_logger


def adapt_generic_csv(csv_path: Path, image_root: Path, cfg: dict) -> list[dict]:
    """Generic CSV adapter expecting columns: patient_id, image_path, label/anatomy."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            label = str(r.get("label") or r.get("abnormal") or "").strip().lower()
            if label in {"1", "positive", "abnormal", "fracture", "true"}:
                answer = "yes"
                abnormality = "fracture"
            elif label in {"0", "negative", "normal", "false"}:
                answer = "no"
                abnormality = "normal"
            else:
                answer = label
                abnormality = r.get("abnormality", "")
            img_rel = r.get("image_path") or r.get("path")
            rows.append(
                {
                    "sample_id": r.get("sample_id", f"sample_{i:05d}"),
                    "patient_id": r.get("patient_id", f"patient_{i:05d}"),
                    "image_path": str(Path(img_rel)),
                    "question": "Is there evidence of a fracture?",
                    "answer": answer,
                    "question_type": "closed",
                    "answer_type": "yes_no",
                    "anatomy": r.get("anatomy", cfg.get("task", "")),
                    "abnormality": abnormality,
                    "mask_path": r.get("mask_path"),
                    "bbox": json.loads(r["bbox"]) if r.get("bbox") else None,
                    "synthetic_qa": True,
                    "template": "Is there evidence of a fracture?",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Dataset YAML under configs/datasets/")
    parser.add_argument("--csv", default=None, help="Optional local CSV to convert")
    args = parser.parse_args()
    logger = get_logger("prepare_dataset")
    cfg = load_yaml(ROOT / args.config)
    name = cfg.get("name", "custom")
    logger.info(
        "Dataset=%s license=%s — ensure you accepted terms before use.",
        name,
        cfg.get("license"),
    )

    out_ann = ROOT / cfg["paths"]["annotation_out"]
    ensure_dir(out_ann.parent)

    if args.csv:
        rows = adapt_generic_csv(Path(args.csv), ROOT / cfg["paths"].get("image_out", "data/images"), cfg)
        split_cfg = cfg.get("split", {})
        if split_cfg.get("strategy", "").startswith("patient"):
            rows = patient_level_split(
                rows,
                train_ratio=float(split_cfg.get("train_ratio", 0.7)),
                val_ratio=float(split_cfg.get("val_ratio", 0.15)),
                test_ratio=float(split_cfg.get("test_ratio", 0.15)),
                seed=int(split_cfg.get("seed", 42)),
            )
        write_jsonl(out_ann, rows)
        logger.info("Wrote %s rows to %s", len(rows), out_ann)
    else:
        logger.warning(
            "No --csv provided. Place licensed raw data under %s and re-run with a converter.",
            cfg["paths"].get("raw_root"),
        )
        logger.info("For a runnable demo, use: python scripts/generate_qa.py")


if __name__ == "__main__":
    main()
