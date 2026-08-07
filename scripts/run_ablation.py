#!/usr/bin/env python
"""Run ablation configurations V/T/L combinations."""

from __future__ import annotations

import argparse
import csv
import copy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.utils.io import ensure_dir, load_yaml, save_json
from bonemedvqa.utils.logger import get_logger


EXPERIMENT_FLAGS = {
    "Baseline": (False, False, False),
    "V": (True, False, False),
    "T": (False, True, False),
    "L": (False, False, True),
    "VT": (True, True, False),
    "VL": (True, False, True),
    "TL": (False, True, True),
    "VTL": (True, True, True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/lightweight.yaml")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["Baseline", "V", "T", "L", "VT", "VL", "TL", "VTL"],
    )
    parser.add_argument("--epochs", type=int, default=1, help="Override epochs for quick ablation")
    parser.add_argument("--dry-run", action="store_true", help="Only write configs, do not train")
    args = parser.parse_args()
    logger = get_logger("run_ablation")

    base = load_yaml(ROOT / args.base_config)
    abl_dir = ensure_dir(ROOT / "outputs" / "ablation")
    rows = []

    for name in args.experiments:
        key = name.replace("+", "").replace(" ", "")
        if key not in EXPERIMENT_FLAGS and name not in EXPERIMENT_FLAGS:
            logger.warning("Unknown experiment %s — skip", name)
            continue
        flags = EXPERIMENT_FLAGS.get(name) or EXPERIMENT_FLAGS[key]
        v, t, l = flags
        cfg = copy.deepcopy(base)
        cfg["experiment"]["name"] = f"ablation_{name}"
        cfg["model"]["use_visual_prompt"] = v
        cfg["model"]["use_textual_prompt"] = t
        cfg["model"]["use_latent_prompt"] = l
        if l:
            cfg["model"].setdefault("latent_prompt", {})["enabled"] = True
            cfg["loss"]["lambda_latent"] = cfg.get("loss", {}).get("lambda_latent", 0.2) or 0.2
        else:
            cfg["model"].setdefault("latent_prompt", {})["enabled"] = False
            cfg["loss"]["lambda_latent"] = 0.0
        cfg["train"]["epochs"] = args.epochs
        cfg["output"]["checkpoint_dir"] = str(abl_dir / name / "checkpoints")
        cfg["output"]["log_dir"] = str(abl_dir / name / "logs")
        cfg_path = abl_dir / f"config_{name}.yaml"
        # save as json-compatible yaml via json dump then note
        save_json(cfg_path.with_suffix(".json"), cfg)
        logger.info("Prepared %s -> V=%s T=%s L=%s", name, v, t, l)

        metrics = {
            "Experiment": name,
            "Visual": "Yes" if v else "No",
            "Textual": "Yes" if t else "No",
            "Latent": "Yes" if l else "No",
            "Accuracy": "",
            "Macro F1": "",
            "BERTScore": "",
            "IoU": "",
            "ECE": "",
            "Trainable Params": "",
        }

        if not args.dry_run:
            # Train via train_full using JSON config bridge
            # Write a temporary YAML-like by reusing load through json path in train? 
            # train_full expects yaml — write minimal yaml manually
            import yaml

            ypath = abl_dir / f"config_{name}.yaml"
            with open(ypath, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            cmd = [sys.executable, str(ROOT / "scripts" / "train_full.py"), "--config", str(ypath.relative_to(ROOT))]
            logger.info("Running: %s", " ".join(cmd))
            subprocess.run(cmd, cwd=str(ROOT), check=False)
            # Fill params from model config estimate after train if metrics exist
            hist = Path(cfg["output"]["log_dir"]) / "history.json"
            if hist.exists():
                import json

                history = json.loads(hist.read_text(encoding="utf-8"))
                if history:
                    last = history[-1]
                    metrics["Accuracy"] = last.get("val_accuracy", "")
                    metrics["Macro F1"] = last.get("val_macro_f1", "")
        rows.append(metrics)

    csv_path = abl_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (fill metrics after real runs; empty cells mean not yet evaluated)", csv_path)


if __name__ == "__main__":
    main()
