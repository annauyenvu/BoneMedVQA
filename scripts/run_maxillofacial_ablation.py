#!/usr/bin/env python
"""Run maxillofacial ablation and fill results CSV with benchmark metrics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.utils.io import ensure_dir, load_yaml, read_jsonl
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
    parser.add_argument("--base-config", default="configs/maxillofacial.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--experiments", nargs="+", default=["Baseline", "V", "T", "VT", "VTL"])
    args = parser.parse_args()
    logger = get_logger("maxillofacial_ablation")
    base = load_yaml(ROOT / args.base_config)
    abl_dir = ensure_dir(ROOT / "outputs" / "maxillofacial" / "ablation")
    rows_out = []

    import yaml

    for name in args.experiments:
        if name not in EXPERIMENT_FLAGS:
            continue
        v, t, l = EXPERIMENT_FLAGS[name]
        cfg = copy.deepcopy(base)
        cfg["experiment"]["name"] = f"mf_ablation_{name}"
        cfg["model"]["use_visual_prompt"] = v
        cfg["model"]["use_textual_prompt"] = t
        cfg["model"]["use_latent_prompt"] = l
        cfg["model"].setdefault("latent_prompt", {})["enabled"] = l
        cfg["train"]["epochs"] = args.epochs
        out_sub = abl_dir / name
        cfg["output"]["dir"] = str(out_sub)
        cfg["output"]["checkpoint_dir"] = str(out_sub / "checkpoints")
        cfg["output"]["log_dir"] = str(out_sub / "logs")
        ypath = abl_dir / f"config_{name}.yaml"
        with open(ypath, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        cmd = [sys.executable, str(ROOT / "scripts" / "train_full.py"), "--config", str(ypath.relative_to(ROOT))]
        logger.info("Training ablation %s", name)
        subprocess.run(cmd, cwd=str(ROOT), check=False)

        bench_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_benchmark.py"),
            "--config",
            str(ypath.relative_to(ROOT)),
            "--checkpoint",
            str((out_sub / "checkpoints" / "best.pt").relative_to(ROOT)),
        ]
        subprocess.run(bench_cmd, cwd=str(ROOT), check=False)

        report_path = out_sub / "benchmark_results.json"
        direct_acc, complex_tx, macro_f1 = "", "", ""
        if report_path.exists():
            data = json.loads(report_path.read_text(encoding="utf-8"))
            rm = data.get("reasoning_metrics", {}).get("by_reasoning_level", {})
            direct_acc = rm.get("direct_recognition", {}).get("closed_accuracy_pct", "")
            complex_tx = rm.get("complex_reasoning", {}).get("treatment_recall_pct", "")

        hist_path = out_sub / "logs" / "history.json"
        if hist_path.exists():
            hist = json.loads(hist_path.read_text(encoding="utf-8"))
            if hist:
                macro_f1 = hist[-1].get("val_macro_f1", "")

        rows_out.append(
            {
                "Experiment": name,
                "Visual": "Yes" if v else "No",
                "Textual": "Yes" if t else "No",
                "Latent": "Yes" if l else "No",
                "Direct Acc (%)": direct_acc,
                "Complex TX Recall (%)": complex_tx,
                "Macro F1": macro_f1,
            }
        )

    csv_path = abl_dir / "ablation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    logger.info("Wrote %s", csv_path)


if __name__ == "__main__":
    main()
