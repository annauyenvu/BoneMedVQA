#!/usr/bin/env python
"""Evaluate a checkpoint on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.collators import BoneMedVQACollator
from bonemedvqa.data.datasets import BoneMedVQADataset, LabelVocab
from bonemedvqa.evaluation.calibration import expected_calibration_error, reliability_bins
from bonemedvqa.evaluation.closed_metrics import compute_closed_metrics
from bonemedvqa.evaluation.error_analysis import analyze_errors
from bonemedvqa.models.model_factory import build_model
from bonemedvqa.prompting.textual_prompt import TextualPromptBuilder
from bonemedvqa.training.checkpoint import load_checkpoint
from bonemedvqa.utils.device import resolve_device
from bonemedvqa.utils.io import ensure_dir, load_yaml, read_jsonl, save_json, write_jsonl
from bonemedvqa.utils.logger import get_logger
from bonemedvqa.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    logger = get_logger("evaluate")
    cfg = load_yaml(ROOT / args.config)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    device = resolve_device(cfg.get("device"))

    rows = read_jsonl(ROOT / cfg["data"]["annotation_path"])
    vocab = LabelVocab.from_rows(rows)
    prompt_builder = TextualPromptBuilder()
    ds = BoneMedVQADataset(
        rows,
        image_root=ROOT / cfg["data"].get("image_root", "data/images"),
        split=args.split,
        vocab=vocab,
        image_size=int(cfg.get("model", {}).get("visual_encoder", {}).get("image_size", 224)),
        textual_prompt_builder=prompt_builder,
        train=False,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=BoneMedVQACollator())

    model = build_model(cfg, num_classes=len(vocab), id_to_label=vocab.id_to_label).to(device)
    ckpt = ROOT / args.checkpoint
    if ckpt.exists():
        load_checkpoint(ckpt, model, map_location=device)
        logger.info("Loaded %s", ckpt)
    else:
        logger.warning("Checkpoint missing (%s); evaluating random init for smoke only.", ckpt)

    model.eval()
    y_true, y_pred, y_prob, confs, correct_flags = [], [], [], [], []
    records = []
    threshold = float(cfg.get("calibration", {}).get("confidence_threshold", 0.55))

    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch_dev)
            preds = out["pred"].cpu().tolist()
            labels = batch["labels"].tolist()
            probs = out["probs"].cpu()
            conf = out["confidence"].cpu().tolist()
            for i, (p, y) in enumerate(zip(preds, labels)):
                if y < 0:
                    continue
                y_true.append(y)
                y_pred.append(p)
                y_prob.append(probs[i].tolist())
                confs.append(conf[i])
                correct_flags.append(p == y)
                abstained = conf[i] < threshold
                records.append(
                    {
                        "sample_id": batch["sample_ids"][i],
                        "question": batch["questions"][i],
                        "gold": vocab.decode(y),
                        "pred": vocab.decode(p),
                        "confidence": conf[i],
                        "abstained": abstained,
                        "anatomy": "",
                    }
                )

    metrics = compute_closed_metrics(y_true, y_pred, y_prob)
    metrics["ece"] = expected_calibration_error(confs, correct_flags)
    metrics["trainable_params"] = model.count_trainable_parameters()
    metrics["checkpoint"] = str(ckpt)
    metrics["split"] = args.split
    metrics["warning"] = "Metrics are only meaningful after real training on licensed data."

    out_dir = ensure_dir(ROOT / "outputs")
    save_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions" / "test_predictions.jsonl", records)
    df = analyze_errors(records)
    df.to_csv(out_dir / "error_analysis.csv", index=False)

    # Confusion matrix figure
    fig_dir = ensure_dir(out_dir / "figures")
    cm = np.array(metrics.get("confusion_matrix", [[0]]))
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Reliability diagram
    bins = reliability_bins(confs, correct_flags)
    xs, ys = [], []
    for b in bins["bins"]:
        if b["confidence"] is not None:
            xs.append(b["confidence"])
            ys.append(b["accuracy"])
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    if xs:
        plt.scatter(xs, ys)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability (ECE={metrics['ece']:.3f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "reliability_diagram.png", dpi=150)
    plt.close()

    logger.info("Metrics: %s", metrics)
    logger.info("Wrote outputs/metrics.json and figures/")


if __name__ == "__main__":
    main()
