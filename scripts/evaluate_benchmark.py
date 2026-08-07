#!/usr/bin/env python
"""Full benchmark: direct recognition, complex reasoning, KG treatment, explainability."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.collators import BoneMedVQACollator
from bonemedvqa.data.datasets import BoneMedVQADataset, LabelVocab
from bonemedvqa.evaluation.clinical_metrics import compute_clinical_metrics, compute_explainability_metrics
from bonemedvqa.evaluation.reasoning_metrics import compute_reasoning_metrics
from bonemedvqa.knowledge_graph.treatment_advisor import TreatmentAdvisor
from bonemedvqa.models.model_factory import build_model
from bonemedvqa.prompting.textual_prompt import TextualPromptBuilder
from bonemedvqa.training.checkpoint import load_checkpoint
from bonemedvqa.utils.device import resolve_device
from bonemedvqa.utils.io import ensure_dir, load_yaml, read_jsonl, save_json, write_jsonl
from bonemedvqa.utils.logger import get_logger
from bonemedvqa.utils.seed import set_seed


TX_KEYWORDS = {
    "tx_closed_reduction": ["closed reduction", "imf", "intermaxillary"],
    "tx_open_reduction": ["orif", "open reduction", "internal fixation"],
    "tx_refer_omfs": ["referral", "maxillofacial surgery", "omfs", "specialist"],
    "tx_tmj_reduction": ["tmj reduction", "manual reduction"],
    "tx_surgical_extraction": ["surgical extraction", "extraction"],
    "tx_endodontic": ["endodontic", "root canal", "drainage"],
    "tx_observation": ["observation", "follow-up", "monitor"],
    "tx_soft_diet": ["soft diet", "analgesia"],
}


def extract_treatment_ids(text: str) -> list[str]:
    t = text.lower()
    found = []
    for tx_id, keys in TX_KEYWORDS.items():
        if any(k in t for k in keys):
            found.append(tx_id)
    return found


def normalize_closed_answer(text: str, vocab: LabelVocab) -> int | None:
    t = text.strip().lower()
    if t in vocab.label_to_id:
        return vocab.label_to_id[t]
    if t in {"yes", "true", "positive"}:
        return vocab.label_to_id.get("yes")
    if t in {"no", "false", "negative"}:
        return vocab.label_to_id.get("no")
    for label in vocab.label_to_id:
        if label in t or t in label:
            return vocab.label_to_id[label]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maxillofacial.yaml")
    parser.add_argument("--checkpoint", default="outputs/maxillofacial/checkpoints/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-config", default="configs/datasets/maxillofacial_vqa.yaml")
    args = parser.parse_args()
    logger = get_logger("evaluate_benchmark")
    cfg = load_yaml(ROOT / args.config)
    set_seed(int(cfg.get("experiment", {}).get("seed", 42)))
    device = resolve_device(cfg.get("device"))

    ann_path = ROOT / cfg["data"]["annotation_path"]
    if not ann_path.exists():
        logger.info("Annotations missing — generating maxillofacial dataset...")
        import subprocess

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "generate_maxillofacial_qa.py")],
            cwd=str(ROOT),
            check=True,
        )

    rows = read_jsonl(ann_path)
    test_rows = [r for r in rows if r.get("split") == args.split]
    if not test_rows:
        test_rows = [r for r in rows if r.get("split") == "test"]
    row_by_id = {r["sample_id"]: r for r in test_rows}
    logger.info("Evaluating %d test samples", len(test_rows))

    vocab = LabelVocab.from_rows(rows)
    prompt_builder = TextualPromptBuilder(language=cfg.get("model", {}).get("textual_prompt", {}).get("language", "vi"))
    ds = BoneMedVQADataset(
        test_rows,
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
        logger.info("Loaded checkpoint %s", ckpt)
    else:
        logger.warning("Checkpoint not found (%s) — using random init (run train first)", ckpt)

    kg_enabled = bool(cfg.get("knowledge_graph", {}).get("enabled", False))
    advisor = TreatmentAdvisor(language=cfg.get("knowledge_graph", {}).get("language", "vi")) if kg_enabled else None
    threshold = float(cfg.get("calibration", {}).get("confidence_threshold", 0.55))

    model.eval()
    eval_records = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch_dev)
            preds = out["pred"].cpu().tolist()
            probs = out["probs"].cpu()
            confs = out["confidence"].cpu().tolist()
            for i in range(len(batch["sample_ids"])):
                sid = batch["sample_ids"][i]
                row = row_by_id.get(sid, {})
                gold = row.get("answer", "")
                pred_label = vocab.decode(preds[i])
                conf = confs[i]
                qtype = row.get("question_type", "closed")
                rec = {
                    "sample_id": batch["sample_ids"][i],
                    "question": batch["questions"][i],
                    "gold": gold,
                    "pred": pred_label,
                    "confidence": conf,
                    "abstained": conf < threshold,
                    "question_type": qtype,
                    "reasoning_level": row.get("reasoning_level", "unknown"),
                    "anatomy": row.get("anatomy"),
                    "abnormality": row.get("abnormality"),
                    "finding_id": row.get("finding_id"),
                    "treatment_gold": row.get("treatment_gold", []),
                    "activated_prompts": {
                        "visual": bool(cfg.get("model", {}).get("use_visual_prompt")),
                        "textual": bool(cfg.get("model", {}).get("use_textual_prompt")),
                        "latent": bool(cfg.get("model", {}).get("use_latent_prompt")),
                    },
                    "has_explanation": bool(cfg.get("model", {}).get("use_visual_prompt")),
                }
                if qtype == "closed":
                    rec["gold_id"] = normalize_closed_answer(gold, vocab)
                    rec["pred_id"] = preds[i]
                    rec["prob"] = probs[i].tolist()
                if advisor and qtype == "open":
                    suggestion = advisor.suggest(
                        anatomy=row.get("anatomy"),
                        abnormality=row.get("abnormality"),
                        answer=pred_label,
                        question=row.get("question", ""),
                        confidence=conf,
                    )
                    rec["treatment_pred"] = [t["id"] for t in suggestion.get("treatments", [])]
                    rec["kg_summary"] = suggestion.get("summary_vi") or suggestion.get("summary_en")
                    # Open answer proxy: KG primary treatment text
                    if suggestion.get("treatments"):
                        rec["pred"] = suggestion["treatments"][0]["label_en"]
                eval_records.append(rec)

    reasoning = compute_reasoning_metrics(eval_records)
    clinical = compute_clinical_metrics(eval_records)
    explain = compute_explainability_metrics(eval_records)

    results = {
        "dataset": "maxillofacial_vqa",
        "split": args.split,
        "n_samples": len(eval_records),
        "checkpoint": str(ckpt),
        "knowledge_graph_enabled": kg_enabled,
        "reasoning_metrics": reasoning,
        "clinical_metrics": clinical,
        "explainability_metrics": explain,
        "profile": cfg.get("experiment", {}).get("profile", "maxillofacial"),
    }

    out_dir = ensure_dir(ROOT / cfg.get("output", {}).get("dir", "outputs/maxillofacial"))
    save_json(out_dir / "benchmark_results.json", results)
    write_jsonl(out_dir / "benchmark_predictions.jsonl", eval_records)
    logger.info("Benchmark complete → %s", out_dir / "benchmark_results.json")
    logger.info(
        "Direct acc: %s%% | Complex treatment recall: %s%%",
        reasoning.get("by_reasoning_level", {}).get("direct_recognition", {}).get("closed_accuracy_pct"),
        reasoning.get("by_reasoning_level", {}).get("complex_reasoning", {}).get("treatment_recall_pct"),
    )


if __name__ == "__main__":
    main()
