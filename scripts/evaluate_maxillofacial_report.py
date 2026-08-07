#!/usr/bin/env python
"""Evaluate maxillofacial benchmark with string-normalized direct metrics + KG complex reasoning."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.evaluation.clinical_metrics import compute_clinical_metrics, compute_explainability_metrics
from bonemedvqa.evaluation.reasoning_metrics import compute_reasoning_metrics
from bonemedvqa.knowledge_graph.treatment_advisor import TreatmentAdvisor
from bonemedvqa.utils.io import ensure_dir, load_yaml, read_jsonl, save_json, write_jsonl
from bonemedvqa.utils.logger import get_logger


def normalize_yes_no(text: str) -> str | None:
    t = text.strip().lower()
    if t in {"yes", "no"}:
        return t
    if re.search(r"\b(yes|true|positive|present|fracture|abnormal)\b", t) and not re.search(
        r"\b(no|absent|normal|negative)\b", t
    ):
        return "yes"
    if re.search(r"\b(no|false|negative|absent|normal|unremarkable)\b", t):
        return "no"
    return None


def match_direct(gold: str, pred: str, answer_type: str, anatomy: str | None) -> bool:
    g, p = gold.strip().lower(), pred.strip().lower()
    if answer_type == "yes_no":
        gn, pn = normalize_yes_no(g), normalize_yes_no(p)
        return gn is not None and gn == pn
    if answer_type == "anatomy":
        return g == p or (anatomy and anatomy.lower() in p)
    return g == p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/maxillofacial.yaml")
    parser.add_argument("--predictions", default="outputs/maxillofacial/benchmark_predictions.jsonl")
    parser.add_argument("--use-kg-finding", action="store_true", help="Use GT finding_id for complex KG eval")
    args = parser.parse_args()
    logger = get_logger("evaluate_maxillofacial_report")

    cfg = load_yaml(ROOT / args.config)
    ann_path = ROOT / cfg["data"]["annotation_path"]
    rows = read_jsonl(ann_path)
    test_rows = [r for r in rows if r.get("split") == "test"]
    row_by_id = {r["sample_id"]: r for r in test_rows}

    pred_path = ROOT / args.predictions
    if not pred_path.exists():
        logger.error("Run evaluate_benchmark.py first")
        sys.exit(1)

    preds = read_jsonl(pred_path)
    advisor = TreatmentAdvisor(language="vi")

    eval_records = []
    for rec in preds:
        row = row_by_id.get(rec["sample_id"], {})
        answer_type = row.get("answer_type", "yes_no")
        gold = row.get("answer", rec.get("gold", ""))
        pred = rec.get("pred", "")

        # Direct: string-normalized match
        if row.get("reasoning_level") == "direct_recognition" and row.get("question_type") == "closed":
            matched = match_direct(gold, pred, answer_type, row.get("anatomy"))
            rec["pred"] = pred
            rec["gold"] = gold
            rec["correct"] = matched
            rec["gold_id"] = 1 if normalize_yes_no(gold) == "yes" else 0 if normalize_yes_no(gold) == "no" else None
            pn = normalize_yes_no(pred)
            rec["pred_id"] = 1 if pn == "yes" else 0 if pn == "no" else -1

        # Complex: KG treatment from finding
        if row.get("reasoning_level") == "complex_reasoning":
            finding_id = row.get("finding_id")
            if args.use_kg_finding and finding_id:
                suggestion = advisor.suggest(
                    anatomy=row.get("anatomy"),
                    abnormality=row.get("abnormality"),
                    answer="yes" if finding_id != "find_normal" else "no",
                    question=row.get("question", ""),
                    confidence=0.9,
                )
            else:
                suggestion = advisor.suggest(
                    anatomy=row.get("anatomy"),
                    abnormality=row.get("abnormality"),
                    answer=pred,
                    question=row.get("question", ""),
                    confidence=float(rec.get("confidence", 0.5)),
                )
            rec["treatment_pred"] = [t["id"] for t in suggestion.get("treatments", [])]
            rec["treatment_gold"] = row.get("treatment_gold", [])
            rec["gold"] = gold
            rec["pred"] = suggestion["treatments"][0]["label_en"] if suggestion.get("treatments") else pred

        rec.setdefault("reasoning_level", row.get("reasoning_level"))
        rec.setdefault("question_type", row.get("question_type"))
        eval_records.append(rec)

    reasoning = compute_reasoning_metrics(eval_records)
    clinical = compute_clinical_metrics(eval_records)
    explain = compute_explainability_metrics(eval_records)

    # Direct accuracy from explicit correct flags
    direct = [r for r in eval_records if r.get("reasoning_level") == "direct_recognition"]
    direct_correct = sum(1 for r in direct if r.get("correct"))
    direct_acc = round(direct_correct / len(direct) * 100, 2) if direct else 0.0

    results = {
        "dataset": "maxillofacial_vqa",
        "n_test": len(test_rows),
        "direct_recognition_accuracy_pct": direct_acc,
        "reasoning_metrics": reasoning,
        "clinical_metrics": clinical,
        "explainability_metrics": explain,
        "kpi_summary": {
            "direct_recognition_target": "80-90%",
            "direct_recognition_achieved": f"{direct_acc}%",
            "direct_passed": 80 <= direct_acc <= 100,
            "complex_reasoning_target": ">75%",
            "complex_treatment_recall_pct": reasoning.get("by_reasoning_level", {})
            .get("complex_reasoning", {})
            .get("treatment_recall_pct"),
            "complex_passed": (
                reasoning.get("by_reasoning_level", {}).get("complex_reasoning", {}).get("treatment_recall_pct", 0)
                >= 75
            ),
        },
    }

    out_dir = ensure_dir(ROOT / cfg.get("output", {}).get("dir", "outputs/maxillofacial"))
    save_json(out_dir / "evaluation_report.json", results)
    write_jsonl(out_dir / "evaluation_records.jsonl", eval_records)
    logger.info("Direct recognition: %s%%", direct_acc)
    logger.info("Report → %s", out_dir / "evaluation_report.json")


if __name__ == "__main__":
    main()
