"""Stratified metrics for direct recognition vs complex reasoning."""

from __future__ import annotations

from typing import Any

from bonemedvqa.evaluation.closed_metrics import compute_closed_metrics
from bonemedvqa.evaluation.generation_metrics import exact_match_single, token_f1_single


def stratify_by_reasoning(records: list[dict[str, Any]]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"direct_recognition": [], "complex_reasoning": [], "unknown": []}
    for r in records:
        level = r.get("reasoning_level") or r.get("metadata", {}).get("reasoning_level", "unknown")
        buckets.setdefault(level, []).append(r)
    return buckets


def compute_reasoning_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute KPIs split by reasoning level.

    Expected record keys: reasoning_level, question_type, gold, pred, confidence,
    treatment_gold (optional), treatment_pred (optional).
    """
    buckets = stratify_by_reasoning(records)
    out: dict[str, Any] = {"overall": {}, "by_reasoning_level": {}, "targets": {}}

    all_closed_true, all_closed_pred, all_closed_prob = [], [], []
    for rec in records:
        if rec.get("question_type") == "closed" and rec.get("gold_id") is not None:
            all_closed_true.append(rec["gold_id"])
            all_closed_pred.append(rec.get("pred_id", -1))
            if rec.get("prob") is not None:
                all_closed_prob.append(rec["prob"])

    if all_closed_true:
        out["overall"]["closed"] = compute_closed_metrics(all_closed_true, all_closed_pred, all_closed_prob or None)

    for level, subset in buckets.items():
        if not subset or level == "unknown":
            continue
        level_metrics: dict[str, Any] = {"n": len(subset)}

        closed = [r for r in subset if r.get("question_type") == "closed" and r.get("gold_id") is not None]
        if closed:
            yt = [r["gold_id"] for r in closed]
            yp = [r.get("pred_id", -1) for r in closed]
            probs = [r["prob"] for r in closed if r.get("prob") is not None]
            level_metrics["closed"] = compute_closed_metrics(yt, yp, probs if len(probs) == len(closed) else None)
            level_metrics["closed_accuracy_pct"] = round(level_metrics["closed"]["accuracy"] * 100, 2)

        open_rows = [r for r in subset if r.get("question_type") == "open"]
        if open_rows:
            em_scores = [exact_match_single(r.get("gold", ""), r.get("pred", "")) for r in open_rows]
            f1_scores = [token_f1_single(r.get("gold", ""), r.get("pred", "")) for r in open_rows]
            tx_hits = []
            for r in open_rows:
                gold_tx = set(r.get("treatment_gold") or [])
                pred_tx = set(r.get("treatment_pred") or [])
                if gold_tx:
                    tx_hits.append(len(gold_tx & pred_tx) / len(gold_tx))
            level_metrics["open"] = {
                "exact_match": float(sum(em_scores) / len(em_scores)),
                "token_f1": float(sum(f1_scores) / len(f1_scores)),
                "n": len(open_rows),
            }
            if tx_hits:
                level_metrics["treatment_recall"] = float(sum(tx_hits) / len(tx_hits))
                level_metrics["treatment_recall_pct"] = round(level_metrics["treatment_recall"] * 100, 2)

        out["by_reasoning_level"][level] = level_metrics

    # Target pass/fail
    direct = out["by_reasoning_level"].get("direct_recognition", {})
    complex_ = out["by_reasoning_level"].get("complex_reasoning", {})
    direct_acc = direct.get("closed_accuracy_pct")
    complex_tx = complex_.get("treatment_recall_pct") or complex_.get("open", {}).get("token_f1", 0) * 100

    out["targets"] = {
        "direct_recognition": {
            "target_range": "80-90%",
            "achieved_pct": direct_acc,
            "passed": direct_acc is not None and 80 <= direct_acc <= 100,
        },
        "complex_reasoning": {
            "target_min": ">75%",
            "achieved_pct": round(complex_tx, 2) if complex_tx else None,
            "passed": complex_tx is not None and complex_tx >= 75,
        },
    }
    return out
