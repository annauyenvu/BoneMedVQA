"""Clinical utility and explainability evaluation metrics."""

from __future__ import annotations

from typing import Any


def score_treatment_alignment(record: dict[str, Any]) -> float:
    """Fraction of gold treatment IDs matched by prediction."""
    gold = set(record.get("treatment_gold") or [])
    pred = set(record.get("treatment_pred") or [])
    if not gold:
        return 1.0 if not pred else 0.0
    return len(gold & pred) / len(gold)


def compute_clinical_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate clinical relevance and explainability proxy metrics."""
    tx_records = [r for r in records if r.get("treatment_gold")]
    tx_scores = [score_treatment_alignment(r) for r in tx_records]

    explained = [r for r in records if r.get("heatmap_url") or r.get("mask_url") or r.get("has_explanation")]
    abstained = [r for r in records if r.get("abstained")]

    low_conf = [r for r in records if float(r.get("confidence", 1.0)) < 0.55]

    return {
        "n": len(records),
        "treatment_alignment_mean": float(sum(tx_scores) / len(tx_scores)) if tx_scores else None,
        "treatment_alignment_pct": round(sum(tx_scores) / len(tx_scores) * 100, 2) if tx_scores else None,
        "clinical_relevance_target": ">majority reasonable (expert validation)",
        "explainability_coverage_pct": round(len(explained) / len(records) * 100, 2) if records else 0.0,
        "abstention_rate_pct": round(len(abstained) / len(records) * 100, 2) if records else 0.0,
        "low_confidence_rate_pct": round(len(low_conf) / len(records) * 100, 2) if records else 0.0,
        "notes": (
            "Treatment alignment uses KG ID overlap. "
            "Clinical relevance requires expert review for publication."
        ),
    }


def compute_explainability_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Proxy metrics for explainability (overlay availability, prompt activation)."""
    if not records:
        return {"n": 0}

    visual_on = sum(1 for r in records if (r.get("activated_prompts") or {}).get("visual"))
    textual_on = sum(1 for r in records if (r.get("activated_prompts") or {}).get("textual"))
    latent_on = sum(1 for r in records if (r.get("activated_prompts") or {}).get("latent"))
    has_overlay = sum(1 for r in records if r.get("heatmap_url") or r.get("mask_url"))

    return {
        "n": len(records),
        "visual_prompt_rate_pct": round(visual_on / len(records) * 100, 2),
        "textual_prompt_rate_pct": round(textual_on / len(records) * 100, 2),
        "latent_prompt_rate_pct": round(latent_on / len(records) * 100, 2),
        "overlay_available_pct": round(has_overlay / len(records) * 100, 2),
        "interpretability_note": (
            "Attention/ROI overlays are supportive visualizations, not causal explanations."
        ),
    }
