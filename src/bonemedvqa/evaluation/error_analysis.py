"""Error analysis helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def analyze_errors(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build an error analysis table from prediction records.

    Expected keys per record: sample_id, question, gold, pred, confidence, abstained.
    """
    rows = []
    for r in records:
        gold = str(r.get("gold", "")).lower()
        pred = str(r.get("pred", "")).lower()
        conf = float(r.get("confidence", 0.0))
        abstained = bool(r.get("abstained", False))
        error_type = "correct"
        if abstained:
            error_type = "abstained"
        elif gold != pred:
            if gold in {"no", "normal"} and pred in {"yes", "abnormal", "fracture"}:
                error_type = "false_positive_pathology"
            elif gold in {"yes", "abnormal", "fracture"} and pred in {"no", "normal"}:
                error_type = "false_negative_pathology"
            else:
                error_type = "misclassification"
        rows.append(
            {
                "sample_id": r.get("sample_id"),
                "question": r.get("question"),
                "gold": gold,
                "pred": pred,
                "confidence": conf,
                "abstained": abstained,
                "error_type": error_type,
                "anatomy": r.get("anatomy"),
            }
        )
    return pd.DataFrame(rows)
