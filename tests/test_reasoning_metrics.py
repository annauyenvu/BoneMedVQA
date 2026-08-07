"""Tests for reasoning-level evaluation metrics."""

from bonemedvqa.evaluation.reasoning_metrics import compute_reasoning_metrics


def test_reasoning_metrics_targets():
    records = [
        {
            "reasoning_level": "direct_recognition",
            "question_type": "closed",
            "gold_id": 1,
            "pred_id": 1,
            "prob": [0.1, 0.9],
        },
        {
            "reasoning_level": "direct_recognition",
            "question_type": "closed",
            "gold_id": 0,
            "pred_id": 0,
            "prob": [0.8, 0.2],
        },
        {
            "reasoning_level": "complex_reasoning",
            "question_type": "open",
            "gold": "open reduction internal fixation",
            "pred": "open reduction internal fixation (ORIF)",
            "treatment_gold": ["tx_open_reduction"],
            "treatment_pred": ["tx_open_reduction", "tx_refer_omfs"],
        },
    ]
    metrics = compute_reasoning_metrics(records)
    assert metrics["by_reasoning_level"]["direct_recognition"]["closed_accuracy_pct"] == 100.0
    assert metrics["targets"]["direct_recognition"]["passed"] is True
    assert metrics["by_reasoning_level"]["complex_reasoning"]["treatment_recall_pct"] == 100.0
