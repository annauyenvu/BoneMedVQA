"""Confidence / abstention helpers."""

from __future__ import annotations

from typing import Any


ABSTAIN_MESSAGE_VI = (
    "Không đủ độ tin cậy để đưa ra câu trả lời. "
    "Vui lòng kiểm tra lại ảnh hoặc tham khảo ý kiến chuyên gia y tế."
)

ABSTAIN_MESSAGE_EN = (
    "Insufficient confidence to answer. "
    "Please re-check the image or consult a medical expert."
)

WARNING = (
    "Kết quả chỉ có mục đích nghiên cứu và tham khảo, "
    "không thay thế kết luận của bác sĩ hoặc chuyên gia y tế. "
    "/ For research use only. Not a medical diagnosis."
)


def calibrate_confidence(raw_confidence: float, temperature: float = 1.0) -> float:
    """Simple temperature-aware confidence clamp (logits already softened upstream)."""
    # Placeholder for post-hoc temperature scaling applied at logit level.
    return float(min(1.0, max(0.0, raw_confidence)))


def apply_abstention(
    answer: str,
    confidence: float,
    threshold: float = 0.55,
    language: str = "vi",
) -> dict[str, Any]:
    abstained = confidence < threshold
    msg = ABSTAIN_MESSAGE_VI if language.startswith("vi") else ABSTAIN_MESSAGE_EN
    return {
        "answer": msg if abstained else answer,
        "confidence": confidence,
        "abstained": abstained,
        "warning": WARNING,
    }
