"""Localization metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2 * inter / denom)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def pointing_game_accuracy(heatmap: np.ndarray, gt_mask: np.ndarray) -> float:
    """1 if argmax of heatmap falls inside GT mask."""
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    if gt_mask.shape != heatmap.shape:
        return 0.0
    return float(gt_mask[y, x] > 0)


def compute_localization_metrics(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    pred_boxes: list[list[float]] | None = None,
    gt_boxes: list[list[float]] | None = None,
) -> dict[str, Any]:
    if not pred_masks:
        return {"iou": 0.0, "dice": 0.0, "n": 0}
    ious = [iou_score(p, g) for p, g in zip(pred_masks, gt_masks)]
    dices = [dice_score(p, g) for p, g in zip(pred_masks, gt_masks)]
    out: dict[str, Any] = {
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "localization_recall": float(np.mean([i > 0.5 for i in ious])),
        "n": len(ious),
    }
    if pred_boxes and gt_boxes and len(pred_boxes) == len(gt_boxes):
        out["box_iou"] = float(np.mean([box_iou(a, b) for a, b in zip(pred_boxes, gt_boxes)]))
    return out
