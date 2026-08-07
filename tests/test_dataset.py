"""Dataset and split tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.datasets import LabelVocab, patient_level_split
from bonemedvqa.data.validators import validate_annotations


def test_patient_level_split_no_leak():
    rows = []
    for p in range(10):
        for i in range(2):
            rows.append(
                {
                    "sample_id": f"s{p}_{i}",
                    "patient_id": f"p{p}",
                    "image_path": f"x{p}_{i}.png",
                    "question": "Is there a fracture?",
                    "answer": "yes" if p % 2 == 0 else "no",
                }
            )
    split_rows = patient_level_split(rows, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0)
    by_patient = {}
    for r in split_rows:
        pid = r["patient_id"]
        if pid in by_patient:
            assert by_patient[pid] == r["split"]
        by_patient[pid] = r["split"]
    assert set(by_patient.values()) == {"train", "val", "test"}


def test_label_vocab():
    vocab = LabelVocab(["Yes", "no", "YES", "wrist"])
    assert len(vocab) == 3
    assert vocab.encode("yes") == vocab.encode("YES")
    assert vocab.decode(vocab.encode("wrist")) == "wrist"


def test_validate_annotations_detects_leak():
    rows = [
        {"sample_id": "a", "patient_id": "p1", "image_path": "missing.png", "question": "q", "answer": "yes", "split": "train"},
        {"sample_id": "b", "patient_id": "p1", "image_path": "missing.png", "question": "q2", "answer": "no", "split": "test"},
    ]
    report = validate_annotations(rows, check_images=False)
    assert report.patient_split_leaks
