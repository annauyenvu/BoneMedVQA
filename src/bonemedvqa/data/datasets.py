"""Unified BoneMedVQA dataset and patient-level splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from bonemedvqa.data.transforms import build_transforms, load_image
from bonemedvqa.utils.io import read_jsonl


class LabelVocab:
    """Bidirectional answer label vocabulary."""

    def __init__(self, labels: list[str] | None = None):
        labels = labels or []
        uniq = sorted({str(x).strip().lower() for x in labels if str(x).strip()})
        self.label_to_id = {lab: i for i, lab in enumerate(uniq)}
        self.id_to_label = {i: lab for lab, i in self.label_to_id.items()}

    def __len__(self) -> int:
        return len(self.label_to_id)

    def encode(self, label: str) -> int:
        key = str(label).strip().lower()
        if key not in self.label_to_id:
            raise KeyError(f"Unknown label: {label}")
        return self.label_to_id[key]

    def decode(self, idx: int) -> str:
        return self.id_to_label[int(idx)]

    def to_dict(self) -> dict[str, Any]:
        return {"label_to_id": self.label_to_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelVocab":
        vocab = cls()
        vocab.label_to_id = {str(k): int(v) for k, v in d["label_to_id"].items()}
        vocab.id_to_label = {i: lab for lab, i in vocab.label_to_id.items()}
        return vocab

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> "LabelVocab":
        return cls([r.get("answer", "") for r in rows])


def patient_level_split(
    rows: list[dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Assign split labels by patient_id with no patient overlap.

    If rows already have consistent patient-level splits, they are preserved
    after a leak check. Otherwise ratios are applied.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pid = str(r.get("patient_id") or r.get("sample_id"))
        by_patient[pid].append(r)

    # If all rows already have split and no leak, keep them.
    existing = all("split" in r for r in rows)
    if existing:
        split_of: dict[str, str] = {}
        leak = False
        for pid, items in by_patient.items():
            splits = {str(x.get("split")) for x in items}
            if len(splits) != 1:
                leak = True
                break
            split_of[pid] = next(iter(splits))
        if not leak:
            # still ensure returned copies
            return [dict(r) for r in rows]

    rng = random.Random(seed)
    patient_ids = sorted(by_patient.keys())
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train : n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val :])

    out: list[dict[str, Any]] = []
    for pid, items in by_patient.items():
        if pid in train_ids:
            split = "train"
        elif pid in val_ids:
            split = "val"
        else:
            split = "test"
        for item in items:
            row = dict(item)
            row["split"] = split
            out.append(row)
    return out


def resolve_image_path(image_path: str, image_root: str | Path | None) -> Path:
    p = Path(image_path)
    if p.exists():
        return p
    if image_root is not None:
        cand = Path(image_root) / image_path
        if cand.exists():
            return cand
        cand2 = Path(image_root) / p.name
        if cand2.exists():
            return cand2
    raise FileNotFoundError(f"Cannot resolve image path: {image_path}")


class BoneMedVQADataset(Dataset):
    """PyTorch dataset over unified JSONL annotations."""

    def __init__(
        self,
        annotations: str | Path | list[dict[str, Any]],
        image_root: str | Path | None = None,
        split: str | None = "train",
        vocab: LabelVocab | None = None,
        image_size: int = 224,
        train: bool | None = None,
        apply_clahe: bool = False,
        textual_prompt_builder: Any | None = None,
    ):
        if isinstance(annotations, (str, Path)):
            rows = read_jsonl(annotations)
        else:
            rows = list(annotations)

        if split is not None:
            rows = [r for r in rows if str(r.get("split", "")).lower() == split.lower()]

        self.rows = rows
        self.image_root = Path(image_root) if image_root else None
        self.vocab = vocab or LabelVocab.from_rows(rows)
        self.train = train if train is not None else (split == "train")
        self.transform = build_transforms(
            image_size=image_size,
            train=self.train,
            apply_clahe=apply_clahe,
        )
        self.textual_prompt_builder = textual_prompt_builder

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        img_path = resolve_image_path(row["image_path"], self.image_root)
        image = load_image(img_path)
        tfm = self.transform(image)
        pixel_values = tfm["pixel_values"]

        question = str(row.get("question", ""))
        if self.textual_prompt_builder is not None:
            prompt_text = self.textual_prompt_builder.build(
                question=question,
                question_type=row.get("question_type", "closed"),
                anatomy=row.get("anatomy"),
                abnormality=row.get("abnormality"),
                answer_type=row.get("answer_type"),
            )
        else:
            prompt_text = question

        answer = str(row.get("answer", "")).strip().lower()
        label = self.vocab.encode(answer) if answer in self.vocab.label_to_id else -100

        sample = {
            "sample_id": row.get("sample_id", str(idx)),
            "patient_id": row.get("patient_id", ""),
            "pixel_values": pixel_values,
            "question": question,
            "prompt_text": prompt_text,
            "answer_text": answer,
            "label": torch.tensor(label, dtype=torch.long),
            "question_type": row.get("question_type", "closed"),
            "anatomy": row.get("anatomy", ""),
            "abnormality": row.get("abnormality", ""),
            "bbox": row.get("bbox"),
            "mask_path": row.get("mask_path"),
            "resize_meta": tfm["resize_meta"],
            "image_path": str(img_path),
        }
        return sample
