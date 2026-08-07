"""Dataset validation and patient-leak checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class ValidationReport:
    """Structured dataset validation report."""

    n_samples: int = 0
    n_patients: int = 0
    missing_images: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    missing_answers: list[str] = field(default_factory=list)
    invalid_bboxes: list[str] = field(default_factory=list)
    empty_masks: list[str] = field(default_factory=list)
    duplicate_questions: list[str] = field(default_factory=list)
    patient_split_leaks: list[str] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)
    split_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_images
            or self.corrupt_images
            or self.missing_answers
            or self.invalid_bboxes
            or self.patient_split_leaks
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_annotations(
    rows: list[dict[str, Any]],
    image_root: str | Path | None = None,
    mask_root: str | Path | None = None,
    check_images: bool = True,
) -> ValidationReport:
    """Validate unified annotation schema and patient-level splits."""
    report = ValidationReport(n_samples=len(rows))
    patients = {r.get("patient_id") for r in rows if r.get("patient_id")}
    report.n_patients = len(patients)

    q_counter: Counter[str] = Counter()
    split_patients: dict[str, set[str]] = defaultdict(set)
    label_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()

    for row in rows:
        sid = str(row.get("sample_id", "<unknown>"))
        answer = row.get("answer")
        if answer is None or str(answer).strip() == "":
            report.missing_answers.append(sid)
        else:
            label_counter[str(answer).strip().lower()] += 1

        split = str(row.get("split", "unknown"))
        split_counter[split] += 1
        pid = str(row.get("patient_id", ""))
        if pid:
            split_patients[split].add(pid)

        q = str(row.get("question", "")).strip().lower()
        key = f"{row.get('image_path','')}|{q}"
        q_counter[key] += 1

        bbox = row.get("bbox")
        if bbox is not None:
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                report.invalid_bboxes.append(sid)
            else:
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1 or min(bbox) < 0:
                    report.invalid_bboxes.append(sid)

        if check_images and image_root is not None and row.get("image_path"):
            img_path = Path(image_root) / Path(row["image_path"]).name
            if not Path(row["image_path"]).is_absolute():
                # Prefer relative path under image_root / as stored
                candidates = [
                    Path(row["image_path"]),
                    Path(image_root) / row["image_path"],
                    img_path,
                ]
            else:
                candidates = [Path(row["image_path"])]
            found = next((p for p in candidates if p.exists()), None)
            if found is None:
                report.missing_images.append(sid)
            else:
                try:
                    with Image.open(found) as im:
                        im.verify()
                    # reopen after verify
                    with Image.open(found) as im:
                        w, h = im.size
                    if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        if x2 > w or y2 > h:
                            report.invalid_bboxes.append(sid)
                except Exception:
                    report.corrupt_images.append(sid)

        mask_path = row.get("mask_path")
        if mask_path and mask_root is not None:
            mp = Path(mask_path)
            if not mp.exists():
                mp = Path(mask_root) / Path(mask_path).name
            if mp.exists():
                try:
                    import numpy as np

                    arr = np.array(Image.open(mp))
                    if arr.max() == 0:
                        report.empty_masks.append(sid)
                except Exception:
                    report.empty_masks.append(sid)

    report.duplicate_questions = [k for k, c in q_counter.items() if c > 1]
    report.label_counts = dict(label_counter)
    report.split_counts = dict(split_counter)

    # Patient leak across splits
    seen: dict[str, str] = {}
    for split, pids in split_patients.items():
        for pid in pids:
            if pid in seen and seen[pid] != split:
                report.patient_split_leaks.append(f"{pid}:{seen[pid]}->{split}")
            else:
                seen[pid] = split

    # Class imbalance warning
    if label_counter:
        total = sum(label_counter.values())
        for lab, c in label_counter.items():
            if c / total < 0.05:
                report.warnings.append(f"Rare label '{lab}' = {c}/{total}")

    return report
