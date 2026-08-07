#!/usr/bin/env python
"""Generate maxillofacial X-ray-like VQA dataset with reasoning levels and KG labels."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bonemedvqa.data.datasets import patient_level_split
from bonemedvqa.utils.io import ensure_dir, load_yaml, write_jsonl
from bonemedvqa.utils.seed import set_seed

ANATOMY_LABELS = {
    "mandible": "Mandible (lower jaw)",
    "maxilla": "Maxilla (upper jaw)",
    "tmj": "Temporomandibular joint",
    "zygoma": "Zygomatic bone",
    "dental": "Dental arch",
}

FINDING_KG = {
    "fracture_mandible": "find_mandible_fracture",
    "fracture_maxilla": "find_maxilla_fracture",
    "fracture_zygoma": "find_zygoma_fracture",
    "dislocation_tmj": "find_tmj_dislocation",
    "impacted_tooth": "find_impacted_tooth",
    "periapical_lesion": "find_periapical_lesion",
    "normal": "find_normal",
}

TREATMENT_GOLD = {
    "find_mandible_fracture": ["tx_closed_reduction", "tx_open_reduction", "tx_refer_omfs"],
    "find_maxilla_fracture": ["tx_open_reduction", "tx_refer_omfs"],
    "find_zygoma_fracture": ["tx_open_reduction", "tx_refer_omfs"],
    "find_tmj_dislocation": ["tx_tmj_reduction", "tx_soft_diet"],
    "find_impacted_tooth": ["tx_surgical_extraction", "tx_observation"],
    "find_periapical_lesion": ["tx_endodontic", "tx_surgical_extraction"],
    "find_normal": ["tx_observation"],
}


def synthesize_panoramic(rng: random.Random, anatomy: str, abnormal: str, size: int = 384):
    """Create a panoramic-like synthetic maxillofacial radiograph."""
    arr = rng.randint(25, 45) + np.random.randn(size, size) * 6
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, int(size * 0.55)

    # U-shaped mandible arch
    draw.arc([cx - 120, cy - 40, cx + 120, cy + 100], 200, 340, fill=(210, 210, 210), width=8)
    # Maxilla band
    draw.arc([cx - 100, cy - 80, cx + 100, cy + 20], 160, 380, fill=(190, 190, 190), width=6)
    # TMJ dots
    draw.ellipse([cx - 125, cy - 10, cx - 105, cy + 10], fill=(200, 200, 200))
    draw.ellipse([cx + 105, cy - 10, cx + 125, cy + 10], fill=(200, 200, 200))
    # Teeth
    for i in range(-4, 5):
        tx = cx + i * 18
        draw.rectangle([tx - 5, cy + 15, tx + 5, cy + 35], fill=(230, 230, 230))

    bbox = None
    finding_id = "find_normal"
    abnormality = "normal"

    if abnormal != "normal":
        if abnormal == "fracture":
            if anatomy == "maxilla":
                abnormality = "fracture"
                finding_id = "find_maxilla_fracture"
                x1, y1 = cx - 30, cy - 60
            elif anatomy == "zygoma":
                finding_id = "find_zygoma_fracture"
                x1, y1 = cx + 70, cy - 50
            else:
                finding_id = "find_mandible_fracture"
                x1, y1 = cx - 20, cy + 40
            x2, y2 = x1 + 50, y1 + 25
            draw.line([(x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2 + 3)], fill=(30, 30, 30), width=3)
            bbox = [x1, y1, x2, y2]
        elif abnormal == "dislocation":
            finding_id = "find_tmj_dislocation"
            abnormality = "dislocation"
            x1, y1, x2, y2 = cx + 100, cy - 15, cx + 130, cy + 15
            draw.ellipse([x1, y1, x2, y2], outline=(80, 80, 200), width=3)
            bbox = [x1, y1, x2, y2]
        elif abnormal == "impacted_tooth":
            finding_id = "find_impacted_tooth"
            abnormality = "impacted_tooth"
            x1, y1, x2, y2 = cx + 60, cy + 20, cx + 85, cy + 55
            draw.polygon([(x2, y1), (x2 + 10, y2), (x1, y2)], fill=(180, 180, 180))
            bbox = [x1, y1, x2 + 10, y2]
        elif abnormal == "periapical_lesion":
            finding_id = "find_periapical_lesion"
            abnormality = "periapical_lesion"
            x1, y1 = cx - 10, cy + 38
            draw.ellipse([x1 - 8, y1 + 8, x1 + 8, y1 + 24], fill=(100, 100, 100))
            bbox = [x1 - 8, y1 + 8, x1 + 8, y1 + 24]

    return img, bbox, finding_id, abnormality


DIRECT_TEMPLATES = [
    ("Is there a fracture in this maxillofacial X-ray?", "closed", "yes_no"),
    ("Is there evidence of TMJ dislocation?", "closed", "yes_no"),
    ("Is an impacted tooth visible?", "closed", "yes_no"),
    ("What anatomical region is primarily shown?", "closed", "anatomy"),
    ("Is there a periapical lesion?", "closed", "yes_no"),
]

COMPLEX_TEMPLATES = [
    (
        "Based on the image, what treatment approach would be most appropriate?",
        "open",
        "treatment",
    ),
    (
        "Explain how the fracture location affects the choice between closed reduction and ORIF.",
        "open",
        "reasoning",
    ),
    (
        "What follow-up imaging or review timeline is recommended for this finding?",
        "open",
        "followup",
    ),
    (
        "Combine visual findings and anatomy to justify referral to oral maxillofacial surgery.",
        "open",
        "referral",
    ),
]


def build_qa_rows(
    rng: random.Random,
    patient_id: str,
    image_rel: str,
    anatomy: str,
    finding_id: str,
    abnormality: str,
    bbox,
    sample_idx: int,
) -> list[dict]:
    rows = []
    is_abnormal = finding_id != "find_normal"

    # Direct recognition questions
    q1 = f"Is there a {abnormality.replace('_', ' ')} in this {anatomy} X-ray?"
    if abnormality == "normal":
        q1 = f"Is there an abnormality in this {anatomy} region?"
    rows.append(
        {
            "sample_id": f"mf_{sample_idx:04d}_d0",
            "patient_id": patient_id,
            "image_path": image_rel,
            "question": q1,
            "answer": "yes" if is_abnormal else "no",
            "question_type": "closed",
            "answer_type": "yes_no",
            "reasoning_level": "direct_recognition",
            "anatomy": anatomy,
            "abnormality": abnormality,
            "finding_id": finding_id,
            "treatment_gold": TREATMENT_GOLD.get(finding_id, []),
            "bbox": bbox,
            "mask_path": None,
            "synthetic_qa": True,
            "domain": "maxillofacial",
        }
    )
    sample_idx += 1

    rows.append(
        {
            "sample_id": f"mf_{sample_idx:04d}_d1",
            "patient_id": patient_id,
            "image_path": image_rel,
            "question": "What anatomical region is primarily shown?",
            "answer": anatomy,
            "question_type": "closed",
            "answer_type": "anatomy",
            "reasoning_level": "direct_recognition",
            "anatomy": anatomy,
            "abnormality": abnormality,
            "finding_id": finding_id,
            "treatment_gold": TREATMENT_GOLD.get(finding_id, []),
            "bbox": bbox,
            "mask_path": None,
            "synthetic_qa": True,
            "domain": "maxillofacial",
        }
    )
    sample_idx += 1

    # Complex reasoning — treatment
    tx_gold = TREATMENT_GOLD.get(finding_id, ["tx_observation"])
    primary_tx = tx_gold[0] if tx_gold else "tx_observation"
    tx_answer_map = {
        "tx_closed_reduction": "closed reduction with intermaxillary fixation",
        "tx_open_reduction": "open reduction internal fixation (ORIF)",
        "tx_refer_omfs": "referral to oral and maxillofacial surgery",
        "tx_tmj_reduction": "manual TMJ reduction",
        "tx_surgical_extraction": "surgical extraction",
        "tx_endodontic": "endodontic therapy or drainage",
        "tx_observation": "observation with follow-up if symptomatic",
        "tx_soft_diet": "soft diet and analgesia",
    }
    rows.append(
        {
            "sample_id": f"mf_{sample_idx:04d}_c0",
            "patient_id": patient_id,
            "image_path": image_rel,
            "question": "What treatment approach is most appropriate for this finding?",
            "answer": tx_answer_map.get(primary_tx, "observation"),
            "question_type": "open",
            "answer_type": "treatment",
            "reasoning_level": "complex_reasoning",
            "anatomy": anatomy,
            "abnormality": abnormality,
            "finding_id": finding_id,
            "treatment_gold": tx_gold,
            "bbox": bbox,
            "mask_path": None,
            "synthetic_qa": True,
            "domain": "maxillofacial",
        }
    )
    sample_idx += 1

    rows.append(
        {
            "sample_id": f"mf_{sample_idx:04d}_c1",
            "patient_id": patient_id,
            "image_path": image_rel,
            "question": "Explain the clinical reasoning linking the imaging finding to the recommended management.",
            "answer": (
                f"The {anatomy} finding ({abnormality}) maps to KG pathway {finding_id}; "
                f"primary management: {tx_answer_map.get(primary_tx, 'observation')}."
            ),
            "question_type": "open",
            "answer_type": "reasoning",
            "reasoning_level": "complex_reasoning",
            "anatomy": anatomy,
            "abnormality": abnormality,
            "finding_id": finding_id,
            "treatment_gold": tx_gold,
            "bbox": bbox,
            "mask_path": None,
            "synthetic_qa": True,
            "domain": "maxillofacial",
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/maxillofacial_vqa.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    cfg = load_yaml(ROOT / args.config)
    qa = cfg.get("qa", {})
    n_patients = int(qa.get("n_patients", 50))
    images_per_patient = int(qa.get("images_per_patient", 2))
    anatomies = qa.get("anatomies", list(ANATOMY_LABELS.keys()))
    abnormality_pool = qa.get("abnormalities", ["fracture", "normal"])

    img_dir = ensure_dir(ROOT / cfg["paths"]["image_out"])
    ann_path = ROOT / cfg["paths"]["annotation_out"]
    ensure_dir(ann_path.parent)

    rng = random.Random(args.seed)
    all_rows = []
    sample_idx = 0

    for p in range(n_patients):
        patient_id = f"mf_patient_{p:03d}"
        for j in range(images_per_patient):
            anatomy = rng.choice(anatomies)
            if anatomy in {"mandible", "maxilla", "zygoma"}:
                abn = rng.choice(["fracture", "normal", "fracture"])
            elif anatomy == "tmj":
                abn = rng.choice(["dislocation", "normal"])
            elif anatomy == "dental":
                abn = rng.choice(["impacted_tooth", "periapical_lesion", "normal"])
            else:
                abn = rng.choice(abnormality_pool)

            img, bbox, finding_id, abnormality = synthesize_panoramic(rng, anatomy, abn)
            fname = f"{patient_id}_img{j}.png"
            img.save(img_dir / fname)
            rel = str(Path("data/images/maxillofacial") / fname).replace("\\", "/")
            rows = build_qa_rows(rng, patient_id, rel, anatomy, finding_id, abnormality, bbox, sample_idx)
            all_rows.extend(rows)
            sample_idx += len(rows)

    split_cfg = cfg.get("split", {})
    all_rows = patient_level_split(
        all_rows,
        train_ratio=float(split_cfg.get("train_ratio", 0.6)),
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        test_ratio=float(split_cfg.get("test_ratio", 0.2)),
        seed=int(split_cfg.get("seed", args.seed)),
    )
    write_jsonl(ann_path, all_rows)
    n_direct = sum(1 for r in all_rows if r.get("reasoning_level") == "direct_recognition")
    n_complex = sum(1 for r in all_rows if r.get("reasoning_level") == "complex_reasoning")
    print(f"Wrote {len(all_rows)} maxillofacial QA rows to {ann_path}")
    print(f"  direct_recognition: {n_direct}, complex_reasoning: {n_complex}")


if __name__ == "__main__":
    main()
