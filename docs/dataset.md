# Dataset Guide

## Unified schema

```json
{
  "sample_id": "sample_001",
  "patient_id": "patient_001",
  "image_path": "data/images/image_001.png",
  "question": "Is there a fracture?",
  "answer": "yes",
  "question_type": "closed",
  "answer_type": "yes_no",
  "anatomy": "wrist",
  "abnormality": "fracture",
  "mask_path": null,
  "bbox": [x1, y1, x2, y2],
  "split": "train",
  "synthetic_qa": true
}
```

## Supported adapters (license must be checked by user)

| Dataset | Task | Annotations | License note |
|---------|------|-------------|--------------|
| MURA | Abnormality | Study labels | Stanford research terms — accept before download |
| FracAtlas | Fracture | Labels / boxes | Typically CC-BY-4.0 — verify on source page |
| GRAZPEDWRI-DX | Pediatric wrist fracture | Labels / boxes | Typically CC-BY-4.0 — verify; pediatric PHI care |
| VQA-RAD | General Med-VQA | QA pairs | Check HF / original terms |
| SLAKE | Bilingual Med-VQA | QA + masks | Check HF / original terms |
| synthetic_demo | Smoke tests | Generated | Apache-2.0 project-local |

**This repository does not auto-download clinical datasets.**

## Patient-level split

Use `patient_level_split()` — never place the same `patient_id` in more than one of train/val/test.

## Synthetic QA

When QA is template-generated from metadata:

- set `synthetic_qa: true`
- store `template`
- keep a small manually reviewed subset for quality control
- never leak answers via filenames in evaluation reporting
