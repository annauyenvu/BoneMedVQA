# Training Guide

## Stage 0 — data validation

```bash
python scripts/generate_qa.py --config configs/datasets/synthetic_demo.yaml
python scripts/validate_dataset.py --config configs/datasets/synthetic_demo.yaml
```

## Stage 1 — baseline

```bash
python scripts/train_baseline.py --config configs/baseline.yaml
```

## Stage 2–5 — full / lightweight with prompts

```bash
python scripts/precompute_masks.py --config configs/lightweight.yaml
python scripts/train_full.py --config configs/lightweight.yaml
python scripts/train_full.py --config configs/full_model.yaml
```

## Loss weights

Configured under `loss:` in YAML:

- `lambda_classification`
- `lambda_generation`
- `lambda_alignment`
- `lambda_latent`
- `lambda_localization`

## Checkpoints

- `outputs/checkpoints/best.pt` — best validation metric
- `outputs/checkpoints/last.pt` — latest epoch
- Resume via `train.resume` in config
